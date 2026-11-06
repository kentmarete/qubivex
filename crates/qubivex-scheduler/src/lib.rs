//! Decides which requests run, and when. It knows nothing of HTTP, Tokio or models: a
//! host enters a request, says what it costs, and is told the outcome through a
//! callback. An application embedding the runtime uses it the same way the gateway does.
//!
//! - **Bounded.** Running plus waiting never exceeds `max_running + max_queue`. Past
//!   that, [`Scheduler::enter`] refuses at once, so a burst costs no memory.
//! - **Priorities.** Higher priorities run first. A waiter gains one level per
//!   [`Limits::aging`], so low priorities are delayed, not starved.
//! - **Fairness.** Within a level, the client with the fewest running requests goes
//!   first, then the oldest request.
//! - **Memory admission.** [`Admission`] decides what fits. A request that could never
//!   fit is refused without waiting. One that does not fit now waits at the head and
//!   nothing overtakes it, so a large request cannot be starved by small ones
//!   ([`MemoryPolicy::Refuse`] refuses it instead).
//! - **Cleanup.** A [`Ticket`] leaves the queue when dropped, and a [`Grant`] frees its
//!   slot and memory when dropped, on every path, including a panic.
//!
//! There are no timers and no threads. The host enforces its own queue timeout by
//! dropping the ticket. Callbacks run outside the lock, on whichever thread caused the
//! decision, so they must not block.

use std::collections::HashMap;
#[cfg(not(loom))]
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

#[cfg(loom)]
use loom::sync::{Arc, Mutex, MutexGuard};
#[cfg(loom)]
use std::sync::PoisonError;

/// How urgent a request is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Background work.
    Low,
    /// The default.
    #[default]
    Normal,
    /// Goes ahead of everything else.
    High,
}

impl Priority {
    const fn level(self) -> u32 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
        }
    }
}

/// What to do when a request's memory is not available yet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryPolicy {
    /// Wait for running requests to free it.
    #[default]
    Queue,
    /// Refuse the request.
    Refuse,
}

/// A scheduler's limits.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Requests that run at once (at least 1).
    pub max_running: usize,
    /// Requests that may wait beyond those running.
    pub max_queue: usize,
    /// How long a waiter takes to gain one priority level. `None`: never.
    pub aging: Option<Duration>,
    /// What to do when memory is not available yet.
    pub memory: MemoryPolicy,
}

/// Decides whether a request's cost fits. The cost's unit is the host's.
pub trait Admission: Send + Sync + 'static {
    /// What a running request holds. Dropping it returns the room.
    type Hold: Send + 'static;
    /// Why a request does not fit.
    type Refusal: Send + 'static;

    /// Whether `cost` could fit with nothing else running.
    ///
    /// # Errors
    /// The refusal, when it never could.
    fn check(&self, cost: u64) -> Result<(), Self::Refusal>;

    /// Takes room for `cost`, alongside everything already held.
    ///
    /// # Errors
    /// The refusal, when it does not fit now.
    fn try_hold(&self, cost: u64) -> Result<Self::Hold, Self::Refusal>;
}

/// Who is asking and how urgently.
#[derive(Clone, Debug, Default)]
pub struct RequestInfo {
    /// How urgent the request is.
    pub priority: Priority,
    /// The client, for fairness. Requests with no client count as one client.
    pub client: Option<String>,
}

/// Why a request could not enter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnterError {
    /// The limit of running plus waiting requests is reached.
    Full,
    /// The scheduler is closed.
    Closed,
}

/// How a waiting request ended.
pub enum Outcome<A: Admission> {
    /// It may run. Dropping the grant frees its slot and memory.
    Granted(Grant<A>),
    /// Its memory does not fit.
    Refused(A::Refusal),
    /// The scheduler closed while it waited.
    Closed,
}

/// Counts for metrics and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Requests running now.
    pub running: usize,
    /// Requests waiting now.
    pub queued: usize,
    /// Most requests ever running at once.
    pub peak_running: usize,
    /// Most requests ever waiting at once.
    pub peak_queued: usize,
    /// Requests that have been granted.
    pub granted: u64,
    /// Requests refused because the scheduler was full.
    pub rejected: u64,
}

type Notify<A> = Box<dyn FnOnce(Outcome<A>) + Send>;

struct Ready<A: Admission> {
    cost: u64,
    notify: Notify<A>,
}

struct Waiter<A: Admission> {
    id: u64,
    priority: Priority,
    client: String,
    since: Instant,
    /// Set once the host knows the cost. Until then the waiter holds its place only.
    ready: Option<Ready<A>>,
}

struct State<A: Admission> {
    waiters: Vec<Waiter<A>>,
    running: usize,
    /// Running requests per client, with no entry at zero.
    clients: HashMap<String, usize>,
    next_id: u64,
    closed: bool,
    stats: Stats,
}

struct Shared<A: Admission> {
    admission: A,
    limits: Limits,
    state: Mutex<State<A>>,
}

/// Outcomes decided under the lock, delivered after it is released.
type Deliveries<A> = Vec<(Notify<A>, Outcome<A>)>;

fn deliver<A: Admission>(list: Deliveries<A>) {
    for (notify, outcome) in list {
        notify(outcome);
    }
}

impl<A: Admission> Shared<A> {
    fn lock(&self) -> MutexGuard<'_, State<A>> {
        // Every update below leaves the state consistent, so a poisoned lock is usable.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The priority level `w` has reached by `now`.
    fn level(&self, w: &Waiter<A>, now: Instant) -> u32 {
        let top = Priority::High.level();
        let aged = match self.limits.aging {
            Some(a) if !a.is_zero() => {
                let steps = now.saturating_duration_since(w.since).as_nanos() / a.as_nanos();
                u32::try_from(steps).unwrap_or(top)
            }
            _ => 0,
        };
        w.priority.level().saturating_add(aged).min(top)
    }

    /// The index of the waiter to run next, among those whose cost is known.
    fn next_waiter(&self, st: &State<A>, now: Instant) -> Option<usize> {
        let order = |w: &Waiter<A>| {
            let running = st.clients.get(&w.client).copied().unwrap_or(0);
            (std::cmp::Reverse(self.level(w, now)), running, w.id)
        };
        st.waiters
            .iter()
            .enumerate()
            .filter(|(_, w)| w.ready.is_some())
            .min_by_key(|(_, w)| order(w))
            .map(|(i, _)| i)
    }
}

/// Runs waiters while slots and memory allow.
fn dispatch<A: Admission>(this: &Arc<Shared<A>>, st: &mut State<A>) -> Deliveries<A> {
    let mut out: Deliveries<A> = Vec::new();
    let now = Instant::now();
    while st.running < this.limits.max_running.max(1) {
        let Some(i) = this.next_waiter(st, now) else {
            break;
        };
        let cost = st.waiters[i].ready.as_ref().map_or(0, |r| r.cost);
        match this.admission.try_hold(cost) {
            Ok(hold) => {
                let w = st.waiters.remove(i);
                st.running += 1;
                *st.clients.entry(w.client.clone()).or_default() += 1;
                st.stats.granted += 1;
                st.stats.peak_running = st.stats.peak_running.max(st.running);
                if let Some(ready) = w.ready {
                    let grant = Grant {
                        hold: Some(hold),
                        client: w.client,
                        shared: Arc::clone(this),
                    };
                    out.push((ready.notify, Outcome::Granted(grant)));
                }
            }
            Err(refusal) if this.limits.memory == MemoryPolicy::Refuse => {
                if let Some(ready) = st.waiters.remove(i).ready {
                    out.push((ready.notify, Outcome::Refused(refusal)));
                }
            }
            // The head waits for room. Nothing overtakes it, or small requests could
            // starve it for as long as they keep arriving.
            Err(_) => break,
        }
    }
    out
}

/// One model's or one queue's scheduler. Clones share it.
pub struct Scheduler<A: Admission> {
    shared: Arc<Shared<A>>,
}

impl<A: Admission> Clone for Scheduler<A> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<A: Admission> Scheduler<A> {
    /// A scheduler that admits through `admission`.
    #[must_use]
    pub fn new(admission: A, limits: Limits) -> Self {
        Self {
            shared: Arc::new(Shared {
                admission,
                limits,
                state: Mutex::new(State {
                    waiters: Vec::new(),
                    running: 0,
                    clients: HashMap::new(),
                    next_id: 0,
                    closed: false,
                    stats: Stats::default(),
                }),
            }),
        }
    }

    /// Takes a place in the queue, before the host spends anything on the request.
    ///
    /// # Errors
    /// [`EnterError::Full`] past the limit, [`EnterError::Closed`] after [`Self::close`].
    pub fn enter(&self, info: RequestInfo) -> Result<Ticket<A>, EnterError> {
        let limits = &self.shared.limits;
        let mut st = self.shared.lock();
        if st.closed {
            return Err(EnterError::Closed);
        }
        let limit = limits.max_running.max(1).saturating_add(limits.max_queue);
        if st.running + st.waiters.len() >= limit {
            st.stats.rejected += 1;
            return Err(EnterError::Full);
        }
        let id = st.next_id;
        st.next_id += 1;
        st.waiters.push(Waiter {
            id,
            priority: info.priority,
            client: info.client.unwrap_or_default(),
            since: Instant::now(),
            ready: None,
        });
        st.stats.peak_queued = st.stats.peak_queued.max(st.waiters.len());
        drop(st);
        Ok(Ticket {
            id,
            shared: Arc::clone(&self.shared),
        })
    }

    /// Refuses new requests and ends every waiter with [`Outcome::Closed`]. Running
    /// requests carry on.
    pub fn close(&self) {
        let mut st = self.shared.lock();
        st.closed = true;
        let waiters = std::mem::take(&mut st.waiters);
        drop(st);
        for w in waiters {
            if let Some(ready) = w.ready {
                (ready.notify)(Outcome::Closed);
            }
        }
    }

    /// Current counts.
    #[must_use]
    pub fn stats(&self) -> Stats {
        let st = self.shared.lock();
        Stats {
            running: st.running,
            queued: st.waiters.len(),
            ..st.stats
        }
    }

    /// The limits this scheduler runs with.
    #[must_use]
    pub fn limits(&self) -> Limits {
        self.shared.limits
    }
}

/// A place in the queue. Dropping it leaves the queue; that is how a host cancels a
/// waiting request or enforces a queue timeout.
pub struct Ticket<A: Admission> {
    id: u64,
    shared: Arc<Shared<A>>,
}

impl<A: Admission> Ticket<A> {
    /// Says what the request costs. `notify` is called exactly once with the outcome,
    /// unless the ticket is dropped first, on some thread and outside the scheduler's
    /// lock. Call this once: a second call's `notify` may be dropped or told
    /// [`Outcome::Closed`].
    pub fn ready(&self, cost: u64, notify: impl FnOnce(Outcome<A>) + Send + 'static) {
        let notify: Notify<A> = Box::new(notify);
        if let Err(refusal) = self.shared.admission.check(cost) {
            let mut st = self.shared.lock();
            st.waiters.retain(|w| w.id != self.id);
            let out = dispatch(&self.shared, &mut st);
            drop(st);
            notify(Outcome::Refused(refusal));
            deliver(out);
            return;
        }
        let mut st = self.shared.lock();
        let Some(w) = st.waiters.iter_mut().find(|w| w.id == self.id) else {
            // Only `close` removes a waiter this ticket has not.
            drop(st);
            notify(Outcome::Closed);
            return;
        };
        if w.ready.is_some() {
            return;
        }
        w.ready = Some(Ready { cost, notify });
        let out = dispatch(&self.shared, &mut st);
        drop(st);
        deliver(out);
    }
}

impl<A: Admission> Drop for Ticket<A> {
    fn drop(&mut self) {
        let mut st = self.shared.lock();
        let before = st.waiters.len();
        st.waiters.retain(|w| w.id != self.id);
        // Whoever was behind this waiter may be able to run now.
        let out = if st.waiters.len() == before {
            Vec::new()
        } else {
            dispatch(&self.shared, &mut st)
        };
        drop(st);
        deliver(out);
    }
}

/// Permission to run. Dropping it frees the slot and the memory, and runs whoever is
/// next.
pub struct Grant<A: Admission> {
    hold: Option<A::Hold>,
    client: String,
    shared: Arc<Shared<A>>,
}

impl<A: Admission> Grant<A> {
    /// What this request holds.
    #[must_use]
    pub fn hold(&self) -> &A::Hold {
        match &self.hold {
            Some(h) => h,
            // `hold` is only emptied by `drop`.
            None => unreachable!("a grant holds until it is dropped"),
        }
    }
}

impl<A: Admission> Drop for Grant<A> {
    fn drop(&mut self) {
        // The room goes back first, so the next request can use it.
        drop(self.hold.take());
        let mut st = self.shared.lock();
        st.running -= 1;
        if let Some(n) = st.clients.get_mut(&self.client) {
            *n -= 1;
            if *n == 0 {
                st.clients.remove(&self.client);
            }
        }
        let out = dispatch(&self.shared, &mut st);
        drop(st);
        deliver(out);
    }
}
