//! Model checks of the scheduler's slot and memory accounting. Run with
//! `RUSTFLAGS="--cfg loom" cargo test -p qubivex-scheduler --test loom --release`.

#![cfg(loom)]
#![allow(clippy::unwrap_used)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU64, Ordering};
use qubivex_scheduler::{Admission, Limits, MemoryPolicy, Outcome, RequestInfo, Scheduler};

/// One unit of memory: at most one request can hold it.
struct One(Arc<AtomicU64>);

struct Held(Arc<AtomicU64>);

impl Drop for Held {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Admission for One {
    type Hold = Held;
    type Refusal = ();
    fn check(&self, _: u64) -> Result<(), ()> {
        Ok(())
    }
    fn try_hold(&self, _: u64) -> Result<Held, ()> {
        self.0
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .map(|_| Held(Arc::clone(&self.0)))
            .map_err(|_| ())
    }
}

fn scheduler(max_running: usize, used: &Arc<AtomicU64>) -> Scheduler<One> {
    let limits = Limits {
        max_running,
        max_queue: 2,
        aging: None,
        memory: MemoryPolicy::Queue,
    };
    Scheduler::new(One(Arc::clone(used)), limits)
}

/// Each of two requests runs to its end: neither is lost when one finishes just as the
/// other queues.
#[test]
fn a_finishing_request_never_strands_a_waiter() {
    loom::model(|| {
        let used = Arc::new(AtomicU64::new(0));
        let s = scheduler(1, &used);
        let finished = Arc::new(AtomicU64::new(0));
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let (s, finished) = (s.clone(), Arc::clone(&finished));
                loom::thread::spawn(move || {
                    let t = s.enter(RequestInfo::default()).unwrap();
                    t.ready(1, move |o| {
                        if let Outcome::Granted(g) = o {
                            finished.fetch_add(1, Ordering::SeqCst);
                            drop(g);
                        }
                    });
                    drop(t);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        // A dropped ticket may have left the queue before it ran, but never both
        // waiting and holding on: the scheduler ends empty.
        let st = s.stats();
        assert_eq!((st.running, st.queued), (0, 0));
        assert_eq!(used.load(Ordering::SeqCst), 0);
    });
}

/// With two slots and one unit of memory, the memory is the limit: a waiter runs when
/// a holder gives it back, whichever order they meet the lock in.
#[test]
fn memory_given_back_wakes_a_waiter() {
    loom::model(|| {
        let used = Arc::new(AtomicU64::new(0));
        let s = scheduler(2, &used);
        let first = s.enter(RequestInfo::default()).unwrap();
        let held = Arc::new(loom::sync::Mutex::new(None));
        let h = Arc::clone(&held);
        first.ready(1, move |o| {
            if let Outcome::Granted(g) = o {
                *h.lock().unwrap() = Some(g);
            }
        });
        let second = s.enter(RequestInfo::default()).unwrap();
        let ran = Arc::new(AtomicU64::new(0));
        let r = Arc::clone(&ran);
        let release = loom::thread::spawn(move || drop(held.lock().unwrap().take()));
        second.ready(1, move |o| {
            if matches!(o, Outcome::Granted(_)) {
                r.fetch_add(1, Ordering::SeqCst);
            }
        });
        release.join().unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 1);
        assert_eq!(used.load(Ordering::SeqCst), 0);
        drop((first, second));
    });
}

/// A ticket dropped at the moment it is granted leaks neither slot nor memory.
#[test]
fn a_ticket_dropped_while_granted_leaks_nothing() {
    loom::model(|| {
        let used = Arc::new(AtomicU64::new(0));
        let s = scheduler(1, &used);
        let t = s.enter(RequestInfo::default()).unwrap();
        let (a, b) = (s.clone(), s.clone());
        let enter = loom::thread::spawn(move || {
            if let Ok(t2) = a.enter(RequestInfo::default()) {
                t2.ready(1, drop);
            }
        });
        let leave = loom::thread::spawn(move || {
            t.ready(1, drop);
            drop(t);
            drop(b);
        });
        enter.join().unwrap();
        leave.join().unwrap();
        let st = s.stats();
        assert_eq!((st.running, st.queued), (0, 0));
        assert_eq!(used.load(Ordering::SeqCst), 0);
    });
}
