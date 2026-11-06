//! The scheduler's rules, one at a time, against a counting memory budget.

#![cfg(not(loom))]
#![allow(clippy::unwrap_used)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use qubivex_scheduler::{
    Admission, EnterError, Grant, Limits, MemoryPolicy, Outcome, Priority, RequestInfo, Scheduler,
    Ticket,
};

/// A budget of units.
struct Budget {
    limit: u64,
    used: Arc<AtomicU64>,
}

struct Units(u64, Arc<AtomicU64>);

impl Drop for Units {
    fn drop(&mut self) {
        self.1.fetch_sub(self.0, Ordering::SeqCst);
    }
}

impl Admission for Budget {
    type Hold = Units;
    type Refusal = &'static str;

    fn check(&self, cost: u64) -> Result<(), &'static str> {
        if cost > self.limit {
            Err("never")
        } else {
            Ok(())
        }
    }

    fn try_hold(&self, cost: u64) -> Result<Units, &'static str> {
        if self.used.load(Ordering::SeqCst) + cost > self.limit {
            return Err("not now");
        }
        self.used.fetch_add(cost, Ordering::SeqCst);
        Ok(Units(cost, Arc::clone(&self.used)))
    }
}

type Sched = Scheduler<Budget>;
/// Outcomes in arrival order of the decisions, named by the test.
type Log = Arc<Mutex<Vec<(&'static str, Option<Grant<Budget>>, Option<&'static str>)>>>;

fn limits(max_running: usize, max_queue: usize) -> Limits {
    Limits {
        max_running,
        max_queue,
        aging: None,
        memory: MemoryPolicy::Queue,
    }
}

fn sched(l: Limits, budget: u64) -> (Sched, Arc<AtomicU64>) {
    let used = Arc::new(AtomicU64::new(0));
    let b = Budget {
        limit: budget,
        used: Arc::clone(&used),
    };
    (Scheduler::new(b, l), used)
}

fn info(priority: Priority, client: Option<&str>) -> RequestInfo {
    RequestInfo {
        priority,
        client: client.map(str::to_owned),
    }
}

fn submit(s: &Sched, log: &Log, name: &'static str, i: RequestInfo, cost: u64) -> Ticket<Budget> {
    let t = s.enter(i).unwrap();
    let log = Arc::clone(log);
    t.ready(cost, move |o| {
        let entry = match o {
            Outcome::Granted(g) => (name, Some(g), None),
            Outcome::Refused(r) => (name, None, Some(r)),
            Outcome::Closed => (name, None, Some("closed")),
        };
        log.lock().unwrap().push(entry);
    });
    t
}

fn names(log: &Log) -> Vec<&'static str> {
    log.lock().unwrap().iter().map(|e| e.0).collect()
}

/// Ends the oldest granted request.
fn finish_first(log: &Log) {
    let g = log
        .lock()
        .unwrap()
        .iter_mut()
        .find_map(|e| e.1.take())
        .unwrap();
    drop(g);
}

#[test]
fn running_plus_waiting_never_passes_the_limit() {
    let (s, _) = sched(limits(2, 3), 100);
    let log = Log::default();
    let mut tickets = Vec::new();
    for _ in 0..5 {
        tickets.push(submit(&s, &log, "r", RequestInfo::default(), 1));
    }
    for _ in 0..10 {
        assert_eq!(
            s.enter(RequestInfo::default()).err(),
            Some(EnterError::Full)
        );
    }
    let st = s.stats();
    assert_eq!((st.running, st.queued, st.rejected), (2, 3, 10));
    assert_eq!((st.peak_running, st.granted), (2, 2));
}

#[test]
fn a_place_taken_before_the_cost_is_known_counts_against_the_limit() {
    let (s, _) = sched(limits(1, 0), 100);
    let t = s.enter(RequestInfo::default()).unwrap();
    assert_eq!(
        s.enter(RequestInfo::default()).err(),
        Some(EnterError::Full)
    );
    drop(t);
    assert!(s.enter(RequestInfo::default()).is_ok());
}

#[test]
fn a_finished_request_lets_the_next_one_run() {
    let (s, _) = sched(limits(1, 5), 100);
    let log = Log::default();
    let _a = submit(&s, &log, "a", RequestInfo::default(), 1);
    let _b = submit(&s, &log, "b", RequestInfo::default(), 1);
    assert_eq!(names(&log), ["a"]);
    finish_first(&log);
    assert_eq!(names(&log), ["a", "b"]);
    assert_eq!(s.stats().running, 1);
}

#[test]
fn higher_priorities_run_first_and_arrival_breaks_ties() {
    let (s, _) = sched(limits(1, 5), 100);
    let log = Log::default();
    let _first = submit(&s, &log, "first", RequestInfo::default(), 1);
    let _low = submit(&s, &log, "low", info(Priority::Low, None), 1);
    let _n1 = submit(&s, &log, "n1", RequestInfo::default(), 1);
    let _n2 = submit(&s, &log, "n2", RequestInfo::default(), 1);
    let _high = submit(&s, &log, "high", info(Priority::High, None), 1);
    for _ in 0..4 {
        finish_first(&log);
    }
    assert_eq!(names(&log), ["first", "high", "n1", "n2", "low"]);
}

#[test]
fn a_waiter_ages_past_newer_higher_priorities() {
    let mut l = limits(1, 5);
    l.aging = Some(Duration::from_millis(5));
    let (s, _) = sched(l, 100);
    let log = Log::default();
    let _first = submit(&s, &log, "first", RequestInfo::default(), 1);
    let _low = submit(&s, &log, "low", info(Priority::Low, None), 1);
    std::thread::sleep(Duration::from_millis(15));
    let _high = submit(&s, &log, "high", info(Priority::High, None), 1);
    finish_first(&log);
    assert_eq!(names(&log), ["first", "low"]);
}

#[test]
fn without_aging_a_low_priority_stays_behind() {
    let (s, _) = sched(limits(1, 5), 100);
    let log = Log::default();
    let _first = submit(&s, &log, "first", RequestInfo::default(), 1);
    let _low = submit(&s, &log, "low", info(Priority::Low, None), 1);
    std::thread::sleep(Duration::from_millis(5));
    let _high = submit(&s, &log, "high", info(Priority::High, None), 1);
    finish_first(&log);
    assert_eq!(names(&log), ["first", "high"]);
}

#[test]
fn the_client_with_fewest_running_goes_first() {
    let (s, _) = sched(limits(2, 5), 100);
    let log = Log::default();
    let a = || info(Priority::Normal, Some("a"));
    let b = || info(Priority::Normal, Some("b"));
    let _a1 = submit(&s, &log, "a1", a(), 1);
    let _a2 = submit(&s, &log, "a2", a(), 1);
    let _a3 = submit(&s, &log, "a3", a(), 1);
    let _a4 = submit(&s, &log, "a4", a(), 1);
    let _b1 = submit(&s, &log, "b1", b(), 1);
    finish_first(&log);
    // Client a still runs a2, and client b runs nothing, so b1 passes a3 and a4.
    assert_eq!(names(&log), ["a1", "a2", "b1"]);
}

#[test]
fn a_request_that_could_never_fit_is_refused_at_once() {
    let (s, _) = sched(limits(1, 5), 10);
    let log = Log::default();
    let _a = submit(&s, &log, "a", RequestInfo::default(), 1);
    let _big = submit(&s, &log, "big", RequestInfo::default(), 11);
    let entries = log.lock().unwrap();
    assert_eq!(entries[1].0, "big");
    assert_eq!(entries[1].2, Some("never"));
    drop(entries);
    assert_eq!(s.stats().queued, 0);
}

#[test]
fn a_request_waits_for_memory_and_nothing_overtakes_it() {
    let (s, used) = sched(limits(4, 5), 10);
    let log = Log::default();
    let _a = submit(&s, &log, "a", RequestInfo::default(), 8);
    let _big = submit(&s, &log, "big", RequestInfo::default(), 5);
    // This one fits now, but running it could starve `big` for ever.
    let _small = submit(&s, &log, "small", RequestInfo::default(), 1);
    assert_eq!(names(&log), ["a"]);
    finish_first(&log);
    assert_eq!(names(&log), ["a", "big", "small"]);
    assert_eq!(used.load(Ordering::SeqCst), 6);
}

#[test]
fn the_refuse_policy_refuses_what_does_not_fit_now_and_keeps_going() {
    let mut l = limits(4, 5);
    l.memory = MemoryPolicy::Refuse;
    let (s, _) = sched(l, 10);
    let log = Log::default();
    let _a = submit(&s, &log, "a", RequestInfo::default(), 8);
    let _big = submit(&s, &log, "big", RequestInfo::default(), 5);
    let _small = submit(&s, &log, "small", RequestInfo::default(), 1);
    let entries = log.lock().unwrap();
    assert_eq!(entries[1].2, Some("not now"));
    assert!(entries[2].1.is_some());
}

#[test]
fn a_dropped_ticket_leaves_the_queue_and_lets_others_through() {
    let (s, used) = sched(limits(4, 5), 10);
    let log = Log::default();
    let _a = submit(&s, &log, "a", RequestInfo::default(), 8);
    let big = submit(&s, &log, "big", RequestInfo::default(), 5);
    let _small = submit(&s, &log, "small", RequestInfo::default(), 1);
    assert_eq!(s.stats().queued, 2);
    // A client went away, or its queue timeout passed.
    drop(big);
    assert_eq!(names(&log), ["a", "small"]);
    assert_eq!(used.load(Ordering::SeqCst), 9);
}

#[test]
fn dropping_a_ticket_after_its_grant_changes_nothing() {
    let (s, _) = sched(limits(1, 1), 10);
    let log = Log::default();
    let t = submit(&s, &log, "a", RequestInfo::default(), 1);
    drop(t);
    assert_eq!(s.stats().running, 1);
    finish_first(&log);
    assert_eq!(s.stats().running, 0);
}

#[test]
fn a_grant_nobody_received_frees_itself() {
    let (s, used) = sched(limits(1, 1), 10);
    let t = s.enter(RequestInfo::default()).unwrap();
    // The notification is dropped, as when a host's channel has closed.
    t.ready(3, drop);
    assert_eq!((s.stats().running, used.load(Ordering::SeqCst)), (0, 0));
}

#[test]
fn memory_returns_before_the_next_request_is_tried() {
    let (s, used) = sched(limits(1, 1), 4);
    let log = Log::default();
    let _a = submit(&s, &log, "a", RequestInfo::default(), 4);
    let _b = submit(&s, &log, "b", RequestInfo::default(), 4);
    finish_first(&log);
    assert_eq!(names(&log), ["a", "b"]);
    assert_eq!(used.load(Ordering::SeqCst), 4);
}

#[test]
fn closing_ends_waiters_and_lets_running_requests_finish() {
    let (s, used) = sched(limits(1, 3), 10);
    let log = Log::default();
    let _a = submit(&s, &log, "a", RequestInfo::default(), 1);
    let _b = submit(&s, &log, "b", RequestInfo::default(), 1);
    let unready = s.enter(RequestInfo::default()).unwrap();
    s.close();
    assert_eq!(names(&log), ["a", "b"]);
    assert_eq!(log.lock().unwrap()[1].2, Some("closed"));
    assert_eq!(s.stats().running, 1);
    assert_eq!(
        s.enter(RequestInfo::default()).err(),
        Some(EnterError::Closed)
    );
    // A request that was still working out its cost learns it, too.
    let closed = Arc::new(Mutex::new(false));
    let c = Arc::clone(&closed);
    unready.ready(1, move |o| {
        *c.lock().unwrap() = matches!(o, Outcome::Closed);
    });
    assert!(*closed.lock().unwrap());
    finish_first(&log);
    assert_eq!((s.stats().running, used.load(Ordering::SeqCst)), (0, 0));
}

#[test]
fn a_grant_dropped_by_a_panic_is_released() {
    let (s, used) = sched(limits(1, 1), 10);
    let log = Log::default();
    let _a = submit(&s, &log, "a", RequestInfo::default(), 2);
    let g = log.lock().unwrap()[0].1.take().unwrap();
    let r = std::thread::spawn(move || {
        let _g = g;
        panic!("generation failed");
    })
    .join();
    assert!(r.is_err());
    assert_eq!((s.stats().running, used.load(Ordering::SeqCst)), (0, 0));
}

#[test]
fn many_threads_never_exceed_the_limits_and_everything_comes_back() {
    let (s, used) = sched(limits(3, 4), 10);
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let s = s.clone();
            let used = Arc::clone(&used);
            std::thread::spawn(move || {
                let mut done = 0;
                for i in 0..200_u32 {
                    let Ok(t) = s.enter(RequestInfo::default()) else {
                        continue;
                    };
                    let (tx, rx) = std::sync::mpsc::sync_channel(1);
                    t.ready(u64::from(i % 3) + 1, move |o| {
                        let _ = tx.send(o);
                    });
                    if i % 5 == 0 {
                        drop(t); // Gave up waiting.
                        continue;
                    }
                    if let Outcome::Granted(g) = rx.recv().unwrap() {
                        assert!(used.load(Ordering::SeqCst) <= 10);
                        std::thread::yield_now();
                        drop(g);
                        done += 1;
                    }
                }
                done
            })
        })
        .collect();
    let done: u32 = threads.into_iter().map(|t| t.join().unwrap()).sum();
    assert!(done > 0);
    let st = s.stats();
    assert_eq!((st.running, st.queued), (0, 0));
    assert!(st.peak_running <= 3 && st.peak_queued <= 7, "{st:?}");
    assert_eq!(used.load(Ordering::SeqCst), 0);
}
