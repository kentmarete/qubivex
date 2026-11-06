//! Fetching files into a directory: resumable, verified, with bounded concurrency.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha1::Sha1;
use sha2::{Digest, Sha256};

use crate::error::io_err;
use crate::pull::{Progress, PullOptions};
use crate::{Checksum, DownloadError, ModelProvider, RemoteFile, RepoListing, Selection};

const BUFFER: usize = 1 << 20;
/// How often a backoff looks at the cancel flag.
const CANCEL_POLL: Duration = Duration::from_millis(50);

/// Exponential backoff with jitter, honouring a server's `Retry-After` up to the cap.
pub(crate) struct Backoff<'a> {
    opts: &'a PullOptions,
    cancelled: &'a dyn Fn() -> bool,
    attempt: u32,
}

impl<'a> Backoff<'a> {
    pub(crate) fn new(opts: &'a PullOptions, cancelled: &'a dyn Fn() -> bool) -> Self {
        Self {
            opts,
            cancelled,
            attempt: 0,
        }
    }

    fn reset(&mut self) {
        self.attempt = 0;
    }

    /// Waits before the next attempt, or gives up. Anything but
    /// [`DownloadError::Transient`] goes straight back to the caller.
    fn wait(
        &mut self,
        what: &str,
        e: DownloadError,
        report: &dyn Fn(u32, Duration, &str),
    ) -> Result<(), DownloadError> {
        let DownloadError::Transient {
            reason,
            retry_after,
        } = e
        else {
            return Err(e);
        };
        self.attempt += 1;
        if self.attempt >= self.opts.attempts {
            return Err(DownloadError::Exhausted {
                what: what.to_owned(),
                attempts: self.attempt,
                last: reason,
            });
        }
        let doubled = self
            .opts
            .backoff
            .saturating_mul(1 << (self.attempt - 1).min(16));
        let delay = retry_after.unwrap_or(doubled + jitter(doubled));
        let delay = delay.min(self.opts.max_backoff);
        report(self.attempt, delay, &reason);
        let mut left = delay;
        while !left.is_zero() {
            if (self.cancelled)() {
                return Err(DownloadError::Cancelled);
            }
            let step = left.min(CANCEL_POLL);
            std::thread::sleep(step);
            left -= step;
        }
        Ok(())
    }
}

/// Up to a quarter of `of`, so concurrent workers do not retry in step.
fn jitter(of: Duration) -> Duration {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    of.mul_f64(f64::from(nanos % 1000) / 4000.0)
}

/// Runs `f` until it succeeds, fails finally, or attempts run out.
pub(crate) fn retry<T>(
    what: &str,
    opts: &PullOptions,
    cancelled: &dyn Fn() -> bool,
    mut f: impl FnMut() -> Result<T, DownloadError>,
) -> Result<T, DownloadError> {
    let mut backoff = Backoff::new(opts, cancelled);
    loop {
        if cancelled() {
            return Err(DownloadError::Cancelled);
        }
        match f() {
            Err(e) => backoff.wait(what, e, &|_, _, _| {})?,
            ok => return ok,
        }
    }
}

enum Hasher {
    None,
    Sha256(Sha256),
    GitBlob(Sha1),
}

impl Hasher {
    fn new(file: &RemoteFile) -> Self {
        match file.checksum {
            None => Self::None,
            Some(Checksum::Sha256(_)) => Self::Sha256(Sha256::new()),
            Some(Checksum::GitBlobSha1(_)) => {
                let mut h = Sha1::new();
                h.update(format!("blob {}\0", file.size));
                Self::GitBlob(h)
            }
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        match self {
            Self::None => {}
            Self::Sha256(h) => h.update(bytes),
            Self::GitBlob(h) => h.update(bytes),
        }
    }

    fn hex(self) -> String {
        let bytes: Vec<u8> = match self {
            Self::None => Vec::new(),
            Self::Sha256(h) => h.finalize().to_vec(),
            Self::GitBlob(h) => h.finalize().to_vec(),
        };
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}

fn expected(file: &RemoteFile) -> Option<&str> {
    match &file.checksum {
        None => None,
        Some(Checksum::Sha256(h) | Checksum::GitBlobSha1(h)) => Some(h),
    }
}

/// One directory's transfers.
pub(crate) struct Transfer<'a> {
    pub(crate) provider: &'a dyn ModelProvider,
    pub(crate) listing: &'a RepoListing,
    pub(crate) dir: &'a Path,
    pub(crate) opts: &'a PullOptions,
    pub(crate) cancel: &'a AtomicBool,
    pub(crate) progress: &'a (dyn Fn(Progress) + Sync),
    /// Set when one file failed for good, to stop the others.
    pub(crate) failed: AtomicBool,
}

impl Transfer<'_> {
    fn stopped(&self) -> bool {
        self.cancel.load(Ordering::Relaxed) || self.failed.load(Ordering::Relaxed)
    }

    fn check(&self) -> Result<(), DownloadError> {
        if self.stopped() {
            Err(DownloadError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn report(&self, file: &Selection, done: u64) {
        (self.progress)(Progress::File {
            file: file.name.clone(),
            done,
            total: file.remote.size,
        });
    }

    /// Hashes the first bytes of `reader`, returning how many were read.
    fn absorb(
        &self,
        reader: &mut File,
        hasher: &mut Hasher,
        path: &Path,
    ) -> Result<u64, DownloadError> {
        let mut buffer = vec![0; BUFFER];
        let mut total = 0;
        loop {
            self.check()?;
            let n = reader.read(&mut buffer).map_err(io_err(path))?;
            if n == 0 {
                return Ok(total);
            }
            hasher.update(&buffer[..n]);
            total += n as u64;
        }
    }

    /// True if `path` is already the complete, verified file.
    fn complete(&self, path: &Path, file: &RemoteFile) -> Result<bool, DownloadError> {
        match fs::symlink_metadata(path) {
            Ok(m) if m.is_file() && m.len() == file.size => {}
            Ok(_) => return Ok(false),
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(io_err(path)(e)),
        }
        let mut hasher = Hasher::new(file);
        let mut f = File::open(path).map_err(io_err(path))?;
        self.absorb(&mut f, &mut hasher, path)?;
        Ok(expected(file).is_none_or(|e| hasher.hex() == e))
    }

    /// One request's worth of a file, from `*offset`. Read failures are transient: what
    /// arrived stays, and the next request continues after it.
    fn receive(
        &self,
        sel: &Selection,
        part: (&mut File, &Path),
        hasher: &mut Hasher,
        offset: &mut u64,
    ) -> Result<(), DownloadError> {
        let (file, path) = part;
        let size = sel.remote.size;
        let mut opened = self
            .provider
            .open(self.listing, &sel.remote.path, *offset)?;
        if opened.start != *offset {
            // The server ignored the range: start again from its first byte.
            file.set_len(0).map_err(io_err(path))?;
            file.seek(SeekFrom::Start(0)).map_err(io_err(path))?;
            *hasher = Hasher::new(&sel.remote);
            *offset = 0;
            self.report(sel, 0);
        }
        let mut buffer = vec![0; BUFFER];
        while *offset < size {
            self.check()?;
            let want = usize::try_from(size - *offset).map_or(BUFFER, |left| left.min(BUFFER));
            let n = opened.body.read(&mut buffer[..want]).map_err(|e| {
                DownloadError::transient(format!("{}: {}", sel.name, read_failure(&e)))
            })?;
            if n == 0 {
                return Err(DownloadError::transient(format!(
                    "{}: the body ended at {} of {size} bytes",
                    sel.name, *offset
                )));
            }
            file.write_all(&buffer[..n]).map_err(io_err(path))?;
            hasher.update(&buffer[..n]);
            *offset += n as u64;
            self.report(sel, *offset);
        }
        if matches!(opened.body.read(&mut [0; 1]), Ok(n) if n > 0) {
            return Err(DownloadError::TooLarge {
                file: sel.name.clone(),
                size,
            });
        }
        Ok(())
    }

    fn fetch(&self, sel: &Selection) -> Result<(), DownloadError> {
        let dest = self.dir.join(&sel.name);
        let part = self.dir.join(format!("{}.part", sel.name));
        let size = sel.remote.size;
        if self.complete(&dest, &sel.remote)? {
            self.report(sel, size);
            return Ok(());
        }
        match fs::remove_file(&dest) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(io_err(&dest)(e)),
            _ => {}
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&part)
            .map_err(io_err(&part))?;
        if file.metadata().map_err(io_err(&part))?.len() > size {
            file.set_len(0).map_err(io_err(&part))?;
        }
        let mut hasher = Hasher::new(&sel.remote);
        // Resuming hashes what is already here, so the final check covers the whole file.
        let mut offset = self.absorb(&mut file, &mut hasher, &part)?;
        self.report(sel, offset);

        let what = format!("{} in {}", sel.remote.path, self.listing.repository);
        let cancelled = || self.stopped();
        let mut backoff = Backoff::new(self.opts, &cancelled);
        let retrying = |attempt, delay, reason: &str| {
            (self.progress)(Progress::Retrying {
                file: sel.name.clone(),
                attempt,
                delay,
                reason: reason.to_owned(),
            });
        };
        let mut furthest = offset;
        let mut mismatched = false;
        loop {
            self.check()?;
            if offset < size {
                if let Err(e) = self.receive(sel, (&mut file, &part), &mut hasher, &mut offset) {
                    if matches!(e, DownloadError::TooLarge { .. }) {
                        fs::remove_file(&part).map_err(io_err(&part))?;
                    }
                    // A request that moved the file forward is not a failure. Budgets end
                    // requests on healthy links, and a bad link should only give up when
                    // it stops moving, so this continues at once with fresh attempts.
                    if offset > furthest {
                        furthest = offset;
                        backoff.reset();
                        if matches!(e, DownloadError::Transient { .. }) {
                            continue;
                        }
                    }
                    backoff.wait(&what, e, &retrying)?;
                    continue;
                }
            }
            let actual = std::mem::replace(&mut hasher, Hasher::new(&sel.remote)).hex();
            match expected(&sel.remote) {
                Some(want) if want != actual => {
                    // An earlier run's tail can be wrong, so one clean download is allowed.
                    file.set_len(0).map_err(io_err(&part))?;
                    file.seek(SeekFrom::Start(0)).map_err(io_err(&part))?;
                    offset = 0;
                    self.report(sel, 0);
                    if mismatched {
                        fs::remove_file(&part).map_err(io_err(&part))?;
                        return Err(DownloadError::Checksum {
                            file: sel.name.clone(),
                            expected: want.to_owned(),
                            actual,
                        });
                    }
                    mismatched = true;
                }
                _ => break,
            }
        }
        file.sync_all().map_err(io_err(&part))?;
        fs::rename(&part, &dest).map_err(io_err(&dest))?;
        Ok(())
    }

    /// Fetches every file on up to `concurrency` threads. The first final failure stops
    /// the rest; a caller's cancel stops them all.
    pub(crate) fn run(&self, files: &[Selection]) -> Result<(), DownloadError> {
        let mut order: Vec<&Selection> = files.iter().collect();
        // Small files first: a bad token or a broken repository shows up at once.
        order.sort_by_key(|s| s.remote.size);
        let queue = Mutex::new(VecDeque::from(order));
        let first = Mutex::new(None);
        let workers = self.opts.concurrency.clamp(1, files.len().max(1));
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| {
                    loop {
                        let next = queue.lock().map_or(None, |mut q| q.pop_front());
                        let Some(file) = next else { return };
                        if let Err(e) = self.fetch(file) {
                            let cancelled = matches!(e, DownloadError::Cancelled);
                            if let Ok(mut slot) = first.lock() {
                                if slot.is_none() && !cancelled {
                                    *slot = Some(e);
                                }
                            }
                            if !cancelled {
                                self.failed.store(true, Ordering::Relaxed);
                            }
                            return;
                        }
                    }
                });
            }
        });
        if let Some(e) = first.into_inner().ok().flatten() {
            return Err(e);
        }
        if self.cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        Ok(())
    }
}

/// A read failure's kind only: ureq's message for one can name the URL.
fn read_failure(e: &io::Error) -> String {
    format!("connection error: {}", e.kind())
}
