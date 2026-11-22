//! Checkpoints: the condensed note on a conversation's older turns, kept so the next
//! request of the conversation reuses it unchanged (ADR-021 §2.7).
//!
//! A checkpoint is found by a key over the cache scope, the model and the messages it
//! covers, so a different scope, model or history never finds it. The store holds no
//! messages, only the excerpts the note quotes; the client owns its history.
//!
//! Stores never fail a request: a read or write error is reported to the caller, which
//! builds the note from scratch.

use std::collections::HashMap;
use std::fmt;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::Message;

/// One quoted message of a note.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteEntry {
    /// The message's index in the conversation.
    pub message: usize,
    /// Its role.
    pub role: String,
    /// The excerpts, verbatim, as the note prints them.
    pub text: String,
    /// Sentences and code blocks quoted.
    pub excerpts: usize,
    /// Sentences and code blocks the message has.
    pub units: usize,
}

/// A note on the first `covered` messages of a conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Messages the note covers: the conversation's first `covered`.
    pub covered: usize,
    /// Its entries, in message order.
    pub entries: Vec<NoteEntry>,
}

impl Checkpoint {
    fn bytes(&self) -> usize {
        64 + self
            .entries
            .iter()
            .map(|e| 32 + e.role.len() + e.text.len())
            .sum::<usize>()
    }
}

/// A checkpoint's key: SHA-256 over the scope, the namespace and the messages.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Key(pub [u8; 32]);

impl Key {
    /// Lowercase hex.
    #[must_use]
    pub fn hex(&self) -> String {
        use fmt::Write as _;
        let mut s = String::with_capacity(64);
        for b in self.0 {
            let _ = write!(s, "{b:02x}");
        }
        s
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Key({})", &self.hex()[..12])
    }
}

/// The key of every prefix of `messages` up to `upto` messages: element `i` covers the
/// first `i`. Lengths prefix every field, so no two different inputs share a key.
pub(crate) fn prefix_keys(
    namespace: &str,
    scope: Option<&str>,
    messages: &[Message],
    upto: usize,
) -> Vec<Key> {
    let mut h = Sha256::new();
    let field = |h: &mut Sha256, b: &[u8]| {
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    };
    field(&mut h, b"qubivex-context-checkpoint-v1");
    field(&mut h, namespace.as_bytes());
    match scope {
        None => field(&mut h, b"\0none"),
        Some(s) => {
            field(&mut h, b"\0scope");
            field(&mut h, s.as_bytes());
        }
    }
    let mut out = Vec::with_capacity(upto + 1);
    out.push(Key(h.clone().finalize().into()));
    for m in &messages[..upto.min(messages.len())] {
        field(&mut h, m.role.as_bytes());
        field(&mut h, m.content.as_bytes());
        out.push(Key(h.clone().finalize().into()));
    }
    out
}

/// A store failed.
#[derive(Debug, Error)]
#[error("checkpoint store: {0}")]
pub struct StoreError(pub String);

/// Counters of a store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct StoreStats {
    /// Checkpoints held.
    pub entries: usize,
    /// Their approximate size, in bytes.
    pub bytes: u64,
    /// Lookups that found one.
    pub hits: u64,
    /// Lookups that did not.
    pub misses: u64,
    /// Checkpoints removed for room or age.
    pub evicted: u64,
}

/// Where checkpoints are kept. Implementations are shared by every request.
pub trait CheckpointStore: Send + Sync + fmt::Debug {
    /// The checkpoint under `key`, if there is one.
    ///
    /// # Errors
    /// The store could not be read.
    fn get(&self, key: &Key) -> Result<Option<Checkpoint>, StoreError>;
    /// Keeps `cp` under `key`, replacing any.
    ///
    /// # Errors
    /// The store could not be written; it then holds the old checkpoint or none.
    fn put(&self, key: &Key, cp: &Checkpoint) -> Result<(), StoreError>;
    /// Its counters.
    fn stats(&self) -> StoreStats;
}

/// Checkpoints in memory, least recently used evicted past a byte limit.
#[derive(Debug)]
pub struct MemoryStore {
    max_bytes: usize,
    inner: Mutex<MemoryInner>,
}

#[derive(Debug, Default)]
struct MemoryInner {
    map: HashMap<Key, (Checkpoint, usize, u64)>,
    bytes: usize,
    tick: u64,
    hits: u64,
    misses: u64,
    evicted: u64,
}

impl MemoryStore {
    /// The default limit: 64 MiB.
    pub const DEFAULT_BYTES: usize = 64 << 20;

    /// A store holding at most `max_bytes` of checkpoints.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            inner: Mutex::new(MemoryInner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MemoryInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl CheckpointStore for MemoryStore {
    fn get(&self, key: &Key) -> Result<Option<Checkpoint>, StoreError> {
        let mut g = self.lock();
        g.tick += 1;
        let tick = g.tick;
        let found = g.map.get_mut(key).map(|e| {
            e.2 = tick;
            e.0.clone()
        });
        if found.is_some() {
            g.hits += 1;
        } else {
            g.misses += 1;
        }
        Ok(found)
    }

    fn put(&self, key: &Key, cp: &Checkpoint) -> Result<(), StoreError> {
        let size = cp.bytes();
        if size > self.max_bytes {
            return Ok(());
        }
        let mut g = self.lock();
        g.tick += 1;
        let tick = g.tick;
        if let Some(old) = g.map.insert(*key, (cp.clone(), size, tick)) {
            g.bytes -= old.1;
        }
        g.bytes += size;
        while g.bytes > self.max_bytes {
            let Some(oldest) = g.map.iter().min_by_key(|(_, v)| v.2).map(|(k, _)| *k) else {
                break;
            };
            if let Some(v) = g.map.remove(&oldest) {
                g.bytes -= v.1;
                g.evicted += 1;
            }
        }
        Ok(())
    }

    fn stats(&self) -> StoreStats {
        let g = self.lock();
        StoreStats {
            entries: g.map.len(),
            bytes: g.bytes as u64,
            hits: g.hits,
            misses: g.misses,
            evicted: g.evicted,
        }
    }
}

/// Checkpoints as files in a directory, one JSON file each, so they survive a restart.
///
/// A write goes to a temporary file that is then renamed over the old one: a crash
/// leaves the old checkpoint or the new one, never part of one. A file that does not
/// parse is removed and counts as a miss. Files older than the age limit, then the
/// oldest past the size limit, are removed when the store opens and every
/// [`FileStore::SWEEP_EVERY`] writes.
#[derive(Debug)]
pub struct FileStore {
    dir: PathBuf,
    max_bytes: u64,
    max_age: Duration,
    writes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evicted: AtomicU64,
    /// Serialises sweeps; reads and writes do not take it.
    sweeping: Mutex<()>,
}

impl FileStore {
    /// The default size limit: 256 MiB.
    pub const DEFAULT_BYTES: u64 = 256 << 20;
    /// The default age limit: 7 days.
    pub const DEFAULT_AGE: Duration = Duration::from_secs(7 * 24 * 3600);
    /// Writes between sweeps.
    pub const SWEEP_EVERY: u64 = 32;

    /// Opens (creating it if needed) a store in `dir` and applies the limits.
    ///
    /// # Errors
    /// The directory cannot be created or read.
    pub fn open(dir: &Path, max_bytes: u64, max_age: Duration) -> Result<Self, StoreError> {
        std::fs::create_dir_all(dir)
            .map_err(|e| StoreError(format!("cannot create the store directory: {e}")))?;
        // Checkpoints quote conversations: only this user may list or read them.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
                .map_err(|e| StoreError(format!("cannot restrict the store directory: {e}")))?;
        }
        let s = Self {
            dir: dir.to_owned(),
            max_bytes,
            max_age,
            writes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
            sweeping: Mutex::new(()),
        };
        s.sweep()?;
        Ok(s)
    }

    fn path(&self, key: &Key) -> PathBuf {
        self.dir.join(format!("{}.json", key.hex()))
    }

    /// The checkpoint files, with size and modification time.
    fn files(&self) -> Result<Vec<(PathBuf, u64, SystemTime)>, StoreError> {
        let rd = std::fs::read_dir(&self.dir)
            .map_err(|e| StoreError(format!("cannot read the store directory: {e}")))?;
        let mut out = Vec::new();
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            // Leftover temporary files from a crash are removed too.
            let ours = name.ends_with(".json") || name.starts_with(".tmp-");
            if !ours {
                continue;
            }
            if let Ok(m) = e.metadata() {
                let t = m.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                out.push((p, m.len(), t));
            }
        }
        Ok(out)
    }

    /// Applies the age and size limits.
    ///
    /// # Errors
    /// The directory cannot be read.
    pub fn sweep(&self) -> Result<(), StoreError> {
        let _g = self
            .sweeping
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = SystemTime::now();
        let mut files = self.files()?;
        let remove = |p: &Path| {
            if std::fs::remove_file(p).is_ok() {
                self.evicted.fetch_add(1, Ordering::Relaxed);
            }
        };
        files.retain(|(p, _, t)| {
            let old = now.duration_since(*t).unwrap_or_default() >= self.max_age;
            let tmp = p
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(".tmp-"))
                && now.duration_since(*t).unwrap_or_default() > Duration::from_secs(60);
            if old || tmp {
                remove(p);
            }
            !(old || tmp)
        });
        files.sort_by_key(|f| f.2);
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        for (p, len, _) in &files {
            if total <= self.max_bytes {
                break;
            }
            remove(p);
            total = total.saturating_sub(*len);
        }
        Ok(())
    }
}

/// Temporary file names: unique within the process, and the process id across them.
static TMP: AtomicU64 = AtomicU64::new(0);

impl CheckpointStore for FileStore {
    fn get(&self, key: &Key) -> Result<Option<Checkpoint>, StoreError> {
        let p = self.path(key);
        let bytes = match std::fs::read(&p) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            Err(e) => return Err(StoreError(format!("cannot read a checkpoint: {e}"))),
        };
        if let Ok(cp) = serde_json::from_slice::<Checkpoint>(&bytes) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            // Recently used files survive the size limit longest.
            if let Ok(f) = std::fs::File::options().append(true).open(&p) {
                let _ = f.set_modified(SystemTime::now());
            }
            Ok(Some(cp))
        } else {
            let _ = std::fs::remove_file(&p);
            self.misses.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }

    fn put(&self, key: &Key, cp: &Checkpoint) -> Result<(), StoreError> {
        let json = serde_json::to_vec(cp).map_err(|e| StoreError(e.to_string()))?;
        let tmp = self.dir.join(format!(
            ".tmp-{}-{}-{}",
            key.hex(),
            std::process::id(),
            TMP.fetch_add(1, Ordering::Relaxed)
        ));
        let write = || -> std::io::Result<()> {
            let mut o = std::fs::File::options();
            o.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                o.mode(0o600);
            }
            let mut f = o.open(&tmp)?;
            f.write_all(&json)?;
            f.sync_all()?;
            std::fs::rename(&tmp, self.path(key))
        };
        if let Err(e) = write() {
            let _ = std::fs::remove_file(&tmp);
            return Err(StoreError(format!("cannot write a checkpoint: {e}")));
        }
        if self.writes.fetch_add(1, Ordering::Relaxed) % Self::SWEEP_EVERY == Self::SWEEP_EVERY - 1
        {
            self.sweep()?;
        }
        Ok(())
    }

    fn stats(&self) -> StoreStats {
        let files = self.files().unwrap_or_default();
        StoreStats {
            entries: files.len(),
            bytes: files.iter().map(|f| f.1).sum(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evicted: self.evicted.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp(n: usize, text: &str) -> Checkpoint {
        Checkpoint {
            covered: n,
            entries: vec![NoteEntry {
                message: 0,
                role: "user".into(),
                text: text.into(),
                excerpts: 1,
                units: 1,
            }],
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "qubivex-context-{name}-{}-{}",
            std::process::id(),
            TMP.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn keys_separate_scopes_namespaces_and_histories() {
        let m = [Message::new("user", "ab"), Message::new("assistant", "c")];
        let k = prefix_keys("m", None, &m, 2);
        assert_eq!(k.len(), 3);
        assert_ne!(k[1], k[2]);
        assert_ne!(k[2], prefix_keys("m", Some("alice"), &m, 2)[2]);
        assert_ne!(
            prefix_keys("m", Some("alice"), &m, 2)[2],
            prefix_keys("m", Some("bob"), &m, 2)[2]
        );
        assert_ne!(k[2], prefix_keys("other", None, &m, 2)[2]);
        // Field boundaries count: "ab"+"c" is not "a"+"bc".
        let shifted = [Message::new("user", "a"), Message::new("assistant", "bc")];
        assert_ne!(k[2], prefix_keys("m", None, &shifted, 2)[2]);
        // A prefix's key does not depend on what follows.
        assert_eq!(k[1], prefix_keys("m", None, &m[..1], 1)[1]);
    }

    #[test]
    fn the_memory_store_evicts_the_least_recently_used() {
        let one = cp(1, &"x".repeat(100)).bytes();
        let s = MemoryStore::new(one * 2);
        let k = |i: u8| Key([i; 32]);
        s.put(&k(1), &cp(1, &"x".repeat(100))).unwrap();
        s.put(&k(2), &cp(2, &"x".repeat(100))).unwrap();
        assert!(s.get(&k(1)).unwrap().is_some()); // 1 is now the most recent
        s.put(&k(3), &cp(3, &"x".repeat(100))).unwrap();
        assert!(s.get(&k(2)).unwrap().is_none());
        assert_eq!(s.get(&k(1)).unwrap().unwrap().covered, 1);
        assert_eq!(s.get(&k(3)).unwrap().unwrap().covered, 3);
        let st = s.stats();
        assert_eq!((st.entries, st.evicted), (2, 1));
    }

    #[test]
    fn the_file_store_survives_a_reopen_and_drops_corrupt_files() {
        let d = scratch("reopen");
        let k = Key([7; 32]);
        {
            let s = FileStore::open(&d, FileStore::DEFAULT_BYTES, FileStore::DEFAULT_AGE).unwrap();
            s.put(&k, &cp(4, "kept")).unwrap();
        }
        let s = FileStore::open(&d, FileStore::DEFAULT_BYTES, FileStore::DEFAULT_AGE).unwrap();
        assert_eq!(s.get(&k).unwrap(), Some(cp(4, "kept")));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&d), 0o700, "the directory is private");
            assert_eq!(mode(&s.path(&k)), 0o600, "checkpoints are private");
        }
        // A torn file (written by something else) is a miss and goes away.
        let bad = Key([8; 32]);
        std::fs::write(s.path(&bad), b"{\"covered\":").unwrap();
        assert_eq!(s.get(&bad).unwrap(), None);
        assert!(!s.path(&bad).exists());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_file_store_keeps_within_its_size_and_age() {
        let d = scratch("limits");
        let s = FileStore::open(&d, 400, FileStore::DEFAULT_AGE).unwrap();
        for i in 0..10u8 {
            s.put(&Key([i; 32]), &cp(1, &"y".repeat(100))).unwrap();
        }
        s.sweep().unwrap();
        assert!(s.stats().bytes <= 400, "{:?}", s.stats());
        assert!(s.stats().evicted > 0);
        // Everything is older than an age limit of zero.
        let aged = FileStore::open(&d, u64::MAX, Duration::ZERO).unwrap();
        assert_eq!(aged.stats().entries, 0);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
