//! A tokenizer's vocabulary as a byte trie in preorder, for mask walks.

/// Every token's bytes in a byte trie, laid out in preorder so a mask is one linear
/// pass that skips dead subtrees. End-of-sequence ids are kept apart: they match no
/// bytes and are allowed exactly when the grammar accepts.
#[derive(Debug)]
pub struct TokenTrie {
    vocab: usize,
    /// Per node (node 0 is the root): the byte on the edge into it.
    byte: Box<[u8]>,
    /// Per node: its depth (the root is 0).
    depth: Box<[u32]>,
    /// Per node: the first node after its subtree.
    skip: Box<[u32]>,
    /// Per node `i`: its tokens are `toks[tok_off[i]..tok_off[i + 1]]`.
    tok_off: Box<[u32]>,
    toks: Box<[u32]>,
    max_depth: usize,
    bytes: Box<[Option<Box<[u8]>>]>,
    eos: Box<[u32]>,
}

impl TokenTrie {
    /// A trie over `bytes[id]` for every id (`None`, or empty bytes, for tokens that
    /// never match, such as special tokens). `eos` ids never match bytes.
    #[must_use]
    pub fn new(bytes: Vec<Option<Vec<u8>>>, eos: &[u32]) -> Self {
        let vocab = bytes.len();
        let mut list: Vec<(&[u8], u32)> = bytes
            .iter()
            .zip(0u32..)
            .filter(|&(_, id)| !eos.contains(&id))
            .filter_map(|(b, id)| b.as_deref().filter(|b| !b.is_empty()).map(|b| (b, id)))
            .collect();
        list.sort_unstable();
        let mut byte = vec![0u8];
        let mut depth = vec![0u32];
        let mut owner: Vec<u32> = Vec::with_capacity(list.len());
        let mut path: Vec<u32> = Vec::new();
        let mut prev: &[u8] = &[];
        for &(b, _) in &list {
            let lcp = prev.iter().zip(b).take_while(|(x, y)| x == y).count();
            path.truncate(lcp);
            for (k, &c) in b.iter().enumerate().skip(lcp) {
                path.push(u32::try_from(byte.len()).unwrap_or(u32::MAX));
                byte.push(c);
                depth.push(u32::try_from(k + 1).unwrap_or(u32::MAX));
            }
            owner.push(path.last().copied().unwrap_or(0));
            prev = b;
        }
        let n = byte.len();
        let mut tok_off = vec![0u32; n + 1];
        for &o in &owner {
            tok_off[o as usize + 1] += 1;
        }
        for i in 0..n {
            tok_off[i + 1] += tok_off[i];
        }
        let toks: Vec<u32> = list.iter().map(|&(_, id)| id).collect();
        let mut skip = vec![u32::try_from(n).unwrap_or(u32::MAX); n];
        let mut open: Vec<usize> = Vec::new();
        for j in 0..n {
            while let Some(&t) = open.last() {
                if depth[t] >= depth[j] {
                    skip[t] = u32::try_from(j).unwrap_or(u32::MAX);
                    open.pop();
                } else {
                    break;
                }
            }
            open.push(j);
        }
        let max_depth = depth.iter().copied().max().unwrap_or(0) as usize;
        Self {
            vocab,
            byte: byte.into(),
            depth: depth.into(),
            skip: skip.into(),
            tok_off: tok_off.into(),
            toks: toks.into(),
            max_depth,
            bytes: bytes.into_iter().map(|b| b.map(Into::into)).collect(),
            eos: eos.into(),
        }
    }

    /// Number of token ids (the mask's length in bits).
    #[must_use]
    pub const fn vocab(&self) -> usize {
        self.vocab
    }

    /// A token's bytes (`None` if it never matches).
    #[must_use]
    pub fn bytes(&self, id: u32) -> Option<&[u8]> {
        self.bytes
            .get(id as usize)?
            .as_deref()
            .filter(|b| !b.is_empty())
    }

    /// Whether `id` ends the sequence.
    #[must_use]
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    pub(crate) fn eos(&self) -> &[u32] {
        &self.eos
    }

    pub(crate) const fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub(crate) fn nodes(&self) -> usize {
        self.byte.len()
    }

    /// (byte, depth, skip) of node `i`.
    pub(crate) fn node(&self, i: usize) -> (u8, usize, usize) {
        (self.byte[i], self.depth[i] as usize, self.skip[i] as usize)
    }

    pub(crate) fn tokens_at(&self, i: usize) -> &[u32] {
        &self.toks[self.tok_off[i] as usize..self.tok_off[i + 1] as usize]
    }
}
