use std::io::{self, Write};

/// A writer that emits exactly `len` bytes and swallows anything beyond:
/// the framing integrity invariant. The length comes from one read of the
/// inode, and exactly that many bytes reach fast-import's `data <n>`
/// block — a file growing mid-stream must never let agent bytes become
/// fast-import commands. A short input is zero-padded; both directions
/// are reported as a mismatch by [`ExactLen::mismatch`].
pub struct ExactLen<W: Write> {
    inner: W,
    remaining: u64,
    overflowed: bool,
    padded: bool,
}

impl<W: Write> ExactLen<W> {
    pub fn new(inner: W, len: u64) -> Self {
        Self {
            inner,
            remaining: len,
            overflowed: false,
            padded: false,
        }
    }

    /// Zero-pads a short input to the framed length and flushes.
    pub fn finish(&mut self) -> io::Result<()> {
        let zeros = [0u8; 8192];
        while self.remaining > 0 {
            let take = (self.remaining.min(zeros.len() as u64)) as usize;
            self.inner.write_all(&zeros[..take])?;
            self.remaining -= take as u64;
            self.padded = true;
        }
        self.inner.flush()
    }

    /// Whether the input did not match the framed length.
    pub fn mismatch(&self) -> bool {
        self.overflowed || self.padded || self.remaining > 0
    }

    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for ExactLen<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            // Excess input is dropped, not passed on.
            self.overflowed = true;
            return Ok(buf.len());
        }
        let take = (buf.len() as u64).min(self.remaining) as usize;
        self.inner.write_all(&buf[..take])?;
        self.remaining -= take as u64;
        if take < buf.len() {
            self.overflowed = true;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_len_passes_exact_content() {
        let mut out = Vec::new();
        let mismatch = {
            let mut framed = ExactLen::new(&mut out, 5);
            framed.write_all(b"hello").unwrap();
            framed.finish().unwrap();
            framed.mismatch()
        };
        assert_eq!(out, b"hello");
        assert!(!mismatch);
    }

    #[test]
    fn exact_len_drops_excess_bytes() {
        let mut out = Vec::new();
        let mismatch = {
            let mut framed = ExactLen::new(&mut out, 5);
            // The excess is a valid fast-import command sequence: it must
            // never reach the stream.
            framed.write_all(b"hell").unwrap();
            framed.write_all(b"o\nreset refs/heads/main\n").unwrap();
            framed.finish().unwrap();
            framed.mismatch()
        };
        assert_eq!(out, b"hello");
        assert!(mismatch);
    }

    #[test]
    fn exact_len_zero_pads_short_input() {
        let mut out = Vec::new();
        let mismatch = {
            let mut framed = ExactLen::new(&mut out, 8);
            framed.write_all(b"hi").unwrap();
            framed.finish().unwrap();
            framed.mismatch()
        };
        assert_eq!(out, b"hi\0\0\0\0\0\0");
        assert!(mismatch);
    }
}
