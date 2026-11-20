use std::fs;
use std::io::{self, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// The line-ending statistics git's `gather_stats` collects, and its
/// binary heuristic: any NUL, a lone CR, or too many non-printables
/// (non-printables exceeding printables over 128). Feeding chunk by chunk
/// keeps the CR carry across chunk boundaries exact.
#[derive(Default)]
pub struct TextStat {
    cr: u64,
    crlf: u64,
    nul: bool,
    printable: u64,
    nonprintable: u64,
    carry_cr: bool,
}

impl TextStat {
    pub fn gather(content: &[u8]) -> Self {
        let mut stats = Self::default();
        stats.feed(content);
        stats
    }

    pub fn feed(&mut self, chunk: &[u8]) {
        for &byte in chunk {
            match byte {
                b'\r' => self.cr += 1,
                b'\n' => {
                    if self.carry_cr {
                        self.crlf += 1;
                    }
                }
                0x7f => self.nonprintable += 1,
                // \b, \t, ESC and FF count as printable, like git.
                0x08 | 0x09 | 0x1b | 0x0c => self.printable += 1,
                byte if byte < 32 => {
                    if byte == 0 {
                        self.nul = true;
                    }
                    self.nonprintable += 1;
                }
                _ => self.printable += 1,
            }
            self.carry_cr = byte == b'\r';
        }
    }

    pub fn is_binary(&self) -> bool {
        let lone_cr = self.cr - self.crlf;
        self.nul || lone_cr > 0 || (self.printable >> 7) < self.nonprintable
    }

    pub fn has_crlf(&self) -> bool {
        self.crlf > 0
    }

    /// The number of CRLF pairs, which is also how many bytes a CRLF-to-LF
    /// conversion removes.
    pub fn crlf_pairs(&self) -> u64 {
        self.crlf
    }
}

/// git's binary heuristic over content.
pub fn is_binary(content: &[u8]) -> bool {
    TextStat::gather(content).is_binary()
}

/// Whether a reference blob suppresses an auto conversion, git's
/// `has_crlf_in_index`: not binary and carrying CRLF (the `i/crlf` and
/// `i/mixed` cases).
pub fn reference_has_crlf(content: &[u8]) -> bool {
    let stats = TextStat::gather(content);
    !stats.is_binary() && stats.has_crlf()
}

/// Strips the CR of every CRLF pair, git's `crlf_to_git` output step.
pub fn crlf_to_git(content: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(content.len());
    let mut previous_cr = false;
    for &byte in content {
        if byte == b'\n' && previous_cr {
            out.pop();
        }
        out.push(byte);
        previous_cr = byte == b'\r';
    }
    out
}

/// A bounded spool for normalized content: `data <n>` needs the exact
/// length up front, so a converting file is materialized — in memory up
/// to [`SPOOL_MEMORY_LIMIT`] bytes, in a temp file above it, preserving
/// the bounded-memory invariant.
pub enum Spool {
    Memory(Vec<u8>),
    Temp {
        file: fs::File,
        path: PathBuf,
        len: u64,
    },
}

/// In-memory spool threshold; per-file opens dominate on Windows
/// (~180 microseconds each), so small files stay in memory.
pub const SPOOL_MEMORY_LIMIT: u64 = 8 * 1024 * 1024;

static SPOOL_COUNTER: AtomicU64 = AtomicU64::new(0);

impl Spool {
    pub fn create() -> io::Result<Self> {
        Ok(Self::Memory(Vec::new()))
    }

    pub fn write(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            Self::Memory(data) => {
                if data.len() as u64 + buf.len() as u64 > SPOOL_MEMORY_LIMIT {
                    // Move the memory out first: the new value replaces
                    // the whole enum, so it cannot borrow from itself.
                    let drained = std::mem::take(data);
                    let mut temp = Self::make_temp_file()?;
                    temp.write(&drained)?;
                    temp.write(buf)?;
                    *self = temp;
                } else {
                    data.extend_from_slice(buf);
                }
                Ok(())
            }
            Self::Temp { file, len, .. } => {
                file.write_all(buf)?;
                *len += buf.len() as u64;
                Ok(())
            }
        }
    }

    fn make_temp_file() -> io::Result<Self> {
        let index = SPOOL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("ext4kit-spool-{}-{index}.tmp", std::process::id()));
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        Ok(Self::Temp { file, path, len: 0 })
    }

    /// The spooled byte count.
    pub fn len(&self) -> u64 {
        match self {
            Self::Memory(data) => data.len() as u64,
            Self::Temp { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Streams the spooled content in chunks to `sink`. The temp file is
    /// reopened for reading; Rust's open shares read access on Windows.
    pub fn for_each_chunk(&self, mut sink: impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        const CHUNK: usize = 512 * 1024;
        match self {
            Self::Memory(data) => {
                for chunk in data.chunks(CHUNK) {
                    sink(chunk)?;
                }
                Ok(())
            }
            Self::Temp { path, len, .. } => {
                let mut reader = BufReader::with_capacity(CHUNK, fs::File::open(path)?);
                let mut remaining = *len;
                let mut buffer = vec![0u8; CHUNK];
                while remaining > 0 {
                    let take = (remaining as usize).min(CHUNK);
                    reader.read_exact(&mut buffer[..take])?;
                    sink(&buffer[..take])?;
                    remaining -= take as u64;
                }
                Ok(())
            }
        }
    }

    /// Streams the spooled content through a CRLF-to-LF conversion to
    /// `sink`. The byte count is the spool length minus the CRLF pair
    /// count the caller already computed for the `data <n>` header, so
    /// the output must be exactly that long: a CRLF pair straddling a
    /// chunk boundary is handled by withholding the trailing CR until
    /// the next chunk's first byte is known — emitting it up front would
    /// push one byte past the framing and let image bytes reach
    /// fast-import's command parser.
    pub fn stream_crlf_to_git(
        &self,
        mut sink: impl FnMut(&[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        // A CR withheld at the end of the previous chunk, not yet emitted.
        let mut withheld_cr = false;
        self.for_each_chunk(|chunk| {
            let mut start = 0;
            if withheld_cr {
                if chunk.first() == Some(&b'\n') {
                    // The withheld CR pairs with this LF: drop it.
                    sink(b"\n")?;
                    start = 1;
                } else {
                    // A lone CR: it stays in the output.
                    sink(b"\r")?;
                }
                withheld_cr = false;
            }
            let mut previous_cr = false;
            for (index, &byte) in chunk.iter().enumerate().skip(start) {
                if byte == b'\n' && previous_cr {
                    // Drop the CR at index - 1, which this chunk still holds.
                    sink(&chunk[start..index - 1])?;
                    sink(b"\n")?;
                    start = index + 1;
                }
                previous_cr = byte == b'\r';
            }
            let mut end = chunk.len();
            if previous_cr && end > start {
                // The chunk's last byte is a CR that may pair with the
                // next chunk's LF: withhold it rather than emit it now.
                withheld_cr = true;
                end -= 1;
            }
            sink(&chunk[start..end])?;
            Ok(())
        })?;
        // A CR at the very end of the content is a lone CR and stays.
        if withheld_cr {
            sink(b"\r")?;
        }
        Ok(())
    }
}

impl Write for Spool {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Spool::write(self, buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Memory(_) => Ok(()),
            Self::Temp { file, .. } => file.flush(),
        }
    }
}

impl Drop for Spool {
    fn drop(&mut self) {
        if let Self::Temp { path, .. } = self {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_detection_follows_gits_heuristic() {
        assert!(is_binary(b"a\0b"));
        assert!(is_binary(b"lone\rcr"));
        assert!(!is_binary(b"line\r\nline\r\n"));
        // One non-printable per 128 printables stays text.
        let mut mostly_text = vec![b'a'; 1280];
        mostly_text[500] = 0x01;
        assert!(!is_binary(&mostly_text));
        let mut too_dirty = vec![b'a'; 128];
        too_dirty[50] = 0x01;
        assert!(is_binary(&too_dirty));
    }

    #[test]
    fn reference_crlf_detection() {
        assert!(reference_has_crlf(b"a\r\nb\r\nc"));
        // Mixed line endings count as carrying CRLF.
        assert!(reference_has_crlf(b"a\r\nb\nc"));
        assert!(!reference_has_crlf(b"a\nb\nc"));
        // A binary reference does not suppress the conversion.
        assert!(!reference_has_crlf(b"a\r\n\0b\r\n"));
    }

    #[test]
    fn crlf_conversion_matches_the_streaming_form() {
        for content in [
            &b""[..],
            b"a",
            b"a\r\nb",
            b"a\rb\r\n",
            b"\r\n\r\n",
            b"ends with cr\r",
            b"cr at chunk end\r\nand more",
        ] {
            assert_eq!(crlf_to_git(content).len() as u64, {
                let stat = TextStat::gather(content);
                content.len() as u64 - stat.crlf_pairs()
            });
            let spool = Spool::Memory(content.to_vec());
            let mut streamed = Vec::new();
            spool
                .stream_crlf_to_git(|chunk| {
                    streamed.extend_from_slice(chunk);
                    Ok(())
                })
                .unwrap();
            assert_eq!(streamed, crlf_to_git(content), "{content:?}");
        }
    }

    #[test]
    fn crlf_pairs_straddling_chunk_boundaries_stay_inside_the_framing() {
        // A CR as the last byte of one 512 KiB chunk with its LF opening
        // the next: without withholding the CR, one byte too many would
        // leave the data <n> block and become fast-import commands. The
        // trailing lone CR exercises the end-of-content flush.
        const CHUNK: usize = 512 * 1024;
        let mut content = vec![b'a'; CHUNK - 1];
        content.push(b'\r');
        content.push(b'\n');
        content.extend_from_slice(b"mid\rdle\r\n");
        content.push(b'\r'); // a lone CR to withhold at the very end
        let expected = crlf_to_git(&content);
        let stat = TextStat::gather(&content);
        assert_eq!(
            expected.len() as u64,
            content.len() as u64 - stat.crlf_pairs()
        );
        let spool = Spool::Memory(content.clone());
        let mut streamed = Vec::new();
        spool
            .stream_crlf_to_git(|chunk| {
                streamed.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(streamed.len(), expected.len());
        assert_eq!(streamed, expected);
    }

    #[test]
    fn the_spool_promotes_to_a_temp_file_above_the_limit() {
        let mut spool = Spool::create().unwrap();
        spool.write(&vec![0u8; 1024]).unwrap();
        assert!(matches!(spool, Spool::Memory(_)));
        assert_eq!(spool.len(), 1024);
        spool
            .write(&vec![0u8; SPOOL_MEMORY_LIMIT as usize])
            .unwrap();
        assert!(matches!(spool, Spool::Temp { .. }));
        // The temp file reads back exactly what was written, in chunks.
        let mut read_back = 0u64;
        spool
            .for_each_chunk(|chunk| {
                read_back += chunk.len() as u64;
                Ok(())
            })
            .unwrap();
        assert_eq!(read_back, SPOOL_MEMORY_LIMIT + 1024);
    }
}
