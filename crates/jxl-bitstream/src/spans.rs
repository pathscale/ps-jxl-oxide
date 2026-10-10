use std::ops::Range;
use std::sync::Arc;

use crate::{BitstreamResult, Error};

#[derive(Clone)]
struct Span {
    buffer: Arc<[u8]>,
    range: Range<usize>,
    logical_start: usize,
}

/// An append-only logical stream backed by immutable physical byte ranges.
///
/// Cloning or slicing copies only descriptors and retains the caller's buffers.
#[derive(Clone, Default)]
pub struct Spans {
    spans: Vec<Span>,
    len: usize,
}

impl std::fmt::Debug for Spans {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Spans")
            .field("len", &self.len)
            .field("ranges", &self.spans.len())
            .finish()
    }
}

impl Spans {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends a physical range, coalescing adjacent ranges of the same buffer.
    pub fn append(&mut self, buffer: Arc<[u8]>, range: Range<usize>) -> BitstreamResult<()> {
        if range.start > range.end || range.end > buffer.len() {
            return Err(Error::ValidationFailed("physical span out of bounds"));
        }
        let added = range.end - range.start;
        let len = self.len.checked_add(added)
            .filter(|len| len.checked_mul(8).is_some())
            .ok_or(Error::ValidationFailed("logical stream length overflow"))?;
        if added == 0 {
            return Ok(());
        }
        if let Some(last) = self.spans.last_mut()
            && Arc::ptr_eq(&last.buffer, &buffer)
            && last.range.end == range.start
        {
            last.range.end = range.end;
            self.len = len;
            return Ok(());
        }
        self.spans.try_reserve(1).map_err(|e| Error::Io(std::io::Error::other(e)))?;
        self.spans.push(Span {
            buffer,
            range,
            logical_start: self.len,
        });
        self.len = len;
        Ok(())
    }

    /// Appends another stream's ranges without copying its bytes.
    pub fn append_spans(&mut self, other: &Self) -> BitstreamResult<()> {
        self.len.checked_add(other.len)
            .filter(|len| len.checked_mul(8).is_some())
            .ok_or(Error::ValidationFailed("logical stream length overflow"))?;
        for span in &other.spans {
            self.append(Arc::clone(&span.buffer), span.range.clone())?;
        }
        Ok(())
    }

    /// Validates and maps a logical range to physical ranges.
    pub fn slice(&self, range: Range<usize>) -> BitstreamResult<Self> {
        if range.start > range.end {
            return Err(Error::ValidationFailed("reversed logical range"));
        }
        if range.end > self.len {
            return Err(Error::Io(std::io::ErrorKind::UnexpectedEof.into()));
        }
        let mut out = Self::new();
        if range.is_empty() {
            return Ok(out);
        }
        let (mut index, _) = self.locate(range.start);
        let mut position = range.start;
        while position < range.end {
            let span = &self.spans[index];
            let physical_start = span.range.start + (position - span.logical_start);
            let count = (span.range.end - physical_start).min(range.end - position);
            out.append(Arc::clone(&span.buffer), physical_start..physical_start + count)?;
            position += count;
            index += 1;
        }
        Ok(out)
    }

    pub fn section(&self, range: Range<usize>) -> BitstreamResult<Section> {
        Section::new(range.start, self.slice(range)?)
    }

    pub fn chunks(&self) -> impl Iterator<Item = &[u8]> {
        self.spans.iter().map(|span| &span.buffer[span.range.clone()])
    }

    pub(crate) fn span_bytes(&self, index: usize) -> Option<&[u8]> {
        self.spans.get(index).map(|span| &span.buffer[span.range.clone()])
    }

    pub(crate) fn locate(&self, position: usize) -> (usize, &[u8]) {
        if position >= self.len {
            return (self.spans.len(), &[]);
        }
        let index = self.spans.partition_point(|span| span.logical_start <= position) - 1;
        let span = &self.spans[index];
        let start = span.range.start + (position - span.logical_start);
        (index, &span.buffer[start..span.range.end])
    }

    /// Reads at most the requested prefix into bounded caller-owned scratch.
    pub(crate) fn copy_prefix(&self, position: usize, output: &mut [u8]) -> usize {
        let (mut index, mut bytes) = self.locate(position);
        let mut written = 0;
        while written < output.len() && !bytes.is_empty() {
            let count = bytes.len().min(output.len() - written);
            output[written..written + count].copy_from_slice(&bytes[..count]);
            written += count;
            index += 1;
            bytes = self.span_bytes(index).unwrap_or(&[]);
        }
        written
    }
}

/// A logical start/length and the physical spans covering that section.
#[derive(Debug, Clone, Default)]
pub struct Section {
    logical_start: usize,
    spans: Spans,
}

impl Section {
    pub fn new(logical_start: usize, spans: Spans) -> BitstreamResult<Self> {
        logical_start.checked_add(spans.len())
            .filter(|end| end.checked_mul(8).is_some())
            .ok_or(Error::ValidationFailed("section extent overflow"))?;
        Ok(Self { logical_start, spans })
    }

    pub fn range(&self) -> Range<usize> {
        self.logical_start..self.logical_start + self.len()
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn spans(&self) -> &Spans {
        &self.spans
    }

    /// Adds a contiguous logical suffix from a validated source stream.
    pub fn append_from(&mut self, source: &Spans, range: Range<usize>) -> BitstreamResult<()> {
        if !self.is_empty() && self.range().end != range.start {
            return Err(Error::ValidationFailed("noncontiguous section append"));
        }
        let suffix = source.slice(range.clone())?;
        let start = if self.is_empty() { range.start } else { self.logical_start };
        start.checked_add(self.len()).and_then(|end| end.checked_add(suffix.len()))
            .filter(|end| end.checked_mul(8).is_some())
            .ok_or(Error::ValidationFailed("section extent overflow"))?;
        self.spans.append_spans(&suffix)?;
        self.logical_start = start;
        Ok(())
    }
}

