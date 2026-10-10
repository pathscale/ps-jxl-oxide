//! Types for JPEG XL container format.

use alloc::sync::Arc;

use crate::{BitstreamResult, Error, Spans};

mod box_header;
mod parse;

pub use box_header::*;
pub use parse::*;

/// Parser that detects the kind of bitstream and emits parser events.
#[derive(Debug, Default)]
pub struct ContainerParser {
    state: DetectState,
    jxlp_index_state: JxlpIndexState,
    source: Spans,
    cursor: usize,
    previous_consumed_bytes: usize,
    failed: bool,
    finalized: bool,
}

#[derive(Debug, Default, Clone)]
enum DetectState {
    #[default]
    WaitingSignature,
    WaitingBoxHeader,
    WaitingJxlpIndex(ContainerBoxHeader),
    InAuxBox {
        header: ContainerBoxHeader,
        brotli_box_type: Option<ContainerBoxType>,
        bytes_left: Option<usize>,
    },
    InCodestream {
        kind: BitstreamKind,
        bytes_left: Option<usize>,
        pending_no_more_aux_box: bool,
    },
}

/// Structure of the decoded bitstream.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BitstreamKind {
    /// Decoder can't determine structure of the bitstream.
    Unknown,
    /// Bitstream is a direct JPEG XL codestream without box structure.
    BareCodestream,
    /// Bitstream is a JPEG XL container with box structure.
    Container,
    /// Bitstream is not a valid JPEG XL image.
    Invalid,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
enum JxlpIndexState {
    #[default]
    Initial,
    SingleJxlc,
    Jxlp(u32),
    JxlpFinished,
}

impl ContainerParser {
    /// Creates a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the kind of this bitstream currently recognized.
    pub fn kind(&self) -> BitstreamKind {
        match self.state {
            DetectState::WaitingSignature => BitstreamKind::Unknown,
            DetectState::WaitingBoxHeader
            | DetectState::WaitingJxlpIndex(..)
            | DetectState::InAuxBox { .. } => BitstreamKind::Container,
            DetectState::InCodestream { kind, .. } => kind,
        }
    }

    /// Appends an owned chunk and receives parser events.
    ///
    /// Exhaust the iterator before feeding another chunk. All input is retained;
    /// incomplete signatures, headers and indexes must not be resubmitted.
    pub fn feed_bytes(&mut self, input: Arc<[u8]>) -> BitstreamResult<ParseEvents<'_>> {
        if self.failed || self.finalized {
            return Err(Error::InvalidBox);
        }
        let len = input.len();
        self.source.append(input, 0..len)?;
        self.previous_consumed_bytes = len;
        Ok(ParseEvents::new(self))
    }

    /// Returns how many bytes were accepted by the previous feed.
    pub fn previous_consumed_bytes(&self) -> usize {
        self.previous_consumed_bytes
    }

    /// Validates that EOF is at a complete container boundary.
    pub fn finalize(&mut self) -> BitstreamResult<()> {
        if self.finalized {
            return Ok(());
        }
        if self.failed {
            return Err(Error::InvalidBox);
        }
        if self.cursor != self.source.len() {
            return Err(Error::UnexpectedEof);
        }
        if matches!(self.jxlp_index_state, JxlpIndexState::Jxlp(_)) {
            return Err(Error::UnexpectedEof);
        }
        let complete = match self.state {
            DetectState::WaitingBoxHeader => {
                !matches!(self.jxlp_index_state, JxlpIndexState::Initial)
            }
            DetectState::InAuxBox {
                ref header,
                brotli_box_type,
                bytes_left: None,
            } => {
                !matches!(self.jxlp_index_state, JxlpIndexState::Initial)
                    && (header.box_type() != ContainerBoxType::BROTLI_COMPRESSED
                        || brotli_box_type.is_some())
            }
            DetectState::InCodestream {
                kind,
                bytes_left: None,
                ..
            } => kind != BitstreamKind::Invalid,
            _ => false,
        };
        if !complete {
            return Err(Error::UnexpectedEof);
        }
        self.finalized = true;
        Ok(())
    }
}
