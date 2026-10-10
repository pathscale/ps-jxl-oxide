use super::box_header::*;
use super::{BitstreamKind, ContainerParser, DetectState, JxlpIndexState};
use crate::consts;
use crate::error::{BitstreamResult, Error};
use crate::Section;

/// Iterator that emits events over retained immutable input ranges.
#[derive(Debug)]
pub struct ParseEvents<'inner> {
    inner: &'inner mut ContainerParser,
    finished: bool,
}

impl<'inner> ParseEvents<'inner> {
    pub(super) fn new(parser: &'inner mut ContainerParser) -> Self {
        Self { inner: parser, finished: false }
    }

    fn take_payload(&mut self, count: usize) -> BitstreamResult<Section> {
        let start = self.inner.cursor;
        let end = start.checked_add(count).ok_or(Error::InvalidBox)?;
        let section = self.inner.source.section(start..end)?;
        self.inner.cursor = end;
        Ok(section)
    }

    fn prefix(&self, output: &mut [u8]) -> usize {
        self.inner.source.copy_prefix(self.inner.cursor, output)
    }

    fn emit_single(&mut self) -> BitstreamResult<Option<ParseEvent>> {
        loop {
            match &mut self.inner.state {
                DetectState::InCodestream { pending_no_more_aux_box: pending @ true, .. } => {
                    *pending = false;
                    return Ok(Some(ParseEvent::NoMoreAuxBox));
                }
                DetectState::InCodestream { bytes_left: Some(0), .. } => {
                    self.inner.state = DetectState::WaitingBoxHeader;
                    continue;
                }
                DetectState::InAuxBox { header, brotli_box_type, bytes_left: Some(0) } => {
                    let ty = brotli_box_type.unwrap_or(header.box_type());
                    self.inner.state = DetectState::WaitingBoxHeader;
                    return Ok(Some(ParseEvent::AuxBoxEnd(ty)));
                }
                _ => {}
            }

            let available = self.inner.source.len() - self.inner.cursor;
            if available == 0 {
                return Ok(None);
            }

            match self.inner.state.clone() {
                DetectState::WaitingSignature => {
                    let mut signature = [0u8; 12];
                    let mut count = self.prefix(&mut signature[..2]);
                    let mut kind = None;
                    if count == 2 && signature[..2] == consts::CODESTREAM_SIG {
                        kind = Some(BitstreamKind::BareCodestream);
                    } else if consts::CONTAINER_SIG.starts_with(&signature[..count]) {
                        count = self.prefix(&mut signature);
                        if signature[..count] == consts::CONTAINER_SIG {
                            self.inner.cursor += consts::CONTAINER_SIG.len();
                            self.inner.state = DetectState::WaitingBoxHeader;
                            return Ok(Some(ParseEvent::BitstreamKind(BitstreamKind::Container)));
                        }
                        if !consts::CONTAINER_SIG.starts_with(&signature[..count]) {
                            kind = Some(BitstreamKind::Invalid);
                        }
                    } else if !consts::CODESTREAM_SIG.starts_with(&signature[..count]) {
                        kind = Some(BitstreamKind::Invalid);
                    }
                    let Some(kind) = kind else { return Ok(None); };
                    self.inner.state = DetectState::InCodestream {
                        kind,
                        bytes_left: None,
                        pending_no_more_aux_box: true,
                    };
                    return Ok(Some(ParseEvent::BitstreamKind(kind)));
                }

                DetectState::WaitingBoxHeader => {
                    let mut bytes = [0u8; 16];
                    let mut count = self.prefix(&mut bytes[..8]);
                    if count == 8 && bytes[..4] == [0, 0, 0, 1] {
                        count = self.prefix(&mut bytes);
                    }
                    let HeaderParseResult::Done { header, header_size } =
                        ContainerBoxHeader::parse(&bytes[..count])?
                    else { return Ok(None); };
                    self.inner.cursor += header_size;
                    let ty = header.box_type();
                    let bytes_left = header.box_size()
                        .map(|size| usize::try_from(size).map_err(|_| Error::InvalidBox))
                        .transpose()?;
                    if ty == ContainerBoxType::CODESTREAM {
                        if self.inner.jxlp_index_state != JxlpIndexState::Initial {
                            return Err(Error::InvalidBox);
                        }
                        self.inner.jxlp_index_state = JxlpIndexState::SingleJxlc;
                        self.inner.state = DetectState::InCodestream {
                            kind: BitstreamKind::Container,
                            bytes_left,
                            pending_no_more_aux_box: bytes_left.is_none(),
                        };
                    } else if ty == ContainerBoxType::PARTIAL_CODESTREAM {
                        if matches!(bytes_left, Some(0..=3)) {
                            return Err(Error::InvalidBox);
                        }
                        self.inner.jxlp_index_state = match self.inner.jxlp_index_state {
                            JxlpIndexState::Initial => JxlpIndexState::Jxlp(0),
                            JxlpIndexState::Jxlp(index) => JxlpIndexState::Jxlp(
                                index.checked_add(1)
                                    .filter(|index| *index <= 0x7fffffff)
                                    .ok_or(Error::InvalidBox)?,
                            ),
                            _ => return Err(Error::InvalidBox),
                        };
                        self.inner.state = DetectState::WaitingJxlpIndex(header);
                    } else {
                        let brotli_compressed = ty == ContainerBoxType::BROTLI_COMPRESSED;
                        if brotli_compressed && matches!(bytes_left, Some(0..=3)) {
                            return Err(Error::InvalidBox);
                        }
                        self.inner.state = DetectState::InAuxBox {
                            header,
                            brotli_box_type: None,
                            bytes_left,
                        };
                        if !brotli_compressed {
                            return Ok(Some(ParseEvent::AuxBoxStart {
                                ty,
                                brotli_compressed: false,
                                last_box: bytes_left.is_none(),
                            }));
                        }
                    }
                }

                DetectState::WaitingJxlpIndex(header) => {
                    let mut bytes = [0u8; 4];
                    if self.prefix(&mut bytes) != 4 {
                        return Ok(None);
                    }
                    let index = u32::from_be_bytes(bytes);
                    let is_last = index & 0x80000000 != 0;
                    let index = index & 0x7fffffff;
                    if self.inner.jxlp_index_state != JxlpIndexState::Jxlp(index) {
                        return Err(Error::InvalidBox);
                    }
                    self.inner.cursor += 4;
                    if is_last {
                        self.inner.jxlp_index_state = JxlpIndexState::JxlpFinished;
                    }
                    let bytes_left = header.box_size()
                        .map(|size| {
                            let size = size.checked_sub(4).ok_or(Error::InvalidBox)?;
                            usize::try_from(size).map_err(|_| Error::InvalidBox)
                        })
                        .transpose()?;
                    self.inner.state = DetectState::InCodestream {
                        kind: BitstreamKind::Container,
                        bytes_left,
                        pending_no_more_aux_box: bytes_left.is_none(),
                    };
                }

                DetectState::InCodestream { kind, bytes_left, pending_no_more_aux_box } => {
                    let count = bytes_left.unwrap_or(available).min(available);
                    let payload = self.take_payload(count)?;
                    self.inner.state = DetectState::InCodestream {
                        kind,
                        bytes_left: bytes_left.map(|left| left - count),
                        pending_no_more_aux_box,
                    };
                    return Ok(Some(ParseEvent::Codestream(payload)));
                }

                DetectState::InAuxBox { header, brotli_box_type: None, bytes_left }
                    if header.box_type() == ContainerBoxType::BROTLI_COMPRESSED =>
                {
                    let mut ty = [0u8; 4];
                    if self.prefix(&mut ty) != 4 {
                        return Ok(None);
                    }
                    if &ty[..3] == b"jxl" || &ty == b"brob" || &ty == b"jbrd" {
                        return Err(Error::ValidationFailed(
                            "brob box, jxl boxes and jbrd box cannot be Brotli-compressed",
                        ));
                    }
                    self.inner.cursor += 4;
                    let ty = ContainerBoxType(ty);
                    self.inner.state = DetectState::InAuxBox {
                        header,
                        brotli_box_type: Some(ty),
                        bytes_left: bytes_left.map(|left| left - 4),
                    };
                    return Ok(Some(ParseEvent::AuxBoxStart {
                        ty,
                        brotli_compressed: true,
                        last_box: bytes_left.is_none(),
                    }));
                }

                DetectState::InAuxBox { header, brotli_box_type, bytes_left } => {
                    let ty = brotli_box_type.unwrap_or(header.box_type());
                    let count = bytes_left.unwrap_or(available).min(available);
                    let payload = self.take_payload(count)?;
                    self.inner.state = DetectState::InAuxBox {
                        header,
                        brotli_box_type,
                        bytes_left: bytes_left.map(|left| left - count),
                    };
                    return Ok(Some(ParseEvent::AuxBoxData(ty, payload)));
                }
            }
        }
    }
}

impl Iterator for ParseEvents<'_> {
    type Item = BitstreamResult<ParseEvent>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.emit_single() {
            Ok(Some(event)) => Some(Ok(event)),
            Ok(None) => {
                self.finished = true;
                None
            }
            Err(error) => {
                self.finished = true;
                self.inner.failed = true;
                Some(Err(error))
            }
        }
    }
}

/// Parser event emitted by [`ParseEvents`].
#[derive(Debug)]
pub enum ParseEvent {
    /// Bitstream structure is detected.
    BitstreamKind(BitstreamKind),
    /// Codestream ranges into the retained source, excluding container bytes.
    Codestream(Section),
    /// No more auxiliary box will come after this event.
    NoMoreAuxBox,
    /// New auxiliary box starts.
    AuxBoxStart {
        /// Type of this box.
        ty: ContainerBoxType,
        /// Indicates whether the payload is Brotli-compressed.
        brotli_compressed: bool,
        /// Indicates whether this box is the last box of the bitstream.
        last_box: bool,
    },
    /// Partial auxiliary payload ranges into the retained source.
    AuxBoxData(ContainerBoxType, Section),
    /// Current auxiliary box has ended.
    AuxBoxEnd(ContainerBoxType),
}

