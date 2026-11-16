//! This crate provides types related to JPEG XL frames.
//!
//! A JPEG XL image contains one or more frames. A frame represents single unit of image that can
//! be displayed or referenced by other frames.
//!
//! A frame consists of a few components:
//! - [Frame header][FrameHeader].
//! - [Table of contents (TOC)][data::Toc].
//! - Actual frame data, in the following order, potentially permuted as specified in the TOC:
//!   - one [`LfGlobal`],
//!   - [`num_lf_groups`] [`LfGroup`]'s, in raster order,
//!   - one [`HfGlobal`], potentially empty for Modular frames, and
//!   - [`num_passes`] times [`num_groups`] [pass groups][data::decode_pass_group], in raster
//!     order.
//!
//! [`num_lf_groups`]: FrameHeader::num_lf_groups
//! [`num_groups`]: FrameHeader::num_groups
//! [`num_passes`]: header::Passes::num_passes
#![cfg_attr(not(feature = "std"), no_std)]
use alloc::vec::Vec;

extern crate alloc;

mod math;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;

use jxl_bitstream::{Bitstream, Section, Spans};
use jxl_grid::AllocTracker;
use jxl_image::ImageHeader;
use jxl_oxide_common::Bundle;

pub mod data;
mod error;
pub mod filter;
pub mod header;

pub use error::{Error, Result};
pub use header::FrameHeader;
use jxl_modular::Sample;
use jxl_modular::{MaConfig, image::TransformedModularSubimage};

use crate::data::*;

/// JPEG XL frame.
///
/// A frame represents a single unit of image that can be displayed or referenced by other frames.
#[derive(Debug)]
pub struct Frame {
    tracker: Option<AllocTracker>,
    image_header: Arc<ImageHeader>,
    header: FrameHeader,
    toc: Toc,
    header_data: Option<Section>,
    data: Vec<GroupData>,
    reading_data_index: usize,
    pass_shifts: BTreeMap<u32, (i32, i32)>,
}

/// Single-entry section indexing owned by one parser or render coordinator.
/// Keep this state with the same frame and input generation.
#[derive(Debug, Default, Clone)]
pub struct FrameParser {
    lf_group: usize,
    hf_global: usize,
    pass_group: usize,
    has_error: usize,
}

#[derive(Debug)]
struct GroupData {
    toc_group: TocGroup,
    bytes: Section,
}

impl From<TocGroup> for GroupData {
    fn from(value: TocGroup) -> Self {
        Self {
            toc_group: value,
            bytes: Section::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameContext<'a> {
    pub image_header: Arc<ImageHeader>,
    pub tracker: Option<&'a AllocTracker>,
}

impl Bundle<FrameContext<'_>> for Frame {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, ctx: FrameContext) -> Result<Self> {
        let (base_offset, header) = Self::parse_header(bitstream, &ctx.image_header)?;
        let toc = Toc::parse(bitstream, &header)?;
        Self::from_header(bitstream, ctx, base_offset, header, toc)
    }
}

impl Frame {
    fn parse_header(
        bitstream: &mut Bitstream,
        image_header: &ImageHeader,
    ) -> Result<(usize, FrameHeader)> {
        bitstream.zero_pad_to_byte()?;
        let base_offset = bitstream.num_read_bits() / 8;
        let header = FrameHeader::parse(bitstream, image_header)?;

        let width = header.width as u64;
        let height = header.height as u64;
        if width > (1 << 30) {
            tracing::error!(width, "Frame width too large; limit is 2^30");
            return Err(jxl_bitstream::Error::ProfileConformance("frame width too large").into());
        }
        if height > (1 << 30) {
            tracing::error!(width, "Frame height too large; limit is 2^30");
            return Err(jxl_bitstream::Error::ProfileConformance("frame height too large").into());
        }
        if (width * height) > (1 << 40) {
            tracing::error!(
                area = width * height,
                "Frame area (width * height) too large; limit is 2^40"
            );
            return Err(jxl_bitstream::Error::ProfileConformance("frame area too large").into());
        }

        let has_extra = !header.ec_blending_info.is_empty();
        for blending_info in core::iter::once(&header.blending_info).chain(&header.ec_blending_info)
        {
            if blending_info.mode.use_alpha() && has_extra {
                let alpha_idx = blending_info.alpha_channel as usize;
                let Some(alpha_ec_info) = image_header.metadata.ec_info.get(alpha_idx) else {
                    tracing::error!(?blending_info, "blending_info.alpha_channel out of range");
                    return Err(jxl_bitstream::Error::ValidationFailed(
                        "blending_info.alpha_channel out of range",
                    )
                    .into());
                };
                if !alpha_ec_info.is_alpha() {
                    tracing::error!(
                        ?blending_info,
                        ?alpha_ec_info,
                        "blending_info.alpha_channel is not the type of Alpha",
                    );
                    return Err(jxl_bitstream::Error::ValidationFailed(
                        "blending_info.alpha_channel is not the type of Alpha",
                    )
                    .into());
                }
            }
        }

        if header.flags.use_lf_frame() && header.lf_level >= 4 {
            return Err(jxl_bitstream::Error::ValidationFailed("lf_level out of range").into());
        }

        let color_upsampling_shift = header.upsampling.trailing_zeros();
        for (ec_upsampling, ec_info) in header
            .ec_upsampling
            .iter()
            .zip(image_header.metadata.ec_info.iter())
        {
            let ec_upsampling_shift = ec_upsampling.trailing_zeros();
            let dim_shift = ec_info.dim_shift;

            if ec_upsampling_shift + dim_shift < color_upsampling_shift {
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "EC upsampling < color upsampling, which is invalid",
                )
                .into());
            }

            if ec_upsampling_shift + dim_shift > 6 {
                tracing::error!(
                    ec_upsampling,
                    dim_shift = ec_info.dim_shift,
                    "Cumulative EC upsampling factor is too large"
                );
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "cumulative EC upsampling factor is too large",
                )
                .into());
            }

            let actual_dim_shift = ec_upsampling_shift + dim_shift - color_upsampling_shift;

            if actual_dim_shift > 7 + header.group_size_shift {
                return Err(jxl_bitstream::Error::ValidationFailed("dim_shift too large").into());
            }
        }

        if header.width == 0 || header.height == 0 {
            return Err(jxl_bitstream::Error::ValidationFailed(
                "Invalid crop dimensions for frame: zero width or height",
            )
            .into());
        }

        Ok((base_offset, header))
    }

    fn from_header(
        bitstream: &Bitstream,
        ctx: FrameContext<'_>,
        base_offset: usize,
        header: FrameHeader,
        mut toc: Toc,
    ) -> Result<Self> {
        let FrameContext {
            image_header,
            tracker,
        } = ctx;
        let tracker = tracker.cloned();
        toc.adjust_offsets(base_offset)?;
        let header_data = bitstream
            .source_section(base_offset..bitstream.num_read_bits() / 8)
            .transpose()?;
        let data = toc.iter_bitstream_order().map(GroupData::from).collect();

        let passes = &header.passes;
        let mut pass_shifts = BTreeMap::new();
        let mut maxshift = 3i32;
        for (&downsample, &last_pass) in passes.downsample.iter().zip(&passes.last_pass) {
            let minshift = downsample.trailing_zeros() as i32;
            pass_shifts.insert(last_pass, (minshift, maxshift));
            maxshift = minshift;
        }
        pass_shifts.insert(header.passes.num_passes - 1, (0i32, maxshift));

        Ok(Self {
            tracker,
            image_header,
            header,
            toc,
            header_data,
            data,
            reading_data_index: 0,
            pass_shifts,
        })
    }
}

/// Resumes a frame header and TOC over an append-only codestream.
#[derive(Debug, Default)]
pub struct FrameHeaderParser {
    toc: Option<(FrameHeader, TocParser)>,
    next_bit: usize,
    retry_at: usize,
    #[cfg(test)]
    header_attempts: usize,
}

impl FrameHeaderParser {
    /// Returns a complete frame and its absolute header end, or waits for more bytes.
    /// Resume with the same frame start, context and append-only source until completion.
    pub fn parse(
        &mut self,
        source: &Spans,
        frame_start: usize,
        ctx: FrameContext<'_>,
    ) -> Result<Option<(Frame, usize)>> {
        let available = source.len().saturating_sub(frame_start);
        if available == 0 || available < self.retry_at {
            return Ok(None);
        }
        let mut bitstream = Bitstream::from_spans_at(source, frame_start)?;
        if self.toc.is_none() {
            #[cfg(test)]
            {
                self.header_attempts += 1;
            }
            // Ordinary fields are format-bounded. Incomplete name and extension
            // payloads supply the exact minimum extent needed to try again.
            let header = match Frame::parse_header(&mut bitstream, &ctx.image_header) {
                Ok((_, header)) => header,
                Err(error) if error.unexpected_eof() => {
                    self.retry_at = bitstream.required_byte_len().max(available + 1);
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            self.next_bit = bitstream.num_read_bits();
            let toc = TocParser::new(&header)?;
            self.toc = Some((header, toc));
        } else {
            bitstream.skip_bits(self.next_bit)?;
        }

        let (header, toc) = self.toc.as_mut().unwrap();
        let toc = match toc.parse(&mut bitstream, header) {
            Ok(toc) => toc,
            Err(error) if error.unexpected_eof() => {
                self.next_bit = bitstream.num_read_bits();
                self.retry_at = available + 1;
                return Ok(None);
            }
            Err(error) => {
                *self = Self::default();
                return Err(error);
            }
        };
        let (header, _) = self.toc.take().unwrap();
        self.next_bit = 0;
        self.retry_at = 0;
        let frame = Frame::from_header(&bitstream, ctx, 0, header, toc)?;
        Ok(Some((frame, frame_start + bitstream.num_read_bits() / 8)))
    }
}

impl Frame {
    #[inline]
    pub fn alloc_tracker(&self) -> Option<&AllocTracker> {
        self.tracker.as_ref()
    }

    pub fn image_header(&self) -> &ImageHeader {
        &self.image_header
    }

    pub fn clone_image_header(&self) -> Arc<ImageHeader> {
        Arc::clone(&self.image_header)
    }

    /// Returns the frame header.
    pub fn header(&self) -> &FrameHeader {
        &self.header
    }

    /// Returns the TOC.
    ///
    /// See the documentation of [`Toc`] for details.
    pub fn toc(&self) -> &Toc {
        &self.toc
    }

    pub fn pass_shifts(&self) -> &BTreeMap<u32, (i32, i32)> {
        &self.pass_shifts
    }

    /// Compressed frame header and TOC ranges, when parsed from spans.
    pub fn header_data(&self) -> Option<&Section> {
        self.header_data.as_ref()
    }

    pub fn data(&self, group: TocGroupKind) -> Option<&Section> {
        let idx = self.toc.group_index_bitstream_order(group);
        self.data.get(idx).map(|b| &b.bytes)
    }
}

impl Frame {
    /// Retains available ranges and returns the number of logical bytes consumed.
    pub fn feed_spans(&mut self, source: &Spans, range: core::ops::Range<usize>) -> Result<usize> {
        if range.start > range.end || range.end > source.len() {
            return Err(
                jxl_bitstream::Error::ValidationFailed("frame input range out of bounds").into(),
            );
        }
        let mut position = range.start;
        while let Some(group_data) = self.data.get_mut(self.reading_data_index) {
            let size = usize::try_from(group_data.toc_group.size)
                .map_err(|_| jxl_bitstream::Error::ValidationFailed("group size overflow"))?;
            let bytes_left = size.checked_sub(group_data.bytes.len()).ok_or(
                jxl_bitstream::Error::ValidationFailed("group data exceeds TOC size"),
            )?;
            let count = bytes_left.min(range.end - position);
            group_data
                .bytes
                .append_from(source, position..position + count)?;
            position += count;
            if count < bytes_left {
                break;
            }
            self.reading_data_index += 1;
        }
        Ok(position - range.start)
    }

    #[inline]
    pub fn current_loading_group(&self) -> Option<TocGroup> {
        self.toc.iter_bitstream_order().nth(self.reading_data_index)
    }

    #[inline]
    pub fn is_loading_done(&self) -> bool {
        self.reading_data_index >= self.data.len()
    }
}

impl Frame {
    pub fn try_parse_lf_global<S: Sample>(&self) -> Option<Result<LfGlobal<S>>> {
        self.try_parse_lf_global_with_parser(&mut FrameParser::default())
    }

    pub fn try_parse_lf_global_with_parser<S: Sample>(
        &self,
        parser: &mut FrameParser,
    ) -> Option<Result<LfGlobal<S>>> {
        Some(if self.toc.is_single_entry() {
            if parser.has_error != 0 {
                return Some(Err(Error::HadError));
            }

            let group = self.data.first()?;
            let loaded = self.reading_data_index != 0;
            let mut bitstream = Bitstream::from_section(&group.bytes);
            let lf_global = LfGlobal::parse(
                &mut bitstream,
                LfGlobalParams::new(
                    &self.image_header,
                    &self.header,
                    self.tracker.as_ref(),
                    !loaded,
                ),
            );
            match lf_global {
                Ok(lf_global) => {
                    parser.lf_group = bitstream.num_read_bits();
                    Ok(lf_global)
                }
                Err(e) if !loaded && e.unexpected_eof() => Err(e),
                Err(e) => {
                    parser.has_error = 1;
                    Err(e)
                }
            }
        } else {
            let idx = self.toc.group_index_bitstream_order(TocGroupKind::LfGlobal);
            let group = self.data.get(idx)?;
            let allow_partial = group.bytes.len() < group.toc_group.size as usize;

            let mut bitstream = Bitstream::from_section(&group.bytes);
            LfGlobal::parse(
                &mut bitstream,
                LfGlobalParams::new(
                    &self.image_header,
                    &self.header,
                    self.tracker.as_ref(),
                    allow_partial,
                ),
            )
        })
    }

    pub fn try_parse_lf_group<S: Sample>(
        &self,
        lf_global_vardct: Option<&LfGlobalVarDct>,
        global_ma_config: Option<&MaConfig>,
        mlf_group: Option<TransformedModularSubimage<S>>,
        lf_group_idx: u32,
    ) -> Option<Result<LfGroup<S>>> {
        self.try_parse_lf_group_with_parser(
            lf_global_vardct,
            global_ma_config,
            mlf_group,
            lf_group_idx,
            &mut FrameParser::default(),
        )
    }

    pub fn try_parse_lf_group_with_parser<S: Sample>(
        &self,
        lf_global_vardct: Option<&LfGlobalVarDct>,
        global_ma_config: Option<&MaConfig>,
        mlf_group: Option<TransformedModularSubimage<S>>,
        lf_group_idx: u32,
        parser: &mut FrameParser,
    ) -> Option<Result<LfGroup<S>>> {
        if self.toc.is_single_entry() {
            if parser.has_error != 0 {
                return Some(Err(Error::HadError));
            }

            if lf_group_idx != 0 {
                return None;
            }

            let group = self.data.first()?;
            let loaded = self.reading_data_index != 0;
            let mut bitstream = Bitstream::from_section(&group.bytes);
            let offset = parser.lf_group;
            if offset == 0 {
                let lf_global = self.try_parse_lf_global_with_parser::<S>(parser).unwrap();
                if let Err(e) = lf_global {
                    return Some(Err(e));
                }
            }
            let offset = parser.lf_group;
            if let Err(error) = bitstream.skip_bits(offset) {
                return Some(Err(error.into()));
            }

            let result = LfGroup::parse(
                &mut bitstream,
                LfGroupParams {
                    frame_header: &self.header,
                    quantizer: lf_global_vardct.map(|x| &x.quantizer),
                    global_ma_config,
                    mlf_group,
                    lf_group_idx,
                    allow_partial: !loaded,
                    tracker: self.tracker.as_ref(),
                },
            );

            match result {
                Ok(result) => {
                    parser.hf_global = bitstream.num_read_bits();
                    Some(Ok(result))
                }
                Err(e) if !loaded && e.unexpected_eof() => None,
                Err(e) => {
                    parser.has_error = 2;
                    Some(Err(e))
                }
            }
        } else {
            let idx = self
                .toc
                .group_index_bitstream_order(TocGroupKind::LfGroup(lf_group_idx));
            let group = self.data.get(idx)?;
            let allow_partial = group.bytes.len() < group.toc_group.size as usize;

            let mut bitstream = Bitstream::from_section(&group.bytes);
            let result = LfGroup::parse(
                &mut bitstream,
                LfGroupParams {
                    frame_header: &self.header,
                    quantizer: lf_global_vardct.map(|x| &x.quantizer),
                    global_ma_config,
                    mlf_group,
                    lf_group_idx,
                    allow_partial,
                    tracker: self.tracker.as_ref(),
                },
            );
            if allow_partial && result.is_err() {
                return None;
            }
            Some(result)
        }
    }

    pub fn try_parse_hf_global<S: Sample>(
        &self,
        cached_lf_global: Option<&LfGlobal<S>>,
    ) -> Option<Result<HfGlobal>> {
        self.try_parse_hf_global_with_parser(cached_lf_global, &mut FrameParser::default())
    }

    pub fn try_parse_hf_global_with_parser<S: Sample>(
        &self,
        cached_lf_global: Option<&LfGlobal<S>>,
        parser: &mut FrameParser,
    ) -> Option<Result<HfGlobal>> {
        let is_modular = self.header.encoding == header::Encoding::Modular;

        if self.toc.is_single_entry() {
            if parser.has_error != 0 {
                return Some(Err(Error::HadError));
            }

            let group = self.data.first()?;
            let loaded = self.reading_data_index != 0;
            let mut bitstream = Bitstream::from_section(&group.bytes);
            let offset = parser.hf_global;
            let lf_global = if cached_lf_global.is_none() && (offset == 0 || !is_modular) {
                match self.try_parse_lf_global_with_parser(parser)? {
                    Ok(lf_global) => Some(lf_global),
                    Err(e) => return Some(Err(e)),
                }
            } else {
                None
            };
            let lf_global = cached_lf_global.or(lf_global.as_ref());

            if offset == 0 {
                let lf_global = lf_global.unwrap();
                let mut gmodular = match lf_global.gmodular.try_clone() {
                    Ok(gmodular) => gmodular,
                    Err(e) => return Some(Err(e)),
                };
                let groups = gmodular
                    .modular
                    .image_mut()
                    .map(|x| x.prepare_groups(&self.pass_shifts))
                    .transpose();
                let groups = match groups {
                    Ok(groups) => groups,
                    Err(e) => return Some(Err(e.into())),
                };
                let mlf_group = groups.and_then(|mut x| x.lf_groups.pop());
                let lf_group = self
                    .try_parse_lf_group_with_parser(
                        lf_global.vardct.as_ref(),
                        lf_global.gmodular.ma_config(),
                        mlf_group,
                        0,
                        parser,
                    )
                    .ok_or(jxl_bitstream::Error::UnexpectedEof.into())
                    .and_then(|x| x);
                if let Err(e) = lf_group {
                    return Some(Err(e));
                }
            }
            let offset = parser.hf_global;

            if self.header.encoding == header::Encoding::Modular {
                parser.pass_group = offset;
                return None;
            }

            if let Err(error) = bitstream.skip_bits(offset) {
                return Some(Err(error.into()));
            }
            let lf_global = lf_global.unwrap();
            let result = HfGlobal::parse(
                &mut bitstream,
                HfGlobalParams::new(
                    &self.image_header.metadata,
                    &self.header,
                    lf_global,
                    self.tracker.as_ref(),
                ),
            );

            Some(match result {
                Ok(result) => {
                    parser.pass_group = bitstream.num_read_bits();
                    Ok(result)
                }
                Err(e) if !loaded && e.unexpected_eof() => Err(e),
                Err(e) => {
                    parser.has_error = 3;
                    Err(e)
                }
            })
        } else {
            if self.header.encoding == header::Encoding::Modular {
                return None;
            }

            let idx = self.toc.group_index_bitstream_order(TocGroupKind::HfGlobal);
            let group = self.data.get(idx)?;
            if group.bytes.len() < group.toc_group.size as usize {
                return None;
            }

            let mut bitstream = Bitstream::from_section(&group.bytes);
            let lf_global = if cached_lf_global.is_none() {
                match self.try_parse_lf_global_with_parser(parser)? {
                    Ok(lf_global) => Some(lf_global),
                    Err(e) => return Some(Err(e)),
                }
            } else {
                None
            };
            let lf_global = cached_lf_global.or(lf_global.as_ref()).unwrap();
            let params = HfGlobalParams::new(
                &self.image_header.metadata,
                &self.header,
                lf_global,
                self.tracker.as_ref(),
            );
            Some(HfGlobal::parse(&mut bitstream, params))
        }
    }

    pub fn pass_group_bitstream(
        &self,
        pass_idx: u32,
        group_idx: u32,
    ) -> Option<Result<PassGroupBitstream<'_>>> {
        self.pass_group_bitstream_with_parser(pass_idx, group_idx, &mut FrameParser::default())
    }

    pub fn pass_group_bitstream_with_parser(
        &self,
        pass_idx: u32,
        group_idx: u32,
        parser: &mut FrameParser,
    ) -> Option<Result<PassGroupBitstream<'_>>> {
        Some(if self.toc.is_single_entry() {
            if parser.has_error != 0 {
                return Some(Err(Error::HadError));
            }

            if pass_idx != 0 || group_idx != 0 {
                return None;
            }

            let group = self.data.first()?;
            let loaded = self.reading_data_index != 0;
            let mut bitstream = Bitstream::from_section(&group.bytes);
            let mut offset = parser.pass_group;
            if offset == 0 {
                if let Some(Err(e)) = self.try_parse_hf_global_with_parser::<i32>(None, parser) {
                    return Some(Err(e));
                }
                offset = parser.pass_group;
                if offset == 0 {
                    return None;
                }
            }
            if let Err(error) = bitstream.skip_bits(offset) {
                return Some(Err(error.into()));
            }

            Ok(PassGroupBitstream {
                bitstream,
                partial: !loaded,
            })
        } else {
            let idx = self
                .toc
                .group_index_bitstream_order(TocGroupKind::GroupPass {
                    pass_idx,
                    group_idx,
                });
            let group = self.data.get(idx)?;
            let partial = group.bytes.len() < group.toc_group.size as usize;

            let bitstream = Bitstream::from_section(&group.bytes);
            Ok(PassGroupBitstream { bitstream, partial })
        })
    }
}

#[derive(Debug)]
pub struct PassGroupBitstream<'buf> {
    pub bitstream: Bitstream<'buf>,
    pub partial: bool,
}

impl Frame {
    /// Adjusts the cropping region of the image to the actual decoding region of the frame.
    ///
    /// The cropping region of the *image* needs to be adjusted to be used in a *frame*, for a few
    /// reasons:
    /// - A frame may be blended to the canvas with offset, which makes the image and the frame
    ///   have different coordinates.
    /// - Some filters reference other samples, which requires padding to the region.
    ///
    /// This method takes care of those and adjusts the given region appropriately.
    pub fn adjust_region(&self, (left, top, width, height): &mut (u32, u32, u32, u32)) {
        if self.header.have_crop {
            *left = left.saturating_add_signed(-self.header.x0);
            *top = top.saturating_add_signed(-self.header.y0);
        };

        let mut padding = 0u32;
        if self.header.restoration_filter.gab.enabled() {
            tracing::debug!("Gabor-like filter requires padding of 1 pixel");
            padding = 1;
        }
        if self.header.restoration_filter.epf.enabled() {
            tracing::debug!("Edge-preserving filter requires padding of 3 pixels");
            padding = 3;
        }
        if padding > 0 {
            let delta_w = (*left).min(padding);
            let delta_h = (*top).min(padding);
            *left -= delta_w;
            *top -= delta_h;
            *width += delta_w + padding;
            *height += delta_h + padding;
        }
    }
}

#[cfg(test)]
mod frame_header_retry_tests {
    use super::*;

    #[test]
    fn frame_header_retry_preserves_one_shot_toc_eof() {
        const IMAGE_HEADER: [u8; 5] = [0xff, 0x0a, 0xf8, 0x9f, 0x01];
        let context = FrameContext {
            image_header: Arc::new(ImageHeader::parse(&mut Bitstream::new(&IMAGE_HEADER), ()).unwrap()),
            tracker: None,
        };
        let mut bits = Vec::new();
        let mut write = |value: u32, width: usize| {
            bits.extend((0..width).map(|bit| ((value >> bit) & 1) as u8));
        };
        write(0b001011, 6); // Default frame, permuted TOC, no LZ77, simple one-cluster map.
        write(1, 1); // Prefix coding.
        write(1, 6); // Hybrid integer split exponent 1, no token bits.
        write(7, 8); // Nine-symbol alphabet.
        write(0b1001, 4); // Simple histogram with three symbols.
        write(4, 4); // Token 4 has code 0.
        write(0, 4); // Token 0 has code 10.
        write(3, 4); // Token 3 has code 11.
        write(3, 2); // End token 3, followed by two hybrid bits.
        write(0, 2); // End value 4. The split falls between these two bits.
        for _ in 0..4 {
            write(1, 2); // Token 0, yielding identity Lehmer digits.
        }
        bits.resize(bits.len().next_multiple_of(8), 0);
        bits.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // First group size 1.
        bits.resize((bits.len() + 6 * 12).next_multiple_of(8), 0);
        let bytes: Arc<[u8]> = Arc::from(
            IMAGE_HEADER
                .into_iter()
                .chain(bits.chunks_exact(8).map(|byte| {
                    byte.iter().enumerate().fold(0, |acc, (i, bit)| acc | (*bit << i))
                }))
                .collect::<Vec<_>>(),
        );
        let frame_start = IMAGE_HEADER.len();
        let split = frame_start + 5;
        let mut source = Spans::new();
        source.append(Arc::clone(&bytes), 0..split).unwrap();
        let mut short = Bitstream::from_spans(&source);
        short.skip_bits(frame_start * 8).unwrap();
        let error = Frame::parse(&mut short, context.clone()).unwrap_err();
        assert!(error.unexpected_eof(), "{error:?}");
        let mut parser = FrameHeaderParser::default();
        assert!(parser.parse(&source, frame_start, context.clone()).unwrap().is_none());

        source.append(Arc::clone(&bytes), split..bytes.len()).unwrap();
        let (resumed, end) = parser.parse(&source, frame_start, context.clone()).unwrap().unwrap();
        let mut input = Bitstream::from_spans(&source);
        input.skip_bits(frame_start * 8).unwrap();
        let whole = Frame::parse(&mut input, context).unwrap();
        assert_eq!(end, bytes.len());
        for frame in [resumed, whole] {
            assert_eq!(frame.header_data().unwrap().range(), frame_start..end);
            assert_eq!(frame.toc().bookmark(), end - frame_start);
            assert_eq!(frame.toc().total_byte_size(), 1);
        }
    }

    #[test]
    fn frame_header_retry_skips_incomplete_name_and_extension() {
        const IMAGE_HEADER: [u8; 5] = [0xff, 0x0a, 0xf8, 0x9f, 0x01];
        const NAME_LEN: usize = 1024;
        const EXTENSION_BITS: usize = 32_768;
        let image_header = Arc::new(
            ImageHeader::parse(&mut Bitstream::new(&IMAGE_HEADER), ()).unwrap(),
        );
        let context = FrameContext {
            image_header,
            tracker: None,
        };
        let mut bits = Vec::new();
        let mut write = |value: usize, width: usize| {
            bits.extend((0..width).map(|bit| ((value >> bit) & 1) as u8));
        };
        write(0, 1 + 2 + 1 + 2); // Non-default regular VarDCT frame, zero flags.
        write(0, 2 + 3 + 3 + 2 + 1 + 2); // Upsampling, quantizers, passes, crop, blend.
        write(1, 1); // Last frame.
        write(3, 2);
        write(NAME_LEN - 48, 10);
        for _ in 0..NAME_LEN {
            write(b'x' as usize, 8);
        }
        write(1, 1); // Default restoration filter.
        write(1, 2);
        write(0, 4); // One extension.
        write(3, 2);
        write(0, 12);
        write(1, 1);
        write(EXTENSION_BITS >> 12, 8);
        write(0, 1);
        bits.resize(bits.len() + EXTENSION_BITS, 0);
        bits.push(0); // Unpermuted TOC.
        bits.resize(bits.len().next_multiple_of(8), 0);
        bits.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // LF global size 1.
        bits.resize((bits.len() + 6 * 12).next_multiple_of(8), 0);

        let header: Vec<u8> = bits
            .chunks_exact(8)
            .map(|byte| byte.iter().enumerate().fold(0, |acc, (i, bit)| acc | (*bit << i)))
            .collect();
        let frame_start = IMAGE_HEADER.len();
        let mut source = Spans::new();
        source.append(Arc::from(IMAGE_HEADER), 0..frame_start).unwrap();
        let mut parser = FrameHeaderParser::default();
        for (i, &byte) in header.iter().enumerate() {
            source.append(Arc::from([byte]), 0..1).unwrap();
            let result = parser.parse(&source, frame_start, context.clone()).unwrap();
            if i + 1 == header.len() {
                let (frame, end) = result.unwrap();
                assert_eq!(frame.header().name.as_bytes(), &[b'x'; NAME_LEN]);
                assert_eq!(frame.toc().bookmark(), header.len());
                assert_eq!(end, frame_start + header.len());
                assert_eq!(frame.header_data().unwrap().range(), frame_start..end);
            } else {
                assert!(result.is_none());
            }
        }

        // Only the short structural prefix/suffix and the two declared ends
        // trigger fresh FrameHeader parses, even with a byte per feed.
        assert!(parser.header_attempts <= 12, "{}", parser.header_attempts);
    }
}
