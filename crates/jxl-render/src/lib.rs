//! This crate is the core of jxl-oxide that provides JPEG XL renderer.
use std::sync::Arc;

use jxl_bitstream::Bitstream;
use jxl_color::{
    ColorEncodingWithProfile, ColorManagementSystem, ColorTransform, ColourEncoding, ColourSpace,
    EnumColourEncoding, RenderingIntent, TransferFunction,
};
use jxl_frame::{Frame, FrameContext};
use jxl_grid::AllocTracker;
use jxl_image::{ImageHeader, ImageMetadata};

use jxl_oxide_common::Bundle;
use jxl_threadpool::JxlThreadPool;

mod blend;
mod coordinator;
mod error;
mod features;
mod filter;
mod image;
mod modular;
mod region;
mod render;
mod state;
mod util;
mod vardct;

pub use error::{Error, Result};
pub use features::render_spot_color;
pub use image::{ImageBuffer, ImageWithRegion};
pub use region::Region;
use state::*;

/// Render context that tracks loaded and rendered frames.
pub struct RenderContext {
    image_header: Arc<ImageHeader>,
    pool: JxlThreadPool,
    tracker: Option<AllocTracker>,
    pub(crate) frames: Vec<Arc<IndexedFrame>>,
    pub(crate) renders_wide: Vec<Option<RenderCache<i32>>>,
    pub(crate) renders_narrow: Vec<Option<RenderCache<i16>>>,
    pub(crate) rendered_frames: Vec<Option<Arc<ImageWithRegion>>>,
    pub(crate) keyframes: Vec<usize>,
    pub(crate) keyframe_in_progress: Option<usize>,
    pub(crate) refcounts: Vec<usize>,
    pub(crate) frame_deps: Vec<FrameDependence>,
    pub(crate) lf_frame: [usize; 4],
    pub(crate) reference: [usize; 4],
    pub(crate) loading_frame: Option<Arc<IndexedFrame>>,
    pub(crate) loading_render_cache_wide: Option<RenderCache<i32>>,
    pub(crate) loading_render_cache_narrow: Option<RenderCache<i16>>,
    pub(crate) loading_region: Option<Region>,
    requested_image_region: Region,
    embedded_icc: Vec<u8>,
    requested_color_encoding: ColorEncodingWithProfile,
    cms: Box<dyn ColorManagementSystem + Send + Sync>,
    cached_transform: Option<Arc<ColorTransform>>,
    force_wide_buffers: bool,
}

impl std::fmt::Debug for RenderContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderContext").finish_non_exhaustive()
    }
}

impl RenderContext {
    pub fn builder() -> RenderContextBuilder {
        RenderContextBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct RenderContextBuilder {
    embedded_icc: Vec<u8>,
    pool: Option<JxlThreadPool>,
    tracker: Option<AllocTracker>,
    force_wide_buffers: bool,
}

impl RenderContextBuilder {
    pub fn embedded_icc(mut self, icc: Vec<u8>) -> Self {
        self.embedded_icc = icc;
        self
    }

    pub fn pool(mut self, pool: JxlThreadPool) -> Self {
        self.pool = Some(pool);
        self
    }

    pub fn alloc_tracker(mut self, tracker: AllocTracker) -> Self {
        self.tracker = Some(tracker);
        self
    }

    pub fn force_wide_buffers(mut self, force_wide_buffers: bool) -> Self {
        self.force_wide_buffers = force_wide_buffers;
        self
    }

    pub fn build(self, image_header: Arc<ImageHeader>) -> Result<RenderContext> {
        let color_encoding = &image_header.metadata.colour_encoding;
        let requested_color_encoding = match color_encoding {
            ColourEncoding::Enum(encoding) => ColorEncodingWithProfile::new(encoding.clone()),
            ColourEncoding::IccProfile(color_space) => {
                let header_is_gray = *color_space == ColourSpace::Grey;

                let parsed_icc = match ColorEncodingWithProfile::with_icc(&self.embedded_icc) {
                    Ok(parsed_icc) => {
                        let icc_is_gray = parsed_icc.is_grayscale();
                        if header_is_gray != icc_is_gray {
                            tracing::error!(
                                header_is_gray,
                                icc_is_gray,
                                "Color channel mismatch between header and ICC profile"
                            );
                            return Err(jxl_bitstream::Error::ValidationFailed(
                                "Color channel mismatch between header and ICC profile",
                            )
                            .into());
                        }

                        let is_supported_icc = parsed_icc.icc_profile().is_empty();
                        if !is_supported_icc {
                            tracing::trace!(
                                "Failed to convert embedded ICC into enum color encoding"
                            );
                        }

                        let allow_parsed_icc =
                            !image_header.metadata.xyb_encoded || is_supported_icc;
                        allow_parsed_icc.then_some(parsed_icc)
                    }
                    Err(e) => {
                        tracing::warn!(%e, "Malformed embedded ICC profile");
                        None
                    }
                };

                if let Some(profile) = parsed_icc {
                    profile
                } else if header_is_gray {
                    ColorEncodingWithProfile::new(EnumColourEncoding::gray_srgb(
                        RenderingIntent::Relative,
                    ))
                } else {
                    ColorEncodingWithProfile::new(EnumColourEncoding::srgb(
                        RenderingIntent::Relative,
                    ))
                }
            }
        };

        tracing::debug!(
            default_color_encoding = ?requested_color_encoding,
            "Setting default output color encoding"
        );

        let full_image_region = Region::with_size(
            image_header.width_with_orientation(),
            image_header.height_with_orientation(),
        );

        Ok(RenderContext {
            image_header,
            tracker: self.tracker,
            pool: self.pool.unwrap_or_else(JxlThreadPool::none),
            frames: Vec::new(),
            renders_wide: Vec::new(),
            renders_narrow: Vec::new(),
            rendered_frames: Vec::new(),
            keyframes: Vec::new(),
            keyframe_in_progress: None,
            refcounts: Vec::new(),
            frame_deps: Vec::new(),
            lf_frame: [usize::MAX; 4],
            reference: [usize::MAX; 4],
            loading_frame: None,
            loading_render_cache_wide: None,
            loading_render_cache_narrow: None,
            loading_region: None,
            requested_image_region: full_image_region,
            embedded_icc: self.embedded_icc,
            requested_color_encoding,
            cms: Box::new(jxl_color::NullCms),
            cached_transform: None,
            force_wide_buffers: self.force_wide_buffers,
        })
    }
}

impl RenderContext {
    #[inline]
    pub fn alloc_tracker(&self) -> Option<&AllocTracker> {
        self.tracker.as_ref()
    }
}

impl RenderContext {
    #[inline]
    pub fn set_cms(&mut self, cms: impl ColorManagementSystem + Send + Sync + 'static) {
        self.cached_transform = None;
        self.cms = Box::new(cms);
    }

    pub fn suggested_hdr_tf(&self) -> Option<TransferFunction> {
        let tf = match &self.image_header.metadata.colour_encoding {
            ColourEncoding::Enum(e) => e.tf,
            ColourEncoding::IccProfile(_) => {
                let icc = self.embedded_icc().unwrap();
                jxl_color::icc::icc_tf(icc)?
            }
        };

        match tf {
            TransferFunction::Pq | TransferFunction::Hlg => Some(tf),
            _ => None,
        }
    }

    #[inline]
    pub fn request_color_encoding(&mut self, encoding: ColorEncodingWithProfile) {
        self.cached_transform = None;
        self.requested_color_encoding = encoding;
    }

    #[inline]
    pub fn requested_color_encoding(&self) -> &ColorEncodingWithProfile {
        &self.requested_color_encoding
    }

    #[inline]
    pub fn request_image_region(&mut self, image_region: Region) {
        self.requested_image_region = image_region;
        self.reset_cache();
    }

    #[inline]
    pub fn image_region(&self) -> Region {
        self.requested_image_region
    }
}

impl RenderContext {
    /// Returns the image width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.image_header.size.width
    }

    /// Returns the image height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.image_header.size.height
    }

    #[inline]
    pub fn embedded_icc(&self) -> Option<&[u8]> {
        (!self.embedded_icc.is_empty()).then_some(&self.embedded_icc)
    }

    /// Returns the number of loaded keyframes in the context.
    #[inline]
    pub fn loaded_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    /// Returns the number of loaded frames in the context, including frames that are not shown
    /// directly.
    #[inline]
    pub fn loaded_frames(&self) -> usize {
        self.frames.len()
    }

    #[inline]
    fn metadata(&self) -> &ImageMetadata {
        &self.image_header.metadata
    }

    #[inline]
    fn narrow_modular(&self) -> bool {
        !self.force_wide_buffers && self.image_header.metadata.modular_16bit_buffers
    }

    fn preserve_current_frame(&mut self) {
        let Some(frame) = self.loading_frame.take() else {
            return;
        };

        let header = frame.header();
        let idx = self.frames.len();

        self.refcounts.push(0);

        let lf = if header.flags.use_lf_frame() {
            let lf = self.lf_frame[header.lf_level as usize];
            self.refcounts[lf] += 1;
            lf
        } else {
            usize::MAX
        };
        for ref_idx in self.reference {
            if ref_idx != usize::MAX {
                self.refcounts[ref_idx] += 1;
            }
        }

        let deps = FrameDependence {
            lf,
            ref_slots: self.reference,
        };

        if header.can_reference() {
            let ref_idx = header.save_as_reference as usize;
            self.reference[ref_idx] = idx;
        }
        if header.lf_level != 0 {
            let lf_idx = header.lf_level as usize - 1;
            self.lf_frame[lf_idx] = idx;
        }

        if header.is_keyframe() {
            self.refcounts[idx] += 1;
            self.keyframes.push(idx);
            self.keyframe_in_progress = None;
        } else if header.frame_type.is_normal_frame() {
            self.keyframe_in_progress = Some(idx);
        }

        self.renders_narrow
            .push(self.loading_render_cache_narrow.take());
        self.renders_wide
            .push(self.loading_render_cache_wide.take());
        self.rendered_frames.push(None);

        self.frames.push(Arc::clone(&frame));
        self.frame_deps.push(deps);
    }

    fn loading_frame(&self) -> Option<&IndexedFrame> {
        let search_from = self
            .keyframe_in_progress
            .or_else(|| self.keyframes.last().map(|x| x + 1))
            .unwrap_or(0);
        self.frames[search_from..]
            .iter()
            .map(|r| &**r)
            .chain(self.loading_frame.as_deref())
            .rev()
            .find(|x| x.header().frame_type.is_progressive_frame())
    }
}

impl RenderContext {
    pub fn load_frame_header(&mut self, bitstream: &mut Bitstream) -> Result<&mut IndexedFrame> {
        if self.loading_frame.is_some() && !self.try_finalize_current_frame() {
            panic!("another frame is still loading");
        }

        let image_header = &self.image_header;

        let bitstream_original = bitstream.clone();
        let frame = match Frame::parse(
            bitstream,
            FrameContext {
                image_header: image_header.clone(),
                tracker: self.tracker.as_ref(),
                pool: self.pool.clone(),
            },
        ) {
            Ok(frame) => frame,
            Err(e) => {
                *bitstream = bitstream_original;
                return Err(e.into());
            }
        };

        let header = frame.header();
        // Check if LF frame exists
        if header.flags.use_lf_frame() && self.lf_frame[header.lf_level as usize] == usize::MAX {
            return Err(Error::UninitializedLfFrame(header.lf_level));
        }

        self.loading_frame = Some(Arc::new(IndexedFrame::new(frame, self.frames.len())));
        Ok(Arc::get_mut(self.loading_frame.as_mut().unwrap()).unwrap())
    }

    pub fn current_loading_frame(&mut self) -> Option<&mut IndexedFrame> {
        self.try_finalize_current_frame();
        self.loading_frame.as_mut().and_then(Arc::get_mut)
    }

    pub fn finalize_current_frame(&mut self) {
        if !self.try_finalize_current_frame() {
            panic!("frame is not fully loaded");
        }
    }

    fn try_finalize_current_frame(&mut self) -> bool {
        if let Some(loading_frame) = &self.loading_frame
            && loading_frame.is_loading_done()
        {
            self.preserve_current_frame();
            return true;
        }
        false
    }
}

impl RenderContext {
    /// Returns the frame with the keyframe index, or `None` if the keyframe does not exist.
    #[inline]
    pub fn keyframe(&self, keyframe_idx: usize) -> Option<&IndexedFrame> {
        if keyframe_idx == self.keyframes.len() {
            self.loading_frame()
        } else if let Some(&idx) = self.keyframes.get(keyframe_idx) {
            Some(&self.frames[idx])
        } else {
            None
        }
    }

    #[inline]
    pub fn frame(&self, frame_idx: usize) -> Option<&IndexedFrame> {
        if self.frames.len() == frame_idx {
            self.loading_frame.as_deref()
        } else {
            self.frames.get(frame_idx).map(|x| &**x)
        }
    }
}

impl RenderContext {
    fn get_previous_frames_visibility<'a>(&'a self, frame: &'a IndexedFrame) -> (usize, usize) {
        let frame_idx = frame.index();
        let (is_keyframe, keyframe_idx) = match self.keyframes.binary_search(&frame_idx) {
            Ok(val) => (true, val),
            // Handle partial rendering. If val != self.keyframes.len(), is_keyframe() should be
            // false, since if not the index should exist in self.keyframes.
            Err(val) => (frame.header().is_keyframe(), val),
        };
        let prev_keyframes = &self.keyframes[..keyframe_idx];

        let visible_frames_num = keyframe_idx + is_keyframe as usize;

        let invisible_frames_num = if is_keyframe {
            0
        } else if prev_keyframes.is_empty() {
            1 + frame_idx
        } else {
            let last_visible_frame = prev_keyframes[keyframe_idx - 1];
            frame_idx - last_visible_frame
        };

        (visible_frames_num, invisible_frames_num)
    }
}

/// Frame with its index in the image.
#[derive(Debug)]
pub struct IndexedFrame {
    f: Frame,
    idx: usize,
}

impl IndexedFrame {
    fn new(frame: Frame, index: usize) -> Self {
        IndexedFrame {
            f: frame,
            idx: index,
        }
    }

    /// Returns the frame index.
    pub fn index(&self) -> usize {
        self.idx
    }
}

impl std::ops::Deref for IndexedFrame {
    type Target = Frame;

    fn deref(&self) -> &Self::Target {
        &self.f
    }
}

impl std::ops::DerefMut for IndexedFrame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.f
    }
}

#[derive(Debug, Copy, Clone)]
struct FrameDependence {
    pub(crate) lf: usize,
    pub(crate) ref_slots: [usize; 4],
}

#[derive(Debug, Clone, Default)]
struct ReferenceFrames {
    pub(crate) lf: Option<Reference>,
    pub(crate) refs: [Option<Reference>; 4],
}

#[derive(Debug, Clone)]
struct Reference {
    pub(crate) frame: Arc<IndexedFrame>,
    pub(crate) image: Arc<ImageWithRegion>,
}
