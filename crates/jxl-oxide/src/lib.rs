//! jxl-oxide is a JPEG XL decoder written in pure Rust. It's internally organized into a few
//! small crates. This crate acts as a blanket and provides a simple interface made from those
//! crates to decode the actual image.
//!
//! # Decoding an image
//!
//! Decoding a JPEG XL image starts with constructing [`JxlImage`]. First create a builder using
//! [`JxlImage::builder`], and use [`read`][JxlImageBuilder::read] to read from a reader that
//! implements [`Read`][std::io::Read]:
//!
//! ```no_run
//! # use jxl_oxide::JxlImage;
//! # let reader = std::io::empty();
//! let image = JxlImage::builder().read(reader).expect("Failed to read image header");
//! println!("{:?}", image.image_header()); // Prints the image header
//! ```
//!
//! For incremental input, create an uninitialized image and append immutable chunks.
//! The decoder retains each chunk, including incomplete container headers.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use jxl_oxide::{JxlImage, InitializeResult};
//! # async fn next_chunk() -> jxl_oxide::Result<Arc<[u8]>> { Ok(Arc::from([])) }
//! # async fn run() -> jxl_oxide::Result<()> {
//! let mut uninit_image = JxlImage::builder().build_uninit();
//! let mut image = loop {
//!     uninit_image.feed(next_chunk().await?)?;
//!     match uninit_image.try_init()? {
//!         InitializeResult::NeedMoreData(uninit) => uninit_image = uninit,
//!         InitializeResult::Initialized(image) => break image,
//!     }
//! };
//! while !image.is_loading_done() {
//!     image.feed(next_chunk().await?)?;
//! }
//! image.finalize()?;
//! # Ok(())
//! # }
//! ```
//!
//! For complete input use [`JxlImageBuilder::read_bytes`] with an `Arc<[u8]>`.
//!
//! Rendering exclusively borrows the image while its coordinator updates frame state.
//! The builder defaults to [`Tasks::inline`]; synchronous render methods poll the async
//! path once and return an error if it suspends. Set [`JxlImageBuilder::tasks`] to use
//! a caller-owned executor and await [`JxlImage::render_frame_async`] (or its cropped
//! and loading variants). The decoder never creates workers. With the `nagoya`
//! feature, [`NagoyaExecutor`] adapts an existing host executor.
//!
//! `JxlImage` parses the image header and embedded ICC profile (if there's any). Use
//! [`JxlImage::render_frame`] to render the image.
//!
//! ```no_run
//! # use jxl_oxide::Render;
//! use jxl_oxide::{JxlImage, RenderResult};
//!
//! # fn present_image(_: Render) {}
//! # fn main() -> jxl_oxide::Result<()> {
//! # let mut image = JxlImage::builder().read(std::io::empty()).unwrap();
//! for keyframe_idx in 0..image.num_loaded_keyframes() {
//!     let render = image.render_frame(keyframe_idx)?;
//!     present_image(render);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Color management
//! jxl-oxide has basic color management support, which enables color transformation between
//! well-known color encodings and parsing simple, matrix-based ICC profiles. However, jxl-oxide
//! alone does not support conversion to and from arbitrary ICC profiles, notably CMYK profiles.
//! This includes converting from embedded ICC profiles.
//!
//! Use [`JxlImage::request_color_encoding`] or [`JxlImage::request_icc`] to set color encoding of
//! rendered images. Conversion to and/or from ICC profiles may occur if you do this; in that case,
//! external CMS need to be set using [`JxlImage::set_cms`].
//!
//! ```no_run
//! # use jxl_oxide::{EnumColourEncoding, JxlImage, RenderingIntent};
//! # use jxl_oxide::NullCms as MyCustomCms;
//! # let reader = std::io::empty();
//! let mut image = JxlImage::builder().read(reader).expect("Failed to read image header");
//! image.set_cms(MyCustomCms);
//!
//! let color_encoding = EnumColourEncoding::display_p3(RenderingIntent::Perceptual);
//! image.request_color_encoding(color_encoding);
//! ```
//!
//! External CMS defaults to [`NullCms`]. Set a custom [`ColorManagementSystem`] implementation
//! to handle arbitrary ICC profiles.
//!
//! ```no_run
//! # use jxl_oxide::{JxlImage, NullCms};
//! # let reader = std::io::empty();
//! let mut image = JxlImage::builder().read(reader).expect("Failed to read image header");
//! image.set_cms(NullCms);
//! ```
//!
//! ## Not using `set_cms` for color management
//! If implementing `ColorManagementSystem` is difficult for your use case, color management can be
//! done separately using ICC profile of rendered images. [`JxlImage::rendered_icc`] returns ICC
//! profile for further processing.
//!
//! ```no_run
//! # use jxl_oxide::Render;
//! use jxl_oxide::{JxlImage, RenderResult};
//!
//! # fn present_image_with_cms(_: Render, _: &[u8]) {}
//! # fn main() -> jxl_oxide::Result<()> {
//! # let mut image = JxlImage::builder().read(std::io::empty()).unwrap();
//! let icc_profile = image.rendered_icc();
//! for keyframe_idx in 0..image.num_loaded_keyframes() {
//!     let render = image.render_frame(keyframe_idx)?;
//!     present_image_with_cms(render, &icc_profile);
//! }
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

use std::sync::Arc;

use jxl_bitstream::{Bitstream, ContainerParser, ParseEvent};
pub use jxl_bitstream::{Section, Spans};
use jxl_frame::FrameContext;
use jxl_image::BitDepth;
use jxl_oxide_common::{Bundle, Name};
use jxl_render::ImageBuffer;
use jxl_render::ImageWithRegion;
use jxl_render::Region;
use jxl_render::{IndexedFrame, RenderContext};

pub use jxl_color::{ColorEncodingWithProfile, ColorManagementSystem, NullCms, PreparedTransform};
pub use jxl_frame::header as frame;
pub use jxl_frame::{Frame, FrameHeader};
pub use jxl_grid::{AlignedGrid, AllocTracker};
pub use jxl_image::color::{self, EnumColourEncoding, RenderingIntent};
pub use jxl_image::{self as image, ExtraChannelType, ImageHeader};
#[cfg(feature = "nagoya")]
pub use jxl_threadpool::NagoyaExecutor;
pub use jxl_threadpool::{Cancelled, Job, JobFuture, TaskExecutor, Tasks};

mod aux_box;
mod fb;
pub use aux_box::{AuxBoxData, AuxBoxList, RawExif};
pub use fb::{FrameBuffer, FrameBufferSample, ImageStream};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;

/// JPEG XL image decoder builder.
#[derive(Debug, Default)]
pub struct JxlImageBuilder {
    tasks: Option<Tasks>,
    tracker: Option<AllocTracker>,
    force_wide_buffers: bool,
}

impl JxlImageBuilder {
    /// Sets the caller-owned executor used by async rendering.
    pub fn tasks(mut self, tasks: Tasks) -> Self {
        self.tasks = Some(tasks);
        self
    }

    /// Sets an allocation tracker.
    pub fn alloc_tracker(mut self, tracker: AllocTracker) -> Self {
        self.tracker = Some(tracker);
        self
    }

    /// Force 32-bit Modular buffers when decoding.
    pub fn force_wide_buffers(mut self, force_wide_buffers: bool) -> Self {
        self.force_wide_buffers = force_wide_buffers;
        self
    }

    /// Consumes the builder, and creates an empty, uninitialized JPEG XL image decoder.
    pub fn build_uninit(self) -> UninitializedJxlImage {
        UninitializedJxlImage {
            tasks: self.tasks.unwrap_or_else(Tasks::inline),
            tracker: self.tracker,
            reader: ContainerParser::new(),
            buffer: Spans::new(),
            aux_boxes: AuxBoxList::new(),
            force_wide_buffers: self.force_wide_buffers,
            retry_at: 0,
            end_of_input: false,
        }
    }

    /// Reads complete input into one owned buffer, then calls `read_bytes`.
    pub fn read(self, mut reader: impl std::io::Read) -> Result<JxlImage> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        self.read_bytes(Arc::from(bytes))
    }

    /// Decodes complete input while retaining ranges into the caller's buffer.
    pub fn read_bytes(self, bytes: Arc<[u8]>) -> Result<JxlImage> {
        let mut uninit = self.build_uninit();
        uninit.feed(bytes)?;
        let mut image = match uninit.try_init()? {
            InitializeResult::Initialized(image) => image,
            InitializeResult::NeedMoreData(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "input ended before parsing image header",
                )
                .into());
            }
        };
        image.finalize()?;
        Ok(image)
    }
}

/// Empty, uninitialized JPEG XL image.
///
/// # Examples
/// ```no_run
/// # fn read_bytes() -> jxl_oxide::Result<std::sync::Arc<[u8]>> { Ok(std::sync::Arc::from([])) }
/// # use jxl_oxide::{JxlImage, InitializeResult};
/// # fn main() -> jxl_oxide::Result<()> {
/// let mut uninit_image = JxlImage::builder().build_uninit();
/// let image = loop {
///     let buf = read_bytes()?;
///     uninit_image.feed_bytes(buf)?;
///     match uninit_image.try_init()? {
///         InitializeResult::NeedMoreData(uninit) => {
///             uninit_image = uninit;
///         }
///         InitializeResult::Initialized(image) => {
///             break image;
///         }
///     }
/// };
/// println!("{:?}", image.image_header());
/// # Ok(())
/// # }
/// ```
pub struct UninitializedJxlImage {
    tasks: Tasks,
    tracker: Option<AllocTracker>,
    reader: ContainerParser,
    buffer: Spans,
    aux_boxes: AuxBoxList,
    force_wide_buffers: bool,
    /// Buffered length before which another header attempt is skipped (geometric backoff).
    retry_at: usize,
    /// Set by `end_of_input`; the next `try_init` always attempts a parse.
    end_of_input: bool,
}

impl UninitializedJxlImage {
    /// Declares that no more input will arrive, so the next `try_init` parses whatever is buffered.
    pub fn end_of_input(&mut self) {
        self.end_of_input = true;
    }

    /// The image header and embedded ICC profile are not resumable: a failed attempt reparses from the start.
    /// Waiting until the buffer grows by half keeps repeated attempts linear in the header size.
    fn need_more(mut self) -> InitializeResult {
        let len = self.buffer.len();
        self.retry_at = len.saturating_add((len / 2).max(4096));
        InitializeResult::NeedMoreData(self)
    }
    /// Appends an immutable chunk. Incomplete input is retained automatically.
    pub fn feed(&mut self, buf: Arc<[u8]>) -> Result<()> {
        self.feed_bytes(buf)?;
        Ok(())
    }

    /// Feeds a chunk and returns its accepted length. Do not resubmit any suffix.
    pub fn feed_bytes(&mut self, buf: Arc<[u8]>) -> Result<usize> {
        let len = buf.len();
        for event in self.reader.feed_bytes(buf)? {
            match event? {
                ParseEvent::BitstreamKind(_) => {}
                ParseEvent::Codestream(section) => {
                    self.buffer.append_spans(section.spans())?;
                }
                aux_box_event => {
                    self.aux_boxes.handle_event(aux_box_event)?;
                }
            }
        }
        Ok(len)
    }

    /// Returns the internal reader.
    #[inline]
    pub fn reader(&self) -> &ContainerParser {
        &self.reader
    }

    /// Try to initialize an image with the data fed into so far.
    ///
    /// # Returns
    /// - `Ok(InitializeResult::Initialized(_))` if the initialization was successful,
    /// - `Ok(InitializeResult::NeedMoreData(_))` if the data was not enough, and
    /// - `Err(_)` if there was a decode error during the initialization, meaning invalid bitstream
    ///   was given.
    pub fn try_init(self) -> Result<InitializeResult> {
        if !self.end_of_input && self.buffer.len() < self.retry_at {
            return Ok(InitializeResult::NeedMoreData(self));
        }
        let mut bitstream = Bitstream::from_spans(&self.buffer);
        let image_header = match ImageHeader::parse(&mut bitstream, ()) {
            Ok(x) => x,
            Err(e) if e.unexpected_eof() => {
                return Ok(self.need_more());
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        let embedded_icc = if image_header.metadata.colour_encoding.want_icc() {
            let icc = match jxl_color::icc::read_icc(&mut bitstream) {
                Ok(x) => x,
                Err(e) if e.unexpected_eof() => {
                    return Ok(self.need_more());
                }
                Err(e) => {
                    return Err(e.into());
                }
            };
            tracing::debug!("Image has an embedded ICC profile");
            let icc = jxl_color::icc::decode_icc(&icc)?;
            Some(icc)
        } else {
            None
        };
        match bitstream.zero_pad_to_byte() {
            Ok(()) => {}
            Err(e) if e.unexpected_eof() => {
                return Ok(self.need_more());
            }
            Err(e) => return Err(e.into()),
        }
        let header_end = bitstream.num_read_bits() / 8;

        let image_header = Arc::new(image_header);
        let skip_bytes = if image_header.metadata.preview.is_some() {
            let frame = match Frame::parse(
                &mut bitstream,
                FrameContext {
                    image_header: image_header.clone(),
                    tracker: self.tracker.as_ref(),
                },
            ) {
                Ok(x) => x,
                Err(e) if e.unexpected_eof() => {
                    return Ok(self.need_more());
                }
                Err(e) => {
                    return Err(e.into());
                }
            };

            let bytes_read = bitstream.num_read_bits() / 8;
            let x = frame.toc().total_byte_size();
            let preview_end =
                bytes_read
                    .checked_add(x)
                    .ok_or(jxl_bitstream::Error::ValidationFailed(
                        "preview extent overflow",
                    ))?;
            if self.buffer.len() < preview_end {
                return Ok(self.need_more());
            }

            x
        } else {
            0usize
        };

        let bytes_read = (bitstream.num_read_bits() / 8)
            .checked_add(skip_bytes)
            .ok_or(jxl_bitstream::Error::ValidationFailed(
                "image header extent overflow",
            ))?;
        let header_data = self.buffer.section(0..header_end)?;

        let render_spot_color = !image_header.metadata.grayscale();

        let mut builder = RenderContext::builder();
        if let Some(icc) = embedded_icc {
            builder = builder.embedded_icc(icc);
        }
        if let Some(tracker) = self.tracker {
            builder = builder.alloc_tracker(tracker);
        }
        builder = builder.force_wide_buffers(self.force_wide_buffers);
        let ctx = builder.build(image_header.clone())?;

        let mut image = JxlImage {
            tasks: self.tasks.clone(),
            reader: self.reader,
            image_header,
            header_data,
            ctx: Box::new(ctx),
            render_spot_color,
            inner: JxlImageInner {
                end_of_image: false,
                buffer: self.buffer,
                buffer_offset: bytes_read,
                frame_offsets: Vec::new(),
                aux_boxes: self.aux_boxes,
            },
        };
        image.inner.feed_bytes_inner(&mut image.ctx)?;

        Ok(InitializeResult::Initialized(image))
    }
}

/// Initialization result from [`UninitializedJxlImage::try_init`].
pub enum InitializeResult {
    /// The data was not enough. Feed more data into the returned image.
    NeedMoreData(UninitializedJxlImage),
    /// The image is successfully initialized.
    Initialized(JxlImage),
}

/// JPEG XL image.
#[derive(Debug)]
pub struct JxlImage {
    tasks: Tasks,
    reader: ContainerParser,
    image_header: Arc<ImageHeader>,
    header_data: Section,
    ctx: Box<RenderContext>,
    render_spot_color: bool,
    inner: JxlImageInner,
}

/// # Constructors and data-feeding methods
impl JxlImage {
    /// Creates a decoder builder with default options.
    pub fn builder() -> JxlImageBuilder {
        JxlImageBuilder::default()
    }

    /// Reads an image from the reader with default options.
    pub fn read_with_defaults(reader: impl std::io::Read) -> Result<JxlImage> {
        Self::builder().read(reader)
    }

    /// Appends an immutable chunk. Incomplete input is retained automatically.
    pub fn feed(&mut self, buf: Arc<[u8]>) -> Result<()> {
        self.feed_bytes(buf)?;
        Ok(())
    }

    /// Feeds a chunk and returns its accepted length. Do not resubmit any suffix.
    pub fn feed_bytes(&mut self, buf: Arc<[u8]>) -> Result<usize> {
        let len = buf.len();
        for event in self.reader.feed_bytes(buf)? {
            match event? {
                ParseEvent::BitstreamKind(_) => {}
                ParseEvent::Codestream(section) => {
                    self.inner.buffer.append_spans(section.spans())?;
                    self.inner.feed_bytes_inner(&mut self.ctx)?;
                }
                aux_box_event => {
                    self.inner.aux_boxes.handle_event(aux_box_event)?;
                }
            }
        }
        Ok(len)
    }

    /// Signals EOF and validates that the container and all frames are complete.
    ///
    /// This is automatically done by `read()` and `read_bytes()`.
    pub fn finalize(&mut self) -> Result<()> {
        self.reader.finalize()?;
        if !self.inner.end_of_image {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "input ended before loading all image frames",
            )
            .into());
        }
        self.inner.aux_boxes.eof()?;
        Ok(())
    }
}

/// # Image and decoder metadata accessors
impl JxlImage {
    /// Returns the image header.
    #[inline]
    pub fn image_header(&self) -> &ImageHeader {
        &self.image_header
    }

    /// Returns the compressed image header and ICC ranges.
    pub fn header_data(&self) -> &Section {
        &self.header_data
    }

    /// Returns the image width with orientation applied.
    #[inline]
    pub fn width(&self) -> u32 {
        self.image_header.width_with_orientation()
    }

    /// Returns the image height with orientation applied.
    #[inline]
    pub fn height(&self) -> u32 {
        self.image_header.height_with_orientation()
    }

    /// Returns the *original* ICC profile embedded in the image.
    #[inline]
    pub fn original_icc(&self) -> Option<&[u8]> {
        self.ctx.embedded_icc()
    }

    /// Returns the ICC profile that describes rendered images.
    ///
    /// The returned profile will change if different color encoding is specified using
    /// [`request_icc`][Self::request_icc] or
    /// [`request_color_encoding`][Self::request_color_encoding].
    pub fn rendered_icc(&self) -> Vec<u8> {
        let encoding = self.ctx.requested_color_encoding();
        match encoding.encoding() {
            color::ColourEncoding::Enum(encoding) => {
                jxl_color::icc::colour_encoding_to_icc(encoding)
            }
            color::ColourEncoding::IccProfile(_) => encoding.icc_profile().to_vec(),
        }
    }

    /// Returns the CICP tag of the color encoding of rendered images, if there's any.
    #[inline]
    pub fn rendered_cicp(&self) -> Option<[u8; 4]> {
        let encoding = self.ctx.requested_color_encoding();
        encoding.encoding().cicp()
    }

    /// Returns the pixel format of the rendered image.
    pub fn pixel_format(&self) -> PixelFormat {
        let encoding = self.ctx.requested_color_encoding();
        let is_grayscale = encoding.is_grayscale();
        let has_black = encoding.is_cmyk();
        let mut has_alpha = false;
        for ec_info in &self.image_header.metadata.ec_info {
            if ec_info.is_alpha() {
                has_alpha = true;
            }
        }

        match (is_grayscale, has_black, has_alpha) {
            (false, false, false) => PixelFormat::Rgb,
            (false, false, true) => PixelFormat::Rgba,
            (false, true, false) => PixelFormat::Cmyk,
            (false, true, true) => PixelFormat::Cmyka,
            (true, _, false) => PixelFormat::Gray,
            (true, _, true) => PixelFormat::Graya,
        }
    }

    /// Returns what HDR transfer function the image uses, if there's any.
    ///
    /// Returns `None` if the image is not HDR one.
    pub fn hdr_type(&self) -> Option<HdrType> {
        self.ctx.suggested_hdr_tf().and_then(|tf| match tf {
            color::TransferFunction::Pq => Some(HdrType::Pq),
            color::TransferFunction::Hlg => Some(HdrType::Hlg),
            _ => None,
        })
    }

    /// Returns whether the spot color channels will be rendered.
    #[inline]
    pub fn render_spot_color(&self) -> bool {
        self.render_spot_color
    }

    /// Sets whether the spot colour channels will be rendered.
    #[inline]
    pub fn set_render_spot_color(&mut self, render_spot_color: bool) -> &mut Self {
        if render_spot_color && self.image_header.metadata.grayscale() {
            tracing::warn!("Spot colour channels are not rendered on grayscale images");
            return self;
        }
        self.render_spot_color = render_spot_color;
        self
    }

    /// Returns the list of auxiliary boxes in the JPEG XL container.
    ///
    /// The list may contain Exif and XMP metadata.
    pub fn aux_boxes(&self) -> &AuxBoxList {
        &self.inner.aux_boxes
    }

    /// Returns the number of currently loaded keyframes.
    #[inline]
    pub fn num_loaded_keyframes(&self) -> usize {
        self.ctx.loaded_keyframes()
    }

    /// Returns the number of currently loaded frames, including frames that are not displayed
    /// directly.
    #[inline]
    pub fn num_loaded_frames(&self) -> usize {
        self.ctx.loaded_frames()
    }

    /// Returns whether the image is loaded completely, without missing animation keyframes or
    /// partially loaded frames.
    #[inline]
    pub fn is_loading_done(&self) -> bool {
        self.inner.end_of_image
    }

    /// Returns frame data by keyframe index.
    pub fn frame_by_keyframe(&self, keyframe_index: usize) -> Option<&IndexedFrame> {
        self.ctx.keyframe(keyframe_index)
    }

    /// Returns the frame header for the given keyframe index, or `None` if the keyframe does not
    /// exist.
    pub fn frame_header(&self, keyframe_index: usize) -> Option<&FrameHeader> {
        let frame = self.ctx.keyframe(keyframe_index)?;
        Some(frame.header())
    }

    /// Returns frame data by frame index, including frames that are not displayed directly.
    ///
    /// There are some situations where a frame is not displayed directly:
    /// - It may be marked as reference only, and meant to be only used by other frames.
    /// - It may contain LF image (which is 8x downsampled version) of another VarDCT frame.
    /// - Zero duration frame that is not the last frame of image is blended with following frames
    ///   and displayed together.
    pub fn frame(&self, frame_idx: usize) -> Option<&IndexedFrame> {
        self.ctx.frame(frame_idx)
    }

    /// Returns the offset of frame within codestream, in bytes.
    pub fn frame_offset(&self, frame_index: usize) -> Option<usize> {
        self.inner.frame_offsets.get(frame_index).copied()
    }

    /// Returns the caller-owned task executor.
    #[inline]
    pub fn tasks(&self) -> &Tasks {
        &self.tasks
    }

    /// Returns the internal reader.
    pub fn reader(&self) -> &ContainerParser {
        &self.reader
    }
}

/// # Color management methods
impl JxlImage {
    /// Sets color management system implementation to be used by the renderer.
    #[inline]
    pub fn set_cms(&mut self, cms: impl ColorManagementSystem + Send + Sync + 'static) {
        self.ctx.set_cms(cms);
    }

    /// Requests the decoder to render in specific color encoding, described by an ICC profile.
    ///
    /// # Errors
    /// This function will return an error if it cannot parse the ICC profile.
    pub fn request_icc(&mut self, icc_profile: &[u8]) -> Result<()> {
        self.ctx
            .request_color_encoding(ColorEncodingWithProfile::with_icc(icc_profile)?);
        Ok(())
    }

    /// Requests the decoder to render in specific color encoding, described by
    /// `EnumColourEncoding`.
    pub fn request_color_encoding(&mut self, color_encoding: EnumColourEncoding) {
        self.ctx
            .request_color_encoding(ColorEncodingWithProfile::new(color_encoding))
    }
}

/// # Rendering to image buffers
impl JxlImage {
    /// Renders the given keyframe.
    pub fn render_frame(&mut self, keyframe_index: usize) -> Result<Render> {
        jxl_threadpool::poll_once(self.render_frame_async(keyframe_index))?
    }

    /// Renders the given keyframe with the caller's executor.
    pub async fn render_frame_async(&mut self, keyframe_index: usize) -> Result<Render> {
        self.render_frame_cropped_async(keyframe_index).await
    }

    /// Renders the given keyframe with optional cropping region.
    pub fn render_frame_cropped(&mut self, keyframe_index: usize) -> Result<Render> {
        jxl_threadpool::poll_once(self.render_frame_cropped_async(keyframe_index))?
    }

    /// Renders the given keyframe in the current cropping region with the caller's executor.
    pub async fn render_frame_cropped_async(&mut self, keyframe_index: usize) -> Result<Render> {
        let image = self
            .ctx
            .render_keyframe(keyframe_index, self.tasks.clone())
            .await?;

        let image_region = self
            .ctx
            .image_region()
            .apply_orientation(&self.image_header);
        let frame = self.ctx.keyframe(keyframe_index).unwrap();
        let frame_header = frame.header();
        let target_frame_region = image_region.translate(-frame_header.x0, -frame_header.y0);

        let is_cmyk = self.ctx.requested_color_encoding().is_cmyk();
        let result = Render {
            keyframe_index,
            name: frame_header.name.clone(),
            duration: frame_header.duration,
            orientation: self.image_header.metadata.orientation,
            image,
            extra_channels: self.convert_ec_info(),
            target_frame_region,
            color_bit_depth: self.image_header.metadata.bit_depth,
            is_cmyk,
            render_spot_color: self.render_spot_color,
        };
        Ok(result)
    }

    /// Renders the currently loading keyframe.
    pub fn render_loading_frame(&mut self) -> Result<Render> {
        jxl_threadpool::poll_once(self.render_loading_frame_async())?
    }

    /// Renders the currently loading keyframe with the caller's executor.
    pub async fn render_loading_frame_async(&mut self) -> Result<Render> {
        self.render_loading_frame_cropped_async().await
    }

    /// Renders the currently loading keyframe with optional cropping region.
    pub fn render_loading_frame_cropped(&mut self) -> Result<Render> {
        jxl_threadpool::poll_once(self.render_loading_frame_cropped_async())?
    }

    /// Renders the loading keyframe in the current cropping region with the caller's executor.
    pub async fn render_loading_frame_cropped_async(&mut self) -> Result<Render> {
        let (frame, image) = self.ctx.render_loading_keyframe(self.tasks.clone()).await?;
        let frame_header = frame.header();
        let name = frame_header.name.clone();
        let duration = frame_header.duration;

        let image_region = self
            .ctx
            .image_region()
            .apply_orientation(&self.image_header);
        let frame = self
            .ctx
            .frame(self.ctx.loaded_frames())
            .or_else(|| self.ctx.frame(self.ctx.loaded_frames() - 1))
            .unwrap();
        let frame_header = frame.header();
        let target_frame_region = image_region.translate(-frame_header.x0, -frame_header.y0);

        let is_cmyk = self.ctx.requested_color_encoding().is_cmyk();
        let result = Render {
            keyframe_index: self.ctx.loaded_keyframes(),
            name,
            duration,
            orientation: self.image_header.metadata.orientation,
            image,
            extra_channels: self.convert_ec_info(),
            target_frame_region,
            color_bit_depth: self.image_header.metadata.bit_depth,
            is_cmyk,
            render_spot_color: self.render_spot_color,
        };
        Ok(result)
    }

    /// Returns current cropping region (region of interest).
    pub fn current_image_region(&self) -> CropInfo {
        let region = self.ctx.image_region();
        region.into()
    }

    /// Sets the cropping region (region of interest).
    ///
    /// Subsequent rendering methods will crop the image buffer according to the region.
    pub fn set_image_region(&mut self, region: CropInfo) -> &mut Self {
        self.ctx.request_image_region(region.into());
        self
    }
}

/// # Private methods
impl JxlImage {
    fn convert_ec_info(&self) -> Vec<ExtraChannel> {
        self.image_header
            .metadata
            .ec_info
            .iter()
            .map(|ec_info| ExtraChannel {
                ty: ec_info.ty,
                name: ec_info.name.clone(),
                bit_depth: ec_info.bit_depth,
            })
            .collect()
    }
}

#[derive(Debug)]
struct JxlImageInner {
    end_of_image: bool,
    buffer: Spans,
    buffer_offset: usize,
    frame_offsets: Vec<usize>,
    aux_boxes: AuxBoxList,
}

impl JxlImageInner {
    fn feed_bytes_inner(&mut self, ctx: &mut RenderContext) -> Result<()> {
        while !self.end_of_image {
            if ctx.current_loading_frame().is_none() {
                if self.buffer_offset == self.buffer.len() {
                    return Ok(());
                }
                let pending = self.buffer.section(self.buffer_offset..self.buffer.len())?;
                let mut bitstream = Bitstream::from_section(&pending);
                let frame = match ctx.load_frame_header(&mut bitstream) {
                    Ok(frame) => frame,
                    Err(e) if e.unexpected_eof() => return Ok(()),
                    Err(e) => return Err(e.into()),
                };
                let frame_index = frame.index();
                assert_eq!(self.frame_offsets.len(), frame_index);
                self.frame_offsets.push(self.buffer_offset);
                self.buffer_offset = self
                    .buffer_offset
                    .checked_add(bitstream.num_read_bits() / 8)
                    .filter(|offset| *offset <= self.buffer.len())
                    .ok_or(jxl_bitstream::Error::ValidationFailed(
                        "frame header extent overflow",
                    ))?;
            }
            let frame =
                ctx.current_loading_frame()
                    .ok_or(jxl_bitstream::Error::ValidationFailed(
                        "missing loading frame",
                    ))?;
            let count = frame.feed_spans(&self.buffer, self.buffer_offset..self.buffer.len())?;
            self.buffer_offset = self
                .buffer_offset
                .checked_add(count)
                .filter(|offset| *offset <= self.buffer.len())
                .ok_or(jxl_bitstream::Error::ValidationFailed(
                    "frame data extent overflow",
                ))?;
            if !frame.is_loading_done() {
                return Ok(());
            }
            let is_last = frame.header().is_last;
            ctx.finalize_current_frame();
            if is_last {
                self.end_of_image = true;
            }
        }
        Ok(())
    }
}

/// Pixel format of the rendered image.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PixelFormat {
    /// Grayscale, single channel
    Gray,
    /// Grayscale with alpha, two channels
    Graya,
    /// RGB, three channels
    Rgb,
    /// RGB with alpha, four channels
    Rgba,
    /// CMYK, four channels
    Cmyk,
    /// CMYK with alpha, five channels
    Cmyka,
}

impl PixelFormat {
    /// Returns the number of channels of the image.
    #[inline]
    pub fn channels(self) -> usize {
        match self {
            PixelFormat::Gray => 1,
            PixelFormat::Graya => 2,
            PixelFormat::Rgb => 3,
            PixelFormat::Rgba => 4,
            PixelFormat::Cmyk => 4,
            PixelFormat::Cmyka => 5,
        }
    }

    /// Returns whether the image is grayscale.
    #[inline]
    pub fn is_grayscale(self) -> bool {
        matches!(self, Self::Gray | Self::Graya)
    }

    /// Returns whether the image has an alpha channel.
    #[inline]
    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            PixelFormat::Graya | PixelFormat::Rgba | PixelFormat::Cmyka
        )
    }

    /// Returns whether the image has a black channel.
    #[inline]
    pub fn has_black(self) -> bool {
        matches!(self, PixelFormat::Cmyk | PixelFormat::Cmyka)
    }
}

/// HDR transfer function type, returned by [`JxlImage::hdr_type`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum HdrType {
    /// Perceptual quantizer.
    Pq,
    /// Hybrid log-gamma.
    Hlg,
}

/// The result of loading the keyframe.
#[derive(Debug)]
pub enum LoadResult {
    /// The frame is loaded with the given keyframe index.
    Done(usize),
    /// More data is needed to fully load the frame.
    NeedMoreData,
    /// No more frames are present.
    NoMoreFrames,
}

/// The result of loading and rendering the keyframe.
#[derive(Debug)]
pub enum RenderResult {
    /// The frame is rendered.
    Done(Render),
    /// More data is needed to fully render the frame.
    NeedMoreData,
    /// No more frames are present.
    NoMoreFrames,
}

/// The result of rendering a keyframe.
#[derive(Debug)]
pub struct Render {
    keyframe_index: usize,
    name: Name,
    duration: u32,
    orientation: u32,
    image: Arc<ImageWithRegion>,
    extra_channels: Vec<ExtraChannel>,
    target_frame_region: Region,
    color_bit_depth: BitDepth,
    is_cmyk: bool,
    render_spot_color: bool,
}

impl Render {
    /// Returns the keyframe index.
    #[inline]
    pub fn keyframe_index(&self) -> usize {
        self.keyframe_index
    }

    /// Returns the name of the frame.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns how many ticks this frame is presented.
    #[inline]
    pub fn duration(&self) -> u32 {
        self.duration
    }

    /// Returns the orientation of the image.
    #[inline]
    pub fn orientation(&self) -> u32 {
        self.orientation
    }

    /// Creates a stream that writes to borrowed buffer.
    ///
    /// The stream will include black and alpha channels, if exist, in addition to color channels.
    /// Orientation is applied.
    pub fn stream(&self) -> ImageStream<'_> {
        ImageStream::from_render(self, false)
    }

    /// Creates a stream that writes to borrowed buffer.
    ///
    /// The stream will include black channels if exist, but not alpha channels. Orientation is
    /// applied.
    pub fn stream_no_alpha(&self) -> ImageStream<'_> {
        ImageStream::from_render(self, true)
    }

    /// Creates a buffer with interleaved channels, with orientation applied.
    ///
    /// All extra channels are included. Use [`stream`](Render::stream) if only color, black and
    /// alpha channels are needed.
    #[inline]
    pub fn image_all_channels(&self) -> FrameBuffer {
        let fb: Vec<_> = self.image.buffer().iter().collect();
        let mut bit_depth = vec![self.color_bit_depth; self.image.color_channels()];
        for ec in &self.extra_channels {
            bit_depth.push(ec.bit_depth);
        }
        let regions: Vec<_> = self
            .image
            .regions_and_shifts()
            .iter()
            .map(|(region, _)| *region)
            .collect();

        FrameBuffer::from_grids(
            &fb,
            &bit_depth,
            &regions,
            self.target_frame_region,
            self.orientation,
        )
    }

    /// Creates a separate buffer by channel, with orientation applied.
    ///
    /// All extra channels are included.
    pub fn image_planar(&self) -> Vec<FrameBuffer> {
        let grids = self.image.buffer();
        let bit_depth_it = std::iter::repeat_n(self.color_bit_depth, self.image.color_channels())
            .chain(self.extra_channels.iter().map(|ec| ec.bit_depth));
        let region_it = self
            .image
            .regions_and_shifts()
            .iter()
            .map(|(region, _)| *region);

        bit_depth_it
            .zip(region_it)
            .zip(grids)
            .map(|((bit_depth, region), x)| {
                FrameBuffer::from_grids(
                    &[x],
                    &[bit_depth],
                    &[region],
                    self.target_frame_region,
                    self.orientation,
                )
            })
            .collect()
    }

    /// Returns the color channels.
    ///
    /// Orientation is not applied.
    #[inline]
    pub fn color_channels(&self) -> &[ImageBuffer] {
        let color_channels = self.image.color_channels();
        &self.image.buffer()[..color_channels]
    }

    /// Returns the extra channels, potentially including alpha and black channels.
    ///
    /// Orientation is not applied.
    #[inline]
    pub fn extra_channels(&self) -> (&[ExtraChannel], &[ImageBuffer]) {
        let color_channels = self.image.color_channels();
        (&self.extra_channels, &self.image.buffer()[color_channels..])
    }
}

/// Extra channel of the image.
#[derive(Debug)]
pub struct ExtraChannel {
    ty: ExtraChannelType,
    name: Name,
    bit_depth: BitDepth,
}

impl ExtraChannel {
    /// Returns the type of the extra channel.
    #[inline]
    pub fn ty(&self) -> ExtraChannelType {
        self.ty
    }

    /// Returns the name of the channel.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns `true` if the channel is a black channel of CMYK image.
    #[inline]
    pub fn is_black(&self) -> bool {
        matches!(self.ty, ExtraChannelType::Black)
    }

    /// Returns `true` if the channel is an alpha channel.
    #[inline]
    pub fn is_alpha(&self) -> bool {
        matches!(self.ty, ExtraChannelType::Alpha { .. })
    }

    /// Returns `true` if the channel is a spot colour channel.
    #[inline]
    pub fn is_spot_colour(&self) -> bool {
        matches!(self.ty, ExtraChannelType::SpotColour { .. })
    }
}

/// Cropping region information.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct CropInfo {
    pub width: u32,
    pub height: u32,
    pub left: u32,
    pub top: u32,
}

impl From<CropInfo> for jxl_render::Region {
    fn from(value: CropInfo) -> Self {
        Self {
            left: value.left as i32,
            top: value.top as i32,
            width: value.width,
            height: value.height,
        }
    }
}

impl From<jxl_render::Region> for CropInfo {
    fn from(value: jxl_render::Region) -> Self {
        Self {
            left: value.left.max(0) as u32,
            top: value.top.max(0) as u32,
            width: value.width,
            height: value.height,
        }
    }
}
