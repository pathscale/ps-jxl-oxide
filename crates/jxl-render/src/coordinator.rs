use alloc::{boxed::Box, sync::Arc, vec::Vec};

use jxl_color::{
    ColorEncodingWithProfile, ColorTransform, ColourEncoding, EnumColourEncoding, RenderingIntent,
    TransferFunction,
};
use jxl_modular::Sample;
use jxl_threadpool::Tasks;

use crate::{
    Error, ImageWithRegion, IndexedFrame, Reference, ReferenceFrames, Region, RenderContext,
    Result, image, render, state::RenderCache, util,
};

impl RenderContext {
    fn loading_frame_arc(&self) -> Option<Arc<IndexedFrame>> {
        let frame = self.loading_frame()?;
        self.frames
            .get(frame.index())
            .cloned()
            .or_else(|| self.loading_frame.clone())
    }

    fn reference_slots<S: Sample>(
        frame: &IndexedFrame,
        cache: &mut RenderCache<S>,
    ) -> Result<([bool; 4], [bool; 4])> {
        if cache.lf_global.is_none() {
            match frame.try_parse_lf_global() {
                Some(Ok(global)) => cache.lf_global = Some(global),
                Some(Err(error)) if error.unexpected_eof() => {}
                Some(Err(error)) => return Err(error.into()),
                None => {}
            }
        }
        let mut used = [false; 4];
        if let Some(patches) = cache
            .lf_global
            .as_ref()
            .and_then(|global| global.patches.as_ref())
        {
            for patch in &patches.patches {
                used[patch.ref_idx as usize] = true;
            }
        }
        let patch_slots = used;
        let header = frame.header();
        if header.frame_type.is_normal_frame() && !header.resets_canvas {
            for blend in core::iter::once(&header.blending_info).chain(&header.ec_blending_info) {
                used[blend.source as usize] = true;
            }
        }
        Ok((used, patch_slots))
    }

    async fn resolve_references(
        &mut self,
        _frame: &IndexedFrame,
        deps: crate::FrameDependence,
        used: [bool; 4],
        patch_slots: [bool; 4],
        tasks: &Tasks,
    ) -> Result<ReferenceFrames> {
        let mut references = ReferenceFrames::default();
        if deps.lf != usize::MAX {
            let image = Box::pin(self.render_by_index(deps.lf, None, tasks)).await?;
            references.lf = Some(Reference {
                frame: Arc::clone(&self.frames[deps.lf]),
                image,
            });
        }

        // Patch references require their complete saved frame region. Composition references
        // require this frame's oriented output region. Reuse one snapshot for duplicate slots.
        let output_region = util::apply_orientation_to_image_region(
            &self.image_header,
            self.requested_image_region,
        );
        let mut indices: Vec<_> = deps
            .ref_slots
            .iter()
            .copied()
            .enumerate()
            .filter(|(slot, index)| used[*slot] && *index != usize::MAX)
            .map(|(_, index)| index)
            .collect();
        indices.sort_unstable();
        indices.dedup();
        for index in indices {
            let reference_frame = Arc::clone(&self.frames[index]);
            let is_patch = (0..4).any(|slot| deps.ref_slots[slot] == index && patch_slots[slot]);
            let region = if is_patch {
                let header = reference_frame.header();
                Region::with_size(header.width, header.height).translate(header.x0, header.y0)
            } else {
                output_region
            };
            let image = Box::pin(self.render_by_index(index, Some(region), tasks)).await?;
            for slot in 0..4 {
                if used[slot] && deps.ref_slots[slot] == index {
                    references.refs[slot] = Some(Reference {
                        frame: Arc::clone(&reference_frame),
                        image: Arc::clone(&image),
                    });
                }
            }
        }
        Ok(references)
    }

    async fn render_by_index(
        &mut self,
        index: usize,
        oriented_region: Option<Region>,
        tasks: &Tasks,
    ) -> Result<Arc<ImageWithRegion>> {
        if let Some(image) = &self.rendered_frames[index] {
            return Ok(Arc::clone(image));
        }
        let frame = Arc::clone(&self.frames[index]);
        let deps = self.frame_deps[index];
        let image_region = self.requested_image_region;
        let visibility = self.get_previous_frames_visibility(&frame);

        let (mut image, references) = if self.narrow_modular() {
            let mut cache = self.renders_narrow[index]
                .take()
                .unwrap_or_else(|| RenderCache::new(&frame));
            let (used, patch_slots) = Self::reference_slots(&frame, &mut cache)?;
            let references = self
                .resolve_references(&frame, deps, used, patch_slots, tasks)
                .await?;
            let result = render::render_frame(
                &frame,
                references.clone(),
                &mut cache,
                image_region,
                visibility,
                tasks,
            )
            .await;
            if result.is_err() {
                self.renders_narrow[index] = Some(cache);
            }
            (result?, references)
        } else {
            let mut cache = self.renders_wide[index]
                .take()
                .unwrap_or_else(|| RenderCache::new(&frame));
            let (used, patch_slots) = Self::reference_slots(&frame, &mut cache)?;
            let references = self
                .resolve_references(&frame, deps, used, patch_slots, tasks)
                .await?;
            let result = render::render_frame(
                &frame,
                references.clone(),
                &mut cache,
                image_region,
                visibility,
                tasks,
            )
            .await;
            if result.is_err() {
                self.renders_wide[index] = Some(cache);
            }
            (result?, references)
        };
        if !image::composite_preprocess(&frame, &mut image, tasks).await? {
            let region = oriented_region.unwrap_or_else(|| {
                util::apply_orientation_to_image_region(&self.image_header, image_region)
            });
            image::composite(&frame, &mut image, references.refs, region)?;
            self.release_overwritten_reference(&frame, deps);
        }
        let image = Arc::new(image);
        self.rendered_frames[index] = Some(Arc::clone(&image));
        Ok(image)
    }

    fn release_overwritten_reference(
        &mut self,
        frame: &IndexedFrame,
        deps: crate::FrameDependence,
    ) {
        let header = frame.header();
        if header.can_reference() {
            let index = deps.ref_slots[header.save_as_reference as usize];
            if index != usize::MAX && !self.frames[index].header().is_keyframe() {
                self.rendered_frames[index] = None;
            }
        }
    }

    /// Renders the first loaded keyframe with the caller's executor.
    pub async fn render(&mut self, tasks: Tasks) -> Result<Arc<ImageWithRegion>> {
        self.render_keyframe(0, tasks).await
    }

    /// Renders a loaded keyframe and converts it to the requested output encoding.
    pub async fn render_keyframe(
        &mut self,
        keyframe_idx: usize,
        tasks: Tasks,
    ) -> Result<Arc<ImageWithRegion>> {
        let index = *self
            .keyframes
            .get(keyframe_idx)
            .ok_or(Error::IncompleteFrame)?;
        let grid = self.render_by_index(index, None, &tasks).await?;
        let frame = Arc::clone(&self.frames[index]);
        self.postprocess_keyframe(&frame, grid, &tasks).await
    }

    pub async fn render_loading_keyframe(
        &mut self,
        tasks: Tasks,
    ) -> Result<(&IndexedFrame, Arc<ImageWithRegion>)> {
        let mut current_frame_grid = None;
        if self.loading_frame().is_some() {
            match self.render_loading_frame(&tasks).await {
                Ok(grid) => current_frame_grid = Some(grid),
                Err(Error::IncompleteFrame) => {}
                Err(error) => return Err(error),
            }
        }
        let (frame, grid) = if let Some(grid) = current_frame_grid {
            (
                self.loading_frame_arc().ok_or(Error::IncompleteFrame)?,
                Arc::new(grid),
            )
        } else if let Some(index) = self.keyframe_in_progress {
            let grid = self.render_by_index(index, None, &tasks).await?;
            (Arc::clone(&self.frames[index]), grid)
        } else {
            return Err(Error::IncompleteFrame);
        };
        let grid = self.postprocess_keyframe(&frame, grid, &tasks).await?;
        Ok((
            self.frame(frame.index()).ok_or(Error::IncompleteFrame)?,
            grid,
        ))
    }

    pub fn reset_cache(&mut self) {
        self.loading_region = None;
        self.loading_render_cache_wide = None;
        self.loading_render_cache_narrow = None;
        for index in 0..self.frames.len() {
            if self.frames[index].header().frame_type == jxl_frame::header::FrameType::ReferenceOnly
            {
                continue;
            }
            self.renders_wide[index] = None;
            self.renders_narrow[index] = None;
            self.rendered_frames[index] = None;
        }
    }

    async fn render_loading_frame(&mut self, tasks: &Tasks) -> Result<ImageWithRegion> {
        let frame = self.loading_frame_arc().ok_or(Error::IncompleteFrame)?;
        if !frame.header().frame_type.is_progressive_frame() {
            return Err(Error::IncompleteFrame);
        }
        let image_region = self.requested_image_region;
        self.loading_region = Some(util::image_region_to_frame(&frame, image_region, false));
        let header = frame.header();
        let deps = crate::FrameDependence {
            lf: if header.flags.use_lf_frame() {
                self.lf_frame[header.lf_level as usize]
            } else {
                usize::MAX
            },
            ref_slots: self.reference,
        };
        let visibility = self.get_previous_frames_visibility(&frame);
        let (mut image, references) = if self.narrow_modular() {
            let mut cache = self
                .loading_render_cache_narrow
                .take()
                .unwrap_or_else(|| RenderCache::new(&frame));
            let (used, patch_slots) = Self::reference_slots(&frame, &mut cache)?;
            let references = self
                .resolve_references(&frame, deps, used, patch_slots, tasks)
                .await?;
            let result = render::render_frame(
                &frame,
                references.clone(),
                &mut cache,
                image_region,
                visibility,
                tasks,
            )
            .await;
            if result.is_err() {
                self.loading_render_cache_narrow = Some(cache);
            }
            (result?, references)
        } else {
            let mut cache = self
                .loading_render_cache_wide
                .take()
                .unwrap_or_else(|| RenderCache::new(&frame));
            let (used, patch_slots) = Self::reference_slots(&frame, &mut cache)?;
            let references = self
                .resolve_references(&frame, deps, used, patch_slots, tasks)
                .await?;
            let result = render::render_frame(
                &frame,
                references.clone(),
                &mut cache,
                image_region,
                visibility,
                tasks,
            )
            .await;
            if result.is_err() {
                self.loading_render_cache_wide = Some(cache);
            }
            (result?, references)
        };
        if !image::composite_preprocess(&frame, &mut image, tasks).await? {
            let region = util::apply_orientation_to_image_region(&self.image_header, image_region);
            image::composite(&frame, &mut image, references.refs, region)?;
        }
        if header.lf_level > 0 {
            let mut image = image.upsample_lf(header.lf_level)?;
            image.fill_opaque_alpha(&self.image_header.metadata.ec_info);
            Ok(image)
        } else {
            Ok(image)
        }
    }

    fn cache_color_transform(&mut self) -> Result<()> {
        if self.cached_transform.is_some() {
            return Ok(());
        }
        let metadata = self.metadata();
        let frame_color_encoding = if metadata.xyb_encoded {
            ColorEncodingWithProfile::new(EnumColourEncoding::xyb(RenderingIntent::Perceptual))
        } else if let ColourEncoding::Enum(encoding) = &metadata.colour_encoding {
            ColorEncodingWithProfile::new(encoding.clone())
        } else {
            ColorEncodingWithProfile::with_icc(&self.embedded_icc)?
        };
        let mut builder = ColorTransform::builder();
        builder.set_srgb_icc(!self.cms.supports_linear_tf());
        builder.from_pq(self.suggested_hdr_tf() == Some(TransferFunction::Pq));
        let transform = builder.build(
            &frame_color_encoding,
            &self.requested_color_encoding,
            &metadata.opsin_inverse_matrix,
            &metadata.tone_mapping,
            &*self.cms,
        )?;
        self.cached_transform = Some(Arc::new(transform));
        Ok(())
    }

    async fn postprocess_keyframe(
        &mut self,
        frame: &IndexedFrame,
        grid: Arc<ImageWithRegion>,
        tasks: &Tasks,
    ) -> Result<Arc<ImageWithRegion>> {
        if grid.ct_done() {
            return Ok(grid);
        }
        self.cache_color_transform()?;
        let transform = Arc::clone(self.cached_transform.as_ref().unwrap());
        let header = frame.header();
        let metadata = self.metadata();
        if transform.is_noop() && !header.do_ycbcr {
            return Ok(grid);
        }
        let mut grid = grid.try_clone()?;
        if header.do_ycbcr {
            grid.convert_modular_color(metadata.bit_depth)?;
            let [cb, y, cr] = grid.as_color_floats_mut();
            jxl_color::ycbcr_to_rgb([cb.buf_mut(), y.buf_mut(), cr.buf_mut()]);
        }
        if transform.is_noop() {
            grid.remove_color_channels(transform.output_channels());
            return Ok(Arc::new(grid));
        }
        if header.encoded_color_channels() < 3 {
            grid.clone_gray()?;
        }
        grid.convert_modular_color(metadata.bit_depth)?;
        let (color_channels, extra_channels) = grid.buffer_mut().split_at_mut(3);
        let mut channels = Vec::new();
        for channel in color_channels {
            channels.push(channel.as_float_mut().unwrap().buf_mut());
        }
        let mut has_black = false;
        for (channel, info) in extra_channels.iter_mut().zip(&metadata.ec_info) {
            if info.is_black() {
                channels.push(channel.convert_to_float_modular(info.bit_depth)?.buf_mut());
                has_black = true;
                break;
            }
        }
        if has_black {
            for channel in &mut channels {
                for value in &mut **channel {
                    *value = 1.0 - *value;
                }
            }
        }
        let output_channels = transform.run_with_tasks(&mut channels, tasks).await?;
        if output_channels < 3 {
            grid.remove_color_channels(output_channels);
        }
        grid.set_ct_done(true);
        Ok(Arc::new(grid))
    }
}
