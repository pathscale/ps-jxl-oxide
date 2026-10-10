use alloc::sync::Arc;
use jxl_frame::{
    filter::{EdgePreservingFilter, Gabor},
    header::Encoding,
};

use jxl_modular::Sample;
use jxl_threadpool::Tasks;

use crate::{
    Error, ImageWithRegion, IndexedFrame, Reference, ReferenceFrames, Region, Result, blend,
    features, filter, modular, state::RenderCache, util, vardct,
};

pub(crate) async fn render_frame<S: Sample>(
    frame: &Arc<IndexedFrame>,
    reference_frames: ReferenceFrames,
    cache: &mut RenderCache<S>,
    image_region: Region,
    frame_visibility: (usize, usize),
    tasks: &Tasks,
) -> Result<ImageWithRegion> {
    let frame_region = util::image_region_to_frame(frame, image_region, false);
    tracing::debug!(
        index = frame.idx,
        ?image_region,
        ?frame_region,
        "Rendering frame"
    );

    let image_header = frame.image_header();
    let frame_header = frame.header();
    let frame_region = util::pad_lf_region(frame_header, frame_region);

    let upsampled_full_frame_region =
        Region::with_size(frame_header.sample_width(1), frame_header.sample_height(1));
    let upsampling_valid_region = util::pad_upsampling(image_header, frame_header, frame_region)
        .intersection(upsampled_full_frame_region);

    let full_frame_region = Region::with_size(
        frame_header.color_sample_width(),
        frame_header.color_sample_height(),
    );
    let color_padded_region = util::pad_color_region(image_header, frame_header, frame_region)
        .intersection(full_frame_region);

    let mut fb = match frame_header.encoding {
        Encoding::Modular => {
            modular::render_modular(frame, cache, color_padded_region, tasks).await?
        }
        Encoding::VarDct => {
            let result = vardct::render_vardct(
                frame,
                reference_frames.lf.as_ref(),
                cache,
                color_padded_region,
                tasks,
            )
            .await;
            match (result, reference_frames.lf) {
                (Ok(grid), _) => grid,
                (Err(e), Some(lf)) if matches!(e, Error::IncompleteFrame) || e.unexpected_eof() => {
                    let render = lf.image;
                    let mut render = render.upsample_lf(1)?;
                    render.fill_opaque_alpha(&image_header.metadata.ec_info);
                    render
                }
                (Err(e), _) => return Err(e),
            }
        }
    };

    if frame_header.do_ycbcr {
        fb.upsample_jpeg(color_padded_region, image_header.metadata.bit_depth)?;
    }

    let color_channels = fb.color_channels();
    let gabor_enabled = frame_header.restoration_filter.gab.enabled();
    let epf_enabled = frame_header.restoration_filter.epf.enabled();
    if (gabor_enabled || epf_enabled) && !color_padded_region.is_empty() {
        if fb.color_channels() < 3 {
            tracing::trace!("Cloning gray channel");
            fb.clone_gray()?;
        }

        fb.convert_modular_color(image_header.metadata.bit_depth)?;
        let preserve_outside = !gabor_enabled
            && matches!(
                &frame_header.restoration_filter.epf,
                EdgePreservingFilter::Enabled(params) if params.iters == 2
            );
        let mut filtered =
            filter::FilterImage::take_from(&mut fb, color_padded_region, preserve_outside);

        if let Gabor::Enabled(weights) = frame_header.restoration_filter.gab {
            filtered = filter::apply_gabor_like(filtered, weights, tasks).await?;
        }

        if let EdgePreservingFilter::Enabled(epf_params) = &frame_header.restoration_filter.epf {
            filtered =
                filter::apply_epf(filtered, &cache.lf_groups, frame_header, epf_params, tasks)
                    .await?;
        }

        filtered.write_to(&mut fb)?;
    }

    // Truncate cloned gray channels.
    fb.remove_color_channels(color_channels);

    fb.prepare_color_upsampling(frame_header);

    render_features(
        frame,
        &mut fb,
        upsampling_valid_region,
        reference_frames.refs.clone(),
        cache,
        frame_visibility.0,
        frame_visibility.1,
        tasks,
    )
    .await?;

    fb.upsample_nonseparable(
        image_header,
        frame_header,
        upsampling_valid_region,
        false,
        tasks,
    )
    .await?;

    if !frame_header.save_before_ct && !frame_header.is_last {
        util::convert_color_for_record(image_header, frame_header.do_ycbcr, &mut fb, tasks).await?;
    }

    Ok(fb)
}

#[allow(clippy::too_many_arguments)]
async fn render_features<S: Sample>(
    frame: &IndexedFrame,
    grid: &mut ImageWithRegion,
    upsampling_valid_region: Region,
    reference_grids: [Option<Reference>; 4],
    cache: &mut RenderCache<S>,
    visible_frames_num: usize,
    invisible_frames_num: usize,
    tasks: &Tasks,
) -> Result<()> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let Some(lf_global) = cache.lf_global.as_ref() else {
        tracing::trace!("LfGlobal not available, skipping feature rendering");
        return Ok(());
    };
    let base_correlations_xb = lf_global.vardct.as_ref().map(|x| {
        (
            x.lf_chan_corr.base_correlation_x,
            x.lf_chan_corr.base_correlation_b,
        )
    });

    if let Some(patches) = &lf_global.patches {
        grid.upsample_nonseparable(
            image_header,
            frame_header,
            upsampling_valid_region,
            true,
            tasks,
        )
        .await?;

        for patch in &patches.patches {
            let Some(ref_grid) = &reference_grids[patch.ref_idx as usize] else {
                return Err(Error::InvalidReference(patch.ref_idx));
            };
            blend::patch(image_header, grid, &ref_grid.image, patch)?;
        }
    }

    if let Some(splines) = &lf_global.splines {
        if grid.color_channels() == 3 {
            grid.convert_modular_color(image_header.metadata.bit_depth)?;
            features::render_spline(frame_header, grid, splines, base_correlations_xb)?;
        } else {
            tracing::warn!("Cannot render splines on grayscale buffer; skipping");
        }
    }

    if let Some(noise) = &lf_global.noise {
        if grid.color_channels() == 3 {
            grid.convert_modular_color(image_header.metadata.bit_depth)?;
            features::render_noise(
                frame.header(),
                visible_frames_num,
                invisible_frames_num,
                base_correlations_xb,
                grid,
                noise,
                tasks,
            )
            .await?;
        } else {
            tracing::warn!("Cannot render noise on grayscale buffer; skipping");
        }
    }

    Ok(())
}
