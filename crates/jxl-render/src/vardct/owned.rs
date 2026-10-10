use alloc::{boxed::Box, collections::BTreeMap, sync::Arc, vec::Vec};

use jxl_frame::{
    FrameParser,
    data::{HfGlobal, LfGlobal, LfGroup, PassGroupParams, PassGroupParamsVardct},
};
use jxl_grid::AlignedGrid;
use jxl_modular::{ChannelShift, Sample, image::TransformedModularSubimage};
use jxl_threadpool::Tasks;

use crate::{
    Error, ImageWithRegion, IndexedFrame, Reference, Region, RenderCache, Result,
    image::ImageBuffer, modular, util,
};

type Job<T> = Box<dyn FnOnce() -> T + Send + 'static>;

struct CoefficientGroup {
    index: u32,
    channels: [AlignedGrid<f32>; 3],
}

pub(super) async fn render<S: Sample>(
    frame: &Arc<IndexedFrame>,
    lf_frame: Option<&Reference>,
    cache: &mut RenderCache<S>,
    region: Region,
    tasks: &Tasks,
) -> Result<ImageWithRegion> {
    let mut parser = FrameParser::default();
    if !cache
        .lf_global
        .as_ref()
        .is_some_and(|x| !x.gmodular.is_partial())
    {
        cache.lf_global = Some(
            frame
                .try_parse_lf_global_with_parser(&mut parser)
                .ok_or(Error::IncompleteFrame)??,
        );
    }
    let lf_global = Arc::new(cache.lf_global.take().unwrap());
    let mut hf_global = cache.hf_global.take().map(Arc::new);
    let mut lf_groups = Arc::new(core::mem::take(&mut cache.lf_groups));
    let result = render_inner(
        frame,
        lf_frame,
        &lf_global,
        &mut hf_global,
        &mut lf_groups,
        region,
        &mut parser,
        tasks,
    )
    .await;
    // Cancellation can leave a synchronous job retiring with its private snapshot.
    // In that case discard the cache entry and rebuild it on the next render.
    cache.lf_global = Arc::try_unwrap(lf_global).ok();
    cache.hf_global = hf_global.and_then(|global| Arc::try_unwrap(global).ok());
    cache.lf_groups = Arc::try_unwrap(lf_groups).unwrap_or_default();
    result
}

async fn render_inner<S: Sample>(
    frame: &Arc<IndexedFrame>,
    lf_frame: Option<&Reference>,
    lf_global: &Arc<LfGlobal<S>>,
    hf_global: &mut Option<Arc<HfGlobal>>,
    lf_groups: &mut Arc<BTreeMap<u32, LfGroup<S>>>,
    region: Region,
    parser: &mut FrameParser,
    tasks: &Tasks,
) -> Result<ImageWithRegion> {
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();
    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);
    let shifts_cbycr: [_; 3] =
        core::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));
    if lf_frame.is_some() && lf_global.gmodular.is_partial() {
        return Err(Error::IncompleteFrame);
    }

    let mut gmodular = lf_global.gmodular.try_clone()?;
    let lf_global_vardct = lf_global.vardct.as_ref().unwrap();
    let width = frame_header.color_sample_width() as usize;
    let height = frame_header.color_sample_height() as usize;
    let (width_rounded, height_rounded) = {
        let mut bw = width.div_ceil(8);
        let mut bh = height.div_ceil(8);
        let h_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 2);
        let v_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 3);
        if h_upsample {
            bw = bw.div_ceil(2) * 2;
        }
        if v_upsample {
            bh = bh.div_ceil(2) * 2;
        }
        (bw * 8, bh * 8)
    };
    let aligned_region = region.container_aligned(frame_header.group_dim());
    let aligned_lf_region = {
        let aligned_region_div8 = Region {
            left: aligned_region.left / 8,
            top: aligned_region.top / 8,
            width: aligned_region.width / 8,
            height: aligned_region.height / 8,
        };
        if frame_header.flags.skip_adaptive_lf_smoothing() {
            aligned_region_div8
        } else {
            aligned_region_div8.pad(1)
        }
        .container_aligned(frame_header.group_dim())
    };
    let aligned_region = aligned_region.intersection(Region::with_size(
        width_rounded as u32,
        height_rounded as u32,
    ));
    let aligned_lf_region = aligned_lf_region.intersection(Region::with_size(
        width_rounded as u32 / 8,
        height_rounded as u32 / 8,
    ));
    let modular_region =
        modular::compute_modular_region(frame_header, &gmodular, aligned_region, false);
    let modular_lf_region =
        modular::compute_modular_region(frame_header, &gmodular, aligned_lf_region, true)
            .intersection(Region::with_size(
                width_rounded as u32 / 8,
                height_rounded as u32 / 8,
            ));

    let (lf_xyb, returned_modular) = util::load_lf_groups(
        frame,
        lf_global,
        Arc::get_mut(lf_groups).unwrap(),
        gmodular,
        modular_lf_region,
        parser,
        tasks,
    )
    .await?;
    gmodular = returned_modular;

    // A single-entry stream reaches HF-global only after LF parsing has indexed it.
    if hf_global.is_none() {
        let frame = Arc::clone(frame);
        let lf_global = Arc::clone(lf_global);
        let mut task_parser = parser.clone();
        let jobs: Vec<Job<_>> = vec![Box::new(move || {
            let result = frame
                .try_parse_hf_global_with_parser(Some(&*lf_global), &mut task_parser)
                .transpose()
                .map_err(Error::from);
            (result, task_parser)
        })];
        let mut outputs = tasks
            .run_all(jobs)
            .await
            .map_err(|_| Error::Color(jxl_color::Error::TaskCancelled))?;
        let (result, task_parser) = outputs.pop().unwrap();
        *parser = task_parser;
        *hf_global = result?.map(Arc::new);
    }

    let lf_xyb = if let Some(reference) = lf_frame {
        Arc::clone(&reference.image)
    } else {
        let mut lf_xyb = lf_xyb.unwrap();
        if !subsampled {
            super::chroma_from_luma_lf(
                lf_xyb.as_color_floats_mut(),
                &lf_global_vardct.lf_chan_corr,
            );
        }
        if !frame_header.flags.skip_adaptive_lf_smoothing() {
            super::adaptive_lf_smoothing(
                shifts_cbycr,
                lf_xyb.as_color_floats_mut(),
                &lf_global.lf_dequant,
                &lf_global_vardct.quantizer,
            )?;
        }
        Arc::new(lf_xyb)
    };

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let mut groups = Vec::new();
    for row in 0..modular_region.height.div_ceil(group_dim) {
        for col in 0..modular_region.width.div_ceil(group_dim) {
            let group_x = modular_region.left as u32 / group_dim + col;
            let group_y = modular_region.top as u32 / group_dim + row;
            let index = group_y * groups_per_row + group_x;
            if !lf_groups.contains_key(&frame_header.lf_group_idx_from_group_idx(index)) {
                continue;
            }
            let width = (modular_region.width - col * group_dim).min(group_dim);
            let height = (modular_region.height - row * group_dim).min(group_dim);
            let channels = shifts_cbycr.map(|shift| {
                let (w8, h8) = shift.shift_size((width.div_ceil(8), height.div_ceil(8)));
                AlignedGrid::with_alloc_tracker(w8 as usize * 8, h8 as usize * 8, tracker)
            });
            let [x, y, b] = channels;
            groups.push(CoefficientGroup {
                index,
                channels: [x?, y?, b?],
            });
        }
    }

    if let Some(hf_global) = hf_global.as_ref() {
        for pass_idx in 0..frame_header.passes.num_passes {
            if gmodular.modular.image().is_none() {
                let jobs: Vec<Job<_>> = groups
                    .into_iter()
                    .map(|mut group| {
                        let frame = Arc::clone(frame);
                        let lf_global = Arc::clone(lf_global);
                        let hf_global = Arc::clone(hf_global);
                        let lf_groups = Arc::clone(lf_groups);
                        let mut task_parser = parser.clone();
                        Box::new(move || {
                            decode_group(
                                &frame,
                                &lf_global,
                                &hf_global,
                                &lf_groups,
                                &mut group,
                                pass_idx,
                                None,
                                &mut task_parser,
                            )?;
                            Ok(group)
                        }) as Job<Result<CoefficientGroup>>
                    })
                    .collect();
                let results = tasks
                    .run_all(jobs)
                    .await
                    .map_err(|_| Error::Color(jxl_color::Error::TaskCancelled))?;
                groups = results.into_iter().collect::<Result<Vec<_>>>()?;
            } else {
                // Own the aggregate, construct its borrowed views only inside this job,
                // and move the coefficient owners through the pass without copying.
                let task_frame = Arc::clone(frame);
                let task_lf_global = Arc::clone(lf_global);
                let task_hf_global = Arc::clone(hf_global);
                let task_lf_groups = Arc::clone(lf_groups);
                let mut task_parser = parser.clone();
                let jobs: Vec<Job<_>> = vec![Box::new(move || -> Result<_> {
                    let pass_image = gmodular
                        .modular
                        .image_mut()
                        .unwrap()
                        .prepare_groups(task_frame.pass_shifts())?
                        .pass_groups
                        .into_iter()
                        .nth(pass_idx as usize)
                        .unwrap();
                    let mut image_it = pass_image.into_iter().enumerate();
                    let mut first_error = None;
                    for group in &mut groups {
                        let modular = image_it
                            .find(|(index, _)| *index == group.index as usize)
                            .map(|(_, image)| image);
                        let result = decode_group(
                            &task_frame,
                            &task_lf_global,
                            &task_hf_global,
                            &task_lf_groups,
                            group,
                            pass_idx,
                            modular,
                            &mut task_parser,
                        );
                        if first_error.is_none() {
                            first_error = result.err();
                        }
                    }
                    drop(image_it);
                    if let Some(error) = first_error {
                        return Err(error);
                    }
                    Ok((groups, gmodular, task_parser))
                })];
                let mut results = tasks
                    .run_all(jobs)
                    .await
                    .map_err(|_| Error::Color(jxl_color::Error::TaskCancelled))?;
                let (returned_groups, returned_modular, task_parser) = results.pop().unwrap()?;
                groups = returned_groups;
                gmodular = returned_modular;
                *parser = task_parser;
            }
        }
    }

    let jobs: Vec<Job<_>> = groups
        .into_iter()
        .map(|mut group| {
            let frame = Arc::clone(frame);
            let lf_global = Arc::clone(lf_global);
            let hf_global = hf_global.clone();
            let lf_groups = Arc::clone(lf_groups);
            let lf_xyb = Arc::clone(&lf_xyb);
            Box::new(move || {
                reconstruct_group(
                    &frame,
                    &lf_global,
                    hf_global.as_deref(),
                    &lf_groups,
                    &lf_xyb,
                    &mut group,
                    aligned_region,
                    subsampled,
                );
                group
            }) as Job<CoefficientGroup>
        })
        .collect();
    let groups = tasks
        .run_all(jobs)
        .await
        .map_err(|_| Error::Color(jxl_color::Error::TaskCancelled))?;

    // The current framebuffer requires one placement copy. Allocate it after the
    // coefficient stages and release each group as soon as its rows are placed.
    let mut fb = ImageWithRegion::new(3, tracker);
    for shift in shifts_cbycr {
        let (w8, h8) = shift.shift_size((
            modular_region.width.div_ceil(8),
            modular_region.height.div_ceil(8),
        ));
        let buffer = AlignedGrid::with_alloc_tracker(w8 as usize * 8, h8 as usize * 8, tracker)?;
        fb.append_channel_shifted(ImageBuffer::F32(buffer), modular_region, shift);
    }
    for group in groups {
        let left = group.index % groups_per_row * group_dim - modular_region.left as u32;
        let top = group.index / groups_per_row * group_dim - modular_region.top as u32;
        for ((output, input), shift) in fb
            .as_color_floats_mut()
            .into_iter()
            .zip(group.channels)
            .zip(shifts_cbycr)
        {
            let left = (left >> shift.hshift()) as usize;
            let top = (top >> shift.vshift()) as usize;
            for y in 0..input.height() {
                output.get_row_mut(top + y)[left..left + input.width()]
                    .copy_from_slice(input.get_row(y));
            }
        }
    }
    if let Some(image) = gmodular.modular.image_mut() {
        image.prepare_subimage()?.finish();
        fb.extend_from_gmodular(gmodular);
    }
    Ok(fb)
}

#[allow(clippy::too_many_arguments)]
fn decode_group<S: Sample>(
    frame: &IndexedFrame,
    lf_global: &LfGlobal<S>,
    hf_global: &HfGlobal,
    lf_groups: &BTreeMap<u32, LfGroup<S>>,
    group: &mut CoefficientGroup,
    pass_idx: u32,
    modular: Option<TransformedModularSubimage<S>>,
    parser: &mut FrameParser,
) -> Result<()> {
    let frame_header = frame.header();
    let lf_group = &lf_groups[&frame_header.lf_group_idx_from_group_idx(group.index)];
    if lf_group.hf_meta.is_none() {
        return Ok(());
    }
    let bitstream = match frame.pass_group_bitstream_with_parser(pass_idx, group.index, parser) {
        Some(Ok(bitstream)) => bitstream,
        Some(Err(error)) => return Err(error.into()),
        None => return Ok(()),
    };
    let allow_partial = bitstream.partial;
    let mut bitstream = bitstream.bitstream;
    let mut grids = group
        .channels
        .each_mut()
        .map(|grid| grid.as_subgrid_mut().into_i32());
    let result = jxl_frame::data::decode_pass_group(
        &mut bitstream,
        PassGroupParams {
            frame_header,
            lf_group,
            pass_idx,
            group_idx: group.index,
            global_ma_config: lf_global.gmodular.ma_config(),
            modular,
            vardct: Some(PassGroupParamsVardct {
                lf_vardct: lf_global.vardct.as_ref().unwrap(),
                hf_global,
                hf_coeff_output: &mut grids,
            }),
            allow_partial,
            tracker: frame.alloc_tracker(),
        },
    );
    if !allow_partial {
        result?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_group<S: Sample>(
    frame: &IndexedFrame,
    lf_global: &LfGlobal<S>,
    hf_global: Option<&HfGlobal>,
    lf_groups: &BTreeMap<u32, LfGroup<S>>,
    lf_xyb: &ImageWithRegion,
    group: &mut CoefficientGroup,
    aligned_region: Region,
    subsampled: bool,
) {
    let frame_header = frame.header();
    let group_dim = frame_header.group_dim();
    let group_x = group.index % frame_header.groups_per_row();
    let group_y = group.index / frame_header.groups_per_row();
    let lf_group = &lf_groups[&frame_header.lf_group_idx_from_group_idx(group.index)];
    let mut grids = group.channels.each_mut().map(AlignedGrid::as_subgrid_mut);
    let group_region = Region {
        left: (group_x * group_dim) as i32,
        top: (group_y * group_dim) as i32,
        width: group_dim,
        height: group_dim,
    };
    if lf_group.hf_meta.is_none()
        || hf_global.is_none()
        || group_region.intersection(aligned_region).is_empty()
    {
        super::transform_with_lf_grouped(lf_xyb, &mut grids, group.index, frame_header, lf_groups);
        return;
    }
    super::dequant_hf_varblock_grouped(
        &mut grids,
        group.index,
        frame.image_header(),
        frame_header,
        lf_global,
        lf_groups,
        hf_global.unwrap(),
    );
    if !subsampled {
        let hf_meta = lf_group.hf_meta.as_ref().unwrap();
        let cfl_base_x = ((group_x % 8) * group_dim / 64) as usize;
        let cfl_base_y = ((group_y % 8) * group_dim / 64) as usize;
        let gw = grids[0].width().div_ceil(64);
        let gh = grids[0].height().div_ceil(64);
        let x_from_y = hf_meta
            .x_from_y
            .as_subgrid()
            .subgrid(cfl_base_x..cfl_base_x + gw, cfl_base_y..cfl_base_y + gh);
        let b_from_y = hf_meta
            .b_from_y
            .as_subgrid()
            .subgrid(cfl_base_x..cfl_base_x + gw, cfl_base_y..cfl_base_y + gh);
        super::chroma_from_luma_hf_grouped(
            &mut grids,
            &x_from_y,
            &b_from_y,
            &lf_global.vardct.as_ref().unwrap().lf_chan_corr,
        );
    }
    super::transform_with_lf_grouped(lf_xyb, &mut grids, group.index, frame_header, lf_groups);
}
