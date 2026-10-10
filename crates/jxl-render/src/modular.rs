use alloc::{boxed::Box, sync::Arc, vec::Vec};

use jxl_bitstream::{Bitstream, Section};
use jxl_frame::{
    FrameHeader,
    data::{GlobalModular, LfGlobal, LfGlobalParams, LfGroup, TocGroupKind},
};
use jxl_grid::AllocTracker;
use jxl_modular::{
    MaConfig, Sample,
    image::{OwnedModularSubimage, TransformedModularSubimage},
};
use jxl_oxide_common::Bundle;
use jxl_threadpool::Tasks;

use crate::{Error, ImageWithRegion, IndexedFrame, Region, RenderCache, Result};

const GROUP_BATCH: usize = 8;

struct GroupWork<'dest, S: Sample> {
    kind: TocGroupKind,
    stream_index: u32,
    modular: Option<TransformedModularSubimage<'dest, S>>,
}

struct GroupInput {
    section: Section,
    offset: usize,
    stream_index: u32,
    allow_partial: bool,
    ma_config: Option<Arc<MaConfig>>,
    tracker: Option<AllocTracker>,
}

struct GroupOutput<S: Sample> {
    modular: Option<OwnedModularSubimage<S>>,
    result: jxl_frame::Result<()>,
    offset: usize,
}

fn decode_group<S: Sample>(
    input: Arc<GroupInput>,
    mut modular: Option<OwnedModularSubimage<S>>,
) -> GroupOutput<S> {
    let mut bitstream = Bitstream::from_section(&input.section);
    let result = bitstream
        .skip_bits(input.offset)
        .map_err(jxl_frame::Error::from)
        .and_then(|()| {
            if let Some(modular) = &mut modular {
                modular
                    .decode_recursive(
                        &mut bitstream,
                        input.ma_config.as_deref(),
                        input.stream_index,
                        input.allow_partial,
                        input.tracker.as_ref(),
                    )
                    .map_err(jxl_frame::Error::from)?;
            }
            Ok(())
        });
    GroupOutput {
        modular,
        result,
        offset: bitstream.num_read_bits(),
    }
}

fn group_section(frame: &IndexedFrame, kind: TocGroupKind) -> Option<(Section, bool)> {
    let section = frame.data(kind)?.clone();
    let size = frame
        .toc()
        .iter_bitstream_order()
        .find(|group| group.kind == kind)?
        .size;
    let partial = section.len() < size as usize;
    Some((section, partial))
}

pub(crate) async fn render_modular<S: Sample>(
    frame: &IndexedFrame,
    cache: &mut RenderCache<S>,
    region: Region,
    tasks: &Tasks,
) -> Result<ImageWithRegion> {
    let image_header = frame.image_header();
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();
    let xyb_encoded = image_header.metadata.xyb_encoded;
    let single = frame.toc().is_single_entry();
    let mut single_offset = 0;

    // Single-entry sections have no independent LF/pass boundaries. Own the cursor here.
    let single_section = if single {
        let (section, allow_partial) =
            group_section(frame, TocGroupKind::All).ok_or(Error::IncompleteFrame)?;
        let mut bitstream = Bitstream::from_section(&section);
        let parsed = LfGlobal::<S>::parse(
            &mut bitstream,
            LfGlobalParams::new(image_header, frame_header, tracker, allow_partial),
        );
        let lf_global = match parsed {
            Err(error) if allow_partial && error.unexpected_eof() => {
                return Err(Error::IncompleteFrame);
            }
            other => other?,
        };
        single_offset = bitstream.num_read_bits();
        cache.lf_global = Some(lf_global);
        Some((section, allow_partial))
    } else {
        if cache.lf_global.is_none() {
            cache.lf_global = Some(
                frame
                    .try_parse_lf_global()
                    .ok_or(Error::IncompleteFrame)??,
            );
        }
        None
    };

    let lf_global = cache.lf_global.as_ref().unwrap();
    let mut gmodular = lf_global.gmodular.try_clone()?;
    let modular_region = compute_modular_region(frame_header, &gmodular, region, false);
    let global_ma_config = gmodular.ma_config().cloned().map(Arc::new);
    let modular_image = gmodular.modular.image_mut().unwrap();
    let groups = modular_image.prepare_groups(frame.pass_shifts())?;

    // Stable reduction order: LF raster order, then pass-major group raster order.
    let group_dim = frame_header.group_dim();
    let lf_region = modular_region.downsample(3);
    let mut work = Vec::new();
    let mut lf_images = groups.lf_groups.into_iter();
    for group_idx in 0..frame_header.num_lf_groups() {
        let modular = lf_images.next();
        let groups_per_row = frame_header.lf_groups_per_row();
        let group_region = Region {
            left: ((group_idx % groups_per_row) * group_dim) as i32,
            top: ((group_idx / groups_per_row) * group_dim) as i32,
            width: group_dim,
            height: group_dim,
        };
        if !single && group_region.intersection(lf_region).is_empty() {
            continue;
        }
        work.push(GroupWork {
            kind: TocGroupKind::LfGroup(group_idx),
            stream_index: 1 + frame_header.num_lf_groups() + group_idx,
            modular,
        });
    }
    for (pass_idx, pass_image) in groups.pass_groups.into_iter().enumerate() {
        let pass_idx = pass_idx as u32;
        for (group_idx, modular) in pass_image.into_iter().enumerate() {
            let group_idx = group_idx as u32;
            let groups_per_row = frame_header.groups_per_row();
            let group_region = Region {
                left: ((group_idx % groups_per_row) * group_dim) as i32,
                top: ((group_idx / groups_per_row) * group_dim) as i32,
                width: group_dim,
                height: group_dim,
            };
            if !single && group_region.intersection(modular_region).is_empty() {
                continue;
            }
            work.push(GroupWork {
                kind: TocGroupKind::GroupPass {
                    pass_idx,
                    group_idx,
                },
                stream_index: 1
                    + 3 * frame_header.num_lf_groups()
                    + 17
                    + pass_idx * frame_header.num_groups()
                    + group_idx,
                modular: Some(modular),
            });
        }
    }

    let mut work = work.into_iter();
    let mut stop_single = false;
    loop {
        let mut batch = Vec::with_capacity(GROUP_BATCH);
        let mut jobs: Vec<Box<dyn FnOnce() -> GroupOutput<S> + Send + 'static>> =
            Vec::with_capacity(GROUP_BATCH);
        let batch_size = if single { 1 } else { GROUP_BATCH };
        for _ in 0..batch_size {
            let Some(item) = work.next() else {
                break;
            };
            if stop_single {
                continue;
            }
            let source = if let Some((section, partial)) = &single_section {
                Some((section.clone(), *partial, single_offset))
            } else {
                group_section(frame, item.kind).map(|(section, partial)| (section, partial, 0))
            };
            let Some((section, allow_partial, offset)) = source else {
                continue;
            };
            let owned = item
                .modular
                .as_ref()
                .map(|modular| modular.to_owned(tracker))
                .transpose()?;
            let input = Arc::new(GroupInput {
                section,
                offset,
                stream_index: item.stream_index,
                allow_partial,
                ma_config: global_ma_config.clone(),
                tracker: tracker.cloned(),
            });
            jobs.push(Box::new(move || decode_group(input, owned)));
            batch.push((item, allow_partial));
        }
        if jobs.is_empty() {
            if work.len() == 0 {
                break;
            }
            continue;
        }

        let outputs = tasks
            .run_all(jobs)
            .await
            .map_err(|_| Error::Modular(jxl_modular::Error::TaskCancelled))?;
        for ((mut item, allow_partial), mut output) in batch.into_iter().zip(outputs) {
            let is_lf = matches!(item.kind, TocGroupKind::LfGroup(_));
            let decoded = output.result.is_ok();
            if let Err(error) = output.result {
                let suppress = allow_partial && (!is_lf || !single || error.unexpected_eof());
                if !suppress {
                    return Err(error.into());
                }
                if single && is_lf {
                    stop_single = true;
                }
            }
            if single {
                single_offset = output.offset;
            }
            let mut complete = true;
            if decoded {
                if let Some(modular) = &mut output.modular {
                    complete = modular.finish_with_tasks(tasks, tracker).await?;
                }
                if let TocGroupKind::LfGroup(idx) = item.kind {
                    cache.lf_groups.insert(
                        idx,
                        LfGroup {
                            lf_coeff: None,
                            hf_meta: None,
                            partial: !complete,
                        },
                    );
                }
            }
            if let (Some(target), Some(owned)) = (&mut item.modular, output.modular) {
                target.copy_from_owned(owned);
            }
        }
    }

    modular_image
        .prepare_subimage()?
        .finish_with_tasks(tasks, tracker)
        .await?;

    let mut fb = ImageWithRegion::new(frame_header.encoded_color_channels(), tracker);
    fb.extend_from_gmodular(gmodular);

    if xyb_encoded {
        tracing::trace_span!("Dequant XYB")
            .in_scope(|| fb.convert_modular_xyb(&lf_global.lf_dequant))?;
    }

    Ok(fb)
}

#[inline]
pub fn compute_modular_region<S: Sample>(
    frame_header: &FrameHeader,
    gmodular: &GlobalModular<S>,
    region: Region,
    is_lf: bool,
) -> Region {
    if gmodular.modular.has_palette() || gmodular.modular.has_squeeze() {
        let mut width = frame_header.color_sample_width();
        let mut height = frame_header.color_sample_height();
        if is_lf {
            width = width.div_ceil(8);
            height = height.div_ceil(8);
        }
        let width = width.max(region.width.checked_add_signed(region.left).unwrap());
        let height = height.max(region.height.checked_add_signed(region.top).unwrap());
        Region::with_size(width, height)
    } else {
        region
    }
}
