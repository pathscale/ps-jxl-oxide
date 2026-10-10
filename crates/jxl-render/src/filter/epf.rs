use std::{collections::HashMap, sync::Arc};

use jxl_frame::{FrameHeader, data::LfGroup, filter::EpfParams};
use jxl_grid::AlignedGrid;
use jxl_modular::Sample;
use jxl_threadpool::Tasks;

use crate::{Error, Result, util};

use super::owned::{FilterImage, STRIPE_HEIGHT, Stripe};

pub(crate) async fn apply_epf<S: Sample>(
    mut input: FilterImage,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    frame_header: &FrameHeader,
    epf_params: &EpfParams,
    tasks: &Tasks,
) -> Result<FilterImage> {
    let region = input.filter_region;
    assert!(region.left >= 0 && region.top >= 0);
    assert!(region.left % 8 == 0);
    assert!(region.top % 8 == 0);

    let sigma_width = (region.width as usize).div_ceil(8);
    let sigma_height = (region.height as usize).div_ceil(8);
    let mut sigma =
        AlignedGrid::with_alloc_tracker(sigma_width, sigma_height, input.tracker.as_ref())?;
    let num_lf_groups = frame_header.num_lf_groups() as usize;
    let mut sigma_grid_map = vec![None::<&AlignedGrid<f32>>; num_lf_groups];
    for (&lf_group_idx, lf_group) in lf_groups {
        if let Some(hf_meta) = &lf_group.hf_meta {
            sigma_grid_map[lf_group_idx as usize] = Some(&hf_meta.epf_sigma);
        }
    }

    let sigma_group_dim_shift = frame_header.group_dim().trailing_zeros();
    let sigma_group_dim_mask = (frame_header.group_dim() - 1) as usize;
    let groups_per_row = frame_header.lf_groups_per_row() as usize;
    let mut sigma_row = vec![epf_params.sigma_for_modular; sigma_width];
    for y in 0..sigma_height {
        let sigma_y = region.top as usize / 8 + y;
        let sigma_group_y = sigma_y >> sigma_group_dim_shift;
        let sigma_inner_y = sigma_y & sigma_group_dim_mask;
        for (dx, value) in sigma_row.iter_mut().enumerate() {
            let sigma_x = region.left as usize / 8 + dx;
            let sigma_group_x = sigma_x >> sigma_group_dim_shift;
            let sigma_inner_x = sigma_x & sigma_group_dim_mask;
            let sigma_grid_idx = sigma_group_y * groups_per_row + sigma_group_x;
            if let Some(grid) = sigma_grid_map[sigma_grid_idx] {
                let width = grid.width();
                *value = grid.buf()[sigma_inner_y * width + sigma_inner_x];
            }
        }
        // Preserve the inline path's carry when an LF sigma grid is absent.
        sigma.get_row_mut(y).copy_from_slice(&sigma_row);
    }

    let metadata = Arc::new(EpfMetadata {
        sigma,
        params: epf_params.clone(),
    });

    // Every awaited step is a barrier before the next input generation.
    if epf_params.iters == 3 {
        tracing::debug!("Running step 0");
        input = run_epf_step::<0>(input, Arc::clone(&metadata), tasks).await?;
    }

    tracing::debug!("Running step 1");
    input = run_epf_step::<1>(input, Arc::clone(&metadata), tasks).await?;

    if epf_params.iters >= 2 {
        tracing::debug!("Running step 2");
        input = run_epf_step::<2>(input, metadata, tasks).await?;
    }

    Ok(input)
}

struct EpfMetadata {
    sigma: AlignedGrid<f32>,
    params: EpfParams,
}

/// The SIMD kernels only require row access, so rows may belong to different stripes.
#[derive(Clone, Copy)]
pub(super) struct EpfInputRows<'buf> {
    rows: [&'buf [f32]; 7],
}

impl<'buf> EpfInputRows<'buf> {
    #[inline]
    pub(super) fn get_row(&self, y: usize) -> &'buf [f32] {
        self.rows[y]
    }

    // Only the x86_64 SSE4.1 EPF kernel reads single samples.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    #[inline]
    pub(super) fn get(&self, x: usize, y: usize) -> f32 {
        self.rows[y][x]
    }
}

pub(super) struct EpfRow<'buf, 'epf> {
    pub input_rows: [[&'buf [f32]; 7]; 3],
    #[allow(unused)]
    pub merged_input_rows: Option<[EpfInputRows<'buf>; 3]>,
    pub output_rows: [&'buf mut [f32]; 3],
    pub width: usize,
    pub y: usize,
    pub sigma_row: &'buf [f32],
    pub epf_params: &'epf EpfParams,
    pub skip_inner: bool,
}

pub(super) type EpfRowHandler = for<'a, 'b> unsafe fn(EpfRow<'a, 'b>);

async fn run_epf_step<const STEP: usize>(
    input: FilterImage,
    metadata: Arc<EpfMetadata>,
    tasks: &Tasks,
) -> Result<FilterImage> {
    let handle_row_simd = super::impls::epf_row_handler::<STEP>();
    let input = Arc::new(input);
    let width = input.filter_region.width as usize;
    let height = input.filter_region.height as usize;
    let top = input.filter_region.top as usize;
    let output_height = input.output_region.height as usize;
    let left = input.output_left();
    let right = left + width;

    let mut jobs: Vec<Box<dyn FnOnce() -> Stripe + Send + 'static>> =
        Vec::with_capacity(output_height.div_ceil(STRIPE_HEIGHT));
    for base_y in (0..output_height).step_by(STRIPE_HEIGHT) {
        let job_height = (output_height - base_y).min(STRIPE_HEIGHT);
        let mut output = input.allocate_stripe(job_height)?;
        let mut sigma =
            AlignedGrid::with_alloc_tracker(width.div_ceil(8), 1, input.tracker.as_ref())?;
        let input = Arc::clone(&input);
        let metadata = Arc::clone(&metadata);
        jobs.push(Box::new(move || {
            let [output0, output1, output2] = output.each_mut();
            for dy in 0..job_height {
                let output_y = base_y + dy;
                let mut output_rows = [
                    output0.get_row_mut(dy),
                    output1.get_row_mut(dy),
                    output2.get_row_mut(dy),
                ];
                for (channel, row) in output_rows.iter_mut().enumerate() {
                    input.copy_border(channel, output_y, &mut **row);
                }
                let Some(y) = input.filter_y(output_y) else {
                    continue;
                };

                let sigma_row = sigma.buf_mut();
                sigma_row.copy_from_slice(metadata.sigma.get_row(y / 8));
                let input_rows: [[_; 7]; 3] = std::array::from_fn(|channel| {
                    std::array::from_fn(|idx| {
                        let y = util::mirror((y + idx) as isize - 3, height);
                        input.row(channel, y)
                    })
                });
                let merged_input_rows = if y >= 3 && y + 4 <= height {
                    Some(input_rows.map(|rows| EpfInputRows { rows }))
                } else {
                    None
                };

                let [output0, output1, output2] = output_rows;
                let output0 = &mut output0[left..right];
                let output1 = &mut output1[left..right];
                let output2 = &mut output2[left..right];
                let image_y = top + y;
                let mut skip_inner = false;
                if merged_input_rows.is_some()
                    && let Some(handle_row_simd) = handle_row_simd
                {
                    skip_inner = true;
                    let output_rows = [&mut *output0, &mut *output1, &mut *output2];
                    let row = EpfRow {
                        input_rows,
                        merged_input_rows,
                        output_rows,
                        width,
                        y: image_y,
                        sigma_row,
                        epf_params: &metadata.params,
                        skip_inner,
                    };
                    // SAFETY: The selector checks the handler's CPU requirements.
                    unsafe {
                        handle_row_simd(row);
                    }
                }

                let output_rows = [output0, output1, output2];
                let row = EpfRow {
                    input_rows,
                    output_rows,
                    merged_input_rows,
                    width,
                    y: image_y,
                    sigma_row,
                    epf_params: &metadata.params,
                    skip_inner,
                };
                super::impls::generic::epf::epf_row::<STEP>(row);
            }
            output
        }));
    }

    let stripes = tasks
        .run_all(jobs)
        .await
        .map_err(|_| Error::NotSupported("filter task cancelled or returned an invalid output"))?;
    Ok(input.with_stripes(stripes))
}

pub(super) const fn epf_kernel_offsets<const STEP: usize>() -> &'static [(isize, isize)] {
    const EPF_KERNEL_1: [(isize, isize); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    #[rustfmt::skip]
    const EPF_KERNEL_2: [(isize, isize); 12] = [
        (0, -2), (-1, -1), (0, -1), (1, -1),
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (0, 2),
    ];

    if STEP == 0 {
        &EPF_KERNEL_2
    } else if STEP == 1 || STEP == 2 {
        &EPF_KERNEL_1
    } else {
        panic!()
    }
}

pub(super) const fn epf_dist_offsets<const STEP: usize>() -> &'static [(isize, isize)] {
    if STEP == 0 {
        &[(0, -1), (1, 0), (0, 0), (-1, 0), (0, 1)]
    } else if STEP == 1 {
        &[(0, -1), (0, 0), (0, 1), (-1, 0), (1, 0)]
    } else if STEP == 2 {
        &[(0, 0)]
    } else {
        panic!()
    }
}
