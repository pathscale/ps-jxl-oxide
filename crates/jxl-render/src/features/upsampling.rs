use alloc::{boxed::Box, sync::Arc, vec::Vec};

use jxl_grid::{AlignedGrid, AllocTracker};
use jxl_threadpool::Tasks;

use crate::util::mirror;
use crate::{Error, Region};

const STRIPE_HEIGHT: usize = 64;

/// An immutable source channel and the crop used by the upsampling kernel.
#[derive(Clone)]
pub struct UpsamplingInput {
    grid: Arc<AlignedGrid<f32>>,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
}

impl UpsamplingInput {
    pub fn new(
        grid: Arc<AlignedGrid<f32>>,
        left: usize,
        top: usize,
        width: usize,
        height: usize,
    ) -> Self {
        assert!(left <= grid.width() && width <= grid.width() - left);
        assert!(top <= grid.height() && height <= grid.height() - top);
        Self {
            grid,
            left,
            top,
            width,
            height,
        }
    }

    fn from_grid(grid: AlignedGrid<f32>) -> Self {
        let width = grid.width();
        let height = grid.height();
        Self::new(Arc::new(grid), 0, 0, width, height)
    }

    fn padded_row(&self, y: usize) -> Option<&[f32]> {
        let y = if self.height == 1 {
            // Preserve the zero outer top row produced by PaddedGrid.
            if y == 0 {
                return None;
            }
            0
        } else {
            mirror(y as isize - 2, self.height)
        };
        Some(&self.grid.get_row(self.top + y)[self.left..][..self.width])
    }
}

pub async fn upsample(
    grid: UpsamplingInput,
    out_region: &mut Region,
    image_header: &jxl_image::ImageHeader,
    factor: u32,
    tracker: Option<&AllocTracker>,
    tasks: &Tasks,
) -> crate::Result<Option<AlignedGrid<f32>>> {
    let metadata = &image_header.metadata;

    let mut grid = grid;

    let up8 = factor / 3;
    let last_up = factor % 3;
    for _ in 0..up8 {
        grid = UpsamplingInput::from_grid(
            upsample_inner::<8, 210>(grid, &metadata.up8_weight, tracker, tasks).await?,
        );
    }
    let out = match last_up {
        1 => Some(upsample_inner::<2, 15>(grid, &metadata.up2_weight, tracker, tasks).await?),
        2 => Some(upsample_inner::<4, 55>(grid, &metadata.up4_weight, tracker, tasks).await?),
        _ if up8 != 0 => Some(
            Arc::try_unwrap(grid.grid)
                .map_err(|_| Error::NotSupported("upsampling output is still shared"))?,
        ),
        _ => None,
    };
    *out_region = out_region.upsample(factor);
    Ok(out)
}

async fn upsample_inner<const K: usize, const NW: usize>(
    grid: UpsamplingInput,
    weights: &[f32; NW],
    tracker: Option<&AllocTracker>,
    tasks: &Tasks,
) -> crate::Result<AlignedGrid<f32>> {
    assert!((K == 2 && NW == 15) || (K == 4 && NW == 55) || (K == 8 && NW == 210));
    let grid_width = grid.width;
    let grid_height = grid.height;
    let frame_width = grid_width << K.ilog2();
    let frame_height = grid_height << K.ilog2();

    if frame_width == 0 || frame_height == 0 {
        return Ok(AlignedGrid::with_alloc_tracker(
            frame_width,
            frame_height,
            tracker,
        )?);
    }

    // 5x5 kernel
    const PADDING: usize = 2;
    let mut padded_x = AlignedGrid::with_alloc_tracker(grid_width + PADDING * 2, 1, tracker)?;
    for (x, source_x) in padded_x.buf_mut().iter_mut().enumerate() {
        *source_x = if grid_width == 1 {
            // PaddedGrid leaves both outer horizontal padding samples zero.
            if x == 0 || x == 4 { usize::MAX } else { 0 }
        } else {
            mirror(x as isize - PADDING as isize, grid_width)
        };
    }
    let padded_x = Arc::new(padded_x);

    // For K = 8:
    // 0  1  2  3  3h 2h 1h 0h
    // 4  5  6  7  7h 6h 5h 4h
    // 8  9  a  b  bh ah 9h 8h
    // c  d  e  f  fh eh dh ch
    // cv dv ev fv f' e' d' c'
    // 8v 9v av bv b' a' 9' 8'
    // 4v 5v 6v 7v 7' 6' 5' 4'
    // 0v 1v 2v 3v 3' 2' 1' 0'
    let mut weights_quarter = vec![[0.0f32; 25]; K * K / 4];
    let mut weight_idx = 0usize;
    let mat_n = K / 2;
    for y in 0..5 * mat_n {
        let mat_y = y / 5;
        let ky = y % 5;
        for x in y..5 * mat_n {
            let mat_x = x / 5;
            let kx = x % 5;
            let w = weights[weight_idx];
            weight_idx += 1;

            weights_quarter[mat_y * mat_n + mat_x][ky * 5 + kx] = w;
            weights_quarter[mat_x * mat_n + mat_y][kx * 5 + ky] = w;
        }
    }
    let weights_quarter = Arc::new(weights_quarter);

    // One inline job writes the final grid, keeping its padded source scratch bounded.
    let stripe_height = if tasks.is_inline() {
        frame_height
    } else {
        STRIPE_HEIGHT
    };
    let mut jobs: Vec<Box<dyn FnOnce() -> AlignedGrid<f32> + Send + 'static>> =
        Vec::with_capacity(frame_height.div_ceil(stripe_height));
    for base_y in (0..frame_height).step_by(stripe_height) {
        let job_height = (frame_height - base_y).min(stripe_height);
        let mut output = AlignedGrid::with_alloc_tracker(frame_width, job_height, tracker)?;
        let grid = grid.clone();
        let padded_x = Arc::clone(&padded_x);
        let weights_quarter = Arc::clone(&weights_quarter);

        jobs.push(Box::new(move || {
            for block_y in (base_y..base_y + job_height).step_by(STRIPE_HEIGHT) {
                let block_height = (base_y + job_height - block_y).min(STRIPE_HEIGHT);
                // A private padded copy of the source rows this stripe reads, so the kernel
                // indexes one contiguous buffer exactly as the serial version did.
                let first_ref = block_y / K;
                let last_ref = (block_y + block_height - 1) / K;
                let padded_width = grid_width + PADDING * 2;
                let padded_rows = last_ref - first_ref + 5;
                let padded_x = padded_x.buf();
                let mut padded = vec![0.0f32; padded_width * padded_rows];
                for (row_idx, row) in padded.chunks_exact_mut(padded_width).enumerate() {
                    if let Some(source) = grid.padded_row(first_ref + row_idx) {
                        for (sample, &source_x) in row.iter_mut().zip(padded_x) {
                            if source_x != usize::MAX {
                                *sample = source[source_x];
                            }
                        }
                    }
                }

                let grid_buf = output.buf_mut();
                for dy in 0..block_height {
                    let y = block_y + dy;
                    let ref_y = y / K - first_ref;
                    let mat_y = (y % K).min(K - y % K - 1);
                    let flip_v = y % K >= mat_n;
                    for x in 0..frame_width {
                        let ref_x = x / K;
                        let mat_x = (x % K).min(K - x % K - 1);
                        let flip_h = x % K >= mat_n;

                        let kernel = &weights_quarter[mat_y * mat_n + mat_x];
                        let mut sum = 0.0f32;
                        let mut min = f32::INFINITY;
                        let mut max = -f32::INFINITY;
                        for iy in 0..5 {
                            let ky = if flip_v { 4 - iy } else { iy };
                            for ix in 0..5 {
                                let kx = if flip_h { 4 - ix } else { ix };
                                let sample = padded[(ref_y + iy) * padded_width + (ref_x + ix)];
                                sum += kernel[ky * 5 + kx] * sample;
                                min = min.min(sample);
                                max = max.max(sample);
                            }
                        }

                        grid_buf[(y - base_y) * frame_width + x] = if !min.is_finite() {
                            // This happens if all samples are NaN. f32::clamp would panic
                            // in this case.
                            f32::NAN
                        } else {
                            sum.clamp(min, max)
                        };
                    }
                }
            }
            output
        }));
    }

    let mut stripes = tasks.run_all(jobs).await.map_err(|_| {
        Error::NotSupported("upsampling task cancelled or returned an invalid output")
    })?;
    drop(grid);
    drop(padded_x);
    drop(weights_quarter);

    if stripes.len() == 1 {
        return Ok(stripes.pop().unwrap());
    }

    let mut output = AlignedGrid::with_alloc_tracker(frame_width, frame_height, tracker)?;
    for (index, stripe) in stripes.into_iter().enumerate() {
        let base_y = index * STRIPE_HEIGHT;
        for y in 0..stripe.height() {
            output
                .get_row_mut(base_y + y)
                .copy_from_slice(stripe.get_row(y));
        }
    }
    Ok(output)
}
