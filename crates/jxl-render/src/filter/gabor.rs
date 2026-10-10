use std::sync::Arc;

use jxl_threadpool::Tasks;

use crate::{Error, Result};

use super::{
    impls::generic::gabor::gabor_row_edge,
    owned::{FilterImage, STRIPE_HEIGHT, Stripe},
};

pub(crate) async fn apply_gabor_like(
    input: FilterImage,
    weights: [[f32; 2]; 3],
    tasks: &Tasks,
) -> Result<FilterImage> {
    tracing::debug!("Running gaborish");
    let handle_row = super::impls::gabor_row_handler();
    let input = Arc::new(input);
    let height = input.filter_region.height as usize;
    let output_height = input.output_region.height as usize;
    let left = input.output_left();
    let right = left + input.filter_region.width as usize;

    let mut jobs: Vec<Box<dyn FnOnce() -> Stripe + Send + 'static>> =
        Vec::with_capacity(output_height.div_ceil(STRIPE_HEIGHT));
    for base_y in (0..output_height).step_by(STRIPE_HEIGHT) {
        let job_height = (output_height - base_y).min(STRIPE_HEIGHT);
        let mut output = input.allocate_stripe(job_height)?;
        let input = Arc::clone(&input);
        jobs.push(Box::new(move || {
            for (channel, plane) in output.iter_mut().enumerate() {
                for dy in 0..job_height {
                    let output_y = base_y + dy;
                    let output_row = plane.get_row_mut(dy);
                    input.copy_border(channel, output_y, output_row);
                    let Some(y) = input.filter_y(output_y) else {
                        continue;
                    };
                    let output_row = &mut output_row[left..right];
                    if height == 1 {
                        gabor_row_edge(input.row(channel, 0), None, output_row, weights[channel]);
                    } else if y == 0 {
                        gabor_row_edge(
                            input.row(channel, 0),
                            Some(input.row(channel, 1)),
                            output_row,
                            weights[channel],
                        );
                    } else if y == height - 1 {
                        gabor_row_edge(
                            input.row(channel, y),
                            Some(input.row(channel, y - 1)),
                            output_row,
                            weights[channel],
                        );
                    } else {
                        let row = GaborRow {
                            input_rows: [
                                input.row(channel, y - 1),
                                input.row(channel, y),
                                input.row(channel, y + 1),
                            ],
                            output_row,
                            weights: weights[channel],
                        };
                        // SAFETY: The selector checks the handler's CPU requirements.
                        unsafe {
                            handle_row(row);
                        }
                    }
                }
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

pub(super) struct GaborRow<'buf> {
    pub input_rows: [&'buf [f32]; 3],
    pub output_row: &'buf mut [f32],
    pub weights: [f32; 2],
}

pub(super) type GaborRowHandler = for<'a> unsafe fn(GaborRow<'a>);

#[allow(unused)]
pub(crate) fn run_gabor_row_generic(row: GaborRow) {
    super::impls::generic::gabor::run_gabor_row_generic(row)
}

