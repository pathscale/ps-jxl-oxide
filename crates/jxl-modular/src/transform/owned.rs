use std::sync::Arc;

use jxl_grid::{AlignedGrid, AllocTracker, MutableSubgrid};
use jxl_threadpool::Tasks;

use crate::{Error, Result, Sample};

// Four of the original sixteen-row/column stripes per job, eight jobs per batch.
const STRIPE: usize = 64;
const BATCH: usize = 8;

struct Stripe<S: Sample> {
    left: usize,
    top: usize,
    channels: Vec<AlignedGrid<S>>,
}

/// Repack only a bounded batch. Views stay in the coordinator or inside an owned job.
pub(super) async fn run_stripes<S, F>(
    grids: &mut [MutableSubgrid<'_, S>],
    horizontal: bool,
    tasks: &Tasks,
    tracker: Option<&AllocTracker>,
    kernel: F,
) -> Result<()>
where
    S: Sample,
    F: Fn(&mut [MutableSubgrid<'_, S>]) + Send + Sync + 'static,
{
    let Some(first) = grids.first() else {
        return Ok(());
    };
    let width = first.width();
    let height = first.height();
    for grid in grids.iter() {
        assert_eq!(width, grid.width());
        assert_eq!(height, grid.height());
    }
    if width == 0 || height == 0 {
        return Ok(());
    }

    let kernel = Arc::new(kernel);
    let extent = if horizontal { height } else { width };
    let mut start = 0;
    while start < extent {
        let mut jobs: Vec<Box<dyn FnOnce() -> Stripe<S> + Send + 'static>> =
            Vec::with_capacity(BATCH);
        for _ in 0..BATCH {
            if start == extent {
                break;
            }
            let count = (extent - start).min(STRIPE);
            let (left, top, stripe_width, stripe_height) = if horizontal {
                (0, start, width, count)
            } else {
                (start, 0, count, height)
            };
            let mut channels = Vec::with_capacity(grids.len());
            for grid in grids.iter() {
                let mut channel =
                    AlignedGrid::with_alloc_tracker(stripe_width, stripe_height, tracker)?;
                for y in 0..stripe_height {
                    channel.get_row_mut(y).copy_from_slice(
                        &grid.get_row(top + y)[left..left + stripe_width],
                    );
                }
                channels.push(channel);
            }
            let kernel = Arc::clone(&kernel);
            jobs.push(Box::new(move || {
                let mut views: Vec<_> = channels.iter_mut()
                    .map(AlignedGrid::as_subgrid_mut)
                    .collect();
                kernel(&mut views);
                drop(views);
                Stripe { left, top, channels }
            }));
            start += count;
        }

        let outputs = tasks.run_all(jobs).await.map_err(|_| Error::TaskCancelled)?;
        for output in outputs {
            for (grid, channel) in grids.iter_mut().zip(output.channels) {
                let width = channel.width();
                for y in 0..channel.height() {
                    grid.get_row_mut(output.top + y)[output.left..output.left + width]
                        .copy_from_slice(channel.get_row(y));
                }
            }
        }
    }
    Ok(())
}

