use jxl_grid::{AlignedGrid, AllocTracker};

use crate::{ImageBuffer, ImageWithRegion, Region, Result};

pub(super) const STRIPE_HEIGHT: usize = 64;
pub(super) type Stripe = [AlignedGrid<f32>; 3];

enum Storage {
    Planes(Stripe),
    Stripes(Vec<Stripe>),
}

/// Owns the current filter generation without flattening intermediate stripes.
pub(crate) struct FilterImage {
    region: Region,
    pub(super) filter_region: Region,
    pub(super) output_region: Region,
    pub(super) tracker: Option<AllocTracker>,
    storage: Storage,
}

impl FilterImage {
    pub(crate) fn take_from(
        image: &mut ImageWithRegion,
        filter_region: Region,
        preserve_outside: bool,
    ) -> Self {
        let region = image.regions_and_shifts()[0].0;
        assert!(region.contains(filter_region));
        let tracker = image.alloc_tracker().cloned();
        let planes = std::array::from_fn(|channel| {
            let buffer = std::mem::replace(
                &mut image.buffer_mut()[channel],
                ImageBuffer::F32(AlignedGrid::empty()),
            );
            let ImageBuffer::F32(grid) = buffer else {
                panic!("filter input is not F32");
            };
            grid
        });
        Self {
            region,
            filter_region,
            output_region: if preserve_outside {
                region
            } else {
                filter_region
            },
            tracker,
            storage: Storage::Planes(planes),
        }
    }

    pub(super) fn allocate_stripe(&self, height: usize) -> Result<Stripe> {
        let width = self.output_region.width as usize;
        let tracker = self.tracker.as_ref();
        Ok([
            AlignedGrid::with_alloc_tracker(width, height, tracker)?,
            AlignedGrid::with_alloc_tracker(width, height, tracker)?,
            AlignedGrid::with_alloc_tracker(width, height, tracker)?,
        ])
    }

    fn raw_row(&self, channel: usize, y: usize) -> &[f32] {
        match &self.storage {
            Storage::Planes(planes) => planes[channel].get_row(y),
            Storage::Stripes(stripes) => {
                stripes[y / STRIPE_HEIGHT][channel].get_row(y % STRIPE_HEIGHT)
            }
        }
    }

    pub(super) fn row(&self, channel: usize, y: usize) -> &[f32] {
        let left = self.region.left.abs_diff(self.filter_region.left) as usize;
        let top = self.region.top.abs_diff(self.filter_region.top) as usize;
        &self.raw_row(channel, top + y)[left..left + self.filter_region.width as usize]
    }

    pub(super) fn output_left(&self) -> usize {
        self.output_region.left.abs_diff(self.filter_region.left) as usize
    }

    pub(super) fn filter_y(&self, output_y: usize) -> Option<usize> {
        if self.filter_region.is_empty() {
            return None;
        }
        let top = self.output_region.top.abs_diff(self.filter_region.top) as usize;
        output_y
            .checked_sub(top)
            .filter(|&y| y < self.filter_region.height as usize)
    }

    /// Carries unchanged pixels for the two-pass EPF destination region.
    pub(super) fn copy_border(&self, channel: usize, output_y: usize, output: &mut [f32]) {
        let source_left = self.region.left.abs_diff(self.output_region.left) as usize;
        let source_top = self.region.top.abs_diff(self.output_region.top) as usize;
        let source =
            &self.raw_row(channel, source_top + output_y)[source_left..source_left + output.len()];
        if self.filter_y(output_y).is_none() {
            output.copy_from_slice(source);
            return;
        }
        let left = self.output_left();
        let right = left + self.filter_region.width as usize;
        output[..left].copy_from_slice(&source[..left]);
        output[right..].copy_from_slice(&source[right..]);
    }

    pub(super) fn with_stripes(&self, stripes: Vec<Stripe>) -> Self {
        Self {
            region: self.output_region,
            filter_region: self.filter_region,
            output_region: self.output_region,
            tracker: self.tracker.clone(),
            storage: Storage::Stripes(stripes),
        }
    }

    /// Performs the single placement copy required by the contiguous consumer.
    pub(crate) fn write_to(self, image: &mut ImageWithRegion) -> Result<()> {
        let Self {
            region,
            tracker,
            storage,
            ..
        } = self;
        let planes = match storage {
            Storage::Planes(planes) => planes,
            Storage::Stripes(mut stripes) if stripes.len() == 1 => stripes.pop().unwrap(),
            Storage::Stripes(stripes) => {
                let width = region.width as usize;
                let height = region.height as usize;
                let mut planes = [
                    AlignedGrid::with_alloc_tracker(width, height, tracker.as_ref())?,
                    AlignedGrid::with_alloc_tracker(width, height, tracker.as_ref())?,
                    AlignedGrid::with_alloc_tracker(width, height, tracker.as_ref())?,
                ];
                for (index, stripe) in stripes.into_iter().enumerate() {
                    let base_y = index * STRIPE_HEIGHT;
                    for (output, stripe) in planes.iter_mut().zip(stripe) {
                        for y in 0..stripe.height() {
                            output
                                .get_row_mut(base_y + y)
                                .copy_from_slice(stripe.get_row(y));
                        }
                    }
                }
                planes
            }
        };
        for (channel, plane) in planes.into_iter().enumerate() {
            image.replace_channel(channel, ImageBuffer::F32(plane), region);
        }
        Ok(())
    }
}
