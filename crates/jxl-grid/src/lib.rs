//! This crate provides [`AlignedGrid`], used in various places involving images.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{vec, vec::Vec};

mod alloc_tracker;
mod mutable_subgrid;
mod shared_subgrid;
mod simd;
pub use alloc_tracker::*;
pub use mutable_subgrid::*;
pub use shared_subgrid::*;
pub use simd::SimdVector;

#[cfg(all(test, feature = "std"))]
mod test;

/// The error type for failed grid allocation.
#[derive(Debug)]
pub struct OutOfMemory {
    bytes: usize,
}

impl OutOfMemory {
    fn new(bytes: usize) -> Self {
        Self { bytes }
    }

    /// Returns the size of failed allocation in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl core::error::Error for OutOfMemory {}

impl core::fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "failed to allocate {} byte(s)", self.bytes)
    }
}

/// A continuous buffer in the "raster order".
///
/// The beginning of the buffer is aligned so that it can be used in SIMD instructions.
pub struct AlignedGrid<S> {
    width: usize,
    height: usize,
    offset: usize,
    buf: Vec<S>,
    handle: Option<AllocHandle>,
}

impl<S> core::fmt::Debug for AlignedGrid<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AlignedGrid")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

impl<S> AlignedGrid<S> {
    /// Creates a zero-sized "empty" grid.
    #[inline]
    pub fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            offset: 0,
            buf: Vec::new(),
            handle: None,
        }
    }
}

impl<S: Default + Clone> AlignedGrid<S> {
    const ALIGN: usize = 32;

    #[inline]
    fn checked_layout(width: usize, height: usize) -> Result<(usize, usize), OutOfMemory> {
        let size = core::mem::size_of::<S>();
        let len = width
            .checked_mul(height)
            .ok_or_else(|| OutOfMemory::new(usize::MAX))?;
        let padding = (Self::ALIGN - 1) / size;
        let buf_len = len
            .checked_add(padding)
            .ok_or_else(|| OutOfMemory::new(usize::MAX))?;
        let bytes = buf_len
            .checked_mul(size)
            .ok_or_else(|| OutOfMemory::new(usize::MAX))?;
        if bytes > isize::MAX as usize {
            return Err(OutOfMemory::new(bytes));
        }

        Ok((len, buf_len))
    }

    /// Create a new buffer, recording the allocation if a tracker is given.
    #[inline]
    pub fn with_alloc_tracker(
        width: usize,
        height: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<Self, OutOfMemory> {
        let (len, buf_len) = Self::checked_layout(width, height)?;
        let handle = tracker
            .map(|tracker| tracker.alloc::<S>(buf_len))
            .transpose()?;
        let mut buf = vec![S::default(); buf_len];

        let extra = buf.as_ptr() as usize & (Self::ALIGN - 1);
        let offset = ((Self::ALIGN - extra) % Self::ALIGN) / core::mem::size_of::<S>();
        let len_with_offset = len
            .checked_add(offset)
            .ok_or_else(|| OutOfMemory::new(usize::MAX))?;
        buf.resize_with(len_with_offset, S::default);

        Ok(Self {
            width,
            height,
            offset,
            buf,
            handle,
        })
    }

    #[inline]
    fn empty_aligned(
        width: usize,
        height: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<Self, OutOfMemory> {
        let (len, buf_len) = Self::checked_layout(width, height)?;
        let handle = tracker
            .map(|tracker| tracker.alloc::<S>(buf_len))
            .transpose()?;
        let mut buf = Vec::with_capacity(buf_len);

        let extra = buf.as_ptr() as usize & (Self::ALIGN - 1);
        let offset = ((Self::ALIGN - extra) % Self::ALIGN) / core::mem::size_of::<S>();
        buf.resize_with(offset, S::default);

        Ok(Self {
            width,
            height,
            offset,
            buf,
            handle,
        })
    }

    /// Clones the buffer without recording an allocation.
    pub fn clone_untracked(&self) -> Self {
        let mut out = Self::empty_aligned(self.width, self.height, None).unwrap();
        out.buf.extend_from_slice(self.buf());
        out
    }

    /// Tries to clone the buffer, and records the allocation in the same tracker as the original
    /// buffer.
    pub fn try_clone(&self) -> Result<Self, OutOfMemory> {
        let mut out = Self::empty_aligned(self.width, self.height, self.tracker().as_ref())?;
        out.buf.extend_from_slice(self.buf());
        Ok(out)
    }
}

impl<S> AlignedGrid<S> {
    /// Returns the width of the grid.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the grid.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns allocation tracker associated with the grid.
    #[inline]
    pub fn tracker(&self) -> Option<AllocTracker> {
        self.handle.as_ref().map(|handle| handle.tracker())
    }

    /// Returns a reference to the sample at the given location.
    ///
    /// # Panics
    /// Panics if the coordinate is out of bounds.
    #[inline]
    pub fn get_ref(&self, x: usize, y: usize) -> &S {
        let width = self.width;
        let height = self.height;
        let Some(r) = self.try_get_ref(x, y) else {
            panic!("coordinate out of range: ({x}, {y}) not in {width}x{height}");
        };

        r
    }

    /// Returns a reference to the sample at the given location, or `None` if it is out of bounds.
    #[inline]
    pub fn try_get_ref(&self, x: usize, y: usize) -> Option<&S> {
        if x >= self.width || y >= self.height {
            return None;
        }

        Some(&self.buf[y * self.width + x + self.offset])
    }

    /// Returns a mutable reference to the sample at the given location.
    ///
    /// # Panics
    /// Panics if the coordinate is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut S {
        let width = self.width;
        let height = self.height;
        let Some(r) = self.try_get_mut(x, y) else {
            panic!("coordinate out of range: ({x}, {y}) not in {width}x{height}");
        };

        r
    }

    /// Returns a mutable reference to the sample at the given location, or `None` if it is out of
    /// bounds.
    #[inline]
    pub fn try_get_mut(&mut self, x: usize, y: usize) -> Option<&mut S> {
        if x >= self.width || y >= self.height {
            return None;
        }

        Some(&mut self.buf[y * self.width + x + self.offset])
    }

    /// Returns a slice of a row of samples.
    ///
    /// # Panics
    /// Panics if the row index is out of bounds.
    #[inline]
    pub fn get_row(&self, row: usize) -> &[S] {
        let height = self.height;
        let Some(slice) = self.try_get_row(row) else {
            panic!("row index out of range: height is {height} but index is {row}");
        };

        slice
    }

    /// Returns a slice of a row of samples, or `None` if it is out of bounds.
    #[inline]
    pub fn try_get_row(&self, y: usize) -> Option<&[S]> {
        if y >= self.height {
            return None;
        }

        Some(&self.buf[y * self.width + self.offset..][..self.width])
    }

    /// Returns a mutable slice of a row of samples.
    ///
    /// # Panics
    /// Panics if the row index is out of bounds.
    #[inline]
    pub fn get_row_mut(&mut self, row: usize) -> &mut [S] {
        let height = self.height;
        let Some(slice) = self.try_get_row_mut(row) else {
            panic!("row index out of range: height is {height} but index is {row}");
        };

        slice
    }

    /// Returns a mutable slice of a row of samples, or `None` if it is out of bounds.
    #[inline]
    pub fn try_get_row_mut(&mut self, y: usize) -> Option<&mut [S]> {
        if y >= self.height {
            return None;
        }

        Some(&mut self.buf[y * self.width + self.offset..][..self.width])
    }

    /// Returns the immutable slice to the underlying buffer.
    #[inline]
    pub fn buf(&self) -> &[S] {
        &self.buf[self.offset..]
    }

    /// Returns the mutable slice to the underlying buffer.
    #[inline]
    pub fn buf_mut(&mut self) -> &mut [S] {
        &mut self.buf[self.offset..]
    }

    /// Borrows the grid into a `SharedSubgrid`.
    #[inline]
    pub fn as_subgrid(&self) -> SharedSubgrid<'_, S> {
        SharedSubgrid::from(self)
    }

    /// Borrows the grid into a `MutableSubgrid`.
    #[inline]
    pub fn as_subgrid_mut(&mut self) -> MutableSubgrid<'_, S> {
        MutableSubgrid::from(self)
    }
}

impl<V: Copy> AlignedGrid<V> {
    /// Returns a copy of sample at the given location.
    ///
    /// # Panics
    /// Panics if the coordinate is out of range.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> V {
        *self.get_ref(x, y)
    }
}

#[cfg(test)]
mod grid_allocation_bounds_tests {
    use super::{AlignedGrid, AllocTracker, OutOfMemory};

    type Constructor<T> = fn(
        usize,
        usize,
        Option<&AllocTracker>,
    ) -> Result<AlignedGrid<T>, OutOfMemory>;

    fn check_rejected<T: Default + Clone>(
        constructor: Constructor<T>,
        width: usize,
        height: usize,
        expected_bytes: usize,
    ) {
        let err = constructor(width, height, None).unwrap_err();
        assert_eq!(err.bytes(), expected_bytes);

        let tracker = AllocTracker::with_limit(64);
        let err = constructor(width, height, Some(&tracker)).unwrap_err();
        assert_eq!(err.bytes(), expected_bytes);
        drop(tracker.alloc::<u8>(64).expect("rejection must not charge tracker"));
    }

    fn check_all_rejections<T: Default + Clone>(constructor: Constructor<T>) {
        check_rejected(constructor, usize::MAX, 2, usize::MAX);
        check_rejected(constructor, usize::MAX, 1, usize::MAX);
    }

    fn with_tracker_u8(
        width: usize,
        height: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<AlignedGrid<u8>, OutOfMemory> {
        AlignedGrid::with_alloc_tracker(width, height, tracker)
    }

    fn empty_aligned_u8(
        width: usize,
        height: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<AlignedGrid<u8>, OutOfMemory> {
        AlignedGrid::empty_aligned(width, height, tracker)
    }

    fn with_tracker_u64(
        width: usize,
        height: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<AlignedGrid<u64>, OutOfMemory> {
        AlignedGrid::with_alloc_tracker(width, height, tracker)
    }

    fn empty_aligned_u64(
        width: usize,
        height: usize,
        tracker: Option<&AllocTracker>,
    ) -> Result<AlignedGrid<u64>, OutOfMemory> {
        AlignedGrid::empty_aligned(width, height, tracker)
    }

    #[test]
    fn grid_allocation_bounds_reject_overflow_and_vec_ceiling() {
        check_all_rejections(with_tracker_u8);
        check_all_rejections(empty_aligned_u8);
        check_rejected(
            with_tracker_u8,
            isize::MAX as usize,
            1,
            isize::MAX as usize + 31,
        );
        check_rejected(
            empty_aligned_u8,
            isize::MAX as usize,
            1,
            isize::MAX as usize + 31,
        );
        check_rejected(
            with_tracker_u64,
            usize::MAX / 8 + 1,
            1,
            usize::MAX,
        );
        check_rejected(
            empty_aligned_u64,
            usize::MAX / 8 + 1,
            1,
            usize::MAX,
        );
    }

    #[test]
    fn grid_allocation_bounds_preserve_tracker_budget_and_drop_charges() {
        let tracker = AllocTracker::with_limit(31);
        let err = AlignedGrid::<u8>::with_alloc_tracker(1, 1, Some(&tracker)).unwrap_err();
        assert_eq!(err.bytes(), 32);
        drop(tracker.alloc::<u8>(31).expect("failed budget request must not charge"));

        let tracker = AllocTracker::with_limit(32);
        let grid = AlignedGrid::<u8>::with_alloc_tracker(1, 1, Some(&tracker)).unwrap();
        assert!(tracker.alloc::<u8>(1).is_err(), "live grid must retain its full charge");
        drop(grid);
        drop(tracker.alloc::<u8>(32).expect("drop must restore grid budget"));

        let tracker = AllocTracker::with_limit(64);
        let grid = AlignedGrid::<u8>::with_alloc_tracker(1, 1, Some(&tracker)).unwrap();
        let clone = grid.try_clone().unwrap();
        assert!(tracker.alloc::<u8>(1).is_err(), "live original and clone must exhaust the budget");
        drop(clone);
        drop(tracker.alloc::<u8>(32).expect("dropping clone restores its charge"));
        assert!(tracker.alloc::<u8>(33).is_err(), "live original must retain its charge");
        drop(grid);
        drop(tracker.alloc::<u8>(64).expect("dropping original restores its charge"));
    }

    fn check_small_grid<S>()
    where
        S: Default + Clone + Copy + PartialEq + core::fmt::Debug + From<u8>,
    {
        let mut grid = AlignedGrid::<S>::with_alloc_tracker(3, 2, None).unwrap();
        assert_eq!((grid.width(), grid.height()), (3, 2));
        assert_eq!(grid.buf().len(), 6);
        assert!(grid.buf().iter().all(|sample| *sample == S::default()));
        assert_eq!(grid.buf().as_ptr() as usize % 32, 0);
        *grid.get_mut(2, 1) = S::from(17);

        let clone = grid.try_clone().unwrap();
        assert_eq!((clone.width(), clone.height()), (3, 2));
        assert_eq!(clone.buf().len(), 6);
        assert_eq!(clone.get(2, 1), S::from(17));
        assert_eq!(clone.buf().as_ptr() as usize % 32, 0);

        for (width, height) in [(0, 0), (0, 3), (3, 0)] {
            let empty = AlignedGrid::<S>::with_alloc_tracker(width, height, None).unwrap();
            assert_eq!((empty.width(), empty.height()), (width, height));
            assert_eq!(empty.buf().len(), 0);
            assert_eq!(empty.buf().as_ptr() as usize % 32, 0);
            let cloned = empty.try_clone().unwrap();
            assert_eq!((cloned.width(), cloned.height()), (width, height));
            assert_eq!(cloned.buf().len(), 0);
        }
    }

    #[test]
    fn grid_allocation_bounds_keep_small_u8_and_u64_layouts() {
        check_small_grid::<u8>();
        check_small_grid::<u64>();
    }
}
