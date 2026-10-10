#[cfg(all(target_arch = "aarch64", feature = "std"))]
use std::arch::is_aarch64_feature_detected;
#[cfg(all(target_arch = "x86_64", feature = "std"))]
use std::arch::is_x86_feature_detected;
// Without std there is no runtime detection; use the features the target was compiled with.
#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
macro_rules! is_aarch64_feature_detected {
    ($feature:tt) => {
        cfg!(target_feature = $feature)
    };
}
#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
macro_rules! is_x86_feature_detected {
    ($feature:tt) => {
        cfg!(target_feature = $feature)
    };
}
use core::num::Wrapping;

use jxl_grid::MutableSubgrid;

use crate::Sample;

type FnRow<S> = fn(&mut [&mut [S]; 3]);
type UnsafeFnRow<S> = unsafe fn(&mut [&mut [S]; 3]);

pub fn inverse_rct<S: Sample, const TYPE: u32>(
    permutation: u32,
    mut grids: [&mut MutableSubgrid<S>; 3],
) {
    let grid16 = grids.each_mut().map(|g| S::try_as_mutable_subgrid_i16(g));
    if let [Some(a), Some(b), Some(c)] = grid16 {
        let grids = [a, b, c];

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                run_rows_unsafe(permutation, grids, inverse_row_i16_x86_64_avx2::<TYPE>);
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                run_rows_unsafe(permutation, grids, inverse_row_i16_aarch64_neon::<TYPE>);
                return;
            }
        }

        run_rows(permutation, grids, inverse_row_i16_base::<TYPE>);
        return;
    }

    let grid32 = grids.each_mut().map(|g| S::try_as_mutable_subgrid_i32(g));
    if let [Some(a), Some(b), Some(c)] = grid32 {
        let grids = [a, b, c];

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                run_rows_unsafe(permutation, grids, inverse_row_i32_x86_64_avx2::<TYPE>);
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                run_rows_unsafe(permutation, grids, inverse_row_i32_aarch64_neon::<TYPE>);
                return;
            }
        }

        run_rows(permutation, grids, inverse_row_i32_base::<TYPE>);
    }
}

#[inline]
fn run_rows<S: Sample>(permutation: u32, grids: [&mut MutableSubgrid<S>; 3], f: FnRow<S>) {
    // SAFETY: `f` is safe.
    unsafe { run_rows_unsafe(permutation, grids, f) }
}

#[inline(never)]
unsafe fn run_rows_unsafe<S: Sample>(
    permutation: u32,
    grids: [&mut MutableSubgrid<S>; 3],
    f: UnsafeFnRow<S>,
) {
    let width = grids[0].width();
    let height = grids[0].height();
    assert_eq!(width, grids[1].width());
    assert_eq!(width, grids[2].width());
    assert_eq!(height, grids[1].height());
    assert_eq!(height, grids[2].height());
    if width == 0 || height == 0 {
        return;
    }

    let mut grids = grids;
    for y in 0..height {
        let mut rows = grids.each_mut().map(|g| g.get_row_mut(y));
        unsafe {
            f(&mut rows);
        }
        inverse_permute(permutation, rows);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn inverse_row_i16_x86_64_avx2<const TYPE: u32>(rows: &mut [&mut [i16]; 3]) {
    inverse_row_i16_base::<TYPE>(rows);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn inverse_row_i16_aarch64_neon<const TYPE: u32>(rows: &mut [&mut [i16]; 3]) {
    inverse_row_i16_base::<TYPE>(rows);
}

#[inline]
fn inverse_row_i16_base<const TYPE: u32>(rows: &mut [&mut [i16]; 3]) {
    let [a, b, c] = rows;

    for ((ra, rb), rc) in a.iter_mut().zip(&mut **b).zip(&mut **c) {
        let a = Wrapping(*ra);
        let b = Wrapping(*rb);
        let c = Wrapping(*rc);
        let d;
        let e;
        let f;
        if TYPE == 6 {
            let tmp = a - (c >> 1);
            e = c + tmp;
            f = tmp - (b >> 1);
            d = f + b;
        } else {
            d = a;
            f = if TYPE & 1 != 0 { c + a } else { c };
            e = if (TYPE >> 1) == 1 {
                b + a
            } else if (TYPE >> 1) == 2 {
                b + ((a + f) >> 1)
            } else {
                b
            };
        }
        *ra = d.0;
        *rb = e.0;
        *rc = f.0;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn inverse_row_i32_x86_64_avx2<const TYPE: u32>(rows: &mut [&mut [i32]; 3]) {
    inverse_row_i32_base::<TYPE>(rows);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn inverse_row_i32_aarch64_neon<const TYPE: u32>(rows: &mut [&mut [i32]; 3]) {
    inverse_row_i32_base::<TYPE>(rows);
}

#[inline]
fn inverse_row_i32_base<const TYPE: u32>(rows: &mut [&mut [i32]; 3]) {
    let [a, b, c] = rows;

    for ((ra, rb), rc) in a.iter_mut().zip(&mut **b).zip(&mut **c) {
        let a = Wrapping(*ra);
        let b = Wrapping(*rb);
        let c = Wrapping(*rc);
        let d;
        let e;
        let f;
        if TYPE == 6 {
            let tmp = a - (c >> 1);
            e = c + tmp;
            f = tmp - (b >> 1);
            d = f + b;
        } else {
            d = a;
            f = if TYPE & 1 != 0 { c + a } else { c };
            e = if (TYPE >> 1) == 1 {
                b + a
            } else if (TYPE >> 1) == 2 {
                b + ((a + f) >> 1)
            } else {
                b
            };
        }
        *ra = d.0;
        *rb = e.0;
        *rc = f.0;
    }
}

#[inline(always)]
fn inverse_permute<S: Sample>(permutation: u32, rows: [&mut [S]; 3]) {
    let [a, b, c] = rows;
    match permutation {
        1 => {
            a.swap_with_slice(b);
            a.swap_with_slice(c);
        }
        2 => {
            a.swap_with_slice(b);
            b.swap_with_slice(c);
        }
        3 => {
            b.swap_with_slice(c);
        }
        4 => {
            a.swap_with_slice(b);
        }
        5 => {
            a.swap_with_slice(c);
        }
        _ => {}
    }
}
