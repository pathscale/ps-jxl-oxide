//! Float functions core lacks: the inherent methods with std, libm or pure-Rust fallback without it.

#[inline]
pub(crate) fn ceil(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.ceil()
    }
    #[cfg(all(not(feature = "std"), feature = "libm"))]
    {
        libm::ceilf(x)
    }
    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    {
        jxl_oxide_common::math::ceilf(x)
    }
}
