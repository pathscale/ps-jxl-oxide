//! Float functions core lacks: the inherent methods with std, libm or pure-Rust fallback without it.

#[inline]
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.powf(y)
    }
    #[cfg(all(not(feature = "std"), feature = "libm"))]
    {
        libm::powf(x, y)
    }
    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    {
        jxl_oxide_common::math::powf(x, y)
    }
}

#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(all(not(feature = "std"), feature = "libm"))]
    {
        libm::sqrtf(x)
    }
    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    {
        jxl_oxide_common::math::sqrtf(x)
    }
}

#[inline]
pub(crate) fn round(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.round()
    }
    #[cfg(all(not(feature = "std"), feature = "libm"))]
    {
        libm::roundf(x)
    }
    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    {
        jxl_oxide_common::math::roundf(x)
    }
}
