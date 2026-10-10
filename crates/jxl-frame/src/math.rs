//! Float functions core lacks: the inherent methods with std, libm without it.

#[inline]
pub(crate) fn ceil(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.ceil()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::ceilf(x)
    }
}
