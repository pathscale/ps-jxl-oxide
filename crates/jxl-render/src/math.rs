macro_rules! unary {
    ($name:ident, $ty:ty, $method:ident, $libm:ident, $pure:ident) => {
        #[inline]
        pub(crate) fn $name(value: $ty) -> $ty {
            #[cfg(feature = "std")]
            {
                value.$method()
            }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            {
                libm::$libm(value)
            }
            #[cfg(all(not(feature = "std"), not(feature = "libm")))]
            {
                jxl_oxide_common::math::$pure(value)
            }
        }
    };
}

unary!(sqrt, f32, sqrt, sqrtf, sqrtf);
unary!(ln, f32, ln, logf, logf);
unary!(floor, f32, floor, floorf, floorf);
unary!(cos, f32, cos, cosf, cosf);
#[cfg(test)]
unary!(cos64, f64, cos, cos, cos);

#[inline]
pub(crate) fn powf(value: f32, exponent: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        value.powf(exponent)
    }
    #[cfg(all(not(feature = "std"), feature = "libm"))]
    {
        libm::powf(value, exponent)
    }
    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    {
        jxl_oxide_common::math::powf(value, exponent)
    }
}

#[inline]
pub(crate) fn powi(value: f32, exponent: i32) -> f32 {
    #[cfg(feature = "std")]
    {
        value.powi(exponent)
    }
    #[cfg(all(not(feature = "std"), feature = "libm"))]
    {
        libm::pow(value as f64, exponent as f64) as f32
    }
    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    {
        jxl_oxide_common::math::powif(value, exponent)
    }
}

#[inline]
pub(crate) fn mul_add(value: f32, multiplier: f32, addend: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        value.mul_add(multiplier, addend)
    }
    #[cfg(all(not(feature = "std"), feature = "libm"))]
    {
        libm::fmaf(value, multiplier, addend)
    }
    #[cfg(all(not(feature = "std"), not(feature = "libm")))]
    {
        jxl_oxide_common::math::fmaf(value, multiplier, addend)
    }
}
