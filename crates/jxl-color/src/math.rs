//! Scalar float operations with unchanged host math and a pure Rust fallback.

pub(crate) trait FloatMath: Sized {
    fn math_floor(self) -> Self;
    fn math_powf(self, exponent: Self) -> Self;
    fn math_exp(self) -> Self;
    fn math_ln(self) -> Self;
    fn math_log2(self) -> Self;
    fn math_cbrt(self) -> Self;
    fn math_sqrt(self) -> Self;
    fn math_mul_add(self, a: Self, b: Self) -> Self;
}

macro_rules! impl_float_math {
    ($ty:ty, $floor:path, $powf:path, $exp:path, $ln:path, $log2:path, $cbrt:path, $sqrt:path, $mul_add:path) => {
        impl FloatMath for $ty {
            #[inline]
            fn math_floor(self) -> Self {
                #[cfg(feature = "std")]
                {
                    self.floor()
                }
                #[cfg(not(feature = "std"))]
                {
                    $floor(self)
                }
            }

            #[inline]
            fn math_powf(self, exponent: Self) -> Self {
                #[cfg(feature = "std")]
                {
                    self.powf(exponent)
                }
                #[cfg(not(feature = "std"))]
                {
                    $powf(self, exponent)
                }
            }

            #[inline]
            fn math_exp(self) -> Self {
                #[cfg(feature = "std")]
                {
                    self.exp()
                }
                #[cfg(not(feature = "std"))]
                {
                    $exp(self)
                }
            }

            #[inline]
            fn math_ln(self) -> Self {
                #[cfg(feature = "std")]
                {
                    self.ln()
                }
                #[cfg(not(feature = "std"))]
                {
                    $ln(self)
                }
            }

            #[inline]
            fn math_log2(self) -> Self {
                #[cfg(feature = "std")]
                {
                    self.log2()
                }
                #[cfg(not(feature = "std"))]
                {
                    $log2(self)
                }
            }

            #[inline]
            fn math_cbrt(self) -> Self {
                #[cfg(feature = "std")]
                {
                    self.cbrt()
                }
                #[cfg(not(feature = "std"))]
                {
                    $cbrt(self)
                }
            }

            #[inline]
            fn math_sqrt(self) -> Self {
                #[cfg(feature = "std")]
                {
                    self.sqrt()
                }
                #[cfg(not(feature = "std"))]
                {
                    $sqrt(self)
                }
            }

            #[inline]
            fn math_mul_add(self, a: Self, b: Self) -> Self {
                #[cfg(feature = "std")]
                {
                    self.mul_add(a, b)
                }
                #[cfg(not(feature = "std"))]
                {
                    $mul_add(self, a, b)
                }
            }
        }
    };
}

impl_float_math!(
    f32,
    libm::floorf,
    libm::powf,
    libm::expf,
    libm::logf,
    libm::log2f,
    libm::cbrtf,
    libm::sqrtf,
    libm::fmaf
);
impl_float_math!(
    f64,
    libm::floor,
    libm::pow,
    libm::exp,
    libm::log,
    libm::log2,
    libm::cbrt,
    libm::sqrt,
    libm::fma
);
