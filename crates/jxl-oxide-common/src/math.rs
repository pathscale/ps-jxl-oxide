//! Pure-Rust `no_std` IEEE-754 floating-point math polyfills.

#[inline]
pub fn floorf(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() || x == 0.0 || x.abs() >= 8_388_608.0 {
        return x;
    }
    let trunc = (x as i32) as f32;
    if x < trunc { trunc - 1.0 } else { trunc }
}

#[inline]
pub fn floor(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 || x.abs() >= 4_503_599_627_370_496.0 {
        return x;
    }
    let trunc = (x as i64) as f64;
    if x < trunc { trunc - 1.0 } else { trunc }
}

#[inline]
pub fn ceilf(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() || x == 0.0 || x.abs() >= 8_388_608.0 {
        return x;
    }
    let trunc = (x as i32) as f32;
    if x > trunc {
        trunc + 1.0
    } else {
        trunc.copysign(x)
    }
}

#[inline]
pub fn ceil(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 || x.abs() >= 4_503_599_627_370_496.0 {
        return x;
    }
    let trunc = (x as i64) as f64;
    if x > trunc {
        trunc + 1.0
    } else {
        trunc.copysign(x)
    }
}

#[inline]
pub fn roundf(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() || x == 0.0 || x.abs() >= 8_388_608.0 {
        return x;
    }
    let trunc = (x as i32) as f32;
    let frac = x - trunc;
    if frac >= 0.5 {
        trunc + 1.0
    } else if frac <= -0.5 {
        trunc - 1.0
    } else {
        trunc.copysign(x)
    }
}

#[inline]
pub fn round(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 || x.abs() >= 4_503_599_627_370_496.0 {
        return x;
    }
    let trunc = (x as i64) as f64;
    let frac = x - trunc;
    if frac >= 0.5 {
        trunc + 1.0
    } else if frac <= -0.5 {
        trunc - 1.0
    } else {
        trunc.copysign(x)
    }
}

#[inline]
fn round_ties_even_f64(x: f64) -> f64 {
    const TWO_POW_52: f64 = 4_503_599_627_370_496.0;
    if x.is_nan() || x.is_infinite() || x == 0.0 || x.abs() >= TWO_POW_52 {
        return x;
    }
    ((x.abs() + TWO_POW_52) - TWO_POW_52).copysign(x)
}

#[inline]
pub fn sqrt(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    // Rescale subnormals so exponent-halving seed is in normal range.
    let (norm, post_scale) = if x < f64::MIN_POSITIVE {
        (x * 18_014_398_509_481_984.0, 1.0 / 134_217_728.0) // 2^54, 2^-27
    } else {
        (x, 1.0)
    };
    let mut y = f64::from_bits((norm.to_bits() >> 1) + (0x3ff0_0000_0000_0000 >> 1));
    for _ in 0..6 {
        y = 0.5 * (y + norm / y);
    }
    y * post_scale
}

#[inline]
pub fn sqrtf(x: f32) -> f32 {
    sqrt(f64::from(x)) as f32
}

#[inline]
pub fn cbrt(x: f64) -> f64 {
    if x.is_nan() || x == 0.0 || x.is_infinite() {
        return x;
    }
    let abs_x = x.abs();
    let (norm, post_scale) = if abs_x < f64::MIN_POSITIVE {
        (abs_x * 18_014_398_509_481_984.0, 1.0 / 262_144.0) // 2^54, 2^-18
    } else {
        (abs_x, 1.0)
    };
    let mut y = f64::from_bits((norm.to_bits() / 3) + 0x2aa0_0000_0000_0000);
    for _ in 0..6 {
        y = (2.0 * y + norm / (y * y)) / 3.0;
    }
    (y * post_scale).copysign(x)
}

#[inline]
pub fn cbrtf(x: f32) -> f32 {
    cbrt(f64::from(x)) as f32
}

/// Decomposes `x` into `(k, ln(m))` where `x = m * 2^k` and `m in [sqrt(0.5), sqrt(2))`.
#[inline]
fn ln_decomposed(x: f64) -> (i32, f64) {
    if x.is_nan() || x < 0.0 {
        return (0, f64::NAN);
    }
    if x == 0.0 {
        return (0, f64::NEG_INFINITY);
    }
    if x.is_infinite() {
        return (0, f64::INFINITY);
    }
    let (norm, k_bias) = if x < f64::MIN_POSITIVE {
        (x * 18_014_398_509_481_984.0, -54_i32)
    } else {
        (x, 0_i32)
    };
    let bits = norm.to_bits();
    let mut k = ((bits >> 52) as i32) - 1023 + k_bias;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m >= core::f64::consts::SQRT_2 {
        m *= 0.5;
        k += 1;
    }
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let poly = 1.0 / 17.0;
    let poly = 1.0 / 15.0 + u2 * poly;
    let poly = 1.0 / 13.0 + u2 * poly;
    let poly = 1.0 / 11.0 + u2 * poly;
    let poly = 1.0 / 9.0 + u2 * poly;
    let poly = 1.0 / 7.0 + u2 * poly;
    let poly = 1.0 / 5.0 + u2 * poly;
    let poly = 1.0 / 3.0 + u2 * poly;
    let poly = 1.0 + u2 * poly;
    (k, 2.0 * u * poly)
}

#[inline]
pub fn log(x: f64) -> f64 {
    let (k, ln_m) = ln_decomposed(x);
    if ln_m.is_nan() || ln_m.is_infinite() {
        return ln_m;
    }
    f64::from(k) * core::f64::consts::LN_2 + ln_m
}

#[inline]
pub fn logf(x: f32) -> f32 {
    log(f64::from(x)) as f32
}

#[inline]
pub fn log2(x: f64) -> f64 {
    let (k, ln_m) = ln_decomposed(x);
    if ln_m.is_nan() || ln_m.is_infinite() {
        return ln_m;
    }
    f64::from(k) + ln_m * core::f64::consts::LOG2_E
}

#[inline]
pub fn log2f(x: f32) -> f32 {
    log2(f64::from(x)) as f32
}

/// Evaluates `exp(r)` for `|r| <= ln(2)/2`.
#[inline]
fn exp_poly(r: f64) -> f64 {
    let mut p = 1.0 / 6_227_020_800.0; // 1 / 13!
    p = 1.0 / 479_001_600.0 + r * p; // 1 / 12!
    p = 1.0 / 39_916_800.0 + r * p; // 1 / 11!
    p = 1.0 / 3_628_800.0 + r * p; // 1 / 10!
    p = 1.0 / 362_880.0 + r * p; // 1 / 9!
    p = 1.0 / 40_320.0 + r * p; // 1 / 8!
    p = 1.0 / 5_040.0 + r * p; // 1 / 7!
    p = 1.0 / 720.0 + r * p; // 1 / 6!
    p = 1.0 / 120.0 + r * p; // 1 / 5!
    p = 1.0 / 24.0 + r * p; // 1 / 4!
    p = 1.0 / 6.0 + r * p; // 1 / 3!
    p = 0.5 + r * p; // 1 / 2!
    p = 1.0 + r * p; // 1 / 1!
    1.0 + r * p
}

/// Scales `val` by `2^k` for `k` in `[-1074, 1024]`.
#[inline]
fn scale_exp2(val: f64, k: i32) -> f64 {
    if k >= -1022 {
        if k > 1024 {
            return f64::INFINITY;
        }
        if k == 1024 {
            // 2^1024 is not representable, so scale by 2 and then by 2^1023.
            return (val * 2.0) * f64::from_bits(2046_u64 << 52);
        }
        let scale = f64::from_bits(((k + 1023) as u64) << 52);
        val * scale
    } else if k >= -1074 {
        let scale1 = f64::from_bits(((k + 1023 + 512) as u64) << 52);
        let scale2 = f64::from_bits(((1023 - 512) as u64) << 52);
        (val * scale1) * scale2
    } else {
        0.0
    }
}

#[inline]
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if x < -745.133_219_101_941_2 {
        return 0.0;
    }
    const INV_LN2: f64 = core::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let k = round_ties_even_f64(x * INV_LN2);
    let r = (x - k * LN2_HI) - k * LN2_LO;
    scale_exp2(exp_poly(r), k as i32)
}

#[inline]
pub fn expf(x: f32) -> f32 {
    exp(f64::from(x)) as f32
}

#[inline]
fn is_odd_integer(n: f64) -> bool {
    if round_ties_even_f64(n) != n || n.abs() >= 9_007_199_254_740_992.0 {
        return false;
    }
    ((n as i64) & 1) != 0
}

#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    if y == 0.0 || x == 1.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        let is_neg_zero = x.to_bits() == (-0.0_f64).to_bits();
        let neg_odd = is_neg_zero && is_odd_integer(y);
        return if y < 0.0 {
            if neg_odd {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else if neg_odd {
            -0.0
        } else {
            0.0
        };
    }
    if x.is_infinite() {
        let neg_odd = x < 0.0 && is_odd_integer(y);
        return if y < 0.0 {
            if neg_odd { -0.0 } else { 0.0 }
        } else if neg_odd {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if x < 0.0 {
        if round_ties_even_f64(y) == y {
            let mag = exp(y * log(-x));
            return if is_odd_integer(y) { -mag } else { mag };
        }
        return f64::NAN;
    }
    exp(y * log(x))
}

#[inline]
pub fn powf(x: f32, y: f32) -> f32 {
    pow(f64::from(x), f64::from(y)) as f32
}

#[inline]
pub fn powif(x: f32, n: i32) -> f32 {
    let mut exp = n.unsigned_abs();
    let mut base = if n < 0 { 1.0 / f64::from(x) } else { f64::from(x) };
    let mut acc = 1.0_f64;
    while exp > 0 {
        if exp & 1 != 0 {
            acc *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    acc as f32
}

/// Cody-Waite range reduction of `x` modulo `pi/2`.
#[inline]
fn reduce_pi_half(x: f64) -> (i64, f64) {
    // PI_2_A has 21 non-zero mantissa bits so `k * PI_2_A` is exact for |k| < 2^32.
    const PI_2_A: f64 = f64::from_bits(0x3ff9_21fb_5440_0000);
    const PI_2_B: f64 = 6.077_100_506_506_192e-11;
    let k = round_ties_even_f64(x * core::f64::consts::FRAC_2_PI);
    let r = (x - k * PI_2_A) - k * PI_2_B;
    (k as i64, r)
}

#[inline]
fn sin_poly(r: f64) -> f64 {
    let r2 = r * r;
    let mut p = -1.0 / 1_307_674_368_000.0; // -1 / 15!
    p = 1.0 / 6_227_020_800.0 + r2 * p; // 1 / 13!
    p = -1.0 / 39_916_800.0 + r2 * p; // -1 / 11!
    p = 1.0 / 362_880.0 + r2 * p; // 1 / 9!
    p = -1.0 / 5_040.0 + r2 * p; // -1 / 7!
    p = 1.0 / 120.0 + r2 * p; // 1 / 5!
    p = -1.0 / 6.0 + r2 * p; // -1 / 3!
    p = 1.0 + r2 * p;
    r * p
}

#[inline]
fn cos_poly(r: f64) -> f64 {
    let r2 = r * r;
    let mut p = 1.0 / 20_922_789_888_000.0; // 1 / 16!
    p = -1.0 / 87_178_291_200.0 + r2 * p; // -1 / 14!
    p = 1.0 / 479_001_600.0 + r2 * p; // 1 / 12!
    p = -1.0 / 3_628_800.0 + r2 * p; // -1 / 10!
    p = 1.0 / 40_320.0 + r2 * p; // 1 / 8!
    p = -1.0 / 720.0 + r2 * p; // -1 / 6!
    p = 1.0 / 24.0 + r2 * p; // 1 / 4!
    p = -0.5 + r2 * p; // -1 / 2!
    1.0 + r2 * p
}

#[inline]
pub fn cos(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    let (q, r) = reduce_pi_half(x);
    match q & 3 {
        0 => cos_poly(r),
        1 => -sin_poly(r),
        2 => -cos_poly(r),
        _ => sin_poly(r),
    }
}

#[inline]
pub fn cosf(x: f32) -> f32 {
    cos(f64::from(x)) as f32
}

#[inline]
pub fn fmaf(x: f32, a: f32, b: f32) -> f32 {
    (f64::from(x) * f64::from(a) + f64::from(b)) as f32
}

#[inline]
pub fn fma(x: f64, a: f64, b: f64) -> f64 {
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    // Veltkamp split requires |x|, |a| < ~2^996 to avoid intermediate overflow.
    const SPLIT_LIMIT: f64 = 1.0e299;
    if !x.is_finite() || !a.is_finite() || !b.is_finite() || x.abs() > SPLIT_LIMIT || a.abs() > SPLIT_LIMIT {
        return x * a + b;
    }
    let p = x * a;
    if !p.is_finite() || p.abs() < f64::MIN_POSITIVE * 4_503_599_627_370_496.0 {
        return p + b;
    }
    let xc = SPLIT * x;
    let x_hi = xc - (xc - x);
    let x_lo = x - x_hi;
    let ac = SPLIT * a;
    let a_hi = ac - (ac - a);
    let a_lo = a - a_hi;
    let prod_err = ((x_hi * a_hi - p) + x_hi * a_lo + x_lo * a_hi) + x_lo * a_lo;

    let sum = p + b;
    let b_virt = sum - p;
    let p_virt = sum - b_virt;
    let b_err = b - b_virt;
    let p_err = p - p_virt;
    let sum_err = p_err + b_err;

    sum + (sum_err + prod_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_edge_cases() {
        assert_eq!(floorf(2.7), 2.0);
        assert_eq!(floorf(-2.7), -3.0);
        assert_eq!(floorf(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(ceilf(2.1), 3.0);
        assert_eq!(ceilf(-2.1), -2.0);
        assert_eq!(ceilf(-0.5).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(roundf(0.499_999_97), 0.0);
        assert_eq!(roundf(0.5), 1.0);
        assert_eq!(roundf(-0.5), -1.0);
        assert_eq!(roundf(-0.25).to_bits(), (-0.0_f32).to_bits());

        assert_eq!(floor(-2.7), -3.0);
        assert_eq!(ceil(-2.1), -2.0);
        assert_eq!(round(0.5), 1.0);
        assert_eq!(round(-0.5), -1.0);
    }

    #[test]
    fn sqrt_cbrt_accuracy() {
        for &x in &[0.0_f64, 1.0, 2.0, 3.0, 27.0, 1e-12, 1e12, 1e-310] {
            let s = sqrt(x);
            let rel = if x == 0.0 { 0.0 } else { (s * s - x).abs() / x };
            assert!(rel < 1e-15, "sqrt({x}) rel err = {rel}");

            let c = cbrt(x);
            let crel = if x == 0.0 { 0.0 } else { (c * c * c - x).abs() / x };
            assert!(crel < 1e-14, "cbrt({x}) rel err = {crel}");
        }
        assert!((cbrt(-27.0) + 3.0).abs() < 1e-15);
        assert!((cbrtf(-8.0) + 2.0).abs() < 1e-6);
        assert!(sqrt(-1.0).is_nan());
        assert!(sqrtf(-1.0).is_nan());
    }

    #[test]
    fn transcendental_and_fma_accuracy() {
        for &x in &[0.001_f64, 0.5, 1.0, 2.718_281_828, 100.0, 1e-20] {
            let l = log(x);
            let e = exp(l);
            assert!((e - x).abs() / x < 1e-13, "exp(log({x})) = {e}");
            assert!((log2(x) - l * core::f64::consts::LOG2_E).abs() < 1e-13);
        }
        assert!((pow(2.0, 10.0) - 1024.0).abs() < 1e-11);
        assert!((pow(-2.0, 3.0) + 8.0).abs() < 1e-13);
        assert!((powif(3.0, -2) - (1.0 / 9.0)).abs() < 1e-6);
        assert!((cos(0.0) - 1.0).abs() < 1e-15);
        assert!(cos(core::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert!((cosf(core::f32::consts::PI) + 1.0).abs() < 1e-6);

        // Cancellation test for fma
        let x = 1.0e8_f64 + 1.0;
        let a = 1.0e8_f64 - 1.0;
        let b = -1.0e16_f64;
        assert_eq!(fma(x, a, b), -1.0);
        assert!((fmaf(1.5, 2.0, 0.25) - 3.25).abs() < 1e-7);
    }
}
