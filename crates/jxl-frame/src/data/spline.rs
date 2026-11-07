use alloc::vec;
use alloc::vec::Vec;
use jxl_bitstream::{Bitstream, unpack_signed};
use jxl_coding::Decoder;
use jxl_oxide_common::Bundle;

use crate::{FrameHeader, Result};

const MAX_NUM_SPLINES: usize = 1 << 24;
const MAX_NUM_CONTROL_POINTS: usize = 1 << 20;

/// Holds quantized splines
#[derive(Debug)]
pub struct Splines {
    pub quant_splines: Vec<QuantSpline>,
    pub quant_adjust: i32,
}

impl Bundle<&FrameHeader> for Splines {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, header: &FrameHeader) -> Result<Self> {
        let mut decoder = jxl_coding::Decoder::parse(bitstream, 6)?;
        decoder.begin(bitstream)?;

        let encoded_num_splines = decoder.read_varint(bitstream, 2)? as u64;
        let max_num_splines = spline_count_limit(header.width, header.height);
        if encoded_num_splines >= max_num_splines as u64 {
            tracing::error!(
                num_splines = encoded_num_splines,
                max_num_splines,
                "Too many splines"
            );
            return Err(jxl_bitstream::Error::ProfileConformance("too many splines").into());
        }
        // The encoded count is zero-based: the stream always contains one more spline.
        let num_splines = encoded_num_splines as usize + 1;

        let mut start_points = vec![(0i64, 0i64); num_splines];
        let mut prev_point = (
            decoder.read_varint(bitstream, 1)? as i64,
            decoder.read_varint(bitstream, 1)? as i64,
        );
        start_points[0] = prev_point;
        for next_point in &mut start_points[1..] {
            let x = decoder.read_varint(bitstream, 1)?;
            let y = decoder.read_varint(bitstream, 1)?;
            prev_point.0 += unpack_signed(x) as i64;
            prev_point.1 += unpack_signed(y) as i64;
            *next_point = prev_point;
        }

        let quant_adjust = unpack_signed(decoder.read_varint(bitstream, 0)?);

        let mut splines: Vec<QuantSpline> = Vec::with_capacity(num_splines);
        let mut acc_control_points = 0u64;
        for start_point in start_points {
            let spline = QuantSpline::parse(
                bitstream,
                QuantSplineParams::new(
                    start_point,
                    header.width,
                    header.height,
                    &mut decoder,
                    acc_control_points,
                ),
            )?;

            acc_control_points =
                acc_control_points.saturating_add(spline.quant_points.len() as u64);
            splines.push(spline);
        }

        decoder.finalize()?;

        Ok(Self {
            quant_adjust,
            quant_splines: splines,
        })
    }
}

impl Splines {
    pub(crate) fn estimate_area(&self, base_correlation_xb: Option<(f32, f32)>) -> u64 {
        let base_correlation_xb = base_correlation_xb.unwrap_or((0.0, 1.0));
        let corr_x = crate::math::ceil(base_correlation_xb.0.abs()) as u128;
        let corr_b = crate::math::ceil(base_correlation_xb.1.abs()) as u128;
        let quant_adjust = self.quant_adjust;
        let mut total_area = 0u64;

        for quant_spline in &self.quant_splines {
            let mut color_xyb: [u128; 3] = quant_spline.xyb_dct.map(|quant_color_dct| {
                quant_color_dct
                    .into_iter()
                    .map(|q| div_ceil_qa(q.unsigned_abs(), quant_adjust) as u128)
                    .sum()
            });
            color_xyb[0] += corr_x * color_xyb[1];
            color_xyb[2] += corr_b * color_xyb[1];
            let max_color = color_xyb.into_iter().max().unwrap_or(0);
            let log_color = ceil_log2_u128(max_color + 1) as u64;

            let mut width_estimate = 0u64;
            for quant_sigma_dct in quant_spline.sigma_dct {
                let weight =
                    div_ceil_qa(quant_sigma_dct.unsigned_abs(), quant_adjust).saturating_add(1);
                let term = weight.saturating_mul(weight).saturating_mul(log_color);
                width_estimate = width_estimate.saturating_add(term);
            }
            total_area = total_area
                .saturating_add(width_estimate.saturating_mul(quant_spline.manhattan_distance));
        }

        total_area
    }
}

#[inline]
fn spline_pixel_count(width: u32, height: u32) -> u64 {
    width as u64 * height as u64
}

#[inline]
fn spline_count_limit(width: u32, height: u32) -> u64 {
    (MAX_NUM_SPLINES as u64).min(spline_pixel_count(width, height) / 4)
}

#[inline]
fn spline_point_limit(width: u32, height: u32) -> u64 {
    (MAX_NUM_CONTROL_POINTS as u64).min(spline_pixel_count(width, height) / 2)
}

#[inline]
fn ceil_log2_u128(x: u128) -> u32 {
    if x <= 1 {
        0
    } else {
        u128::BITS - (x - 1).leading_zeros()
    }
}

#[inline]
fn div_ceil_qa(dividend: u32, quant_adjust: i32) -> u64 {
    let dividend = dividend as u64;
    if quant_adjust >= 0 {
        let quant_adjust = quant_adjust as u64;
        (8 * dividend + 7 + quant_adjust) / (8 + quant_adjust)
    } else {
        let abs_quant_adjust = quant_adjust.unsigned_abs() as u64;
        dividend + (dividend * abs_quant_adjust).div_ceil(8)
    }
}

struct QuantSplineParams<'d> {
    start_point: (i64, i64),
    width: u32,
    height: u32,
    decoder: &'d mut Decoder,
    acc_control_points: u64,
}

impl<'d> QuantSplineParams<'d> {
    fn new(
        start_point: (i64, i64),
        width: u32,
        height: u32,
        decoder: &'d mut Decoder,
        acc_control_points: u64,
    ) -> Self {
        Self {
            start_point,
            width,
            height,
            decoder,
            acc_control_points,
        }
    }
}

/// Holds delta-endcoded control points coordinates (without starting point) and quantized DCT32 coefficients
#[derive(Debug, Default, Clone)]
pub struct QuantSpline {
    pub quant_points: Vec<(i64, i64)>,
    pub manhattan_distance: u64,
    pub xyb_dct: [[i32; 32]; 3],
    pub sigma_dct: [i32; 32],
}

impl Bundle<QuantSplineParams<'_>> for QuantSpline {
    type Error = crate::Error;

    fn parse(
        bitstream: &mut Bitstream,
        params: QuantSplineParams<'_>,
    ) -> core::result::Result<Self, Self::Error> {
        let QuantSplineParams {
            start_point,
            width,
            height,
            decoder,
            acc_control_points,
        } = params;

        let encoded_num_points = decoder.read_varint(bitstream, 3)? as u64;
        let acc_num_points = acc_control_points.saturating_add(encoded_num_points);
        let max_num_points = spline_point_limit(width, height);
        if acc_num_points > max_num_points {
            tracing::error!(
                num_points = encoded_num_points,
                max_num_points,
                "Too many spline points"
            );
            return Err(jxl_bitstream::Error::ProfileConformance("too many spline points").into());
        }

        let num_points = encoded_num_points as usize;
        let mut quant_points = Vec::with_capacity(1 + num_points);
        let mut cur_value = start_point;
        let mut cur_delta = (0i64, 0i64);
        let mut manhattan_distance = 0u64;
        quant_points.push(cur_value);
        for _ in 0..num_points {
            let prev_value = cur_value;
            let delta_x = unpack_signed(decoder.read_varint(bitstream, 4)?) as i64;
            let delta_y = unpack_signed(decoder.read_varint(bitstream, 4)?) as i64;

            cur_delta.0 =
                cur_delta
                    .0
                    .checked_add(delta_x)
                    .ok_or(jxl_bitstream::Error::ValidationFailed(
                        "control point overflowed",
                    ))?;
            cur_delta.1 =
                cur_delta
                    .1
                    .checked_add(delta_y)
                    .ok_or(jxl_bitstream::Error::ValidationFailed(
                        "control point overflowed",
                    ))?;
            let segment_distance = cur_delta
                .0
                .unsigned_abs()
                .saturating_add(cur_delta.1.unsigned_abs());
            manhattan_distance = manhattan_distance.saturating_add(segment_distance);
            cur_value.0 = cur_value.0.checked_add(cur_delta.0).ok_or(
                jxl_bitstream::Error::ValidationFailed("control point overflowed"),
            )?;
            cur_value.1 = cur_value.1.checked_add(cur_delta.1).ok_or(
                jxl_bitstream::Error::ValidationFailed("control point overflowed"),
            )?;
            if cur_value == prev_value {
                return Err(jxl_bitstream::Error::ValidationFailed(
                    "two consecutive control points have the same value",
                )
                .into());
            }
            quant_points.push(cur_value);
        }

        let mut xyb_dct = [[0; 32]; 3];
        for color_dct in &mut xyb_dct {
            for i in color_dct {
                *i = unpack_signed(decoder.read_varint(bitstream, 5)?);
            }
        }

        let mut sigma_dct = [0; 32];
        for i in &mut sigma_dct {
            *i = unpack_signed(decoder.read_varint(bitstream, 5)?);
        }

        Ok(Self {
            quant_points,
            manhattan_distance,
            xyb_dct,
            sigma_dct,
        })
    }
}

#[cfg(test)]
mod spline_limits_tests {
    use alloc::vec;

    use super::{
        QuantSpline, Splines, ceil_log2_u128, div_ceil_qa, spline_count_limit, spline_pixel_count,
        spline_point_limit,
    };

    #[test]
    fn spline_limits_apply_pixel_and_absolute_caps() {
        assert_eq!(spline_pixel_count(65_536, 65_536), 1u64 << 32);
        assert_eq!(spline_count_limit(65_536, 65_536), 1 << 24);
        assert_eq!(spline_point_limit(65_536, 65_536), 1 << 20);
        assert_eq!(spline_pixel_count(1 << 20, 1), 1u64 << 20);
        assert_eq!(spline_count_limit(1 << 20, 1), 1 << 18);
        assert_eq!(spline_point_limit(1 << 20, 1), 1 << 19);
        assert_eq!(spline_count_limit(1 << 20, 1 << 20), 1 << 24);
        assert_eq!(spline_point_limit(1 << 20, 1 << 20), 1 << 20);
        assert_eq!(
            spline_pixel_count(u32::MAX, u32::MAX),
            (u32::MAX as u64).pow(2)
        );
        assert_eq!(spline_count_limit(u32::MAX, u32::MAX), 1 << 24);
    }

    #[test]
    fn spline_limits_estimate_ordinary_zero_and_saturated_areas() {
        let mut ordinary = QuantSpline::default();
        ordinary.manhattan_distance = 3;
        ordinary.xyb_dct[0][0] = 8;
        let estimate = Splines {
            quant_splines: vec![ordinary.clone()],
            quant_adjust: 0,
        }
        .estimate_area(None);
        assert_eq!(estimate, 3 * 32 * 4);

        let mut correlated = QuantSpline::default();
        correlated.manhattan_distance = 3;
        correlated.xyb_dct[1][0] = 8;
        let correlated = Splines {
            quant_splines: vec![correlated],
            quant_adjust: 0,
        };
        assert_eq!(correlated.estimate_area(Some((2.0, 0.0))), 3 * 32 * 5);
        assert_eq!(correlated.estimate_area(None), 3 * 32 * 4);

        ordinary.manhattan_distance = 0;
        assert_eq!(
            Splines {
                quant_splines: vec![ordinary.clone()],
                quant_adjust: 0
            }
            .estimate_area(None),
            0
        );
        ordinary.manhattan_distance = 7;
        ordinary.xyb_dct = [[0; 32]; 3];
        assert_eq!(
            Splines {
                quant_splines: vec![ordinary],
                quant_adjust: 0
            }
            .estimate_area(None),
            0
        );

        let mut extreme = QuantSpline::default();
        extreme.manhattan_distance = u64::MAX;
        extreme.sigma_dct = [i32::MIN; 32];
        extreme.xyb_dct[1] = [i32::MIN; 32];
        let extreme = Splines {
            quant_splines: vec![extreme.clone(), extreme],
            quant_adjust: i32::MIN,
        };
        assert_eq!(extreme.estimate_area(Some((65_504.0, -65_504.0))), u64::MAX);

        let mut half_max = QuantSpline::default();
        half_max.manhattan_distance = 1 << 58;
        half_max.xyb_dct[0][0] = 1;
        let summed = Splines {
            quant_splines: vec![half_max.clone(), half_max],
            quant_adjust: 0,
        };
        assert_eq!(summed.estimate_area(None), u64::MAX);
        assert_eq!(ceil_log2_u128(1), 0);
        assert_eq!(ceil_log2_u128(5), 3);
        assert!(div_ceil_qa(u32::MAX, i32::MIN) > u32::MAX as u64);
    }
}
