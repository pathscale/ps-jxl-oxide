#![allow(unsafe_op_in_unsafe_fn)]

use alloc::collections::BTreeMap;

use jxl_frame::{
    FrameHeader,
    data::{HfGlobal, LfGlobal, LfGroup},
};
use jxl_grid::{AlignedGrid, MutableSubgrid, SharedSubgrid};
use jxl_image::ImageHeader;
use jxl_modular::{ChannelShift, Sample};
use jxl_vardct::{
    BlockInfo, LfChannelCorrelation, LfChannelDequantization, Quantizer, TransformType,
};

use crate::{ImageWithRegion, IndexedFrame, Reference, Region, RenderCache, Result};

mod dct_common;
mod owned;
mod transform_common;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64 as impls;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64 as impls;

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
mod wasm32;
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
use wasm32 as impls;

mod generic;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
use generic as impls;

pub(crate) async fn render_vardct<S: Sample>(
    frame: &alloc::sync::Arc<IndexedFrame>,
    lf_frame: Option<&Reference>,
    cache: &mut RenderCache<S>,
    region: Region,
    tasks: &jxl_threadpool::Tasks,
) -> Result<ImageWithRegion> {
    owned::render(frame, lf_frame, cache, region, tasks).await
}

pub fn copy_lf_dequant<S: Sample>(
    grid: &mut MutableSubgrid<f32>,
    quantizer: &Quantizer,
    m_lf: f32,
    channel_data: &AlignedGrid<S>,
    extra_precision: u8,
) {
    debug_assert!(extra_precision < 4);
    assert!(grid.width() >= channel_data.width());
    assert!(grid.height() >= channel_data.height());

    let precision_scale = 1i32 << (9 - extra_precision);
    let scale_inv = quantizer.global_scale as u64 * quantizer.quant_lf as u64;
    let scale = (m_lf as f64 * precision_scale as f64 / scale_inv as f64) as f32;

    let width = channel_data.width();
    let height = channel_data.height();
    let buf = channel_data.buf();
    for y in 0..height {
        let row = grid.get_row_mut(y);
        let quant = &buf[y * width..][..width];
        for (out, &q) in row.iter_mut().zip(quant) {
            *out = q.to_i32() as f32 * scale;
        }
    }
}

pub fn adaptive_lf_smoothing(
    shifts: [ChannelShift; 3],
    lf_image: [&mut AlignedGrid<f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) -> Result<()> {
    let scale_inv = quantizer.global_scale as u64 * quantizer.quant_lf as u64;
    let lf_x = (512.0 * lf_dequant.m_x_lf as f64 / scale_inv as f64) as f32;
    let lf_y = (512.0 * lf_dequant.m_y_lf as f64 / scale_inv as f64) as f32;
    let lf_b = (512.0 * lf_dequant.m_b_lf as f64 / scale_inv as f64) as f32;

    let [in_x, in_y, in_b] = lf_image;
    let tracker = in_x.tracker();
    let width = in_x.width().max(in_y.width()).max(in_b.width());
    let height = in_x.height().max(in_y.height()).max(in_b.height());

    let in_x = in_x.buf_mut();
    let in_y = in_y.buf_mut();
    let in_b = in_b.buf_mut();

    impls::adaptive_lf_smoothing_impl(
        width,
        height,
        shifts,
        [in_x, in_y, in_b],
        [lf_x, lf_y, lf_b],
        tracker.as_ref(),
    )
}

pub fn dequant_hf_varblock_grouped<S: Sample>(
    out: &mut [MutableSubgrid<'_, f32>; 3],
    group_idx: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal<S>,
    lf_groups: &BTreeMap<u32, LfGroup<S>>,
    hf_global: &HfGlobal,
) {
    let shifts_cbycr: [_; 3] = core::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let oim = &image_header.metadata.opsin_inverse_matrix;
    let quantizer = &lf_global.vardct.as_ref().unwrap().quantizer;
    let dequant_matrices = &hf_global.dequant_matrices;

    let qm_scale = [
        crate::math::powi(0.8f32, frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        crate::math::powi(0.8f32, frame_header.b_qm_scale as i32 - 2),
    ];

    let quant_bias_numerator = oim.quant_bias_numerator;

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();

    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;

    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;

    let Some(hf_meta) = &lf_group.hf_meta else {
        return;
    };

    let block_info = &hf_meta.block_info;

    let lf_width = (block_info.width() - left_in_lf).min(group_dim as usize / 8);
    let lf_height = (block_info.height() - top_in_lf).min(group_dim as usize / 8);
    let block_info = hf_meta.block_info.as_subgrid().subgrid(
        left_in_lf..(left_in_lf + lf_width),
        top_in_lf..(top_in_lf + lf_height),
    );

    for (channel, coeff) in out.iter_mut().enumerate() {
        let quant_bias = oim.quant_bias[channel];
        let shift = shifts_cbycr[channel];
        for_each_varblocks(
            &block_info,
            shift,
            |VarblockInfo {
                 shifted_bx,
                 shifted_by,
                 dct_select,
                 hf_mul,
             }| {
                let (bw, bh) = dct_select.dct_select_size();
                let left = shifted_bx * 8;
                let top = shifted_by * 8;

                let bw = bw as usize;
                let bh = bh as usize;
                let width = bw * 8;
                let height = bh * 8;

                let need_transpose = dct_select.need_transpose();
                let mul =
                    65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[channel];

                let matrix = if need_transpose {
                    dequant_matrices.get_transposed(channel, dct_select)
                } else {
                    dequant_matrices.get(channel, dct_select)
                };

                let mut coeff = coeff
                    .borrow_mut()
                    .subgrid(left..(left + width), top..(top + height));
                for (y, matrix_row) in matrix.chunks_exact(width).enumerate() {
                    let row = coeff.get_row_mut(y);
                    for (q, &m) in row.iter_mut().zip(matrix_row) {
                        let qn = q.to_bits() as i32;
                        *q = qn as f32;
                        if q.abs() <= 1.0 {
                            *q *= quant_bias;
                        } else {
                            *q -= quant_bias_numerator / *q;
                        }
                        *q *= m;
                        *q *= mul;
                    }
                }
            },
        );
    }
}

pub fn chroma_from_luma_lf(
    coeff_xyb: [&mut AlignedGrid<f32>; 3],
    lf_chan_corr: &LfChannelCorrelation,
) {
    let LfChannelCorrelation {
        colour_factor,
        base_correlation_x,
        base_correlation_b,
        x_factor_lf,
        b_factor_lf,
        ..
    } = *lf_chan_corr;

    let x_factor = x_factor_lf as i32 - 128;
    let b_factor = b_factor_lf as i32 - 128;
    let kx = base_correlation_x + (x_factor as f32 / colour_factor as f32);
    let kb = base_correlation_b + (b_factor as f32 / colour_factor as f32);

    let [x, y, b] = coeff_xyb;
    for ((x, y), b) in x.buf_mut().iter_mut().zip(y.buf()).zip(b.buf_mut()) {
        let y = *y;
        *x += kx * y;
        *b += kb * y;
    }
}

pub fn chroma_from_luma_hf_grouped(
    coeff_xyb: &mut [MutableSubgrid<'_, f32>; 3],
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
) {
    let [coeff_x, coeff_y, coeff_b] = coeff_xyb;

    let gw = coeff_x.width();
    let gh = coeff_x.height();

    for y in 0..gh {
        let x_from_y = x_from_y.get_row(y / 64);
        let b_from_y = b_from_y.get_row(y / 64);

        let coeff_x = coeff_x.get_row_mut(y);
        let coeff_y = coeff_y.get_row(y);
        let coeff_b = coeff_b.get_row_mut(y);

        for (x64, (&kx, &kb)) in x_from_y.iter().zip(b_from_y).enumerate() {
            let kx =
                lf_chan_corr.base_correlation_x + (kx as f32 / lf_chan_corr.colour_factor as f32);
            let kb =
                lf_chan_corr.base_correlation_b + (kb as f32 / lf_chan_corr.colour_factor as f32);

            for dx in 0..((gw - x64 * 64).min(64)) {
                let x = x64 * 64 + dx;
                let coeff_y = coeff_y[x];
                coeff_x[x] += kx * coeff_y;
                coeff_b[x] += kb * coeff_y;
            }
        }
    }
}

pub fn transform_with_lf_grouped<S: Sample>(
    lf: &ImageWithRegion,
    coeff_out: &mut [MutableSubgrid<'_, f32>; 3],
    group_idx: u32,
    frame_header: &FrameHeader,
    lf_groups: &BTreeMap<u32, LfGroup<S>>,
) {
    let lf_regions = <[_; 3]>::try_from(&lf.regions_and_shifts()[..3]).unwrap();
    let [lf_x, lf_y, lf_b] = lf.as_color_floats();
    let shifts_cbycr: [_; 3] = core::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let (group_width, group_height) = frame_header.group_size_for(group_idx);

    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;
    let lf_base_left = group_x * group_dim / 8;
    let lf_base_top = group_y * group_dim / 8;
    let lf = [
        (lf_regions[0], lf_x),
        (lf_regions[1], lf_y),
        (lf_regions[2], lf_b),
    ]
    .map(|((lf_region, shift), lf)| {
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));
        let lf_base_left = lf_base_left as usize;
        let lf_base_top = lf_base_top as usize;

        let lf_base_left = lf_base_left >> shift.hshift();
        let lf_base_top = lf_base_top >> shift.vshift();
        let (lf_width, lf_height) = shift.shift_size((lf_width, lf_height));
        lf.as_subgrid().subgrid(
            lf_base_left..(lf_base_left + lf_width as usize),
            lf_base_top..(lf_base_top + lf_height as usize),
        )
    });

    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;

    let Some(hf_meta) = &lf_group.hf_meta else {
        for (coeff, lf) in coeff_out.iter_mut().zip(lf) {
            for y in 0..coeff.height() {
                let coeff_row = coeff.get_row_mut(y);
                let lf_row = lf.get_row(y / 8);
                for (x, v) in coeff_row.iter_mut().enumerate() {
                    *v = lf_row[x / 8];
                }
            }
        }
        return;
    };

    let block_info = {
        let lf_region = lf_regions[0].0;
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));

        hf_meta.block_info.as_subgrid().subgrid(
            left_in_lf..(left_in_lf + lf_width as usize),
            top_in_lf..(top_in_lf + lf_height as usize),
        )
    };

    impls::transform_varblocks(&lf, coeff_out, shifts_cbycr, &block_info);
}

#[derive(Debug)]
struct VarblockInfo {
    shifted_bx: usize,
    shifted_by: usize,
    dct_select: TransformType,
    hf_mul: i32,
}

#[inline(always)]
fn for_each_varblocks(
    block_info: &SharedSubgrid<BlockInfo>,
    shift: ChannelShift,
    mut f: impl FnMut(VarblockInfo),
) {
    let w8 = block_info.width();
    let h8 = block_info.height();
    let vshift = shift.vshift();
    let hshift = shift.hshift();

    for by in 0..h8 {
        for bx in 0..w8 {
            let BlockInfo::Data { dct_select, hf_mul } = block_info.get(bx, by) else {
                continue;
            };
            let shifted_bx = bx >> hshift;
            let shifted_by = by >> vshift;
            if hshift != 0 || vshift != 0 {
                if (shifted_bx << hshift) != bx || (shifted_by << vshift) != by {
                    continue;
                }
                if !matches!(
                    block_info.get(shifted_bx, shifted_by),
                    BlockInfo::Data { .. }
                ) {
                    continue;
                }
            }

            f(VarblockInfo {
                shifted_bx,
                shifted_by,
                dct_select,
                hf_mul,
            })
        }
    }
}
