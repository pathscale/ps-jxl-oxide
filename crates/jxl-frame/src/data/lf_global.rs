use alloc::vec::Vec;
use jxl_bitstream::Bitstream;
use jxl_grid::AllocTracker;
use jxl_image::ImageHeader;
use jxl_modular::{
    ChannelShift, MaConfig, MaConfigParams, Modular, ModularChannelParams, ModularParams, Sample,
};
use jxl_oxide_common::{Bundle, define_bundle};
use jxl_vardct::{HfBlockContext, LfChannelCorrelation, LfChannelDequantization, Quantizer};

use crate::{FrameHeader, Result, header::Encoding};

use super::{NoiseParameters, Patches, Splines};

#[derive(Debug)]
pub struct LfGlobal<S: Sample> {
    pub patches: Option<Patches>,
    pub splines: Option<Splines>,
    pub noise: Option<NoiseParameters>,
    pub lf_dequant: LfChannelDequantization,
    pub vardct: Option<LfGlobalVarDct>,
    pub gmodular: GlobalModular<S>,
}

#[derive(Debug, Clone, Copy)]
pub struct LfGlobalParams<'a, 'b> {
    pub image_header: &'a ImageHeader,
    pub frame_header: &'a FrameHeader,
    pub tracker: Option<&'b AllocTracker>,
    pub allow_partial: bool,
}

impl<'a, 'b> LfGlobalParams<'a, 'b> {
    pub fn new(
        image_header: &'a ImageHeader,
        frame_header: &'a FrameHeader,
        tracker: Option<&'b AllocTracker>,
        allow_partial: bool,
    ) -> Self {
        Self {
            image_header,
            frame_header,
            tracker,
            allow_partial,
        }
    }
}

impl<S: Sample> Bundle<LfGlobalParams<'_, '_>> for LfGlobal<S> {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: LfGlobalParams) -> Result<Self> {
        let LfGlobalParams {
            image_header,
            frame_header: header,
            ..
        } = params;
        let patches = header
            .flags
            .patches()
            .then(|| -> Result<_> {
                let span = tracing::span!(tracing::Level::TRACE, "Decode Patches");
                let _guard = span.enter();

                let patches = Patches::parse(bitstream, (image_header, header))?;
                let it = patches
                    .patches
                    .iter()
                    .flat_map(|patch| &patch.patch_targets)
                    .flat_map(|target| &target.blending);
                for blending_info in it {
                    if blending_info.mode.use_alpha()
                        && blending_info.alpha_channel as usize
                            >= image_header.metadata.ec_info.len()
                    {
                        return Err(jxl_bitstream::Error::ValidationFailed(
                            "blending_info.alpha_channel out of range",
                        )
                        .into());
                    }
                }
                Ok(patches)
            })
            .transpose()?;
        let splines = header
            .flags
            .splines()
            .then(|| {
                let span = tracing::span!(tracing::Level::TRACE, "Decode Splines");
                let _guard = span.enter();

                Splines::parse(bitstream, header)
            })
            .transpose()?;
        let noise = header
            .flags
            .noise()
            .then(|| {
                let span = tracing::span!(tracing::Level::TRACE, "Decode Noise");
                let _guard = span.enter();

                NoiseParameters::parse(bitstream, ())
            })
            .transpose()?;
        let lf_dequant = LfChannelDequantization::parse(bitstream, ())?;

        let modular_dequants = [
            lf_dequant.m_x_lf_unscaled(),
            lf_dequant.m_y_lf_unscaled(),
            lf_dequant.m_b_lf_unscaled(),
        ];
        if modular_dequants.into_iter().any(|v| v < 1e-8) {
            tracing::error!(?modular_dequants, "Modular dequant weight is too small");
            return Err(jxl_bitstream::Error::ValidationFailed(
                "Modular dequant weight is too small",
            )
            .into());
        }

        let vardct = (header.encoding == crate::header::Encoding::VarDct)
            .then(|| LfGlobalVarDct::parse(bitstream, ()))
            .transpose()?;

        if let Some(splines) = &splines {
            let base_correlation_xb = vardct.as_ref().map(|vardct| {
                let lf_chan_corr = &vardct.lf_chan_corr;
                (
                    lf_chan_corr.base_correlation_x,
                    lf_chan_corr.base_correlation_b,
                )
            });
            let estimated_area = splines.estimate_area(base_correlation_xb);

            // Maximum total_estimated_area_reached for Level 10
            let (level5_limit, max_estimated_area) =
                spline_area_limits(header.width, header.height);
            if let Err(error) = validate_spline_area(estimated_area, max_estimated_area) {
                tracing::error!(
                    estimated_area,
                    max_estimated_area,
                    "Too large estimated area for splines"
                );
                return Err(error);
            }
            // Maximum total_estimated_area_reached for Level 5
            if estimated_area > level5_limit {
                tracing::warn!(
                    "Large estimated_area of splines, expect slower decoding: {}",
                    estimated_area
                );
            }
        }

        let gmodular = GlobalModular::<S>::parse(bitstream, params)?;

        Ok(Self {
            patches,
            splines,
            noise,
            lf_dequant,
            vardct,
            gmodular,
        })
    }
}

define_bundle! {
    #[derive(Debug)]
    pub struct LfGlobalVarDct error(crate::Error) {
        pub quantizer: ty(Bundle(Quantizer)),
        pub hf_block_ctx: ty(Bundle(HfBlockContext)),
        pub lf_chan_corr: ty(Bundle(LfChannelCorrelation)),
    }
}

#[derive(Debug)]
pub struct GlobalModular<S: Sample> {
    pub ma_config: Option<MaConfig>,
    pub modular: Modular<S>,
    extra_channel_from: usize,
    is_partial: bool,
}

impl<S: Sample> GlobalModular<S> {
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            ma_config: self.ma_config.clone(),
            modular: self.modular.try_clone()?,
            extra_channel_from: self.extra_channel_from,
            is_partial: self.is_partial,
        })
    }

    pub fn ma_config(&self) -> Option<&MaConfig> {
        self.ma_config.as_ref()
    }

    pub fn extra_channel_from(&self) -> usize {
        self.extra_channel_from
    }

    pub fn is_partial(&self) -> bool {
        self.is_partial
    }
}

impl<S: Sample> Bundle<LfGlobalParams<'_, '_>> for GlobalModular<S> {
    type Error = crate::Error;

    fn parse(bitstream: &mut Bitstream, params: LfGlobalParams) -> Result<Self> {
        let LfGlobalParams {
            image_header,
            frame_header: header,
            tracker,
            allow_partial,
        } = params;
        let span = tracing::span!(tracing::Level::TRACE, "Decode GlobalModular");
        let _guard = span.enter();

        let num_channels =
            (header.encoded_color_channels() + image_header.metadata.ec_info.len()) as u64;
        let max_global_ma_nodes =
            1024 + header.width as u64 * header.height as u64 * num_channels / 16;
        let max_global_ma_nodes = (1 << 22).min(max_global_ma_nodes) as usize;
        let ma_config_params = MaConfigParams {
            tracker: params.tracker,
            node_limit: max_global_ma_nodes,
            depth_limit: 2048,
        };
        let ma_config = bitstream
            .read_bool()?
            .then(|| MaConfig::parse(bitstream, ma_config_params))
            .transpose()?;

        let color_width = header.color_sample_width();
        let color_height = header.color_sample_height();

        let mut shifts = Vec::new();
        if header.encoding == Encoding::Modular {
            if header.do_ycbcr {
                // Cb, Y, Cr
                shifts.push(ModularChannelParams::jpeg(
                    color_width,
                    color_height,
                    header.jpeg_upsampling,
                    0,
                ));
                shifts.push(ModularChannelParams::jpeg(
                    color_width,
                    color_height,
                    header.jpeg_upsampling,
                    1,
                ));
                shifts.push(ModularChannelParams::jpeg(
                    color_width,
                    color_height,
                    header.jpeg_upsampling,
                    2,
                ));
            } else {
                let channel_param = ModularChannelParams::new(color_width, color_height);
                let channels = header.encoded_color_channels();
                shifts.extend(core::iter::repeat_n(channel_param, channels));
            }
        }

        let extra_channel_from = shifts.len();
        let color_upsampling_shift = header.upsampling.trailing_zeros();

        for (&ec_upsampling, ec_info) in header
            .ec_upsampling
            .iter()
            .zip(image_header.metadata.ec_info.iter())
        {
            let ec_upsampling_shift = ec_upsampling.trailing_zeros();
            let dim_shift = ec_info.dim_shift;
            let actual_dim_shift = ec_upsampling_shift + dim_shift - color_upsampling_shift;

            let shift = ChannelShift::from_shift(actual_dim_shift);
            shifts.push(ModularChannelParams::with_shift(
                color_width,
                color_height,
                shift,
            ));
        }

        let group_dim = header.group_dim();
        let modular_params = ModularParams::with_channels(
            group_dim,
            image_header.metadata.bit_depth.bits_per_sample(),
            shifts,
            ma_config.as_ref(),
            tracker,
        );
        let mut modular = Modular::<S>::parse(bitstream, modular_params)?;
        let mut is_partial = false;
        if let Some(image) = modular.image_mut() {
            let mut gmodular = image.prepare_gmodular()?;
            gmodular.decode(bitstream, 0, allow_partial)?;
            is_partial = gmodular.is_partial();
        }

        Ok(Self {
            ma_config,
            modular,
            extra_channel_from,
            is_partial,
        })
    }
}

#[inline]
fn image_pixel_count(width: u32, height: u32) -> u64 {
    width as u64 * height as u64
}

#[inline]
fn spline_area_limits(width: u32, height: u32) -> (u64, u64) {
    let image_size = image_pixel_count(width, height);
    let level5 = image_size.saturating_mul(8).saturating_add(1u64 << 25);
    let level10 = image_size.saturating_mul(1024).saturating_add(1u64 << 32);
    ((1u64 << 30).min(level5), (1u64 << 42).min(level10))
}

#[inline]
fn validate_spline_area(estimated_area: u64, limit: u64) -> Result<()> {
    if estimated_area > limit {
        Err(jxl_bitstream::Error::ProfileConformance("too large estimated area for splines").into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod spline_limits_tests {
    use super::{image_pixel_count, spline_area_limits, validate_spline_area};

    #[test]
    fn spline_limits_preserve_caps_and_pixel_scaling() {
        assert_eq!(image_pixel_count(0, 0), 0);
        assert_eq!(image_pixel_count(65_536, 65_536), 1u64 << 32);
        assert_eq!(image_pixel_count(1u32 << 20, 1), 1u64 << 20);
        assert_eq!(
            image_pixel_count(u32::MAX, u32::MAX),
            (u32::MAX as u64).pow(2)
        );
        assert_eq!(spline_area_limits(0, 0), (1 << 25, 1 << 32));
        assert_eq!(
            spline_area_limits(1 << 20, 1),
            ((1 << 25) + (1 << 23), (1 << 32) + (1 << 30))
        );
        assert_eq!(spline_area_limits(65_536, 65_536), (1 << 30, 1 << 42));
        assert_eq!(spline_area_limits(1 << 20, 1 << 20), (1 << 30, 1 << 42));
        assert_eq!(spline_area_limits(u32::MAX, u32::MAX), (1 << 30, 1 << 42));
        let limit = spline_area_limits(1 << 20, 1).1;
        assert!(validate_spline_area(limit, limit).is_ok());
        assert!(matches!(
            validate_spline_area(limit + 1, limit),
            Err(crate::Error::Bitstream(
                jxl_bitstream::Error::ProfileConformance("too large estimated area for splines")
            ))
        ));

        let mut extreme = crate::data::QuantSpline::default();
        extreme.manhattan_distance = u64::MAX;
        extreme.sigma_dct = [i32::MAX; 32];
        extreme.xyb_dct[1] = [i32::MAX; 32];
        let splines = crate::data::Splines {
            quant_splines: alloc::vec![extreme],
            quant_adjust: 0,
        };
        let saturated_estimate = splines.estimate_area(None);
        assert_eq!(saturated_estimate, u64::MAX);
        assert!(matches!(
            validate_spline_area(saturated_estimate, limit),
            Err(crate::Error::Bitstream(
                jxl_bitstream::Error::ProfileConformance("too large estimated area for splines")
            ))
        ));
    }
}
