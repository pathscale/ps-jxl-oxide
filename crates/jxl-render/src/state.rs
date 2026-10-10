use alloc::collections::BTreeMap;

use jxl_frame::data::{HfGlobal, LfGlobal, LfGroup};
use jxl_modular::{ChannelShift, Sample};

#[derive(Debug)]
pub struct RenderCache<S: Sample> {
    pub(crate) lf_global: Option<LfGlobal<S>>,
    pub(crate) hf_global: Option<HfGlobal>,
    pub(crate) lf_groups: BTreeMap<u32, LfGroup<S>>,
}

impl<S: Sample> RenderCache<S> {
    pub fn new(frame: &crate::IndexedFrame) -> Self {
        let frame_header = frame.header();
        let jpeg_upsampling = frame_header.jpeg_upsampling;
        let shifts_cbycr: [_; 3] =
            core::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));

        let lf_width = frame_header.color_sample_width().div_ceil(8);
        let lf_height = frame_header.color_sample_height().div_ceil(8);
        let mut whd = [(lf_width, lf_height); 3];
        for ((w, h), shift) in whd.iter_mut().zip(shifts_cbycr) {
            let (shift_w, shift_h) = shift.shift_size((lf_width, lf_height));
            *w = shift_w;
            *h = shift_h;
        }
        Self {
            lf_global: None,
            hf_global: None,
            lf_groups: BTreeMap::new(),
        }
    }
}
