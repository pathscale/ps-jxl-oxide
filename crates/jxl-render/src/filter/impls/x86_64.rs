use crate::filter::{
    epf::EpfRowHandler,
    gabor::{GaborRowHandler, run_gabor_row_generic},
};

mod epf_sse41;
mod gabor_avx2;

pub fn epf_row_handler<const STEP: usize>() -> Option<EpfRowHandler> {
    if is_x86_feature_detected!("sse4.1") {
        Some(epf_sse41::epf_row_x86_64_sse41::<STEP>)
    } else {
        None
    }
}

pub fn gabor_row_handler() -> GaborRowHandler {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        gabor_avx2::run_gabor_row_x86_64_avx2
    } else {
        run_gabor_row_generic
    }
}

