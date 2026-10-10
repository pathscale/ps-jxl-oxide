use crate::filter::{
    epf::EpfRowHandler,
    gabor::{GaborRowHandler, run_gabor_row_generic},
};

mod epf;
mod gabor;

use gabor::run_gabor_row_aarch64_neon;

pub fn epf_row_handler<const STEP: usize>() -> Option<EpfRowHandler> {
    if is_aarch64_feature_detected!("neon") {
        Some(epf::epf_row_aarch64_neon::<STEP>)
    } else {
        None
    }
}

pub fn gabor_row_handler() -> GaborRowHandler {
    if is_aarch64_feature_detected!("neon") {
        run_gabor_row_aarch64_neon
    } else {
        run_gabor_row_generic
    }
}
