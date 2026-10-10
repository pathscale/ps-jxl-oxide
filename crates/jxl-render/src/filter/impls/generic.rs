#![allow(dead_code)]

use crate::filter::{
    epf::EpfRowHandler,
    gabor::{GaborRowHandler, run_gabor_row_generic},
};

pub(crate) mod epf;
pub(crate) mod gabor;

pub fn epf_row_handler<const STEP: usize>() -> Option<EpfRowHandler> {
    None
}

pub fn gabor_row_handler() -> GaborRowHandler {
    run_gabor_row_generic
}
