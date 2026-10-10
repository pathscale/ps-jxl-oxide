use crate::filter::epf::EpfRowHandler;

mod epf;

pub fn epf_row_handler<const STEP: usize>() -> Option<EpfRowHandler> {
    Some(epf::epf_row_wasm32_simd128::<STEP>)
}

pub use super::generic::gabor_row_handler;
