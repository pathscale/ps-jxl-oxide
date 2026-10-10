#[cfg(target_arch = "x86_64")]
macro_rules! is_x86_feature_detected {
    ($feature:tt) => {{
        #[cfg(feature = "std")]
        {
            std::arch::is_x86_feature_detected!($feature)
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(target_feature = $feature)
        }
    }};
}

#[cfg(target_arch = "aarch64")]
macro_rules! is_aarch64_feature_detected {
    ($feature:tt) => {{
        #[cfg(feature = "std")]
        {
            std::arch::is_aarch64_feature_detected!($feature)
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(target_feature = $feature)
        }
    }};
}
