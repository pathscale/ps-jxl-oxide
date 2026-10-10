#[derive(Debug)]
pub(crate) enum Error {
    UnexpectedEof(&'static str),
    InvalidData(&'static str),
    RenderSuspended,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnexpectedEof(message) | Self::InvalidData(message) => f.write_str(message),
            Self::RenderSuspended => {
                f.write_str("render suspended; use the async API with the supplied executor")
            }
        }
    }
}

impl core::error::Error for Error {}
