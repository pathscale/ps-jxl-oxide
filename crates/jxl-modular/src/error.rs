#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Decoded MA tree is invalid.
    InvalidMaTree,
    /// Global MA tree is requested but not available.
    GlobalMaTreeNotAvailable,
    /// Decoded Rct transform parameters are invalid.
    InvalidRctParams,
    /// Decoded Palette transform parameters are invalid.
    InvalidPaletteParams,
    /// Decoded Squeeze transform parameters are invalid.
    InvalidSqueezeParams,
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    Buffer(jxl_grid::OutOfMemory),
    /// An owned task was cancelled or returned an invalid output.
    TaskCancelled,
}

impl From<jxl_bitstream::Error> for Error {
    fn from(err: jxl_bitstream::Error) -> Self {
        Self::Bitstream(err)
    }
}

impl From<jxl_coding::Error> for Error {
    fn from(err: jxl_coding::Error) -> Self {
        Self::Decoder(err)
    }
}

impl From<jxl_grid::OutOfMemory> for Error {
    fn from(err: jxl_grid::OutOfMemory) -> Self {
        Self::Buffer(err)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use Error::*;

        match self {
            InvalidMaTree => write!(f, "invalid meta-adaptive tree"),
            GlobalMaTreeNotAvailable => {
                write!(f, "global meta-adaptive tree requested but unavailable")
            }
            InvalidRctParams => write!(f, "invalid Rct transform parameters"),
            InvalidPaletteParams => write!(f, "invalid Palette transform parameters"),
            InvalidSqueezeParams => write!(f, "invalid Squeeze transform parameters"),
            Bitstream(err) => write!(f, "bitstream error: {err}"),
            Decoder(err) => write!(f, "entropy decoder error: {err}"),
            Buffer(err) => write!(f, "{err}"),
            TaskCancelled => write!(f, "modular task cancelled or returned an invalid output"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        use Error::*;

        match self {
            Bitstream(err) => Some(err),
            Decoder(err) => Some(err),
            Buffer(err) => Some(err),
            _ => None,
        }
    }
}

impl Error {
    pub fn unexpected_eof(&self) -> bool {
        match self {
            Error::Bitstream(e) => e.unexpected_eof(),
            Error::Decoder(e) => e.unexpected_eof(),
            _ => false,
        }
    }
}

pub type Result<T> = core::result::Result<T, Error>;
