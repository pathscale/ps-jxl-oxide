use alloc::boxed::Box;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(jxl_bitstream::Error),
    Decoder(jxl_coding::Error),
    InvalidIccStream(&'static str),
    IccParseFailure(&'static str),
    UnsupportedColorEncoding,
    UnsupportedIccProfile,
    IccProfileEmbedded,
    InvalidEnumColorspace,
    CmsNotAvailable,
    TaskCancelled,
    NotReady,
    InvalidChannelLayout,
    CmsFailure(Box<dyn core::error::Error + Send + Sync + 'static>),
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

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use Error::*;

        match self {
            Bitstream(err) => write!(f, "bitstream error: {err}"),
            Decoder(err) => write!(f, "entropy decoder error: {err}"),
            InvalidIccStream(s) => write!(f, "invalid ICC stream: {s}"),
            IccParseFailure(s) => write!(f, "parsing ICC profile failed: {s}"),
            UnsupportedColorEncoding => write!(f, "unsupported color encoding"),
            UnsupportedIccProfile => write!(f, "unsupported ICC profile"),
            IccProfileEmbedded => write!(f, "embedded ICC profile is signalled, use it instead"),
            InvalidEnumColorspace => write!(f, "unknown colorspace without embedded ICC profile"),
            CmsNotAvailable => write!(f, "color management system is not available"),
            TaskCancelled => write!(f, "color conversion task cancelled"),
            NotReady => write!(f, "color conversion suspended; use the async API"),
            InvalidChannelLayout => write!(f, "color channel lengths differ"),
            CmsFailure(err) => write!(f, "color management system error: {err}"),
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        use Error::*;

        match self {
            Bitstream(err) => Some(err),
            Decoder(err) => Some(err),
            CmsFailure(err) => Some(&**err),
            _ => None,
        }
    }
}

impl Error {
    pub fn unexpected_eof(&self) -> bool {
        if let Error::Bitstream(e) | Error::Decoder(jxl_coding::Error::Bitstream(e)) = self {
            return e.unexpected_eof();
        }
        false
    }
}

pub type Result<T> = core::result::Result<T, Error>;
