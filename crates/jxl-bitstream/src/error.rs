/// The error type for JPEG XL bitstream-level operations.
///
/// Other crates of jxl-oxide may return `ValidationFailed` or `ProfileConformance` errors while
/// validating decoded data.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Error from a standard-library I/O adapter.
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// More input is required to complete the operation.
    UnexpectedEof,
    /// Memory for input span descriptors could not be reserved.
    Allocation(alloc::collections::TryReserveError),
    /// Container was invalid.
    InvalidBox,
    /// `PadZeroToByte` read non-zero bits.
    NonZeroPadding,
    /// Parsed floating point value was Infinity or NaN.
    InvalidFloat,
    /// Parsed value couldn't be represented with the given enum.
    InvalidEnum { name: &'static str, value: u32 },
    /// The bitstream is invalid.
    ValidationFailed(&'static str),
    /// The codestream does not conform to the current decoder profile.
    ProfileConformance(&'static str),
    /// The bitstream couldn't be skipped to the given position, mainly due to the direction being
    /// backwards.
    CannotSkip,
    /// The bistream offsed was not aligned to read byte-aligned data.
    NotAligned,
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            #[cfg(feature = "std")]
            Self::Io(e) => Some(e),
            Self::Allocation(e) => Some(e),
            _ => None,
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Io(e) => {
                write!(f, "I/O error: {e}")
            }
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::Allocation(e) => write!(f, "allocation error: {e}"),
            Self::InvalidBox => write!(f, "invalid container"),
            Self::NonZeroPadding => {
                write!(f, "PadZeroToByte() read non-zero bits")
            }
            Self::InvalidFloat => {
                write!(f, "F16() read NaN or Infinity")
            }
            Self::InvalidEnum { name, value } => {
                write!(f, "Enum({name}) read invalid enum value of {value}")
            }
            Self::ValidationFailed(msg) => {
                write!(f, "bitstream validation failed: {msg}")
            }
            Self::ProfileConformance(msg) => {
                write!(f, "not supported by current profile: {msg}")
            }
            Self::CannotSkip => {
                write!(f, "target bookmark already passed")
            }
            Self::NotAligned => {
                write!(f, "bitstream is unaligned")
            }
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl Error {
    pub fn unexpected_eof(&self) -> bool {
        match self {
            Self::UnexpectedEof => true,
            #[cfg(feature = "std")]
            Self::Io(e) => e.kind() == std::io::ErrorKind::UnexpectedEof,
            _ => false,
        }
    }
}

/// Shorthand for result type of `jxl_bitstream`.
pub type BitstreamResult<T> = core::result::Result<T, Error>;
