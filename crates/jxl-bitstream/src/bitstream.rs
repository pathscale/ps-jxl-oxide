use crate::{BitstreamResult, Error, Section, Spans};

/// Bitstream reader over a borrowed contiguous buffer or immutable spans.
///
/// Implementation is mostly from [jxl-rs].
///
/// [jxl-rs]: https://github.com/libjxl/jxl-rs
#[derive(Clone)]
pub struct Bitstream<'buf> {
    bytes: &'buf [u8],
    spans: Option<&'buf Spans>,
    next_span: usize,
    remaining_bytes: usize,
    logical_start: usize,
    buf: u64,
    num_read_bits: usize,
    remaining_buf_bits: usize,
}

impl core::fmt::Debug for Bitstream<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Bitstream")
            .field(
                "bytes",
                &format_args!(
                    "({} byte{} left)",
                    self.remaining_bytes,
                    if self.remaining_bytes == 1 { "" } else { "s" },
                ),
            )
            .field("buf", &format_args!("0x{:016x}", self.buf))
            .field("num_read_bits", &self.num_read_bits)
            .field("remaining_buf_bits", &self.remaining_buf_bits)
            .finish()
    }
}

impl<'buf> Bitstream<'buf> {
    /// Create a new bitstream reader.
    #[inline]
    pub fn new(bytes: &'buf [u8]) -> Self {
        Self {
            bytes,
            spans: None,
            next_span: 0,
            remaining_bytes: bytes.len(),
            logical_start: 0,
            buf: 0,
            num_read_bits: 0,
            remaining_buf_bits: 0,
        }
    }

    /// Creates a segmented reader without copying any input bytes.
    pub fn from_spans(spans: &'buf Spans) -> Self {
        Self {
            bytes: spans.span_bytes(0).unwrap_or(&[]),
            spans: Some(spans),
            next_span: 1,
            remaining_bytes: spans.len(),
            logical_start: 0,
            buf: 0,
            num_read_bits: 0,
            remaining_buf_bits: 0,
        }
    }

    /// Creates a reader retaining the section's logical origin.
    pub fn from_section(section: &'buf Section) -> Self {
        let mut reader = Self::from_spans(section.spans());
        reader.logical_start = section.range().start;
        reader
    }

    /// Describes bytes from this reader's source, when it is span-backed.
    pub fn source_section(
        &self,
        range: core::ops::Range<usize>,
    ) -> Option<BitstreamResult<Section>> {
        self.spans.map(|spans| {
            let start = self
                .logical_start
                .checked_add(range.start)
                .ok_or(Error::ValidationFailed("section origin overflow"))?;
            Section::new(start, spans.slice(range)?)
        })
    }

    /// Returns the number of bits that are read or skipped.
    #[inline]
    pub fn num_read_bits(&self) -> usize {
        self.num_read_bits
    }
}

impl Bitstream<'_> {
    /// Fills bit buffer from byte buffer.
    #[inline]
    fn refill(&mut self) {
        self.advance_span();
        if let &[b0, b1, b2, b3, b4, b5, b6, b7, ..] = self.bytes {
            let bits = u64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
            self.buf |= bits << self.remaining_buf_bits;
            let read_bytes = (63 - self.remaining_buf_bits) >> 3;
            self.remaining_buf_bits |= 56;
            self.remaining_bytes -= read_bytes;
            // SAFETY: read_bytes < 8, self.bytes.len() >= 8 (from the pattern).
            self.bytes = unsafe {
                core::slice::from_raw_parts(
                    self.bytes.as_ptr().add(read_bytes),
                    self.bytes.len() - read_bytes,
                )
            };
        } else {
            self.refill_slow()
        }
    }

    #[inline]
    fn advance_span(&mut self) {
        if self.bytes.is_empty()
            && let Some(spans) = self.spans
            && let Some(bytes) = spans.span_bytes(self.next_span)
        {
            self.bytes = bytes;
            self.next_span += 1;
        }
    }

    fn skip_bytes(&mut self, count: usize) {
        debug_assert!(count <= self.remaining_bytes);
        self.remaining_bytes -= count;
        if count <= self.bytes.len() {
            self.bytes = &self.bytes[count..];
        } else if let Some(spans) = self.spans {
            let (index, bytes) = spans.locate(spans.len() - self.remaining_bytes);
            self.bytes = bytes;
            self.next_span = index + 1;
        } else {
            unreachable!("contiguous skip exceeded its input");
        }
    }

    #[inline(never)]
    fn refill_slow(&mut self) {
        while self.remaining_buf_bits < 56 {
            self.advance_span();
            let Some((&b, next)) = self.bytes.split_first() else {
                return;
            };

            self.buf |= (b as u64) << self.remaining_buf_bits;
            self.remaining_buf_bits += 8;
            self.bytes = next;
            self.remaining_bytes -= 1;
        }
    }
}

impl Bitstream<'_> {
    /// Peeks bits from bitstream, without consuming them.
    ///
    /// This method refills the bit buffer.
    #[inline]
    pub fn peek_bits(&mut self, n: usize) -> u32 {
        debug_assert!(n <= 32);
        self.refill();
        (self.buf & ((1u64 << n) - 1)) as u32
    }

    /// Peeks bits from bitstream, without consuming them.
    ///
    /// This method refills the bit buffer.
    #[inline]
    pub fn peek_bits_const<const N: usize>(&mut self) -> u32 {
        debug_assert!(N <= 32);
        self.refill();
        (self.buf & ((1u64 << N) - 1)) as u32
    }

    /// Peeks bits from already filled bitstream, without consuming them.
    ///
    /// This method *does not* refill the bit buffer.
    #[inline]
    pub fn peek_bits_prefilled(&mut self, n: usize) -> u32 {
        debug_assert!(n <= 32);
        (self.buf & ((1u64 << n) - 1)) as u32
    }

    /// Peeks bits from already filled bitstream, without consuming them.
    ///
    /// This method *does not* refill the bit buffer.
    #[inline]
    pub fn peek_bits_prefilled_const<const N: usize>(&mut self) -> u32 {
        debug_assert!(N <= 32);
        (self.buf & ((1u64 << N) - 1)) as u32
    }

    /// Consumes bits in bit buffer.
    ///
    /// # Errors
    /// This method returns `Err(Error::UnexpectedEof)` when there are not enough
    /// bits in the bit buffer.
    #[inline]
    pub fn consume_bits(&mut self, n: usize) -> BitstreamResult<()> {
        self.remaining_buf_bits = self
            .remaining_buf_bits
            .checked_sub(n)
            .ok_or(Error::UnexpectedEof)?;
        self.num_read_bits += n;
        self.buf >>= n;
        Ok(())
    }

    /// Consumes bits in bit buffer.
    ///
    /// # Errors
    /// This method returns `Err(Error::UnexpectedEof)` when there are not enough
    /// bits in the bit buffer.
    #[inline]
    pub fn consume_bits_const<const N: usize>(&mut self) -> BitstreamResult<()> {
        self.remaining_buf_bits = self
            .remaining_buf_bits
            .checked_sub(N)
            .ok_or(Error::UnexpectedEof)?;
        self.num_read_bits += N;
        self.buf >>= N;
        Ok(())
    }

    /// Read and consume bits from bitstream.
    #[inline]
    pub fn read_bits(&mut self, n: usize) -> BitstreamResult<u32> {
        let ret = self.peek_bits(n);
        self.consume_bits(n)?;
        Ok(ret)
    }

    #[inline(never)]
    pub fn skip_bits(&mut self, mut n: usize) -> BitstreamResult<()> {
        if let Some(next_remaining_bits) = self.remaining_buf_bits.checked_sub(n) {
            self.num_read_bits += n;
            self.remaining_buf_bits = next_remaining_bits;
            self.buf >>= n;
            return Ok(());
        }

        n -= self.remaining_buf_bits;
        self.num_read_bits += self.remaining_buf_bits;
        self.buf = 0;
        self.remaining_buf_bits = 0;
        let available = self
            .remaining_bytes
            .checked_mul(8)
            .ok_or(Error::ValidationFailed("bitstream length overflow"))?;
        if n > available {
            self.num_read_bits += available;
            self.skip_bytes(self.remaining_bytes);
            return Err(Error::UnexpectedEof);
        }

        self.skip_bytes(n / 8);
        self.num_read_bits += n & !7;
        n %= 8;
        self.refill();
        self.consume_bits(n)
    }

    /// Performs `ZeroPadToByte` as defined in the JPEG XL specification.
    pub fn zero_pad_to_byte(&mut self) -> BitstreamResult<()> {
        let n = (8 - (self.num_read_bits & 7)) & 7;
        if self.read_bits(n)? != 0 {
            Err(Error::NonZeroPadding)
        } else {
            Ok(())
        }
    }
}

impl Bitstream<'_> {
    /// Reads an `U32` as defined in the JPEG XL specification.
    ///
    /// # Example
    ///
    /// ```
    /// use jxl_bitstream::{Bitstream, U};
    ///
    /// let buf = [0b110010];
    /// let mut bitstream = Bitstream::new(&buf);
    /// let val = bitstream.read_u32(1, U(2), 3 + U(4), 19 + U(8)).expect("failed to read data");
    /// assert_eq!(val, 15);
    /// ```
    #[inline]
    pub fn read_u32(
        &mut self,
        d0: impl Into<U32Specifier>,
        d1: impl Into<U32Specifier>,
        d2: impl Into<U32Specifier>,
        d3: impl Into<U32Specifier>,
    ) -> BitstreamResult<u32> {
        let d = match self.read_bits(2)? {
            0 => d0.into(),
            1 => d1.into(),
            2 => d2.into(),
            3 => d3.into(),
            _ => unreachable!(),
        };
        match d {
            U32Specifier::Constant(x) => Ok(x),
            U32Specifier::BitsOffset(offset, n) => {
                self.read_bits(n).map(|x| x.wrapping_add(offset))
            }
        }
    }

    /// Reads an `U64` as defined in the JPEG XL specification.
    pub fn read_u64(&mut self) -> BitstreamResult<u64> {
        let selector = self.read_bits(2)?;
        Ok(match selector {
            0 => 0u64,
            1 => self.read_bits(4)? as u64 + 1,
            2 => self.read_bits(8)? as u64 + 17,
            3 => {
                let mut value = self.read_bits(12)? as u64;
                let mut shift = 12u32;
                while self.read_bits(1)? == 1 {
                    if shift == 60 {
                        value |= (self.read_bits(4)? as u64) << shift;
                        break;
                    }
                    value |= (self.read_bits(8)? as u64) << shift;
                    shift += 8;
                }
                value
            }
            _ => unreachable!(),
        })
    }

    /// Reads a `Bool` as defined in the JPEG XL specification.
    #[inline]
    pub fn read_bool(&mut self) -> BitstreamResult<bool> {
        self.read_bits(1).map(|x| x != 0)
    }

    /// Reads an `F16` as defined in the JPEG XL specification, and convert it to `f32`.
    ///
    /// # Errors
    /// Returns `Error::InvalidFloat` if the value is `NaN` or `Infinity`.
    pub fn read_f16_as_f32(&mut self) -> BitstreamResult<f32> {
        let v = self.read_bits(16)?;
        let neg_bit = (v & 0x8000) << 16;

        if v & 0x7fff == 0 {
            // Zero
            return Ok(f32::from_bits(neg_bit));
        }
        let mantissa = v & 0x3ff; // 10 bits
        let exponent = (v >> 10) & 0x1f; // 5 bits
        if exponent == 0x1f {
            // NaN, Infinity
            Err(Error::InvalidFloat)
        } else if exponent == 0 {
            // Subnormal
            let val = (1.0 / 16384.0) * (mantissa as f32 / 1024.0);
            Ok(if neg_bit != 0 { -val } else { val })
        } else {
            // Normal
            let mantissa = mantissa << 13; // 23 bits
            let exponent = exponent + 112;
            let bitpattern = mantissa | (exponent << 23) | neg_bit;
            Ok(f32::from_bits(bitpattern))
        }
    }

    /// Reads an enum as defined in the JPEG XL specification.
    pub fn read_enum<E: TryFrom<u32>>(&mut self) -> BitstreamResult<E> {
        let v = self.read_u32(0, 1, 2 + U(4), 18 + U(6))?;
        E::try_from(v).map_err(|_| Error::InvalidEnum {
            name: core::any::type_name::<E>(),
            value: v,
        })
    }
}

/// Bit specifier for [`Bitstream::read_u32`].
pub enum U32Specifier {
    Constant(u32),
    BitsOffset(u32, usize),
}

/// Bit count for use in [`Bitstream::read_u32`].
pub struct U(pub usize);

impl From<u32> for U32Specifier {
    fn from(value: u32) -> Self {
        Self::Constant(value)
    }
}

impl From<U> for U32Specifier {
    fn from(value: U) -> Self {
        Self::BitsOffset(0, value.0)
    }
}

impl core::ops::Add<U> for u32 {
    type Output = U32Specifier;

    fn add(self, rhs: U) -> Self::Output {
        U32Specifier::BitsOffset(self, rhs.0)
    }
}
