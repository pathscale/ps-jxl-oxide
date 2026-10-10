use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use crate::CheckResult;

#[derive(Clone, Copy)]
enum Dtype {
    F32,
    F16,
    U8,
    U16,
}

impl Dtype {
    fn parse(descriptor: &str) -> CheckResult<Self> {
        match descriptor {
            "<f4" => Ok(Self::F32),
            "<f2" => Ok(Self::F16),
            "|u1" | "<u1" => Ok(Self::U8),
            "<u2" => Ok(Self::U16),
            "=f4" if cfg!(target_endian = "little") => Ok(Self::F32),
            "=f2" if cfg!(target_endian = "little") => Ok(Self::F16),
            "=u1" => Ok(Self::U8),
            "=u2" if cfg!(target_endian = "little") => Ok(Self::U16),
            _ => Err(format!("unsupported NPY dtype {descriptor:?}")),
        }
    }

    fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F16 | Self::U16 => 2,
            Self::F32 => 4,
        }
    }

    fn sample(self, bytes: &[u8]) -> f32 {
        match self {
            Self::F32 => {
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            }
            Self::F16 => half_to_float(u16::from_le_bytes([bytes[0], bytes[1]])),
            Self::U8 => f32::from(bytes[0]) / 255.0,
            Self::U16 => f32::from(u16::from_le_bytes([bytes[0], bytes[1]])) / 65535.0,
        }
    }
}

fn half_to_float(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    match exponent {
        0 => {
            let value = fraction as f32 / 16_777_216.0;
            if sign == 0 { value } else { -value }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (fraction << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (fraction << 13)),
    }
}

pub struct Npy {
    reader: BufReader<File>,
    path: PathBuf,
    dtype: Dtype,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    frame_samples: usize,
    remaining: usize,
}

impl Npy {
    pub fn open(path: &Path) -> CheckResult<Self> {
        Self::open_inner(path).map_err(|error| format!("{}: {error}", path.display()))
    }

    fn open_inner(path: &Path) -> CheckResult<Self> {
        let file = File::open(path).map_err(|error| error.to_string())?;
        let file_length = file.metadata().map_err(|error| error.to_string())?.len();
        let mut reader = BufReader::new(file);
        let mut prefix = [0u8; 8];
        reader.read_exact(&mut prefix).map_err(|error| error.to_string())?;
        if &prefix[..6] != b"\x93NUMPY" {
            return Err("invalid NPY magic".to_owned());
        }
        let (header_length, prefix_length) = match (prefix[6], prefix[7]) {
            (1, 0) => {
                let mut bytes = [0u8; 2];
                reader.read_exact(&mut bytes).map_err(|error| error.to_string())?;
                (usize::from(u16::from_le_bytes(bytes)), 10usize)
            }
            (2 | 3, 0) => {
                let mut bytes = [0u8; 4];
                reader.read_exact(&mut bytes).map_err(|error| error.to_string())?;
                (
                    usize::try_from(u32::from_le_bytes(bytes))
                        .map_err(|_| "NPY header length overflows usize".to_owned())?,
                    12usize,
                )
            }
            _ => return Err("unsupported NPY version".to_owned()),
        };
        if header_length == 0 || header_length > 1_048_576 {
            return Err("NPY header length is outside the supported range".to_owned());
        }
        let mut header = vec![0u8; header_length];
        reader.read_exact(&mut header).map_err(|error| error.to_string())?;
        let header = std::str::from_utf8(&header)
            .map_err(|_| "NPY header is not valid UTF-8 or ASCII".to_owned())?;
        let (descriptor, shape) = Header::parse(header)?;
        let dtype = Dtype::parse(&descriptor)?;
        let (frames, height, width, channels) = match shape.as_slice() {
            [height, width] => (1, *height, *width, 1),
            [height, width, channels] => (1, *height, *width, *channels),
            [frames, height, width, channels] => (*frames, *height, *width, *channels),
            _ => return Err("expected an HW, HWC, or FHWC NPY shape".to_owned()),
        };
        if frames == 0 || height == 0 || width == 0 || channels == 0 {
            return Err("NPY dimensions must be positive".to_owned());
        }
        let frame_samples = height
            .checked_mul(width)
            .and_then(|value| value.checked_mul(channels))
            .ok_or_else(|| "NPY frame sample count overflows usize".to_owned())?;
        let payload_length = frame_samples
            .checked_mul(frames)
            .and_then(|value| value.checked_mul(dtype.size()))
            .ok_or_else(|| "NPY payload length overflows usize".to_owned())?;
        let total_length = prefix_length
            .checked_add(header_length)
            .and_then(|value| value.checked_add(payload_length))
            .ok_or_else(|| "NPY file length overflows usize".to_owned())?;
        let total_length = u64::try_from(total_length)
            .map_err(|_| "NPY file length overflows u64".to_owned())?;
        if file_length != total_length {
            return Err(format!(
                "NPY file length is {file_length}, expected {total_length}"
            ));
        }
        Ok(Self {
            reader,
            path: path.to_owned(),
            dtype,
            frames,
            height,
            width,
            channels,
            frame_samples,
            remaining: frames,
        })
    }

    pub fn read_frame(&mut self) -> CheckResult<Vec<f32>> {
        if self.remaining == 0 {
            return Err(format!("{}: no reference frames remain", self.path.display()));
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.frame_samples)
            .map_err(|error| format!("cannot allocate reference frame: {error}"))?;
        let mut bytes = [0u8; 8192];
        let size = self.dtype.size();
        while output.len() < self.frame_samples {
            let samples = (self.frame_samples - output.len()).min(bytes.len() / size);
            let bytes = &mut bytes[..samples * size];
            self.reader
                .read_exact(bytes)
                .map_err(|error| format!("{}: {error}", self.path.display()))?;
            for sample in bytes.chunks_exact(size) {
                output.push(self.dtype.sample(sample));
            }
        }
        self.remaining -= 1;
        Ok(output)
    }
}

struct Header<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> Header<'a> {
    fn parse(text: &'a str) -> CheckResult<(String, Vec<usize>)> {
        let mut parser = Self { text, position: 0 };
        let mut descriptor = None;
        let mut fortran = None;
        let mut shape = None;
        parser.expect(b'{')?;
        loop {
            parser.whitespace();
            if parser.take(b'}') {
                break;
            }
            let key = parser.quoted()?;
            parser.expect(b':')?;
            parser.whitespace();
            match key {
                "descr" if descriptor.is_none() => {
                    descriptor = Some(parser.quoted()?.to_owned());
                }
                "fortran_order" if fortran.is_none() => {
                    fortran = Some(if parser.literal("False") {
                        false
                    } else if parser.literal("True") {
                        true
                    } else {
                        return Err("invalid NPY fortran_order".to_owned());
                    });
                }
                "shape" if shape.is_none() => {
                    shape = Some(parser.shape()?);
                }
                _ => return Err(format!("unknown or duplicate NPY header key {key:?}")),
            }
            parser.whitespace();
            if parser.take(b'}') {
                break;
            }
            parser.expect(b',')?;
        }
        parser.whitespace();
        if parser.position != text.len() {
            return Err("trailing NPY header data".to_owned());
        }
        if fortran != Some(false) {
            return Err("NPY arrays must explicitly use C order".to_owned());
        }
        Ok((
            descriptor.ok_or_else(|| "NPY header is missing descr".to_owned())?,
            shape.ok_or_else(|| "NPY header is missing shape".to_owned())?,
        ))
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.position).copied()
    }

    fn whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.position += 1;
        }
    }

    fn take(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, byte: u8) -> CheckResult<()> {
        self.whitespace();
        if self.take(byte) {
            Ok(())
        } else {
            Err(format!(
                "NPY header byte {}: expected {:?}",
                self.position,
                char::from(byte)
            ))
        }
    }

    fn literal(&mut self, literal: &str) -> bool {
        if self.text[self.position..].starts_with(literal) {
            self.position += literal.len();
            true
        } else {
            false
        }
    }

    fn quoted(&mut self) -> CheckResult<&'a str> {
        self.whitespace();
        let quote = self.peek().ok_or_else(|| "truncated NPY string".to_owned())?;
        if !matches!(quote, b'\'' | b'"') {
            return Err("expected a quoted NPY string".to_owned());
        }
        self.position += 1;
        let start = self.position;
        loop {
            match self.peek() {
                Some(byte) if byte == quote => {
                    let value = &self.text[start..self.position];
                    self.position += 1;
                    return Ok(value);
                }
                Some(byte) if byte.is_ascii() && byte >= 0x20 && byte != b'\\' => {
                    self.position += 1;
                }
                _ => return Err("invalid or truncated NPY string".to_owned()),
            }
        }
    }

    fn shape(&mut self) -> CheckResult<Vec<usize>> {
        self.expect(b'(')?;
        let mut dimensions = Vec::new();
        self.whitespace();
        if self.take(b')') {
            return Ok(dimensions);
        }
        loop {
            self.whitespace();
            let start = self.position;
            while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                self.position += 1;
            }
            if start == self.position {
                return Err("expected a nonnegative NPY dimension".to_owned());
            }
            let dimension = self.text[start..self.position]
                .parse::<usize>()
                .map_err(|_| "NPY dimension overflows usize".to_owned())?;
            dimensions.push(dimension);
            if dimensions.len() > 4 {
                return Err("NPY arrays with more than four dimensions are unsupported".to_owned());
            }
            self.whitespace();
            if self.take(b',') {
                self.whitespace();
                if self.take(b')') {
                    break;
                }
            } else {
                self.expect(b')')?;
                break;
            }
        }
        Ok(dimensions)
    }
}
