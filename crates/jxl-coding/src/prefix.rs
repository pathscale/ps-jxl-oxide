//! Prefix code based on Brotli
use alloc::{vec, vec::Vec};

use jxl_bitstream::Bitstream;

use crate::{CodingResult, Error};

const MAX_PREFIX_BITS: usize = 15;
const MAX_TOPLEVEL_BITS: usize = 10;

#[derive(Debug)]
pub struct Histogram {
    toplevel_bits: usize,
    toplevel_mask: u32,
    toplevel_entries: Vec<Entry>,
    second_level_entries: Vec<Entry>,
}

#[derive(Debug, Copy, Clone, Default)]
struct Entry {
    nested: bool,
    bits_or_mask: u8,
    symbol_or_offset: u16,
}

const _: () = {
    ["size of `struct Entry`"][core::mem::size_of::<Entry>() - 4];
};

impl Histogram {
    fn with_code_lengths(code_lengths: Vec<u8>) -> CodingResult<Self> {
        let mut syms_for_length = Vec::with_capacity(MAX_PREFIX_BITS);
        for (sym, len) in code_lengths.into_iter().enumerate() {
            let sym = sym as u16;
            if len > 0 {
                if syms_for_length.len() < len as usize {
                    syms_for_length.resize_with(len as usize, Vec::new);
                }
                syms_for_length[len as usize - 1].push(sym);
            }
        }

        let toplevel_bits = syms_for_length.len().min(MAX_TOPLEVEL_BITS);
        let mut entries = vec![Entry::default(); 1 << toplevel_bits];
        let mut current_bits = 0u16;
        for (idx, syms) in syms_for_length.iter().enumerate().take(toplevel_bits) {
            let shifts = toplevel_bits - 1 - idx;
            for &sym in syms {
                let entry = Entry {
                    nested: false,
                    bits_or_mask: (idx + 1) as u8,
                    symbol_or_offset: sym,
                };
                entries[current_bits as usize..][..(1 << shifts)].fill(entry);
                current_bits += 1u16 << shifts;
            }
        }

        let mut second_level_entries = Vec::new();
        if toplevel_bits < syms_for_length.len() {
            let mut remaining_entries = Vec::new();
            let mut remaining_entry_bits = 0usize;
            for (idx, syms) in syms_for_length.iter().enumerate().skip(toplevel_bits) {
                if syms.is_empty() {
                    continue;
                }

                let chunk_size_bits = idx + 1 - toplevel_bits;
                let chunk_size = 1usize << chunk_size_bits;
                let mut chunk = Vec::with_capacity(chunk_size);
                if !remaining_entries.is_empty() {
                    let mult = 1usize << (chunk_size_bits - remaining_entry_bits);
                    for entry in remaining_entries {
                        for _ in 0..mult {
                            chunk.push(entry);
                        }
                    }
                }
                for &sym in syms {
                    let entry = Entry {
                        nested: false,
                        bits_or_mask: (idx + 1) as u8,
                        symbol_or_offset: sym,
                    };
                    chunk.push(entry);
                    if chunk.len() == chunk_size {
                        entries[current_bits as usize] = Entry {
                            nested: true,
                            bits_or_mask: (chunk_size - 1) as u8,
                            symbol_or_offset: second_level_entries.len() as u16,
                        };
                        vec_reverse_bits(&chunk, &mut second_level_entries);
                        current_bits += 1;
                        chunk = Vec::with_capacity(chunk_size);
                    }
                }
                remaining_entries = chunk;
                remaining_entry_bits = chunk_size_bits;
            }

            if !remaining_entries.is_empty() {
                return Err(Error::InvalidPrefixHistogram);
            }
        }

        if current_bits == 1 << toplevel_bits {
            let mut toplevel_entries = Vec::with_capacity(entries.len());
            vec_reverse_bits(&entries, &mut toplevel_entries);
            Ok(Self {
                toplevel_bits,
                toplevel_mask: (1 << toplevel_bits) - 1,
                toplevel_entries,
                second_level_entries,
            })
        } else {
            Err(Error::InvalidPrefixHistogram)
        }
    }

    fn with_single_symbol(symbol: u16) -> Self {
        let entry = Entry {
            nested: false,
            bits_or_mask: 0,
            symbol_or_offset: symbol,
        };
        Self {
            toplevel_bits: 0,
            toplevel_mask: 0,
            toplevel_entries: vec![entry],
            second_level_entries: Vec::new(),
        }
    }

    pub fn parse(bitstream: &mut Bitstream, alphabet_size: u32) -> CodingResult<Self> {
        match Self::parse_start(bitstream, alphabet_size)? {
            HistogramStart::Complete(histogram) => Ok(histogram),
            HistogramStart::Complex(mut parser) => parser.read::<false>(bitstream),
        }
    }

    fn parse_start(bitstream: &mut Bitstream, alphabet_size: u32) -> CodingResult<HistogramStart> {
        if alphabet_size == 1 {
            return Ok(HistogramStart::Complete(Self::with_single_symbol(0)));
        }

        if alphabet_size > 1u32 << MAX_PREFIX_BITS {
            return Err(Error::PrefixSymbolTooLarge(alphabet_size as usize));
        }

        let hskip = bitstream.read_bits(2)?;
        if hskip == 1 {
            Self::parse_simple(bitstream, alphabet_size).map(HistogramStart::Complete)
        } else {
            ComplexHistogram::new(bitstream, alphabet_size, hskip).map(HistogramStart::Complex)
        }
    }

    fn parse_simple(bitstream: &mut Bitstream, alphabet_size: u32) -> CodingResult<Self> {
        let alphabet_bits = alphabet_size.next_power_of_two().trailing_zeros() as usize;
        let nsym = bitstream.read_bits(2)? + 1;
        let it = match nsym {
            1 => {
                let sym = bitstream.read_bits(alphabet_bits)?;
                if sym >= alphabet_size {
                    return Err(Error::InvalidPrefixHistogram);
                }
                return Ok(Self::with_single_symbol(sym as u16));
            }
            2 => {
                let syms = [
                    0,
                    0,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];

                syms.into_iter().zip([0u8, 0, 1u8, 1])
            }
            3 => {
                let syms = [
                    0,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];

                syms.into_iter().zip([0u8, 1, 2, 2])
            }
            4 => {
                let syms = [
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                    bitstream.read_bits(alphabet_bits)? as usize,
                ];
                let tree_selector = bitstream.read_bool()?;

                if tree_selector {
                    syms.into_iter().zip([1u8, 2, 3, 3])
                } else {
                    syms.into_iter().zip([2u8, 2, 2, 2])
                }
            }
            _ => unreachable!(),
        };

        let mut code_lengths = vec![0u8; alphabet_size as usize];
        for (sym, len) in it {
            if let Some(out) = code_lengths.get_mut(sym) {
                *out = len;
            } else {
                return Err(Error::InvalidPrefixHistogram);
            }
        }
        Self::with_code_lengths(code_lengths)
    }
}

enum HistogramStart {
    Complete(Histogram),
    Complex(ComplexHistogram),
}

#[derive(Debug)]
pub(super) struct ComplexHistogram {
    code_length_histogram: Histogram,
    code_lengths: Vec<u8>,
    next: usize,
    bitacc: usize,
    prev_sym: u8,
    last_nonzero_sym: u8,
    last_repeat_count: usize,
    repeat_count: usize,
    repeat_sym: u8,
    #[cfg(test)]
    symbol_attempts: usize,
}

impl ComplexHistogram {
    pub(super) fn resume(
        parser: &mut Option<Self>,
        bitstream: &mut Bitstream,
        alphabet_size: u32,
    ) -> CodingResult<Histogram> {
        if parser.is_none() {
            let mut input = bitstream.clone();
            let start = Histogram::parse_start(&mut input, alphabet_size)?;
            *bitstream = input;
            match start {
                HistogramStart::Complete(histogram) => return Ok(histogram),
                HistogramStart::Complex(complex) => *parser = Some(complex),
            }
        }

        let histogram = parser.as_mut().unwrap().read::<true>(bitstream)?;
        *parser = None;
        Ok(histogram)
    }

    fn new(
        bitstream: &mut Bitstream,
        alphabet_size: u32,
        hskip: u32,
    ) -> CodingResult<Self> {
        const CODE_LENGTH_ORDER: [usize; 18] =
            [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut code_length_code_lengths = [0u8; 18];
        let mut bitacc = 0usize;

        let mut nonzero_count = 0;
        let mut nonzero_sym = 0;
        for idx in CODE_LENGTH_ORDER.into_iter().skip(hskip as usize) {
            // Read single code length code.
            let base = bitstream.read_u32(0, 4, 3, 8)? as u8;
            let len = if base == 8 {
                if bitstream.read_bool()? {
                    if bitstream.read_bool()? {
                        // 1111
                        5
                    } else {
                        // 0111
                        1
                    }
                } else {
                    // 011
                    2
                }
            } else {
                base
            };

            code_length_code_lengths[idx] = len;
            if len != 0 {
                nonzero_count += 1;
                nonzero_sym = idx;
                bitacc += 32 >> len;

                match bitacc.cmp(&32) {
                    core::cmp::Ordering::Less => {}
                    core::cmp::Ordering::Equal => break,
                    core::cmp::Ordering::Greater => return Err(Error::InvalidPrefixHistogram),
                }
            }
        }

        let code_length_histogram = if nonzero_count == 1 {
            Histogram::with_single_symbol(nonzero_sym as u16)
        } else if bitacc != 32 {
            return Err(Error::InvalidPrefixHistogram);
        } else {
            Histogram::with_code_lengths(code_length_code_lengths.to_vec())?
        };

        Ok(Self {
            code_length_histogram,
            code_lengths: vec![0; alphabet_size as usize],
            next: 0,
            bitacc: 0,
            prev_sym: 8,
            last_nonzero_sym: 8,
            last_repeat_count: 0,
            repeat_count: 0,
            repeat_sym: 0,
            #[cfg(test)]
            symbol_attempts: 0,
        })
    }

    fn read<const RESUME: bool>(&mut self, bitstream: &mut Bitstream) -> CodingResult<Histogram> {
        while self.next < self.code_lengths.len() {
            let len = if self.repeat_count > 0 {
                self.repeat_count -= 1;
                self.repeat_sym
            } else {
                let (sym, repeat_count) = if RESUME {
                    let mut input = bitstream.clone();
                    let symbol = self.read_length_symbol(&mut input)?;
                    *bitstream = input;
                    symbol
                } else {
                    self.read_length_symbol(bitstream)?
                };
                let len = match sym {
                    0 => 0,
                    1..=15 => {
                        self.last_nonzero_sym = sym;
                        sym
                    }
                    16 | 17 => {
                        self.repeat_count = repeat_count;
                        if self.prev_sym == sym {
                            let (factor, offset) = if sym == 16 { (3, 8) } else { (7, 16) };
                            self.repeat_count += self.last_repeat_count * factor - offset;
                            self.last_repeat_count += self.repeat_count;
                        } else {
                            self.last_repeat_count = self.repeat_count;
                        }
                        self.repeat_sym = if sym == 16 { self.last_nonzero_sym } else { 0 };
                        self.repeat_count -= 1;
                        self.repeat_sym
                    }
                    _ => unreachable!(),
                };
                self.prev_sym = sym;
                len
            };

            self.code_lengths[self.next] = len;
            self.next += 1;
            if len != 0 {
                self.bitacc += 1 << MAX_PREFIX_BITS.saturating_sub(len as usize);
                if self.bitacc > 1 << MAX_PREFIX_BITS {
                    return Err(Error::PrefixSymbolTooLarge(self.bitacc));
                } else if self.bitacc == 1 << MAX_PREFIX_BITS && self.repeat_count == 0 {
                    break;
                }
            }
        }

        if self.bitacc != 1 << MAX_PREFIX_BITS || self.repeat_count > 0 {
            return Err(Error::InvalidPrefixHistogram);
        }
        Histogram::with_code_lengths(core::mem::take(&mut self.code_lengths))
    }

    fn read_length_symbol(&mut self, bitstream: &mut Bitstream) -> CodingResult<(u8, usize)> {
        #[cfg(test)]
        {
            self.symbol_attempts += 1;
        }
        let sym = self.code_length_histogram.read_symbol(bitstream)? as u8;
        let repeat_count = match sym {
            16 => bitstream.read_bits(2)? as usize + 3,
            17 => bitstream.read_bits(3)? as usize + 3,
            _ => 0,
        };
        Ok((sym, repeat_count))
    }
}

impl Histogram {
    #[inline(always)]
    pub fn read_symbol(&self, bitstream: &mut Bitstream) -> CodingResult<u32> {
        let Self {
            toplevel_bits,
            toplevel_mask,
            ref toplevel_entries,
            ref second_level_entries,
        } = *self;
        let peeked = bitstream.peek_bits_const::<MAX_PREFIX_BITS>();
        let toplevel_offset = peeked & toplevel_mask;
        let toplevel_entry = toplevel_entries[toplevel_offset as usize];
        if toplevel_entry.nested {
            let chunk_offset = (peeked >> toplevel_bits) & (toplevel_entry.bits_or_mask as u32);
            let second_level_offset = toplevel_entry.symbol_or_offset as u32 + chunk_offset;
            let second_level_entry = second_level_entries[second_level_offset as usize];
            bitstream.consume_bits(second_level_entry.bits_or_mask as usize)?;
            Ok(second_level_entry.symbol_or_offset as u32)
        } else {
            bitstream.consume_bits(toplevel_entry.bits_or_mask as usize)?;
            Ok(toplevel_entry.symbol_or_offset as u32)
        }
    }

    #[inline]
    pub fn single_symbol(&self) -> Option<u32> {
        if let &[
            Entry {
                nested: false,
                bits_or_mask: 0,
                symbol_or_offset: symbol,
            },
        ] = &*self.toplevel_entries
        {
            Some(symbol as u32)
        } else {
            None
        }
    }
}

fn vec_reverse_bits(v: &[Entry], out: &mut Vec<Entry>) {
    let len = v.len();
    debug_assert!(len.is_power_of_two());
    let bits = len.trailing_zeros();
    let shift = usize::BITS - bits;
    for idx in 0..len {
        let rev_idx = idx.reverse_bits() >> shift;
        let entry = v[rev_idx];
        out.push(entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_header_retry_bounds_prefix_symbol_reads() {
        const ALPHABET_SIZE: usize = 1 << MAX_PREFIX_BITS;
        // hskip = 0, code lengths 1, 0, 0, 0, 1 in code-length order.
        let mut input = vec![0x1c, 0x70];
        input.resize(2 + ALPHABET_SIZE / 8, 0);
        input[2] = 1;
        *input.last_mut().unwrap() = 0x80;

        let mut initial = Bitstream::new(&input[..2]);
        let HistogramStart::Complex(mut parser) =
            Histogram::parse_start(&mut initial, ALPHABET_SIZE as u32).unwrap()
        else {
            panic!("expected a complex prefix histogram");
        };
        let mut next_bit = initial.num_read_bits();
        assert_eq!(next_bit, 16);
        for end in 3..=input.len() {
            let mut stream = Bitstream::new(&input[..end]);
            stream.skip_bits(next_bit).unwrap();
            match parser.read::<true>(&mut stream) {
                Err(error) if error.unexpected_eof() => assert!(end < input.len()),
                Ok(histogram) => {
                    assert_eq!(end, input.len());
                    let mut symbols = Bitstream::new(&[0b10]);
                    assert_eq!(histogram.read_symbol(&mut symbols).unwrap(), 0);
                    assert_eq!(
                        histogram.read_symbol(&mut symbols).unwrap(),
                        ALPHABET_SIZE as u32 - 1
                    );
                }
                result => panic!("unexpected histogram result: {result:?}"),
            }
            next_bit = stream.num_read_bits();
        }
        assert_eq!(next_bit, input.len() * 8);
        assert!(parser.symbol_attempts <= ALPHABET_SIZE + input.len());
    }
}
