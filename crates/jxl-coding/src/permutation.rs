use alloc::vec::Vec;

use jxl_bitstream::Bitstream;

/// Read a permutation from the entropy encoded stream.
pub fn read_permutation(
    bitstream: &mut Bitstream,
    decoder: &mut crate::Decoder,
    size: u32,
    skip: u32,
) -> crate::CodingResult<Vec<usize>> {
    PermutationReader::new(size, skip).read_inner::<false>(bitstream, decoder)
}

/// Reads a permutation, retaining decoded integers when the input runs short.
#[derive(Debug)]
pub struct PermutationReader {
    size: u32,
    skip: u32,
    end: Option<u32>,
    lehmer: Vec<u32>,
}

impl PermutationReader {
    pub fn new(size: u32, skip: u32) -> Self {
        Self {
            size,
            skip,
            end: None,
            lehmer: Vec::new(),
        }
    }

    /// On EOF, resume with the same decoder and a bitstream at the last consumed bit.
    pub fn read(
        &mut self,
        bitstream: &mut Bitstream,
        decoder: &mut crate::Decoder,
    ) -> crate::CodingResult<Vec<usize>> {
        self.read_inner::<true>(bitstream, decoder)
    }

    fn read_inner<const RESUME: bool>(
        &mut self,
        bitstream: &mut Bitstream,
        decoder: &mut crate::Decoder,
    ) -> crate::CodingResult<Vec<usize>> {
        let read = |decoder: &mut crate::Decoder, bitstream: &mut Bitstream, ctx| {
            if RESUME {
                decoder.read_checkpointed_integer(bitstream, ctx)
            } else {
                decoder.read_varint(bitstream, ctx)
            }
        };
        let size = self.size;
        let skip = self.skip;
        let end = match self.end {
            Some(end) => end,
            None => {
                let end = read(decoder, bitstream, get_context(size))?;
                if end > size - skip {
                    tracing::error!(size, skip, end, "Invalid permutation");
                    return Err(crate::Error::InvalidPermutation);
                }
                self.end = Some(end);
                end
            }
        };

        while self.lehmer.len() < end as usize {
            let idx = self.lehmer.len() as u32;
            let prev = self.lehmer.last().copied().unwrap_or(0);
            let val = read(decoder, bitstream, get_context(prev))?;
            if val >= size - skip - idx {
                tracing::error!(idx = idx + skip, size, lehmer = val, "Invalid permutation");
                return Err(crate::Error::InvalidPermutation);
            }
            self.lehmer.push(val);
        }

        let mut temp = ((skip as usize)..(size as usize)).collect::<Vec<_>>();
        let mut permutation = Vec::with_capacity(size as usize);
        permutation.extend(0..skip as usize);
        for &idx in &self.lehmer {
            permutation.push(temp.remove(idx as usize));
        }
        permutation.extend(temp);
        Ok(permutation)
    }
}

fn get_context(x: u32) -> u32 {
    crate::add_log2_ceil(x).min(7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_header_retry_restores_incomplete_permutation_integer() {
        // One prefix cluster emits token 2 with no symbol bits. Its hybrid
        // integer reads one additional bit to select the value 2 or 3.
        let mut setup = Bitstream::new(&[0x12, 0x86, 0x10]);
        let mut decoder = crate::Decoder::parse(&mut setup, 8).unwrap();
        let mut reader = PermutationReader::new(8, 0);
        let mut short = Bitstream::new(&[0]);
        short.skip_bits(6).unwrap();

        let error = reader.read(&mut short, &mut decoder).unwrap_err();
        assert!(error.unexpected_eof());
        assert_eq!(short.num_read_bits(), 8);

        let mut complete = Bitstream::new(&[0, 1]);
        complete.skip_bits(short.num_read_bits()).unwrap();
        let order = reader.read(&mut complete, &mut decoder).unwrap();
        assert_eq!(order, [2, 4, 0, 1, 3, 5, 6, 7]);
        assert_eq!(complete.num_read_bits(), 9);
    }

    #[test]
    fn split_ans_integer_restores_seed_and_renormalization() {
        // Equal ANS probabilities for tokens 4 and 5. Seed 2^16
        // forces 16 renormalization bits before either hybrid suffix.
        let mut setup = Bitstream::new(&[0xa0, 0x5c, 0x54, 0, 0x08]);
        let mut decoder = crate::Decoder::parse(&mut setup, 1).unwrap();
        assert_eq!(setup.num_read_bits(), 36);
        let mut whole_decoder = decoder.clone();
        let mut short = Bitstream::new(&[0, 0, 1, 0, 0, 0]);
        let error = decoder.read_checkpointed_integer(&mut short, 0).unwrap_err();
        assert!(error.unexpected_eof(), "{error:?}");
        assert_eq!(short.num_read_bits(), 0);

        let bytes = [0, 0, 1, 0, 0, 0, 1];
        let mut whole = Bitstream::new(&bytes);
        let mut resumed = Bitstream::new(&bytes);
        for expected_position in [49, 50] {
            let expected = whole_decoder.read_varint(&mut whole, 0).unwrap();
            assert_eq!(
                decoder.read_checkpointed_integer(&mut resumed, 0).unwrap(),
                expected
            );
            assert_eq!(resumed.num_read_bits(), expected_position);
            assert_eq!(whole.num_read_bits(), expected_position);
        }
    }
}
