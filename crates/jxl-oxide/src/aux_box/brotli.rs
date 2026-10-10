#[cfg(feature = "std")]
pub(super) type BrotliReader = brotli_decompressor::DecompressorWriter<alloc::vec::Vec<u8>>;

#[cfg(not(feature = "std"))]
mod decoder {
    use alloc::{vec, vec::Vec};
    use core::marker::PhantomData;

    use brotli_decompressor::{
        Allocator, BrotliDecompressStream, BrotliResult, BrotliState, HuffmanCode, SliceWrapper,
        SliceWrapperMut,
    };

    use crate::{Result, error::Error};

    #[derive(Default)]
    pub struct Memory<T>(Vec<T>);

    impl<T> SliceWrapper<T> for Memory<T> {
        fn slice(&self) -> &[T] {
            &self.0
        }
    }

    impl<T> SliceWrapperMut<T> for Memory<T> {
        fn slice_mut(&mut self) -> &mut [T] {
            &mut self.0
        }
    }

    struct Heap<T>(PhantomData<T>);

    impl<T: Clone + Default> Allocator<T> for Heap<T> {
        type AllocatedMemory = Memory<T>;

        fn alloc_cell(&mut self, len: usize) -> Self::AllocatedMemory {
            Memory(vec![T::default(); len])
        }

        fn free_cell(&mut self, _memory: Self::AllocatedMemory) {}
    }

    pub(crate) struct BrotliReader {
        state: BrotliState<Heap<u8>, Heap<u32>, Heap<HuffmanCode>>,
        output: Vec<u8>,
        total_out: usize,
        finished: bool,
        failed: bool,
    }

    impl BrotliReader {
        pub(crate) fn new(output: Vec<u8>, _buffer_size: usize) -> Self {
            Self {
                state: BrotliState::new(Heap(PhantomData), Heap(PhantomData), Heap(PhantomData)),
                output,
                total_out: 0,
                finished: false,
                failed: false,
            }
        }

        pub(crate) fn write_all(&mut self, input: &[u8]) -> Result<()> {
            if self.failed {
                return Err(Error::InvalidData("Brotli decompressor reported an error").into());
            }
            if self.finished {
                return Ok(());
            }
            let mut available_in = input.len();
            let mut input_offset = 0;
            loop {
                let mut buffer = [0u8; 4096];
                let mut available_out = buffer.len();
                let mut output_offset = 0;
                let result = BrotliDecompressStream(
                    &mut available_in,
                    &mut input_offset,
                    input,
                    &mut available_out,
                    &mut output_offset,
                    &mut buffer,
                    &mut self.total_out,
                    &mut self.state,
                );
                self.output.extend_from_slice(&buffer[..output_offset]);
                match result {
                    BrotliResult::ResultSuccess => {
                        self.finished = true;
                        return Ok(());
                    }
                    BrotliResult::NeedsMoreInput => return Ok(()),
                    BrotliResult::NeedsMoreOutput => {}
                    BrotliResult::ResultFailure => {
                        self.failed = true;
                        return Err(
                            Error::InvalidData("Brotli decompressor reported an error").into()
                        );
                    }
                }
            }
        }

        pub(crate) fn flush(&mut self) -> Result<()> {
            self.write_all(&[])
        }

        pub(crate) fn close(&mut self) -> Result<()> {
            self.flush()?;
            if !self.finished {
                return Err(Error::UnexpectedEof("incomplete Brotli auxiliary box").into());
            }
            Ok(())
        }

        pub(crate) fn into_inner(self) -> core::result::Result<Vec<u8>, Vec<u8>> {
            if self.finished && !self.failed {
                Ok(self.output)
            } else {
                Err(self.output)
            }
        }
    }
}

#[cfg(not(feature = "std"))]
pub(super) use decoder::BrotliReader;
