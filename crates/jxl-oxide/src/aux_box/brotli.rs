pub(super) const MAX_DECODED_AUX_BOX_SIZE: usize = 16 * 1024 * 1024;

#[cfg(feature = "std")]
mod std_decoder {
    use alloc::vec::Vec;
    use std::io::{self, Write};

    pub(crate) struct BoundedOutput {
        output: Vec<u8>,
        limit: usize,
    }

    impl BoundedOutput {
        fn new(output: Vec<u8>, limit: usize) -> Self {
            Self { output, limit }
        }
    }

    impl Write for BoundedOutput {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let new_len = self.output.len().checked_add(buf.len()).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Brotli output size overflow")
            })?;
            if new_len > self.limit {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Brotli auxiliary box exceeds decoded size limit",
                ));
            }
            self.output.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    pub(crate) struct BrotliReader {
        writer: brotli_decompressor::DecompressorWriter<BoundedOutput>,
    }

    impl BrotliReader {
        pub(crate) fn with_limit(limit: usize) -> Self {
            Self {
                writer: brotli_decompressor::DecompressorWriter::new(
                    BoundedOutput::new(Vec::new(), limit),
                    4096,
                ),
            }
        }

        pub(crate) fn write_all(&mut self, input: &[u8]) -> crate::Result<()> {
            self.writer.write_all(input)?;
            Ok(())
        }

        pub(crate) fn flush(&mut self) -> crate::Result<()> {
            self.writer.flush()?;
            Ok(())
        }

        pub(crate) fn close(&mut self) -> crate::Result<()> {
            self.writer.close()?;
            Ok(())
        }

        pub(crate) fn into_inner(self) -> core::result::Result<Vec<u8>, Vec<u8>> {
            self.writer
                .into_inner()
                .map(|output| output.output)
                .map_err(|output| output.output)
        }
    }
}

#[cfg(feature = "std")]
pub(super) use std_decoder::BrotliReader;

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
        limit: usize,
        total_out: usize,
        finished: bool,
        failed: bool,
    }

    impl BrotliReader {
        pub(crate) fn with_limit(limit: usize) -> Self {
            Self {
                state: BrotliState::new(Heap(PhantomData), Heap(PhantomData), Heap(PhantomData)),
                output: Vec::new(),
                limit,
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
                let Some(new_len) = self.output.len().checked_add(output_offset) else {
                    self.failed = true;
                    return Err(Error::InvalidData("Brotli output size overflow").into());
                };
                if new_len > self.limit {
                    self.failed = true;
                    return Err(Error::InvalidData("Brotli auxiliary box exceeds decoded size limit").into());
                }
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
                self.failed = true;
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
