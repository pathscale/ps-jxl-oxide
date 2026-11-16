use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};

use jxl_bitstream::Bitstream;

use crate::{
    Coder, CodingResult, Decoder, DecoderInner, Error, IntegerConfig, Lz77, ans, check_clusters,
    inverse_mtf, prefix, read_prefix_alphabet_size,
};

/// Reads an entropy decoder and its initial state over append-only input.
#[derive(Debug)]
pub struct DecoderSetup {
    num_dist: u32,
    allow_lz77: bool,
    lz77: Option<Lz77>,
    cluster_parser: ClusterParser,
    clusters: Option<(u32, Vec<u8>)>,
    configs: Vec<IntegerConfig>,
    distributions: Option<Distributions>,
    decoder: Option<Decoder>,
}

#[derive(Debug)]
enum Distributions {
    Prefix {
        counts: Vec<u32>,
        histograms: Vec<prefix::Histogram>,
        pending: Option<prefix::ComplexHistogram>,
    },
    Ans {
        log_alphabet_size: u32,
        histograms: Vec<ans::Histogram>,
    },
}

impl DecoderSetup {
    /// Creates a parser for a decoder with the given number of distributions.
    pub fn new(num_dist: u32) -> Self {
        Self {
            num_dist,
            allow_lz77: true,
            lz77: None,
            cluster_parser: ClusterParser {
                stage: ClusterStage::Start,
                values: Vec::new(),
            },
            clusters: None,
            configs: Vec::new(),
            distributions: None,
            decoder: None,
        }
    }

    /// Reads a ready decoder. On EOF, append input and resume at the retained bit position.
    /// After success or any other error, discard this parser.
    pub fn read(&mut self, bitstream: &mut Bitstream) -> CodingResult<Decoder> {
        if self.decoder.is_none() {
            self.decoder = Some(self.read_decoder(bitstream)?);
        }
        self.decoder.as_mut().unwrap().begin(bitstream)?;
        Ok(self.decoder.take().unwrap())
    }

    fn read_decoder(&mut self, bitstream: &mut Bitstream) -> CodingResult<Decoder> {
        if self.lz77.is_none() {
            let lz77 = checkpoint(bitstream, |input| {
                if self.allow_lz77 {
                    Lz77::parse(input)
                } else if input.read_bool()? {
                    Err(Error::Lz77NotAllowed)
                } else {
                    Ok(Lz77::Disabled)
                }
            })?;
            if matches!(lz77, Lz77::Enabled { .. }) {
                self.num_dist += 1;
            }
            self.lz77 = Some(lz77);
        }

        if self.clusters.is_none() {
            self.clusters = Some(self.cluster_parser.read(bitstream, self.num_dist)?);
        }
        let num_clusters = self.clusters.as_ref().unwrap().0 as usize;

        if self.distributions.is_none() {
            self.distributions = Some(checkpoint(bitstream, |input| {
                Ok(if input.read_bool()? {
                    Distributions::Prefix {
                        counts: Vec::new(),
                        histograms: Vec::new(),
                        pending: None,
                    }
                } else {
                    Distributions::Ans {
                        log_alphabet_size: input.read_bits(2)? + 5,
                        histograms: Vec::new(),
                    }
                })
            })?);
        }
        let distributions = self.distributions.as_mut().unwrap();
        let log_alphabet_size = match distributions {
            Distributions::Prefix { .. } => 15,
            Distributions::Ans { log_alphabet_size, .. } => *log_alphabet_size,
        };
        while self.configs.len() < num_clusters {
            self.configs.push(checkpoint(bitstream, |input| {
                IntegerConfig::parse(input, log_alphabet_size)
            })?);
        }

        let code = match distributions {
            Distributions::Prefix {
                counts,
                histograms,
                pending,
            } => {
                while counts.len() < num_clusters {
                    counts.push(checkpoint(bitstream, read_prefix_alphabet_size)?);
                }
                while histograms.len() < num_clusters {
                    let count = counts[histograms.len()];
                    histograms.push(prefix::ComplexHistogram::resume(pending, bitstream, count)?);
                }
                Coder::PrefixCode(Arc::new(core::mem::take(histograms)))
            }
            Distributions::Ans {
                log_alphabet_size,
                histograms,
            } => {
                while histograms.len() < num_clusters {
                    // An individual ANS histogram has at most 256 entries.
                    histograms.push(checkpoint(bitstream, |input| {
                        ans::Histogram::parse(input, *log_alphabet_size)
                    })?);
                }
                Coder::Ans {
                    dist: Arc::new(core::mem::take(histograms)),
                    state: 0,
                    initial: true,
                }
            }
        };

        let (_, clusters) = self.clusters.take().unwrap();
        Ok(Decoder {
            lz77: self.lz77.take().unwrap(),
            inner: DecoderInner {
                clusters,
                configs: core::mem::take(&mut self.configs),
                code,
            },
        })
    }
}

#[derive(Debug)]
struct ClusterParser {
    stage: ClusterStage,
    values: Vec<u8>,
}

#[derive(Debug)]
enum ClusterStage {
    Start,
    Simple(usize),
    Setup(Box<DecoderSetup>, bool),
    Encoded(Decoder, bool),
}

impl ClusterParser {
    fn read(&mut self, bitstream: &mut Bitstream, num_dist: u32) -> CodingResult<(u32, Vec<u8>)> {
        if num_dist == 1 {
            return Ok((1, vec![0]));
        }

        if matches!(self.stage, ClusterStage::Start) {
            self.stage = checkpoint(bitstream, |input| {
                Ok(if input.read_bool()? {
                    ClusterStage::Simple(input.read_bits(2)? as usize)
                } else {
                    let use_mtf = input.read_bool()?;
                    let mut setup = DecoderSetup::new(1);
                    setup.allow_lz77 = num_dist > 2;
                    ClusterStage::Setup(Box::new(setup), use_mtf)
                })
            })?;
        }
        if let ClusterStage::Setup(setup, use_mtf) = &mut self.stage {
            self.stage = ClusterStage::Encoded(setup.read(bitstream)?, *use_mtf);
        }

        match &mut self.stage {
            ClusterStage::Simple(nbits) => {
                while self.values.len() < num_dist as usize {
                    self.values.push(bitstream.read_bits(*nbits)? as u8);
                }
            }
            ClusterStage::Encoded(decoder, use_mtf) => {
                while self.values.len() < num_dist as usize {
                    let value = decoder.read_checkpointed_integer(bitstream, 0)?;
                    let cluster = u8::try_from(value).map_err(|_| Error::InvalidCluster(value))?;
                    self.values.push(cluster);
                }
                decoder.finalize()?;
                if *use_mtf {
                    inverse_mtf(&mut self.values);
                }
            }
            _ => unreachable!(),
        }

        check_clusters(core::mem::take(&mut self.values))
    }
}

fn checkpoint<'a, T>(
    bitstream: &mut Bitstream<'a>,
    read: impl FnOnce(&mut Bitstream<'a>) -> CodingResult<T>,
) -> CodingResult<T> {
    let mut input = bitstream.clone();
    let value = read(&mut input)?;
    *bitstream = input;
    Ok(value)
}
