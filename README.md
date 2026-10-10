# jxl-oxide
[![crates.io](https://img.shields.io/crates/v/jxl-oxide.svg)](https://crates.io/crates/jxl-oxide)
[![docs.rs](https://docs.rs/jxl-oxide/badge.svg)](https://docs.rs/crate/jxl-oxide/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/tirr-c/jxl-oxide/build.yml?branch=main)](https://github.com/tirr-c/jxl-oxide/actions/workflows/build.yml?query=branch%3Amain)

A spec-conforming JPEG XL decoder written in pure Rust.

This is the pathscale fork of [tirr-c/jxl-oxide](https://github.com/tirr-c/jxl-oxide), originally developed by Wonwoo Choi and upstream contributors. This stage removes Rayon pools, JPEG bitstream reconstruction, filesystem convenience APIs, `image` integration, and the bundled lcms2 and moxcms adapters. Decode work runs inline through the existing dispatch wrapper; JPEG-transcoded VarDCT pixel decoding, animation and frames, extra channels and alpha, ICC output, orientation, cropping, built-in colour conversion, and the caller-supplied `ColorManagementSystem` boundary remain available. Input still uses `read(impl std::io::Read)` or byte feeding. Later stages will introduce immutable `Arc` inputs, owned task outputs, a nagoya-native asynchronous coordinator, and `no_std + alloc`; crate names remain unchanged in this stage.

---

Dual-licensed under MIT and Apache 2.0.

[conformance]: https://github.com/libjxl/conformance

