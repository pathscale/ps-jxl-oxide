# ps-jxl-oxide

A spec-conforming JPEG XL decoder in pure Rust: the pathscale fork of
[tirr-c/jxl-oxide](https://github.com/tirr-c/jxl-oxide) by Wonwoo Choi and contributors, imported at upstream
commit c0cc4c7. It serves pathscale's PDF toolkit and the chuzz browser, so animation, extra channels and alpha,
ICC output, orientation and cropping stay.

Packages are published as `ps-jxl-*`; their library names stay `jxl_*`, so `use jxl_oxide::JxlImage` is unchanged.

What differs from upstream:

- No thread pool. Parallel work is expressed as owned tasks (`Arc` inputs, owned outputs) that an async
  coordinator spawns through a `TaskExecutor`; `Tasks::inline()` runs them in place, and the `nagoya` feature runs
  them on a host's [nagoya](https://github.com/pathscale/nagoya) executor. Output is bit-identical either way.
- Zero-copy input: the decoder reads `Arc<[u8]>` spans, whole (`read_bytes`) or streamed in chunks (`feed`).
- `no_std + alloc` without the default `std` feature; float functions core lacks come from `libm` there.
- Removed: JPEG bitstream reconstruction, filesystem convenience APIs, the `image` integration, and the lcms2 and
  moxcms adapters (colour management plugs in through the `ColorManagementSystem` trait).

`tools/ps-jxl-check` decodes the [libjxl conformance suite][conformance] and compares every frame against the
reference images with the suite's tolerances. `--hash` prints a hash of every rendered frame so two builds can be
compared bit for bit, `--chunk N` streams the input in N-byte chunks, `--workers N` decodes on a nagoya runtime
(feature `nagoya`) and `--repeat N` reports the fastest of N decodes.

```bash
cargo run --release -p ps-jxl-check --features nagoya -- path/to/conformance --workers 8
```

Dual-licensed under MIT and Apache 2.0, as upstream.

[conformance]: https://github.com/libjxl/conformance
