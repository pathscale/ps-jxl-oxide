# ps-jxl-oxide

A spec-conforming JPEG XL decoder in pure Rust: the pathscale fork of
[tirr-c/jxl-oxide](https://github.com/tirr-c/jxl-oxide) by Wonwoo Choi and contributors, imported at upstream
commit c0cc4c7. It serves pathscale's PDF toolkit and the chuzz browser, so animation, extra channels and alpha,
ICC output, orientation and cropping stay.

Removed so far: the Rayon pool (decode work runs inline through the existing dispatch wrapper), JPEG bitstream
reconstruction, filesystem convenience APIs, the `image` integration, and the lcms2 and moxcms adapters (colour
management plugs in through the `ColorManagementSystem` trait).

Plan: zero-copy input over `Arc<[u8]>` ranges; owned task outputs with an asynchronous coordinator run by
[nagoya](https://github.com/pathscale/nagoya) instead of a thread pool; then `no_std + alloc`.

`tools/ps-jxl-check` decodes the [libjxl conformance suite][conformance] and compares every frame against the
reference images with the suite's tolerances; it also reports milliseconds per decode. Every stage keeps it green.

```bash
cargo run --release -p ps-jxl-check -- path/to/conformance
```

Dual-licensed under MIT and Apache 2.0, as upstream.

[conformance]: https://github.com/libjxl/conformance
