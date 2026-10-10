# ps-jxl-check

A synchronous conformance checker using only the jxl-oxide facade and the Rust standard library.

```sh
cargo run --release -p ps-jxl-check -- /path/to/conformance
cargo run --release -p ps-jxl-check -- /path/to/conformance --only alpha_premultiplied
```

The directory can be either the repository root containing `testcases`, or the `testcases` directory itself. Each case must contain `input.jxl`, `test.json`, and materialized reference NPY and ICC files. The checker does not download objects.

Cases run in name order. Every case produces one stdout line, followed by a suite summary. Diagnostics go to stderr. Any failed case, missing requested case, or suite setup error produces exit code 1.

`frames` reports decoded/expected displayed frames. `max_err` and `rmse` contain per-channel maxima across frames. Each frame is checked independently against every channel's limits. `limit` lists `peak/RMSE` pairs in channel order. `ms` measures one complete decode, including header parsing, ICC request setup, rendering all displayed frames, and extraction of oriented interleaved float pixels. File reads, reference parsing, and numerical comparison are excluded.

## Reference arrays

NPY versions 1.0, 2.0, and 3.0 are supported. Arrays must use C order and little-endian float32, float16, uint8, or uint16 samples. Supported shapes are `(height, width)`, `(height, width, channels)`, and `(frames, height, width, channels)`. Integer samples are normalized by 255 or 65535; float samples retain their values.

The reader validates the header, dimensions, arithmetic, and exact payload length. It reads one reference frame at a time. Nonfinite samples fail comparison.

`reference`, `references`, or `reference_image` can name an NPY file or an ordered array of NPY filenames. A reference object can instead provide `npy`, `file`, or `path`. Without explicit NPY filenames, the checker discovers local `.npy` files, excludes filenames beginning with `preview`, and sorts numbered filenames numerically. Hash references therefore require materialized local NPY files. All referenced paths are relative to the case directory.

## Tolerances and ICC

Tolerances must be explicit in `test.json`. Accepted containers are `thresholds`, `tolerances`, or `error_thresholds`; tolerance fields can also appear at the top level.

An object provides `peak_error` or `max_error`, and `rmse` or `max_rmse`. Each field can be a scalar applying to every channel, or an array with exactly one value per channel. A container can alternatively be an array of per-channel objects or `[peak, rmse]` pairs. Limits must be finite and nonnegative. Missing limits fail the case.

For example:

```json
{
  "reference": "reference.npy",
  "reference_icc": "reference.icc",
  "thresholds": [
    {"peak_error": 0.004, "rmse": 0.0001},
    {"peak_error": 0.004, "rmse": 0.0001},
    {"peak_error": 0.004, "rmse": 0.0001},
    {"peak_error": 0.000976562, "rmse": 0.000976562}
  ]
}
```

`reference_icc` or `icc` specifies the target profile. A reference object can also provide `icc`. Explicit `null` or `"skip"` disables the ICC request. Otherwise the checker requires the named profile, `reference.icc`, or a single discoverable local `.icc` file. A hash names `<hash>.icc` when that file exists, with the same local discovery fallback.

Like the upstream comparison, the checker disables spot-color rendering, requests the reference ICC, compares all extra channels, and requires complete loading and the exact displayed frame count. It uses f64 accumulation for squared error.

Only the retained built-in color conversion is available. A profile requiring an external CMS fails the case; the checker does not substitute another color space or relax tolerances.
