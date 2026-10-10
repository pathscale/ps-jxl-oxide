# Task conversion ledger

Locations below refer to commit 5a67f5f9da34c5692bae1c23693c46e5b4313dde,
before these edits. They are stable lane identifiers rather than current line numbers.
The study's 21 dispatch statements include four jxl-jbr sites already deleted in stage 1.

| Status | Site | Planned owned job |
|---|---|---|
| Converted | crates/jxl-color/src/convert.rs:637 | Arc prepared transform, owned multi-channel 65536-sample chunk; return channels and channel count. |
| Pending | crates/jxl-modular/src/transform.rs:474 | Horizontal squeeze: immutable transform metadata and owned full-width sixteen-row stripe, including averages and residuals. |
| Pending | crates/jxl-modular/src/transform.rs:486 | Vertical squeeze: immutable transform metadata and owned full-height column band; preserve recurrence and account for repacking. |
| Pending | crates/jxl-modular/src/transform/rct.rs:126 | Immutable kernel/permutation and owned matching three-channel stripe. |
| Pending | crates/jxl-render/src/features/noise.rs:165 | Arc source noise groups and owned convolved channel/group tile with private padded row scratch. |
| Pending | crates/jxl-render/src/filter/epf.rs:178 | Arc iteration input, immutable LF sigma metadata and owned three-channel stripe with private sigma scratch. |
| Pending | crates/jxl-render/src/filter/gabor.rs:90 | Arc input image and immutable weights; return owned output stripe including edge handling. |
| Removed | crates/jxl-render/src/lib.rs:499 | LF reference recursion is awaited by the exclusive coordinator; publish one Arc completed image. |
| Removed | crates/jxl-render/src/lib.rs:506 | Reference-slot recursion is awaited in frame-index order; duplicate slots share one Arc image. |
| Removed | crates/jxl-render/src/lib.rs:598 | Loading narrow path uses the same awaited reference coordinator, without detached work. |
| Removed | crates/jxl-render/src/lib.rs:603 | Loading wide path uses the same awaited reference coordinator, without detached work. |
| Pending | crates/jxl-render/src/modular.rs:38 | LF nodes return LF metadata and Modular contributions; retain serial single-entry parsing order. |
| Pending | crates/jxl-render/src/modular.rs:94 | Immutable MA/channel descriptions; return owned recursive Modular output per pass/group. |
| Pending | crates/jxl-render/src/util.rs:230 | Return owned LF group, Modular contribution and optional dequantized LF tile. |
| Pending | crates/jxl-render/src/vardct/mod.rs:156 | Arc LF/frame descriptions; return immutable HF-global snapshot after section indexing permits it. |
| Pending | crates/jxl-render/src/vardct/mod.rs:284 | Arc LF/HF/MA metadata; pass the owned coefficient group through ordered passes and return its Modular contribution. |
| Pending | crates/jxl-render/src/vardct/mod.rs:320 | Reuse the owned coefficient group allocation for reconstruction; Arc LF/HF inputs; return completed pixels. |
| Deleted | crates/jxl-jbr/src/reconstruct.rs:140 | JPEG reconstruction removed in stage 1; no retained job. |
| Deleted | crates/jxl-jbr/src/reconstruct.rs:166 | JPEG reconstruction removed in stage 1; no retained job. |
| Deleted | crates/jxl-jbr/src/reconstruct.rs:253 | JPEG reconstruction removed in stage 1; no retained job. |
| Deleted | crates/jxl-jbr/src/reconstruct.rs:366 | JPEG reconstruction removed in stage 1; no retained job. |

## Colour chunk pattern

The async path reaches ColorTransform::run_with_tasks from both requested-output
conversion and saved-frame conversion. It copies each range of the existing contiguous
planes into owned Vec<f32> channels, captures one immutable Arc<ColorTransform>, and
runs the original operation sequence synchronously inside each job. No borrowed plane,
mutable result slot or lock crosses the executor boundary.

Tasks::run_all submits every job before awaiting any, retains all job futures, and
collects outputs in chunk-index order. The colour coordinator selects the first chunk
error in that order and merges successful channels only after every chunk succeeds.
A cancelled or mistyped output maps to Error::TaskCancelled. An empty input preserves
the original begin-channel count. Chunk size and per-chunk floating-point order are
unchanged. The contiguous caller interface requires the chunk copy and placement copy;
later owned tiled layouts can move storage instead.

Nagoya job futures cancel their join handles when dropped, including on an abandoned
coordinator or a synchronous wrapper's Pending result. Cancellation cannot preempt a
synchronous job already executing; its owned input and output remain safe.

Render state and completed reference snapshots belong to RenderContext, exclusively
borrowed during async rendering. Reference recursion uses Box::pin and never blocks.
Sync facade methods poll the same async path once; Pending returns an error. Use
Tasks::inline() for synchronous callers and async methods for a host executor.

The remaining borrowed fan-outs still use JxlThreadPool's inline wrappers.
render_modular and render_vardct expose async boundaries receiving Tasks, while their
existing synchronous kernels keep nested parsing and borrowed transforms inline until
their lanes supply owned layouts. Their legacy result locks and DCT preparation are
outside the converted task path and remain work for the listed later lanes.

