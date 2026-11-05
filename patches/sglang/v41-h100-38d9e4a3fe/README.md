# DeepSeek V4.1 Flash on H100

This optional runtime patch targets SGLang revision
`38d9e4a3fe4a493b0ae468477a274e83defda81a`. It is independent of the
v0.5.18 spec-capture patch and does not change SpecForge's default dependency.
`manifest.json` pins the source, validated container image, patch hash and
all 42 resulting file hashes. The patch contains its tests and third-party
licenses/source attribution.

Apply to a clean checkout of the pinned SGLang revision:

```bash
# Use an absolute path for PATCH_DIR.
git -C "$SGLANG_SOURCE" apply --check "$PATCH_DIR/h100-runtime.patch"
git -C "$SGLANG_SOURCE" apply "$PATCH_DIR/h100-runtime.patch"
```

Install the patched source into the pinned server environment. The launch
scripts preserve the tested GPFS model path and single-node TP8/EP8 topology;
adjust deployment paths explicitly for another environment. The patch requires
the pinned image's FlashInfer workspace API; a package version string alone
is insufficient to establish compatibility.

The patch adds opt-in SM90 indexer scoring, causal mask/index writeback,
DeepSelect prefill, mHC post/combine/norm and reusable eager MoE scratch.
Scratch is one compatible, same-stream allocation per runtime/device, bounded
by configured prefill capacity and 16 GiB. It does not split MoE token rows.
Graphs and overlapping batches keep their existing allocations. All new
optimization flags default off.

On 8 H100 80GB GPUs, cold TTFT for the same 64,326-token prompt was:

| Script | Cold TTFT median | Scope |
| --- | ---: | --- |
| `launch-best-verified.sh` | 1.544s; 1.504s on repeat | Full computation, CP8, 16K prefill chunks |
| `launch-compact-inference.sh` | 1.355s | Compact128 + prefill Graph; faster decode |
| `launch-compact-w4a8-inference.sh` | 1.035s | Compact128 + CP8 + W4A8 + scratch |

These are inference timings. They are not all-prompt hidden capture timings.
W4A8 changes activation precision and remains an optional inference setting.
No task-quality score was run or inferred from these timings.

## Full-prompt training capture

Install `plugins/sglang-spec-capture` into the server environment. Its writer
is attention TP0 and CP0, so gathered sequences are published once. Request
auxiliary layers 37/38/39 and final hidden with `--return-hidden-states-mode full`
and `--chunked-prefill-size -1`. Full mode disables Compact before model/KV
construction even when Compact was explicitly requested. Do not use the
16K chunked inference scripts for training capture.

Long-input hidden equivalence with the experimental inference flags is not
accepted yet: two 16,385-token comparisons failed, and the controls themselves
also differed across processes. The rejected internal MoE chunking implementation
is absent. Keep all optional inference optimization flags disabled for training
until the documented capture acceptance is satisfied. Real-input MoE workspace comparisons were exact in 1,920 cases, but
long-input whole-model repeatability remained unresolved with DeepSelect off.
The acceptance report records the final diagnosis and remaining limitations separately from inference.

See `acceptance/v41-h100-20261105` in this repository for measurements,
validation and cleanup receipts.
