# Restart full DeNovoSWE using the previously tuned capture path

User authorized restarting the pipeline on 2026-11-05. Old runs and their
sources remain immutable. New UUIDs reference old durable receipts and restore
the latest completed NFC checkpoint, including optimizer, scheduler and RNG.
Updates after that checkpoint are recomputed. Completed feature files remain
in place and are referenced through symlinks, never recaptured for consumption.

Capture: the exact historical fused_indexer.py (DeepInfer/compact flags off),
bf16_fields_zstd_v2 transport, eight codec workers, depth two producer/consumer.
Keep bounded MoE 8192 and int64 MHC/Engram fixes for long contexts. Complete
trajectories are still single unchunked prefills with the same official tokens,
loss masks, aux37/38/39 and final BF16. This is not a claim of 10k tok/s at all
lengths: historical 10k used 16k/64k full-record output on H100. Current runs
add verified 4K slicing and immutable feature SHA checks for the training stream.

Remove the redundant full-record disk write/read: fetch once, run native capture
validation, then write selected slices directly. Each slice is decoded and
compared bitwise before publishing the same durable trajectory receipt. Zstd
wire/frame checksums are disabled as in the tuned path; per-file SHA receipts
remain for compatibility with existing training. No existing slice is replaced.

Each replica first runs the original eight-case GPU indexer regression, then
recaptures its largest already completed trajectory into an isolated parity
directory and compares every saved window's full tensors against old outputs.
Only after this succeeds does it capture the missing trajectories. Depth-two
capacity is checked against worst-case encoded sizes of remaining inputs; the
longest parity request uses depth one and is never truncated.

Training resumes the original fixed slot stream. Reconstruct published order
from immutable blocks and reject changed or duplicate entries. The original
contract and stream path stay in the native checkpoint resume contract. Only
unpublished/new windows can enter future shuffled blocks. Persisted checkpoint
cursor establishes the prefix; audit hashes for that prefix are reconstructed
from the immutable schedule and are explicitly identified as inherited, while
all new updates are observed directly. Full 796 steps and 5896 validation
windows remain required; checkpoint restore does not establish final quality.

CPU regressions cover delayed/failing producer, metadata mismatch, restart with
unchanged existing blocks and complete remaining coverage, and native BF16
slice roundtrips/overlap masks. The same checks run in the target image.
