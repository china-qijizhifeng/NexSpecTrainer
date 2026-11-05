# Northgate V4.1 hidden-state capture

This run starts from NexSpecTrainer main 0ff81d9 and the existing pinned
YiCloud SGLang image 03a08910. It preserves complete trajectories and captures
layer inputs 37/38/39 plus post-final-norm states, with no target prefill
chunking, prefix cache, bounded replay, draft inference, or sequence truncation.
Local MoE calls are bounded to 8192 tokens; int64 MHC/Engram addressing and the
previously validated BF16 fused indexer are scoped to NORTHGATE_LONG_CAPTURE.
These are the same long-input helpers used by the prior 234048-token YiCloud
capture; extrapolation to this dataset's largest trajectory requires live
acceptance. No CP8 runtime optimization is enabled.

prepare.py uses the checkpoint's official V4.1 encoder/tokenizer. Developer
messages become System messages at the same position because V4.1 has no
Developer token. Tools are attached to the first System message (inserted if
necessary); historical reasoning and midstream system messages are retained.
Message and tool-call use_loss flags are explicit. Only repeated reasoning
character spans and confirmed terminal summaries are masked. A token crossing
an excluded character boundary is masked. Per-record IDs/masks are fingerprinted;
input shards are checked against the curation manifest before tokenization.

The requested topology is four independent TP8 H200 replicas (32 GPUs), with
disjoint greedy token-balanced input partitions. Each partition begins with its
shortest complete real trace. No model forward is distributed across replicas.
The native capture writer saves only .ckpt.hybrid files with Brotli 1.2 quality
10 for sign/exponent streams and Zstd for mantissas. collect.py reads every
new file through the native decoder, verifies checksums, exact input IDs/masks,
full shape coverage and finite BF16 values, then writes a durable receipt.
GPU environments install only pinned offline codec wheels and a run-private
copy of the dependency-free plugin. They do not upgrade the serving stack.

SPECFORGE_MAX_CAPTURE_RECORD_BYTES explicitly permits up to 32 GiB per decoded
record; default remains 8 GiB. The same limit is used by disk codecs, wire codec,
transport aggregate checks, and request capacity checks. Set this environment
variable before imports when consuming large files later. It does not enlarge
the Mooncake/model buffers; this launcher configures those separately.

The launcher exits on any capture/read-back failure and stops only its own
server/master processes. RESULT.json is written only after every assigned ID
has a verified hybrid receipt. The platform job and all replicas must also
finish successfully before reporting full dataset completion.
