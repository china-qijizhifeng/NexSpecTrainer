# Capture an external SGLang server into compressed training files

Use `scripts/prepare_server_hidden_states.py` when a separate SGLang service
already hosts the target. The client loads model configuration, sends
pre-tokenized complete episodes, receives the original BF16 tensors through
Mooncake, and compresses them on the CPU before publishing offline files:

```text
SGLang BF16 → Mooncake BF16 → client CPU → .ckpt.zst → offline reader → original BF16
```

This reuses the lossless BF16 fields / Zstd format from
[PR #2](https://github.com/china-qijizhifeng/NexSpecTrainer/pull/2), pinned codec
source `c468c1997cc8ff991b6f4a1867378bed1fff79a5`. Compression reduces disk
storage and disk I/O. Network traffic, SGLang GPU memory, and decoded training
tensor sizes remain unchanged. Direct online Mooncake training also remains
unchanged.

Install the optional codec dependency in the capture client and trainer:

```bash
pip install '.[compression]'
```

Start the SGLang server with the capture plugin or supported patch. Configure
full, unchunked prefill, disabled prefix caching, and the algorithm's target
capture layer IDs. For the Qwen3.5-35B-A3B DSpark configuration used in the
long-trace experiment, these are `1 10 19 28 37`, with auxiliary width 10240
and final width 2048. The CLI resolves these values from the model/draft
configuration; it does not accept a manually duplicated layer list.

Set the server's source buffers before launching SGLang. For example:

```bash
export MOONCAKE_GLOBAL_SEGMENT_SIZE=4294967296
export MOONCAKE_LOCAL_BUFFER_SIZE=4294967296
```

Each 83,542-token Qwen episode contains about 1.91 GiB of uncompressed tensors
(`83542 * (10240 * 2 + 2048 * 2 + 16)` bytes). A 1 GiB source buffer is
insufficient even if the resulting disk file is smaller. The CLI conservatively
compares the complete raw record with the declared source/client capacities
before sending each request. `--server-buffer-bytes` declares the capacity
configured on the server; the client cannot resize or verify a remote server's
buffers. It defaults to 4 GiB, as does `--client-buffer-bytes`.

Input JSONL must contain stable, filesystem-safe IDs and complete, already
chat-templated token sequences. No episode concatenation or truncation occurs:

```json
{"id":"episode-001","input_ids":[1,2,3,4],"loss_mask":[0,1,1,1],"metadata":{"model_revision":"pinned-checkpoint-identity","source":"dataset/revision/row"}}
```

The supplied offline training YAML determines the target and draft config,
strategy, and maximum sequence length. Use a maximum at least as large as the
longest episode; overlength inputs fail. `--target-model-version` is the
operator's pinned identity for the model actually served. The client checks
input `metadata.model_revision` when present; it does not independently hash
remote model shards.

```bash
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_PROTOCOL=tcp

python scripts/prepare_server_hidden_states.py \
  --config /shared/train-long.yaml \
  --server http://127.0.0.1:30000 \
  --samples /shared/complete-episodes.jsonl \
  --output /shared/captured-long \
  --target-model-version pinned-checkpoint-identity \
  --zstd
```

Raw `.ckpt` remains the default. `--zstd-level` defaults to 1;
`--interleave-layers` optionally reorders only the auxiliary feature according
to the resolved capture plan, then reverses that reordering during decoding.
For DSpark, the files contain `input_ids` and `loss_mask` as `[T]`, auxiliary
`hidden_states` as `[1,T,aux_width]`, and
`target_last_hidden_states` as `[1,T,target_width]`. Existing offline training
reads the directory through `data.hidden_states_path`, including mixtures of
raw, gzip, and Zstd feature files.

Only one request and one CPU record are active at a time. The writer waits for
compression and atomic publication before releasing the Mooncake lease and
starting the next episode; it has no unbounded asynchronous tensor queue.
The raw tensors, codec work buffers, and optional interleave copy still need
CPU RAM. Training fully decodes each compressed record in CPU RAM before
normalization; Zstd files cannot be memory mapped. Allow extra RAM per loader
worker, and use a small worker/prefetch count for long episodes.

Use `--resume` with the same output directory and capture plan. Completed
samples are skipped only after comparing input identity and decoding/checking
their stored tensors. A changed checkpoint, layout, codec, or input fails.
A lock rejects concurrent writers. After a process crash, inspect the owning
process before removing its `.capture-writer.lock`; never remove a live
writer's lock. Files ending in `.partial` are not training inputs. A disk or
codec failure propagates and releases the current Mooncake object instead of
silently training on another sample.
