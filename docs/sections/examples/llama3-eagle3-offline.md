# EAGLE3 for Llama 3.1 8B: offline training

Offline training captures target features ahead of time and reads them from
disk while training the draft. It reduces GPU memory pressure during training,
but feature storage can be much larger than the source dataset.

## 1. Prepare ShareGPT

```bash
python ./scripts/prepare_data.py --dataset sharegpt
```

## 2. Capture hidden states

Feature preparation is a data-processing step, not a second training entry
point:

```bash
torchrun --standalone --nproc_per_node 8 \
  scripts/prepare_hidden_states.py \
  --strategy eagle3 \
  --target-model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-config configs/llama3-8B-eagle3.json \
  --data-path ./cache/dataset/sharegpt_train.jsonl \
  --output-path ./cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct \
  --chat-template llama3 \
  --max-length 4096 \
  --tp-size 1 \
  --batch-size 32
```

The output directory now matches the hidden-state path in the checked-in
recipe. It contains the feature checkpoints consumed by the unified trainer
and `vocab_mapping/vocab_mapping.pt`, derived from the same processed corpus.

### Optional lossless activation compression

Install `pip install '.[compression]'` in both the capture and trainer
environments, then append `--zstd` to the capture command. This writes
`.ckpt.zst` records with BF16 sign/exponent/mantissa fields compressed separately
using CPU Zstd level 1. `--zstd-level 1` through `19` controls the tradeoff.
Other supported tensor dtypes retain their bytes and use plain Zstd.

For BF16 auxiliary activations concatenated across capture layers, additionally
pass `--interleave-layers`. The capture plan supplies the ordered layer IDs;
the codec changes `[token, layer, dimension]` to `[token, dimension, layer]`
before compression and reverses it on read. The feature's last dimension must
contain equally sized layer slices. Final-layer targets, IDs and masks are
not interleaved. This option rejects non-BF16 auxiliary features and capture
plans without an auxiliary feature. It changes storage only; the decoded
dtype, shape, values and layer order are identical, including BF16 NaN payloads
at the codec boundary. The capture script still skips NaN-containing records.

Training auto-detects raw `.ckpt`, existing gzip `.ckpt.gz` and new `.ckpt.zst`
files, including mixed directories. `--compress` / `--compression-level` retain
their gzip behavior and cannot be combined with `--zstd`. Resume recognizes
all three extensions and skips existing samples; it does not transcode them.
File listing stays lazy. A corrupt file raises an error when read, including
in the legacy EAGLE3 dataset, rather than substituting another sample.

The version-1 format uses an `NSACTZ01` magic, a bounded JSON manifest with a
SHA-256 checksum, and length-prefixed Zstd frames with content checksums. Tensor
blocks contain at most 1,048,576 elements; BF16 blocks have packed sign bits
(little bit order), exponent bytes and low-seven-bit mantissa bytes. The manifest
records feature names, dtypes, original shapes and optional ordered interleave
layer IDs. Other tensors use raw little-endian bytes per block. No pickle is
used by this format. Writes publish by an atomic rename in the destination
directory; failed writes remove their temporary file. The format supports
little-endian CPUs and dense BF16/FP16/FP32/FP64, signed integer, uint8, bool
and complex64/128 tensors. Tensor strides and shared-storage aliases are not
preserved. The shared reader limits decoded tensor storage to 8 GiB per record
and the manifest to 1 MiB.

Compressed records materialize in CPU RAM and cannot be mmap'ed. Interleave
also needs a tensor-sized reorder buffer, so account for it in capture I/O
queue and trainer prefetch memory. Compression runs after D2H completion in
the existing bounded I/O pool, with one Zstd worker per save. No change is made
to online/RDMA transport. See [research issue #1](https://github.com/china-qijizhifeng/NexSpecTrainer/issues/1)
for the early-layer V4.1 measurements; deeper DSpark layers, training throughput
and acceptance-rate effects have not been measured.

## 3. Use the checked-in run config

The canonical recipe is
[`examples/configs/offline/colocated/llama3.1-8b-eagle3.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/offline/colocated/llama3.1-8b-eagle3.yaml).
It records the same target, feature directory, draft architecture, and trainer
settings used by this walkthrough; edit the checked-in recipe or use dotted
overrides instead of copying its YAML into another document.

The recipe carries a conventional `model.vocab_mapping_path` for deployments
that prepare and share a mapping artifact. For an offline colocated run, leaving
that field empty makes SpecForge count effective tokens in the exact feature
corpus, derive `t2d`/`d2t` deterministically, and cache the reusable mapping
under `data.cache_dir/vocab_mapping`. Equal target and draft vocabularies need
no mapping.

## 4. Train

```bash
specforge train \
  --config examples/configs/offline/colocated/llama3.1-8b-eagle3.yaml \
  model.vocab_mapping_path=./cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct/vocab_mapping/vocab_mapping.pt
```

Omit the final override when the mapping file named by the recipe already
exists. The same recipe supports offline data parallelism through the typed
`deployment.trainer` topology and unified self-launcher.

For long sequences, EAGLE3 offline can instead use USP by setting
`training.attention_backend=usp` and choosing
`training.sp_ulysses_size`/`training.sp_ring_size`. Offline feature training
also supports DFlash, DFlash2, Domino, and DSpark when the feature
checkpoints and draft config use that strategy's contract.
`training.compact_teacher: true` enables
the exact lower-memory teacher projection for offline text EAGLE3. See
[Parallel topologies](../basic_usage/training.md#parallel-topologies) for the
multi-process launch contract.
