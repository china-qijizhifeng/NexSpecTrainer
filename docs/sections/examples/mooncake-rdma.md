# Run compressed capture through Mooncake RDMA

The [external capture client](server-capture-to-disk.md) already accepts
Mooncake transport settings from the environment. Set the same protocol and
device selection on the SGLang capture server and client before starting either
process. This recipe uses CPU source/receive buffers and lossless wire encoding:

```text
SGLang BF16 → CPU encode → Mooncake RDMA → CPU decode → compressed file → training decode
```

## Allocate the RDMA device

On BP, the following Pod-spec fragment supplied device access and the secondary
network in the verified node044 job. A `/dev/infiniband` mount alone did not grant
the required device permissions. Other clusters need their own resource name,
network attachment and device selection.

```yaml
metadata:
  annotations:
    k8s.v1.cni.cncf.io/networks: network-operator/ens108np0
spec:
  activeDeadlineSeconds: 14400
  containers:
    - name: capture
      resources:
        requests:
          infrawaves.com/rdma: "1"
        limits:
          infrawaves.com/rdma: "1"
      securityContext:
        capabilities:
          add: [IPC_LOCK]
      env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
```

Add the image, GPU/CPU/RAM requests, command and storage mounts required by the
application. For Northjob on BP, use project `WeaverRSI`, then verify its current
Volcano queue binding; the validation used `proj-32`.

## Set the transport and budgets

Run the following in both processes' launch environments. The endpoint example
assumes a Mooncake master and both clients in the same Pod; use reachable
endpoints for separate Pods.

```bash
ulimit -l unlimited
unset MC_FORCE_TCP MC_MS_AUTO_DISC MC_USE_TENT MC_USE_TEV1 MC_TE_FILTERS
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_RDMA_DEVICES=mlx5_00
export MOONCAKE_LOCAL_HOSTNAME="${POD_IP:?Supply the Pod IP}"
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_GLOBAL_SEGMENT_SIZE=8589934592
export MOONCAKE_LOCAL_BUFFER_SIZE=8589934592
export MC_STORE_MEMCPY=0
export SGLANG_SPEC_CAPTURE_GPU_PUT=0
export SGLANG_SPEC_CAPTURE_WIRE_ENCODING=bf16_fields_zstd_v1
export SGLANG_SPEC_CAPTURE_WIRE_ZSTD_LEVEL=1
```

For a direct Mooncake API preflight, explicitly register both the source and
destination buffer before `put_from` / `get_into`, check return codes, and keep
the buffers alive until the transfer completes. The repository's feature store
owns that registration for normal capture. Do not infer RDMA success from
device visibility or the requested protocol alone: use two independent
processes, compare transferred bytes and SHA-256, and retain transport handshake
logs from the actual serving image.

Start the server with the capture plugin and the target's full-prefill settings.
Wrap the **server command** with [the persistent cache launcher](kernel-cache.md).
Keep serving and trainer image caches separate. If reusing a populated SGLang
DeepGEMM directory, set both `SGLANG_DG_CACHE_DIR` and `DG_JIT_CACHE_DIR` to that
directory before invoking the wrapper. It enables sampled DeepGEMM prewarm.

Allow a separate cold-start readiness window: the initial V4.1 run needed more
than 20 minutes. The validation used a 5,400-second readiness deadline and a
14,400-second job deadline, while checking that the owned server process remained
alive. `--timeout-s` below bounds a capture request to an already ready service;
it does not extend server startup or a scheduler's job deadline.

```bash
python scripts/prepare_server_hidden_states.py \
  --config /shared/train-v41.yaml \
  --server http://127.0.0.1:30000 \
  --samples /shared/prepared-traces.jsonl \
  --output /shared/captured-traces \
  --target-model-version 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --wire-encoding bf16_fields_zstd_v1 \
  --client-buffer-bytes 8589934592 \
  --server-buffer-bytes 8589934592 \
  --timeout-s 1800 \
  --zstd
```

The prepared JSONL contains the existing token IDs and loss masks; no new
rollouts are needed. Explicit trajectory windows are valid inputs when their
context reset and source token offsets are recorded. The client never silently
cuts a longer record to the configured training length.

`--zstd` and `--hybrid` choose the disk format independently of wire encoding.
The measured V4.1 run used Zstd files. Hybrid's Brotli/Zstd codec has CPU bitwise
round-trip coverage; it was not the disk codec used for that GPU training run.
Both decode back to the original BF16 values before normalization.

## Verified scope

The [V4.1 validation record](https://github.com/china-qijizhifeng/NexSpecTrainer/blob/main/docs/validation/deepseek-v41-20261103.md) reports
same-node, independent-process, CPU-buffer RDMA and real-trajectory capture.
It does not establish cross-node RDMA, GPUDirect, convergence, throughput gains
or strict speculative-output preservation. Keep raw tensor bytes, encoded wire
bytes and compressed file bytes separate when reporting compression.
