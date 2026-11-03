# Reuse GPU kernel compilation and MoE tuning caches

Wrap an existing server or training command with `scripts/with_kernel_cache.py`.
The launcher prepares persistent directories and sets environment defaults
before the command imports GPU libraries. It uses the Python standard library,
preserves command arguments and exit status, and replaces itself with the
command so normal process supervision and signals still work.

```bash
python scripts/with_kernel_cache.py \
  --cache-root /shared/kernel-cache \
  --namespace sglang-c6da252c6837-sm90 \
  -- python -m sglang.launch_server \
  --model-path /shared/models/DeepSeek-V4.1-Flash-1017 \
  --tp-size 8 --ep-size 8 \
  --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4
```

Use the model, capture-plugin environment, capture flags, and remaining server
arguments required by your deployment. This example requires a V4.1-capable
SGLang image; it does not upgrade the repository's pinned SGLang dependency.
The wrapper also accepts `sglang serve ...`, `torchrun ...`, or other commands.
For server-to-file collection, wrap the **server startup**; wrapping only the
capture client cannot change a running server's compilation caches.

Choose a namespace containing the immutable image identity and GPU architecture.
Change it when rebuilding compiler libraries or changing compilation options.
Keep trainer and inference environments separate even on the same GPU type.
Explicit environment variables always take precedence, including deliberate
cache-disable settings. Such overrides can point outside the namespace, so
inspect them before reusing a cache:

```bash
python scripts/with_kernel_cache.py \
  --cache-root /shared/kernel-cache \
  --namespace sglang-c6da252c6837-sm90 --print-env
```

Preview prints only the cache settings and does not create directories or launch
a process. No Torch, Triton, TileLang, or SGLang import is needed.

| Cache | Environment / location under the namespace |
| --- | --- |
| Triton binaries and intermediate code | `TRITON_CACHE_DIR=.../triton` |
| Triton autotune timings, on versions supporting persistence | `TRITON_CACHE_AUTOTUNING=1` |
| TileLang binaries and autotune results | `TILELANG_CACHE_DIR=.../tilelang`; keeps both caches enabled |
| SGLang kernels and MoE tuning | `SGLANG_CACHE_DIR=.../sglang` |
| SGLang JIT extensions / CuTe AOT | `SGLANG_JIT_CACHE_DIR=.../sglang/jit`, `SGLANG_CUTE_AOT_CACHE_DIR=.../sglang/cute_aot` |
| DeepGEMM | Both `SGLANG_DG_CACHE_DIR` and `DG_JIT_CACHE_DIR` point to `.../sglang/deep_gemm` |
| FlashInfer generated/JIT code | `FLASHINFER_WORKSPACE_BASE=.../flashinfer`; the library appends `.cache/flashinfer/<version>/<arch>` |
| Torch compilation / extensions | `TORCHINDUCTOR_CACHE_DIR=.../torchinductor`, `TORCH_EXTENSIONS_DIR=.../torch_extensions` |
| CUDA driver and XDG consumers | `CUDA_CACHE_PATH=.../cuda`, `XDG_CACHE_HOME=.../xdg` |

SGLang can overwrite `DG_JIT_CACHE_DIR` during import using its own
`SGLANG_DG_CACHE_DIR`. The wrapper honors either spelling and sets both to the
same location. Conflicting explicit values fail before launching the command.
FlashInfer's installed AOT/cubin packages remain available; the workspace
directory does not replace those packages. Older library versions may ignore
newer environment settings; inspect the installed version's source and logs.

## Reduce DeepGEMM startup work

The wrapper defaults `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1`. In the V4.1 serving
revision examined below, unchunked prefill (`--chunked-prefill-size -1`) otherwise
sweeps every M from 1 through 65,536 for each encountered kernel type/N/K/group
combination. Fast warmup samples 3,072 M values for that configuration. This is
a reduction in warmup iterations, not a measured 21x startup speedup or 65,536
distinct compiled binaries. Missing specializations can still compile on use.

Keep a full sweep when required by your latency/shape coverage policy:

```bash
SGLANG_JIT_DEEPGEMM_FAST_WARMUP=0 python scripts/with_kernel_cache.py \
  --cache-root /shared/kernel-cache --namespace sglang-c6da252c6837-sm90 \
  -- python -m sglang.launch_server <your-server-arguments>
```

`python -m sglang.compile_deep_gemm` can populate a cache ahead of deployment;
use the same image, environment, and model/parallel/backend arguments. That
utility launches a model server itself: it is not a CPU-only compiler and
should run in a separate scheduled prewarm job. Reusing a live service also
avoids paying model load and process-local initialization again.

## Persist MoE tuning and cover the capture workload

`SGLANG_FLASHINFER_AUTOTUNE_CACHE=1` preserves the selected tactics separately
from compiled kernels. In the examined SGLang revision, files live at:

```text
$SGLANG_CACHE_DIR/flashinfer/autotune/<flashinfer-version>/<sm>/<config-hash>/rank_tp*_pp*_dp*.json
```

Keep the complete set of rank files. SGLang checks synchronized ranks for cache
agreement and discards divergent caches before retuning. Results are saved when
the autotune context completes, which can include intervening DeepGEMM prewarm;
a directory or a completed GEMM progress bar alone does not prove persistence.
The key includes model path, dtype, quantization, backend, parallel topology,
and skip policy. Moving weights to a different path may select a new key.
FlashInfer also checks runtime/compiler-library metadata on load.

Tune with the actual per-rank shapes, expert count, top-k, dtype, and TP/EP
configuration. Decode/verify tuning alone does not cover a 4k-token capture
prefill. The examined DSPARK path skips generic EXTEND autotuning even with
`SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`; its model-specific prefill hook covers
MXFP8 dense GEMMs, not necessarily MoE prefill. The wrapper therefore does not
set this flag or claim to fill that gap. Cover MoE prefill explicitly in a
backend-specific prewarm job, then check that a fresh process loads the saved
tactics for those shapes. Do not profile competing workloads on the same GPUs.

## Use local NVMe with a shared persistent copy

Inside a Kubernetes job, access local disk through a mounted directory, such
as a local PV or a platform-approved `hostPath`. A container's `/tmp` does not
establish that it uses NVMe. On the assigned worker, use `findmnt -T <path>`,
`df -hT <path>`, and `lsblk -o NAME,TYPE,SIZE,ROTA,MODEL,MOUNTPOINTS` to identify
the backing device and capacity. Check shared storage and RAID mappings too;
a nonrotational device alone does not establish the NVMe interface.

For example, after provisioning a dedicated node-local cache directory:

```yaml
volumes:
  - name: kernel-cache
    hostPath:
      path: /data/kernel-cache/your-user
      type: Directory
containers:
  - name: server
    volumeMounts:
      - name: kernel-cache
        mountPath: /kernel-cache
```

This is a Pod-spec fragment. Use `--cache-root /kernel-cache` with the wrapper.
The directory must already exist with suitable permissions. Node-local data
depends on scheduling and node lifetime; it is not shared storage.

Seed a local namespace from a complete GPFS/shared-storage snapshot **before**
starting any ranks. Run compilation and tuning locally, then snapshot the whole
namespace back after all writers finish. Publish through a staging directory
and an atomic rename on the shared filesystem. Do not copy partially written
tuning results, overwrite a namespace in active use, or split local ranks
across independently changing caches. Preserve compiler metadata and the full
directory structure. Keep model weight caching separate from kernel caches.

Validate reuse with a fresh process in the same environment: compare startup
phase timings, inspect tuning-cache load/mismatch messages, and check whether
new compiler work occurs for the expected capture shapes. A cache directory's
existence alone is insufficient. This launcher has CPU subprocess regression
coverage; it does not establish a GPU startup speedup.

Source contracts: [SGLang DeepGEMM warmup](https://github.com/sgl-project/sglang/blob/38d9e4a3fe4a493b0ae468477a274e83defda81a/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py),
[SGLang FlashInfer tuning](https://github.com/sgl-project/sglang/blob/38d9e4a3fe4a493b0ae468477a274e83defda81a/python/sglang/srt/model_executor/runner/flashinfer_autotune.py),
[TileLang 0.1.12](https://github.com/tile-ai/tilelang/blob/v0.1.12/tilelang/env.py),
and [FlashInfer 0.7.2 workspace](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2/flashinfer/jit/env.py).
