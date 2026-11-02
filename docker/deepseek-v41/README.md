# Offline Hopper build on Weaver

Preserve Weaver v1.14.92 at digest
`sha256:cba116d15f4be378c467a5d15157b69e3307e21c3c67342a06481f8f4a1187df`,
including Torch `2.10.0a0+b558c986e8.nv25.11`, TE `2.20.2+6ea2a74`, CUDA 13.0,
cuDNN Frontend 1.26.0 and the existing system libraries. Do not resolve the
trainer's general dependencies over this base.

## Compile the two missing extensions

`Dockerfile.weaver` builds pinned DeepGEMM 2.8.2 and FlashMLA wheels against the
base Torch ABI. Its context contains `native-sources.tar.gz`,
`elfutils-dev-headers.tar.gz`, their hashes and source revisions in
`native-inputs.json`, and `build-weaver.sh`. Compilation uses disconnected pip
with `--no-build-isolation --no-deps`; it does not install Torch or TE.
The `wheels` target exports `/wheels` and `/evidence`.

```sh
docker build --network none --target wheels \
  -t nexspec-weaver-wheels:v41-torch210 -f Dockerfile.weaver .
```

For retained intermediate wheels, run `build-weaver.sh prepare`, then
`build-weaver.sh DeepGEMM` and `build-weaver.sh FlashMLA` in a base-image
container with `/build` mounted read-only and `/opt/nexspec` mounted to an output
directory. Set the same environment variables as `Dockerfile.weaver`.
CCCL must be on the host C++ include path as well as the CUDA include path.

The original DeepGEMM 2.3 exposes grouped GEMM symbols but rejects prefix-sum
layouts on SM90. Preflight therefore requires the pinned 2.8.2 implementation.

## Assemble the runtime overlay

`Dockerfile.runtime.weaver` adds these wheels and source overlays in one layer,
staying within overlay2's layer limit. Its context contains:

- `source-hashes.json` and `trainer-current.tar.gz`, `tile-kernels.tar.gz`,
  `megatron-csa.tar.gz`.
- `overlay-packages.json` and its pinned wheels under `wheelhouse/`:
  TileLang 0.1.15, TVM FFI 0.1.12, torch-c-dlpack-ext 0.1.5,
  z3-solver 4.15.4.0, and CuTe DSL 4.8.1 with its matching library wheels.
- `native-wheels.json` and the newly compiled wheels under `native-wheels/`.
- `assemble-weaver.py` and `smoke.py`.

```sh
docker build --network none \
  -t harbor.local.clusters/bp/nexspec-trainer:v41-weaver-torch210 \
  -f Dockerfile.runtime.weaver .
```

The assembler verifies every archive and wheel hash, then checks Torch/TE
versions and binary paths. Its explicit CuTe DSL path avoids loading Weaver's
older DSL package. The newer Megatron source provides packed CSA; the existing
cuDNN DSA package is reused. No dependency resolver runs during assembly.
TileKernels is a source overlay: its declared `torch>=2.13` requirement is not
used as evidence of an actual operator requirement.

CPU imports and `specforge --help` run during image assembly. Source identities,
package versions, API checks and hashes live in `/opt/nexspec/evidence`.
Runtime entry: `specforge train --config <config.yaml>`. Mount model checkpoints,
datasets, output and JIT caches on shared storage.

## Acceptance

The preserved Weaver stack passed all 17 native checkpoint, kernel and training
checks on an H100 80 GB GPU. This includes grouped GEMM forward/dgrad/wgrad with
an empty expert, fused MoE dispatch/router gradients, private-window attention
with FlashMLA forward and cuDNN DSA backward, and two AdamW steps with layer and
objective recomputation. Job `job-a9d4f30f00bc-20261103003809` used project
`WeaverRSI` and queue `proj-32`; JUnit reports 17 passed in 153.22 seconds.

These are small numerical and integration cases. Full 14.23B draft training,
multi-GPU FSDP, complete target activation capture and exported SGLang inference
remain separate acceptance gates.
