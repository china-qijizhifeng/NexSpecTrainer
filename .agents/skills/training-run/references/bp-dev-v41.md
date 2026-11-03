# BP Dev V4.1 临时运行环境

此路径仅用于正式训练镜像尚未发布/验收时。镜像可用后直接使用镜像内的完整训练栈；以下 GPFS overlay 和任务内 Core 组合应退出默认运行流程。路径与回执为 2026-11-03 的已知状态，运行前检查实际文件。

## 先前镜像与会话

- [接手 specdec 训练调研](codex://threads/01a19c72-18d4-75e0-9717-d4767eea0775) 在 cpu100 构建过 `harbor.local.clusters/bp/nexspec-trainer:v41-weaver-torch210-20261103-r1`。历史记录显示构建完成，但向 Harbor 推送被拒绝。标签名和 Dockerfile 不证明 registry 已有可拉取成品。
- [优化 v4.1 flash prefill](codex://threads/01a19f8e-ce9f-7d90-9b53-c0ea0ce70d92) 记录了 Docker Hub 拉取超时、相同 digest 的公共镜像源及离线 OCI 导入。服务镜像已有原始 OCI 归档；不是另一个训练镜像。
- 训练镜像的构建入口为 `docker/deepseek-v41/README.md`。未来正式镜像需同时满足当前 native forward 与 NFC checkpoint ABI；旧 CSA overlay 与旧 runtime build 不能仅凭曾经构建成功就视为兼容。

## 固定的底座与共享依赖

训练底座：

```text
harbor.local.clusters/bp/weaver-trainer-megatron-bridge:v1.14.92@sha256:cba116d15f4be378c467a5d15157b69e3307e21c3c67342a06481f8f4a1187df
Torch 2.10.0a0+b558c986e8.nv25.11
Transformer Engine 2.20.2+6ea2a74
```

共享目录：

```text
/gpfs/users/linlizhi/nexspec-hopper-validation-20261102/compat-weaver-r2
  overlay/                           DeepGEMM、FlashMLA、CuTe DSL 等
  overlay/nvidia_cutlass_dsl/dsl_packages
  megatron-src/                      已验证的 CSA/MoE forward Core
  tile/                              TileKernels 源码

/gpfs/users/linlizhi/nfc-sharded-pipeline-20261103/source-cpuview-r2/src
  nexflashcheckpoint/                已验收 NFC 实现与编译扩展

/gpfs/users/linlizhi/nexspec-v41-rdma-realtrace-20261103/wheels
  zstandard-0.25.0-...whl             Python 3.12 离线 wheel
```

原生 forward 还要求 DeepGEMM >=2.8.2 的 SM90 prefix-sum grouped GEMM。DeepGEMM 2.3 虽有同名 API，不能完成该布局。保持显式 CuTe DSL 路径，避免 Weaver 较旧包抢先导入。不要 `pip install` 主仓库依赖并让 resolver 升级 Torch/TE。

### NFC 与 CSA 的临时组合

上述 `compat-weaver-r2/megatron-src` 具有 CSA，却不接受 NFC 所需的 `async_strategy`。底座内 `/opt/Nex-Megatron-Bridge/3rdparty/Nex-Megatron/megatron/core/dist_checkpointing` 支持该 ABI，但其 Core 缺少当前 CSA。只调整两个整包的 PYTHONPATH 先后次序无法同时满足两端。

`scripts/prepare_megatron_core.py` 将旧 forward Core 与镜像原样的 checkpoint 子包组合到本任务新目录，保留共享源。它逐文件核查 Python 来源，并固定 checkpoint serialization SHA：

```text
5f745e742b63b4d4f4c406edda9c09605746bd52faa24c1b5fb4446676799a65
```

该组合已在正式底座做 CPU API 导入检查，包含 CSA、MoE 与 NFC；两份辅助脚本也已实际运行并验证重复调用，见 [2026-11-03 回执](validation-20261103.json)。8 H100 的实际训练前向/保存仍需当前运行验收。不要将此临时做法扩展为通用 Megatron backend。

以下命令在实际 trainer 容器内运行。`RUN_ROOT` 是本次新建的 GPFS 目录，`REPO_ROOT` 是挂载的固定源码 checkout，`TRAIN_CONFIG` 是已冻结配置。不要与其他运行共用 RUN_ROOT：

```bash
set -euo pipefail
: "${RUN_ROOT:?set the GPFS run directory}"
: "${REPO_ROOT:?set the pinned checkout directory}"
: "${TRAIN_CONFIG:?set the resolved training YAML}"
SKILL_ROOT="$REPO_ROOT/.agents/skills/training-run"
COMPAT=/gpfs/users/linlizhi/nexspec-hopper-validation-20261102/compat-weaver-r2
NFC_SOURCE=/gpfs/users/linlizhi/nfc-sharded-pipeline-20261103/source-cpuview-r2/src
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8

python3 "$SKILL_ROOT/scripts/prepare_megatron_core.py" \
  --forward-core "$COMPAT/megatron-src/megatron/core" \
  --checkpoint-core /opt/Nex-Megatron-Bridge/3rdparty/Nex-Megatron/megatron/core \
  --output "$RUN_ROOT/train-core"
python3 -m pip install --no-index --no-deps --target "$RUN_ROOT/python-deps" \
  /gpfs/users/linlizhi/nexspec-v41-rdma-realtrace-20261103/wheels/*.whl
export PYTHONPATH="$COMPAT/overlay/nvidia_cutlass_dsl/dsl_packages:$COMPAT/overlay:$RUN_ROOT/train-core:$COMPAT/tile:$REPO_ROOT:$NFC_SOURCE:$RUN_ROOT/python-deps:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export CUDA_CACHE_PATH="$RUN_ROOT/cache/cuda"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export DG_JIT_CACHE_DIR="$RUN_ROOT/cache/deepgemm"
export TILELANG_CACHE_DIR="$RUN_ROOT/cache/tilelang"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" \
  "$DG_JIT_CACHE_DIR" "$TILELANG_CACHE_DIR"

python3 "$SKILL_ROOT/scripts/preflight.py" --gpu --output "$RUN_ROOT/environment.json"
cd "$REPO_ROOT"
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m specforge.cli train --config "$TRAIN_CONFIG"
```

先做 0 GPU CPU 环境检查时，省略 `--gpu`；这不代替 native CUDA preflight 或真实更新检查。多节点运行应按实际 launcher/拓扑配置，不能复用这里的 standalone 参数。

## 独立的 target 服务镜像

原始 SGLang 镜像与训练底座分开：

```text
lmsysorg/sglang@sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa
SGLang source 38d9e4a3fe4a493b0ae468477a274e83defda81a
image config sha256:31d58781877bdcd322d5fcc7a858d427db54f5cbf5a9e6342c5fafc01bc67c6e
```

旧内网 `oss-sglang` 镜像曾缺少 V4.1 配置、ForwardObserver/ResponseCompletion，不可仅按名字替换。按 `plugins/sglang-spec-capture` 的实际契约装插件，禁止对服务端解析安装整个 trainer 依赖。

Docker Hub 曾多次 manifest HEAD 超时。已知公开镜像源 `docker.m.daocloud.io/lmsysorg/sglang@sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa` 在先前查询返回同一 manifest digest；使用前重新验证可达性和身份，不向公开镜像源发送内部 registry 凭据。

离线原始归档位于 `/gpfs/users/linlizhi/nexspec-v41-capture-20261103/offline-image`，`source-receipt.json` 与 `complete.json` 固定 manifest/config、归档 bytes/SHA。部分节点已导入缓存，不代表所有节点都存在。不要将 host containerd socket 挂载/镜像导入作为普通训练作业默认权限；沿用经过核实的运维流程，身份校验后才使用缓存。

## V4.1 数据身份与资源边界

Target：`deepseek-ai/DeepSeek-V4.1-Flash-1017`，revision `1bae0e1088993b9b82de08f59de5be5a34e42170`，共享路径为 `/gpfs/users/linlizhi/models/huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017`。检查 `.download/complete.json`，使用官方 `encoding.encode_messages` 和同 snapshot tokenizer，渲染后 `add_special_tokens=False`。本轮编码为 `thinking_mode="thinking"`、`drop_thinking=False`、`reasoning_effort=75`；这些是本轮参数，后续按用户选择冻结。

原生离线 recipe：`examples/configs/offline/colocated/deepseek-v41-flash-native.yaml`。recipe 是起点，正式训练应确定全量片段数、tail/accumulation、评估及 checkpoint 周期，不能沿用历史 max_steps=3 冒烟值。

- 完整 target 曾使用 8 H100、64 CPU、512 GiB、TP8/EP8、CPU Engram。32 卡采集池可拆为 4 副本；这里只记录单副本和顺序复用卡的流程，不等于 40 卡 RDMA 已验收。
- 完整原生 MTP + FP32 masters/Adam checkpoint 约 199.17 GB，不含冻结的约 510 GB target 权重。NFC 32 GiB/rank pool 合计最高 256 GiB；8 rank 运行例使用 `/dev/shm` 288 GiB、host 640 GiB。实际内存预算还须覆盖 optimizer/loader 等。
- latest/best/写入周转须按完整 checkpoint 规划空闲空间。只有约 249 GiB free 时，不能按正式多份 checkpoint 方案启动；历史 pilot 的 >215,000,000,000 bytes guard 只覆盖其单份最终 checkpoint 和小量特征。
- DeNovoSWE 全量特征约 18.69 TB，最大完整轨迹 234,048 tokens，部分 record 超过 8 GiB。小于 32K 的有限 pilot 不能证明这些长样本已可运行。
- 保留已有的 `/gpfs/users/linlizhi/nexspec-v41-rdma-realtrace-20261103/training-r1/v41-native-real-features-r1-step2`，不能为了本轮空间删除或覆盖该输入 checkpoint。
