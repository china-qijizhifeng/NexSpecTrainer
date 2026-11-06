# First-run failure: 2026-11-06

Job job-7cb0017758b3-20261106155850 reached initial same-set evaluation, before any optimizer update. Both ranks aborted with glibc `free(): double free detected in tcache 2`. No Python traceback was enabled for SIGABRT. Terminal Aborted and exit1 were retained; exact owned job was removed by the terminal monitor.

Static audit: model preparation and frozen tensor hashing completed. Evaluation had entered strategy.forward_loss on its first batch. Native RMS/mHC/rotary/routing use TileKernels, attention uses Megatron CSA/FlashMLA, MoE uses DeepGEMM and TE fused permutation. Gradient checkpointing is conditional on both training mode and grad enabled, so it is inactive on the failed evaluation path. All-anchor enumeration pads every rank to the global maximum, maintaining collective schedules. The ten samples divide evenly across two ranks. Source/init are the same as the completed 32-GPU baseline; CUDA cache contents were copied to task-local paths.

Ranked hypotheses before probe:
1. First native kernel load/compilation: faulthandler and module boundaries will identify a compiler/loader frame.
2. Copied compilation cache is incompatible with current loading: only if stack points there, compare a fresh isolated cache.
3. Initial evaluation no-grad path differs from training: only if crash is in an executing operator, compare direct training forward.

Minimal probe: dspark-overfit10-probe-20261106-r1, same model/data/config/cache, two GPUs, baseline evaluation only, PYTHONFAULTHANDLER=1 and tagged native module entry/exit hooks. No training updates. Record exact failing call before choosing a fix. No shared runtime/source changes.

Probe result: both ranks reproduce SIGABRT. Faulthandler points to torch.distributed.all_gather_into_tensor called from FSDP _all_gather_flat_param / pre_forward_unshard; none of the native module entry hooks ran. This rules out a native forward kernel as the immediate trigger. NCCL 2.28.9 loads the image's NCCL RDMA Plugin v10 / SHARP at /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so, then logs `Failed to initialize NET plugin NCCL RDMA Plugin v10` immediately before process abort. This single-node job disables IB and does not request an RDMA device; the historical 32-card job requested RDMA and initialized this plugin successfully.

Next regression uses a real two-rank all_gather/all_reduce with float32/bfloat16 at sizes 1/4096/1048576, same image and job environment. Compare default external plugin (must reproduce double free) vs NCCL_NET_PLUGIN=none (must pass all operations), changing only this one variable. Full training is gated on both outcomes. A task-local environment change avoids modifying shared libraries, source or existing jobs.
