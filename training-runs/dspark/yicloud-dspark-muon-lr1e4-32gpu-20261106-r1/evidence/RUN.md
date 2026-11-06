# DSpark Muon + AdamW LR 1e-4

2026-11-06 用户要求在上一轮 Muon 基础上开一组 1e-4。

严格复用上一轮仪电 Muon LR 1e-5 的源码 3966a11f345c8fb42b43dfe51e86ec74693deb1d、镜像、特征目录、fresh 初始化 seed42 和训练配方，仅修改 recipe.yaml 的 training.learning_rate 为 1e-4。矩阵 Muon 与其余 AdamW 的 multiplier 均保持 1。4×8 H200，fdj-infra / nex_bp，1326 train +17 validation traces，15104 train windows，136 validation windows（32-rank padding160），globalbatch128/accum4/all anchors/chunk64/FP32 reduction/FULL_SHARD/CPU optimizer offload，warmup3/cosine118，每25步验证保存，最终118步/472训练微批。保留原调度器行为，不混入调度修复。

Run UUID: 407a5537-29d5-4932-97e7-b572674b4c0c。
远端工作根：/mnt/work/shom/dspark-muon-lr1e4-32gpu-20261106-r1。
远端训练根：/mnt/work/shom/training_runs/dspark-muon-lr1e4-32gpu-20261106-r1。
基线训练根：/mnt/work/shom/training_runs/dspark-all-anchors-muon32-20261105-r2。
特征直接复用：/mnt/work/shom/dspark-muon32-20261105-r1/{train,validation}-features。

状态、作业 ID、启动验收见后续回执。不得将提交成功写成真实训练成功；本轮无 checkpoint resume，不操作其他作业。

提交成功：JobId `j-1793948269-442092`，源码 `95d9ef8caa295e6d5ecbfb215589b1e98dcd252c`。终态 reconciler PID 83258，每120秒核对精确作业，不重启、不重投。
