# r10：启用 decode CUDA Graph 与前缀缓存

用户明确要求重启，并参考 DeepInfer 分析 decode 低效。r9 已按精确身份取消，保留36条 target-only 测试结果；旧冻结文件不变。r10从原输入第0条开始独立测量，不混合r9结果，不按已见test输出挑样本或调模型。

单组8H100/64CPU/512Gi，无固定节点，WeaverRSI/proj-32。固定原镜像、原source、r2 exports、四组各247请求和真实EOS1/temp0/max1024。数值差异继续记录，不能称无损或SWE solve score。HTTP/SSE/EOS/计数/finite/输入SHA仍失败即停，不自动重试。

性能调整：Radix前缀缓存开启；独立server隔离四组缓存，保留原请求顺序，统计缓存token。decode graph=full、仅捕获单并发bs1，必须观察实际replay。prefill graph仍disabled：pinned dsv4默认因c4 indexer捕获池压力关闭长上下文prefill图，不强行打开未经容量验证的路径。TP8EP8、固定gamma5、Engram CPU和其他kernel不变；CPU overlap必须启用。保留10对A/A观察，差异继续测量。

首个target-only验证请求在prefill结束后采集8个forward步骤的CPU/GPU trace；64-token上限、真实EOS，trace只在0700目录/0600文件，原始trace不进Git，仅发布kernel/runtime时间汇总。该请求排除在benchmark计时之外。诊断阶段结束发布validation-metrics.json；各组完成发布summary，最终988逐条审计。CPU回归包括真实HTTP/SSE全矩阵、缓存计数、plain-text profiler端点、Graph实际运行证据、身份与生命周期。

worker24h/controller26h。先核所有旧alias消失、前驱取消回执、CPU回归与manifest，controller --check后launch只执行一次。任何launch/started/submit intent存在均禁止再次启动。终态先回执再精确删除本job并确认alias空。不清理其他数据/ckpt，不碰健康baseline A/B或他人资源。heartbeat保持PAUSED。
