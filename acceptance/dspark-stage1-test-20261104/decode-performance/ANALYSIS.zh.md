# decode 性能分析：DeepInfer 对照

参考 DeepInfer `a679bb86ce36e883a98a97733492b686e5b9a08e`（2026-11-03），与实际 SGLang `38d9e4a3fe4a493b0ae468477a274e83defda81a` 对照。结论分为已有实测、源码路径和待trace确认；未测量的项不声称具体贡献。

## 已有实测

r9在node248、8H100、64CPU、512Gi上运行。前32条原序请求平均墙钟112.0秒，输入平均55,846tokens，decode日志中位5.58tokens/s。近15分钟8卡利用率平均69.87%，覆盖60个完整时点。按此样本估计基线247条约7.7小时，非其余组的耗时预测。用户要求重启后精确取消，保留最终36条基线请求；不能当完整组或混入新运行。10对A/A的5对全相同，其余仍有数值差异。

## 为什么原先关闭

- 前缀缓存：让原四组都完整重算输入，消除命中差异；属于隔离实验策略，并非最快服务配置。现在用户明确要求开启，r10同一原序各组独立缓存，报告cached input tokens；速度包含实际命中效应。前缀缓存主要减少prefill，不能把5.58tokens/s的稳定decode低速全部归因于缓存关闭。
- Decode graph：数值定位沿用eager以简化调用链和观察。进入观察式性能测评时仍保留这一约束不合适。r10启用full graph、bs1并验实际replay。
- Prefill graph：该pinned版本`arg_groups/cuda_graph_hook.py::disable_breakable_cudagraph_if_incompatible`明确以“DeepSeek-V4 heavy capture-pool memory pressure”默认关闭非CP的prefill图；c4索引器scratch会钉在capture pool。不能把decode开启解释成64K prefill也能直接开启。

## decode路径的主要差距

| 路径 | DeepInfer参考 | 当前SGLang与H100约束 | 判断/动作 |
|---|---|---|---|
| 发射与图重放 | `worker/cuda_graph_runner.py`捕获forward与输出拷贝，固定buffer重放 | r9逐token eager，每step重复CPU遍历与大量小kernel发射；r10 full decode graph | 已去除明显配置瓶颈，收益待实际trace/计时 |
| CPU/GPU调度重叠 | Q&A描述N+1调度覆盖N的forward/sample，延后CPU后处理 | SGLang已有overlap机制，不能仅凭慢速断言没有异步 | r10核resolved disable_overlap_schedule=false，trace再看空洞/同步 |
| Engram访问 | GPU侧启动单边RDMA从分片host表取行，后台NIC搬运；scale留GPU | r9是GPU映射host页、查表后TP all_reduce；100% huge-page不等于无PCIe/通信开销 | 主机查表/collective是候选，不能未profile就定为主因 |
| Engram驻留 | ENP独立并行，面向大规模部署 | 两层各约24,171MiB/rank，合计约48GiB/rank；其余权重37.99GiB/rank | 全GPU驻留超80GiB，不能直接切GPU placement |
| MoE路径 | SM100 MegaMoE融合dispatch/GEMM/combine | H100 SM90走当前FlashInfer CUTLASS W4A16，非原生Blackwell FP4，另有TP/EP通信 | 硬件与kernel路径不同；不套用SM100专用MegaMoE |
| draft/verify/accept | DSpark图与采样融合，confidence动态裁剪verify长度 | 当前固定gamma5，已有draft graph、采样accept graph与deferred commit实现 | 先验实际命中与trace；不改gamma口径或按test调flag |
| 长prefill重复计算 | SWA bounded replay、KV系统 | 原禁前缀缓存且禁decoder bounded replay，完整约56K输入 | 此次仅开启已支持前缀缓存；bounded replay另需语义验收，不混进本轮修改 |

DeepInfer公开吞吐使用SM100与EP64等部署；H100单组8卡/单请求的latency不能直接和其高并发总吞吐比较。单并发是延迟/接受率实验边界，但会放大小batch的调度和通信成本。

## 本轮如何验证

r10只改缓存与decode graph配置，固定原输入、EOS、模型、后端、拓扑、gamma和顺序。target-only验证请求prefill后采8个forward步CPU/GPU trace（64token真实EOS上限），不计入test吞吐；原始trace私密，只公开kernel与CUDA runtime聚合、无tensor值/输入输出。要求实际decode graph replay证据，并逐条保存缓存命中与累计指标。尚未获得新GPU profile时，不把上述候选写成确定kernel根因或已实现提速。

参考：[DeepInfer Q&A](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)、[CUDA graph runner](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cuda_graph_runner.py)、[SGLang graph配置](https://github.com/sgl-project/sglang/blob/38d9e4a3fe4a493b0ae468477a274e83defda81a/python/sglang/srt/arg_groups/cuda_graph_hook.py)、[Engram实现](https://github.com/sgl-project/sglang/blob/38d9e4a3fe4a493b0ae468477a274e83defda81a/python/sglang/srt/layers/engram.py)。
