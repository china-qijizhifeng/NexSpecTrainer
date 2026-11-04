r7 在正式 test 前的 target-only A/A 检查失败，已停止并精确删除作业，API 确认 alias 不存在。四组 test 请求均为 0，整套测评没有完成。

前两条旧 prompt 和第一条 validation 的 A/A 一致；第一条 validation 同时准确复现 r3 冻结的 61-token EOS 输出。第二条 validation 使用同一 50,527-token 输入（SHA 38db48ae0a9e0cd58315b31edae50f04631466b0557d3d21d26ee7d6ba009296），两次在 zero-based19，即第20个返回token处分歧。输出分别为669与403tokens，两次均以EOS1结束，首个EOS分别在668与402。分歧发生在EOS之前，修复EOS终止协议并未解决服务端重复输出不一致的问题。

原始SSE逐条重放与保存结果一致，计数/EOS契约通过；两次cached_tokens与num_retractions均为0。已排除本次客户端拼包和EOS后强制续写解释。未采集逐步logits、稀疏索引或层级快照，因此不能确认数值kernel根因。静态核对已覆盖greedy采样、dsv4/SM90MoE、归约、topk、缓存与调用链；global deterministic参数仍不支持dsv4，MoE fused finalize默认已关闭，不能盲目切换或称已修复。

已保留原始输入、输出与SSE在私有任务目录，公开归档仅含SHA、计数、失败与清理回执。后续先准备有界的validation-only数值定位、完整静态核查与CPU回归，再考虑新诊断attempt；不自动重提r7，不绕过10条A/A门禁，也不改test选择。
