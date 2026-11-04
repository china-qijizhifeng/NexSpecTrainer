# huangyuqi Weaver 旧 checkpoint 清理

按用户“清理一下weaver huangyuqi 一些老 ckpt”的授权，通过 huangyuqi 自身 Weaver API 删除，未直接移除存储文件或修改数据库。

共删除 9 份已被后续保存点替代的 checkpoint，manifest 数据合计 1,187,640,592,736 bytes（约 1.19 TB）。清理后逐项确认 API deletion_state=deleted、cleanup_pending=false，9 个目录均无剩余 payload 文件。每个模型、每代训练保留最近两份有效保存点；保留所有在用引用、恢复源、派生源、slot-home、未完成保存及非 huangyuqi checkpoint。NexSpecTrainer 的 step25/118、测评输入与服务副本均保留。

先删除 10 月 29 日已 sealed 模型的 epoch1，保留 epoch2/3，恢复数据库连接能力；随后从数据库只读核实候选无引用，通过 API 删除 8 份。清理后一次文件系统观测 free=1,187,659,120,640 bytes。随后共享盘又出现额外回收，不计为本次释放量。检查证据见 cleanup-result.json、delete-receipts.json、bootstrap-delete-summary.json、deletion-plan.json、filesystem-after.json。

独立测评 r4 从未启动 worker。核实旧 controller 已退出、作业仍 Pending/no-worker 后，精确取消 r4 并核 alias 空；r5 在 node090 单次提交，保持原 EOS/1024、四组及冻结输入。当前提交不代表 GPU A/A 或 test 通过，后续以实际 init、worker 和结果审计为准。
