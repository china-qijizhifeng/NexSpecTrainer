# r8 提交身份错误：已撤销

新 r8 目录提交时误沿用 r7 alias，精确新 job 为 `job-d5836fe55d5f-20261104214902`。错误来自准备脚本只替换目录路径，未匹配独立的 `denovo-dspark-test-eval` 命名格式。

已用 PID/start_ticks/精确命令与 Linux pidfd 核对并停止本次 controller，随后仅删除该精确新 job，API 确认 alias 为空。worker 未启动，held-out 请求为 0。原 r7 的历史 job 未重启。

r8 的 869 项冻结 manifest 和文件不改写，不再运行 r8 任一启动入口。完整静态检查 YAML name、JobAlias、controller ALIASES、worker/init 命令、task volume、init mount、API alias/namespace/creator/GPU 字段后，在独立 r9 中补齐 11 种错误变体拒绝回归及完整 CPU 批量检查，再允许单次提交。没有原地重启或自动重试。
