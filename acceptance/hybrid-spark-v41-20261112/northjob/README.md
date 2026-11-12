# Northjob 单 GPU 功能验收

首次提交通过：`gpu_passed=true`，退出码 `0`；运行时间为北京时间 2026-11-12 17:18:31–17:19:18。此报告验证 request-local n-gram、链选择、预算分配和 GPU 历史写入模块；完整 V4.1 模型 serving、SGLang 接线及业务吞吐尚需单独验收。

- alias：`hybrid-spark-kernel-20261112-northjob-r1`
- job ID：`job-5357e3835c5f-20261112171747`
- 项目 / queue：`WeaverRSI` / `proj-32`，1 GPU / 8 CPU / 32 GiB。
- 镜像：`harbor.local.clusters/bp/weaver-trainer-megatron-bridge:v1.14.92@sha256:cba116d15f4be378c467a5d15157b69e3307e21c3c67342a06481f8f4a1187df`。
- 实际环境：NVIDIA H100 80GB HBM3，SM90，driver 535.154.05；Python 3.12.3，torch `2.10.0a0+b558c986e8.nv25.11`，Triton 3.6.0，CUDA 13.0；protobuf 6.33.6 / python backend。
- GPFS：`/gpfs/users/linlizhi/hybrid-spark-kernel-20261112-northjob-r1/`，日志与结果已复制回本目录。

固定源码为 `3c6626db72e6688943457a38043de4102f73a747`，仅上传 `specforge/__init__.py`、`specforge/hybrid/` 和 `tests/gpu/hybrid_spark_acceptance.py`。归档 SHA-256 为 `02b783ec641300c2006a5e3f6a7e1a17ef7096331f493daabb3ac43231cef8c7`，文件指纹见 `source-manifest.json`；本地不保留 tar 二进制。执行命令为：

```sh
python3 tests/gpu/hybrid_spark_acceptance.py --output /gpfs/users/linlizhi/hybrid-spark-kernel-20261112-northjob-r1/kernel-results.json
```

验收覆盖：

- 135 组扫描输入分别使用 int32 / int64 长度，共 270 次与独立 Python brute-force oracle 逐字段对拍，覆盖 tokens、lengths、match_len、agreement、copy_len；包含固定边界、128 组随机输入、8K/32K/64K context、batch129、无效 slot/length、隐藏未来 padding。
- 12 次修改输入后的 CUDA Graph replay，分别覆盖两种长度类型；其中包含校准估计对独立 oracle 的检查。
- 54 组链选择配置，覆盖 batch1/4/17、hybrid/mtp/ngram、contains 两值和价格 0/0.25/1；每组验证 6 个预算，另验证 7 个完全相同得分的预算边界。
- 4 次 GPU history-write graph replay，覆盖 slot 复用、无效 slot、截断与容量边界。
- 本次 harness 总耗时 45.661 秒，含首次编译和功能验证。

6 个 timing 点（context8K/32K/64K × batch1/4）均在编译与 warmup 后用 CUDA event 测量 eager scan 调用，50 次平均约 0.0646–0.0661 ms。计时范围含 eager 调用与分配，不能解释为纯 kernel 时延或整模型加速比，也未与 YiCloud 耗时混合。

与最新 `f5e7164e984289631845e13240a6bd893b891b4e` 的比较：8 个归档文件字节相同，包含 `ngram.py`、`kernels.py`、`selection.py`、`calibration.py` 及验收脚本。`config.py` 新增固定预算优先级计算；`sglang_runtime.py` 调用该计算，并让固定预算运行使用配置价格。新版 `HybridConfig.extra_budget` 的 4 项 CPU 断言另存于 `cpu-extra-budget.json`，全部通过；这两处新版运行时接线不属于本次固定 3c GPU gate 的结论。

| 文件 | 与最新比较 | SHA-256（gate；有变化时为 gate → latest） |
| --- | --- | --- |
| `specforge/__init__.py` | 相同 | `75fdd124964a31d3e130ae97fe84fa282bef85d93005278468722b1164a5f9da` |
| `specforge/hybrid/__init__.py` | 相同 | `d68436894bca14eb97ba0d911bc3a023d6a52d7dde2585b20b228b9948ad14ab` |
| `specforge/hybrid/calibration.py` | 相同 | `8f2ced37f78f2fb700691ca4a3fdd0dd863eb13121335a7bbb742daa6deabad6` |
| `specforge/hybrid/config.py` | 有变化 | `88ff8c5300812dc715b6675cf259afa5b5752716700fe51c218d2ee0fe8af707 → cc04a195026b2833e82746370b9f25fe52b6d10321ef8b618fda841ef5ac67a8` |
| `specforge/hybrid/kernels.py` | 相同 | `b4c810c22b8871b32fb17df6524efce512b0061f62ed3dc15b92b5d03a90bf00` |
| `specforge/hybrid/ngram.py` | 相同 | `af861dcdde9492ffbde59dcace30531a3ac1b845d1ec7a625107b81cd2022844` |
| `specforge/hybrid/runtime_patch.py` | 相同 | `cbd2cf1d95877ed6106edc0698ba55573031ae8f0eb72068c55c3f8c4fe1f975` |
| `specforge/hybrid/selection.py` | 相同 | `44fef9af207b839da09dff476befeaeda460004cf4b6a7644f60f3744de94692` |
| `specforge/hybrid/sglang_runtime.py` | 有变化 | `24caed6c696feada12670e37fbe55924494a99e790f5be00696a93131e5512e2 → 2e83f3a428c304b6eec84b21515a0404ad96f5fdbd69322a4817bd57fff83c06` |
| `tests/gpu/hybrid_spark_acceptance.py` | 相同 | `e7ce7176d260284e161a391451fd55c367e8533826a53235267adea0973ecb09` |

提交 / 清理证据：`submit.stdout.json` 记录 API code200，request ID `ReqID-20261112-171747-397-4bdb`；`status-terminal.json` 记录 Completed；`cleanup-delete.json` 记录仅删除上述精确 alias 和 job ID；`cleanup-confirm.json` 回读 `[]`，确认自有作业已不存在。GPFS 证据保留。

核心结果见 `kernel-results.json`、`runtime.json`、`completion.json`、`worker.log`；审阅过的配置见 `job.yaml`、`run.sh` 和 `upload-receipt.json`。
