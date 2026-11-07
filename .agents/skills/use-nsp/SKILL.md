---
name: use-nsp
description: 使用 nsp 启动、观测、停止和恢复 NexSpecTrainer 训练，并运行独立 loss、准确率、prefix AL 或 serving AL 评测。适用于 NexSpecTrainer 训练及评测操作，不用于只读论文与超参调研。
---

# NexSpecTrainer 训练与评测

NexSpecTrainer 的用户训练和评测任务统一使用 `nsp`。先在实际执行 checkout 中读取 `docs/nsp.md`，核对 `nsp --help`、`nsp train --help` 和 `nsp eval --help`；源码未安装 console script 时使用等价的 `python -m specforge.nsp.cli`。

不要直接用 `specforge train`、自写后台 Python 或临时训练 shell 脚本绕过 Run 记录。底层训练入口只用于 `nsp` 内部及明确要求以原命令为对照的验收。若当前版本没有 nsp，先安装/切换到包含它的已验证版本；遇到 nsp 能力缺口应明确报告或修复，不静默改用旧入口。

## 执行环境

- 在已分配的 GPU 容器/节点内执行 `nsp train --foreground`，让外部调度器持有任务生命期；普通交互执行默认后台，`--watch` 只是提交后的观测。
- `--profile NAME` 是 SSH profile；配置、模型和数据路径按远端主机解释。它不自动申请 YiCloud/Northjob GPU。资源申请遵循相应平台的实际契约，作业 payload 必须调用 nsp。
- Northjob 使用 `linlizhi-bp-dev`，project=`WeaverRSI`；提交前核对项目和当前 queue，历史 queue 为 `proj-32`。YiCloud 不因使用 nsp 而绕过 pool 签名或资源授权。
- nsp 控制命令运行在分布式 launcher 外层。当前 supervisor 版本只管理单节点 trainer；多节点支持未就绪时先扩展 nsp，不能启动每个 rank 各自独立的 Run。
- 固定源码、镜像 digest、模型、数据与 split，使用独立 Run。保留运行所在 profile/root、run_id、attempt、调度器 job identity 和产物路径；不要把提交成功、容器 Running 当作训练完成。

## 标准流程

```bash
nsp --root /shared/nsp train -c train.yaml --plan
nsp --root /shared/nsp train -c train.yaml --name experiment --request-id UNIQUE_ID --watch
nsp --root /shared/nsp list --kind train
nsp --root /shared/nsp show RUN --json
nsp --root /shared/nsp watch RUN --until terminal
nsp --root /shared/nsp logs RUN --follow
nsp --root /shared/nsp checkpoints RUN --json
```

- 提交前先 list/show，避免重复创建。网络超时以相同 `--request-id` 和相同配置重试或查询，不能重新生成 ID 盲目提交。
- watch 的 Ctrl-C 只退出观测；停止任务用 `nsp stop RUN`，继续观测到 cancelled。`--foreground` 的 Ctrl-C/TERM 请求停止并等待清理。
- 恢复使用 `nsp train --resume RUN@step-N`，在原 Run 下创建新 attempt，恢复完整状态。改变 LR/objective/data 的实验使用新 Run 与 recipe 中的 `model.draft_checkpoint_path`，不能冒充 full resume。
- 首次动态失败后暂停同类重试，先检查配置、契约和同类调用链，修复后再验证。

## 评测口径

```bash
nsp --root /shared/nsp eval RUN@step-100 -c eval.yaml \
  --metrics loss,accuracy,prefix-al --group-by ctx-len \
  --ctx-bins '0,4k,8k,16k,32k,64k,128k,inf' --watch
nsp --root /shared/nsp results EVAL_RUN --json
```

- eval 配置包含训练 `recipe`、验证集 `features`；原始完整 feature record 使用 `ctx_origin: record-start`。切片数据不能谎报起点；分 bin 前先保存真实 prefix 关系或扩展 reader。
- `loss` 使用固定 objective，`accuracy` 记录算法预测/reference 口径；`prefix-al` 是离线 `1 + 连续命中`，`al` 是 serving 实测输出 token / verify cycle。禁止把 simulated_acc_len 或 request 平均 AL 伪装为动态 ctx AL。
- overall 始终保留；空 bin 为 null，报告分母、覆盖范围和排除数量。固定 checkpoint，不能在评测期间跟随漂移的 latest。
- serving AL 使用独立匹配该 checkpoint 的 endpoint。动态 ctx bins 需要逐轮 verify telemetry；仅有请求级计数时只能报告 overall，不能自行推测分 bin。
- 按当前 `docs/nsp.md` 能力表检查算法、objective、分布式与 telemetry；不支持的选择应在提交前发现。

## 完成条件

检查真实 optimizer step、有限 loss、最终 completed checkpoint 与 worker completion receipt。评测检查每个所选阶段 complete、overall/bin 的计数和数据版本。恢复已验证、全量覆盖、训练收敛、serving 质量分别报告；短 pilot 不代表全部达标。任务结束后只清理本次分配的资源，保留 Run 回执、日志和产物。
