# BP Dev 与 NorthGate 接入

## 平台与身份

SSH `linlizhi-bp-dev`；Northjob project=`WeaverRSI`、Volcano queue=`proj-32`。提交前读取当前 partition，核对 enabled/sync_state、配额与节点限制。平台的通用 GPU 资源和空闲卡总数不能代替完整节点、CPU/主存、卡型及镜像可用性的核实；实际卡型在已分配 worker 内检查。

Northjob API 历史可用入口为 `http://10.1.2.7:31000`，从 SSH 主机请求并绕过代理。凭据只在远端 `~/.config/northjob/userinfo.conf` 读取；`POST /api/auth/get_token` 得到短期令牌后使用 Bearer 认证，不打印、复制或持久化令牌。可用时复用 `northjob-gpu-test-bp-dev` skill 的 helper；也可审查仓库已有纯 API helper 后使用，不能为调用 helper 而启动历史 controller。

已核对的路由：

| 用途 | 路由 |
| --- | --- |
| 项目/队列 | `GET /api/partitions` |
| 作业列表 | `GET /api/get_jobs`；客户端精确过滤 alias/namespace/creator |
| 提交 YAML | `POST /api/task/create_job_by_yaml`，body为 `{"code": YAML正文}` |
| 删除作业 | `DELETE /api/delete_jobs`，body为精确 `name`/`namespace` |
| 节点 | `GET /api/k8s/nodes/get_nodes`；已见响应顶层 `nodes`，没有 `data` 包装 |
| Service | `GET /api/v1/service/list`、`GET /api/v1/service/detail`、`POST /api/v1/service/create`、`DELETE /api/v1/service/delete` |

Service 路由由该环境前端 `cluster-setting` 源码核对；新环境先核对当前契约。`get_dispatch_nodes` 已返回401，pods/exec/events 曾403，不反复尝试这些接口。默认 kubectl 曾 Unauthorized，使用已授权的平台入口，不能通过其他凭据绕过权限。

使用新 run root、独立 alias/输出/cache，并在提交前独占创建 submit-intent。一次提交后保存 response 与精确 job identity；不确定时只读查询。历史 `job.json` 的 Pending 是提交记录，当前状态以 API 为准；Pending 也可能包含 init 执行。保留已有训练、测评、checkpoint 和活跃依赖；GPFS 按用户要求保留 headroom，历史任务约定850GB。

## 镜像导入与运行权限

先查同 digest 的可用缓存；`imagePullPolicy: Never` 只适用于已证实缓存存在，或同一 job 的受控 init 已完成导入的情况。

确需离线导入时，重新验证 OCI archive 完整 SHA、字节数、manifest/config digest、rootfs diff IDs、实际节点 snapshotter root 和空闲盘。参考流程用 `ctr --local images import`，随后注册 CRI 并用 `inspecti` 核对 exact image identity；API Running 不能替代这些收据。

runtime socket、host ctr/crictl/snapshotter 仅 init 可见，无 privileged；只导入目标镜像，不 prune 或删除其他镜像。主 serving 容器只挂任务输出/私密配置、只读模型和 shm，以适配镜像的非 root UID 运行。不要把其他任务的缓存权限改成可读；独立 cache 可复制原本可读项并跳过 symlink/不可读项，必要时自行编译。

参考环境使用的 OCI archive 约16.645GB、Weaver init digest 与挂载路径保存在 [归档 import 实现](../../../../acceptance/dspark-stage1-test-20261104/serving-baseline/import_worker_image.py)。这些是版本化样例，不应假设未来节点还使用相同 snapshotter 路径。

## 服务入口与认证

用独特 task/replica 标签选择本任务容器，先列出现有 Services，创建自己的 NodePort Service 并由平台分配端口；无需为此启用 hostNetwork。不要复用或改动别人的 Service/端口。实际 Service selector、端口、pod 健康和 NorthGate 到上游的连通性都要验收。

多个 worker 优先使用已核对版本的 SGLang router 和 cache-aware/session affinity。检查实际安装的 help、worker 注册、worker key、router key 与健康探针契约；较新文档不保证旧镜像内安装了兼容 router。保持健康检查，不复制 benchmark 的禁健康检查/盲重试设置；缺少 router 时先选择固定版本并验证，不临时安装 latest 或编写未验收的流式代理。

worker/router key 从模式0600的私密文件读取，并限制其父目录权限。SGLang 参考版本的 args repr、launch command 或 server-info 可能暴露 api_key，不能只隐藏 YAML 就认为已脱敏。归档的 [私密参数 launcher](../../../../acceptance/dspark-stage1-test-20261104/serving-baseline/launch_private.py) 与 [worker 日志/收据脱敏](../../../../acceptance/dspark-stage1-test-20261104/serving-baseline/worker.py) 提供样例；使用时适配新的 root、UID 与 profile，不照抄运行身份。

## NorthGate 上架

NorthGate 是模型网关；NorthJetty 是另一套代理/服务目录。用户要求 NorthGate 时使用其模型接入契约。

先阅读所用 NorthGate checkout 的 `CLAUDE.md`，路由/DTO 查询从 `docs/api-reference.json` 开始。历史环境为 `https://northgate.xiaobei.top`，已验证普通用户 onboarding 可用，管理员 channel API 不可用；不为上架另找管理员凭据。

历史本机 credential store 为 `~/Library/Application Support/Northgate Monitor/config.json`，`api_key` 是接入 access token。只在请求进程内加载；核对当前用户身份，按该版本契约使用 `Authorization` 和 `X-NorthGate-User`，不将某次部署的用户ID写成通用默认值。

- `GET /api/onboarding/models`：确认本用户已有目标，保留无关模型。
- `POST /api/onboarding/check`：检查接入权限、分组等前提；不检查上游健康。
- `POST /api/onboarding/model`：创建或更新本用户目标模型，必须已有可达且完成验收的上游。

参考请求字段如下；endpoint/key 来自本次实际部署，不能把占位符提交出去：

```json
{
  "model_name": "deepseek-v4.1-flash-dspark-baseline",
  "endpoint": "http://VERIFIED_HOST:ALLOCATED_NODEPORT",
  "api_key": "LOAD_FROM_PRIVATE_STORE",
  "type": "openai",
  "groups": ["token-bp-self"],
  "upstream_model_name": "deepseek-v4.1-flash-dspark-baseline"
}
```

`token-bp-self` 是该参考环境已验证分组，新部署按当前授权选择。不要覆盖同账号无关渠道。发布后通过 NorthGate 实际调用并核对 `/v1/models`、chat/SSE、reasoning、工具调用及工具结果往返，以及 `X-NorthGate-Model`/`X-NorthGate-Request-ID`。API接入成功和真实端到端通过分别取证，不把生成内容或凭据提交 Git。

## 生命周期与代码发布

worker 出现新错误后，先静态核查配置/契约/调用链/同类实现并批量回归，再用新 identity 验证。冻结文件或已有 submit-intent 的 attempt 不原样重启。终态先保存 receipt，再核对 alias/name/namespace/creator 精确删除本次资源并确认为空；健康服务按用户要求保留。明确服务的资源租期、上下文/输出边界与退出后的渠道处理。

本项目相关部署修复和临时运行脚本按用户授权提交至 NexSpecTrainer `refs/heads/main`，push 后核对远端 API SHA 和 commit ancestry。main 前进时 fetch 并按依赖合并/变基，不 force-push；有保护规则时走仓库门禁。格式化归档与执行文件的 SHA 分别记录，不改写历史收据，不把只存在本地或功能分支的变更称为已发布。
