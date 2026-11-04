# Flash-1017 / official dSpark baseline for NorthGate

用户授权部署 baseline dSpark + V4.1 Flash、多卡并发布 NorthGate；最新要求先核对 recipe。
已核对官方 cookbook、生成配置、Agentic 原始脚本和模型卡。结论及引用在 RECIPE-REVIEW.md。
这是一套独立线上配置；不得修改、重启或复用 test-r4 的运行身份、冻结协议、结果目录或 controller。
不得启动任何训练阶段、删除原 checkpoint 或停止别人的资源。

本地根 /Users/shom/Downloads/dspark-v41-northgate-20261104-r1；远端根
/gpfs/users/linlizhi/dspark-v41-northgate-20261104-r1。SSH linlizhi-bp-dev。
Northjob project WeaverRSI、queue proj-32，提交前核对当前 partition 和节点配置。
禁止重试已 403 的 pods/exec/events，或已 401 的 get_dispatch_nodes。

计划两个独立 TP8/EP8 worker，共 16H100。先提交 replica-a/node205，通过实际预检、启动及
API 验收后才能提交 replica-b/node207；每个 alias 独占 submit-intent，只提交一次，结果不确定绝不重提。
节点 API 的 generic GPU resource 不能证明是 H100；init 必须实际检查8H100后才导入镜像。
仅候选 node205/207、node_shcheduleable=true，不修改标签、不绕过专属限制。
节点选择需要提交时重新核查；已占用则正常排队，不抢占别人。

固定镜像 c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa；
固定 SGLang38d9e4a3fe4a493b0ae468477a274e83defda81a，核查7项源码SHA与实际版本。
官方模型 snapshot1bae0e1088993b9b82de08f59de5be5a34e42170、官方 bundled DSPARK，
不加载训练 step25/118。复用官方完整下载SHA回执并复核48shard当前尺寸；不能称为本轮重新全量hash。
模型独立 read-only mount。init复用16.645GB已验OCI archive并重新完整hash，验证manifest/config/rootfs/CRI。
runtime socket、host ctr/crictl/snapshotter仅init可见，无privileged、无prune/其他镜像删除。
regular serving只可见本任务、只读官方模型、shm，以UID1006运行。

每worker64CPU/768Gi host RAM/8H100；CPU Engram、memory fraction0.7、DSPARK5、graphs cap64、
chunk4096、max-running8、128K context、prefix cache，reasoning/tool parser auto。
无模拟接受长度或TP同步削减。后端自动解析且必须记录实际值；不强制旧Marlin/Blackwell后端。
真实EOS、请求控制生成参数。7天作业租期是资源上限，尚不是永久服务承诺。
decoder SWA bounded replay不声称与fullprefill数值等价；旧AR/verify greedy差异未定位，lossless=false。

所有配置/脚本格式化后冻结并上传；execution-sha256.json校验，运行中不能修改。
独立cache，不读写旧root私有cache；本任务private/worker-key只由SSH用户创建0600，禁止打印/提交。
launch_private.py只在内存组装key，OS cmdline/YAML无key；父进程对server输出移除实际key。
server-info写出前脱敏。不得把key、原始输入/生成文本、模型权重或checkpoint提交GitHub。

worker先环境/下载身份/源SHA检查，记录environment.json及可用router源码/help；启动server并检查
认证、models、chat、SSE，smoke通过仅到worker_ready。它不代表全部上线验收或NorthGate上架。
上线前还需验证真实reasoning、tool调用及tool结果回填（stream/nonstream）、EOS/finish/spec计数、
有限并发/长prefix和实际router认证/前缀亲和；通过后再注册NorthGate并实际端到端调用。
不把CPU回归/Running/简单smoke称为完成。不预报加速倍数或1M上下文验收。

Northjob平台已验证 Service接口：GET /api/v1/service/list，POST /api/v1/service/create；
由当前前端cluster-setting的实际路由核对。NodePort无需hostNetwork。
服务应只选择本任务独特label，不碰8个既有Service；NorthGate模型名deepseek-v4.1-flash-dspark-baseline，
group token-bp-self，已有用户自助onboarding权限。保持现有Nemotron channel6245。
NorthGate不是NorthJetty，不要混用。部署成功后才可POST /api/onboarding/model，不能把check dry-run当健康验收。

CPU-only probe曾一次提交到node040（该节点镜像缓存已验），若仍Pending且无worker输出，可留证后精确取消；
其检查已纳入实际worker预检，不原样重提probe。别碰node040的test-r4或他人任务。

监控读取replica-*/phase.json、environment.json、image-import/result.json、smoke.json、failure.json、
worker.log和server.log，并按精确alias API核实。正常warmup/排队不重复通知。
首次运行错误必须停止同类动态重试；按AGENTS完整静态核查配置/契约/调用链/同类实现，批量回归后才用新attempt验证。
worker失败或租期终态先保存实际receipt，再核对namespace/creator/name/alias精确删除，确认alias空。
不能因controller不存活原样重提；没有自动提交重试。

临时脚本及后续修复必须必要检查后commit/push NexSpecTrainer refs/heads/main并API SHA/ancestry验证。
No force push。健康运行时保持安静，仅新阶段/失败/资源问题/完成或需决策通知。
GPFS保留850GB headroom。当前尚未创建NorthGate渠道或NodePort服务。
