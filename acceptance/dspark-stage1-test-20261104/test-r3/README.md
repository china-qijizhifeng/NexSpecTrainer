# dSpark test r3: node040 image injection

用户明确要求换节点注入镜像。r2仅排队，没有运行任何test请求；通过精确
PID/start_ticks/cmdline停止r2 controller25507，删除精确Pending作业
job-3f628626865c-20261104091717，读回alias空，记录migration-from-r2.json。
不重启旧controller，不覆盖旧receipt，不重复CPU权重导出。

当前本地 /Users/shom/Downloads/dspark-test-20261104-r3，远端
/gpfs/users/linlizhi/dspark-test-20261104-r3。
新alias denovo-dspark-test-eval-20261104-r3；一次性controller身份在
controller-process.json。started/launch/submit intent存在后绝不能重启/重提。

WeaverRSI/proj-32单pod8H100指定node040。init容器复用先前成功导入流程：
先验证8张H100、官方OCI archive的SHA/size、manifest/config身份、snapshotter
root和本地剩余盘量，然后只导入固定sglang镜像并向CRI注册其确切引用。
不删除、清理或覆盖其他镜像，不执行prune。运行时socket只挂在init容器；
regular serving容器无运行时socket，只挂GPFS和shm。无privileged模式。

固定sglang digest c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa，
config31d58781877bdcd322d5fcc7a858d427db54f5cbf5a9e6342c5fafc01bc67c6e，
archive SHA1c5de44f6bcec849bc241c0ba3b1a2702e6be7e392acb81275c8e4a0dfc8240a。
image-import/result.json只证明镜像导入，不是服务就绪或数值验收。
init失败后不会重试，先检查完整日志、静态契约和同类实现再修复。

通过symlink复用r2已验收export-step25/118、source和被冻结输入，输入及输出预算、
四组服务参数、metrics、旧greedy差异处理规则完全保持r2协议。
新run独立输出与kernel cache（可读r2缓存复制），不写旧run cache。
execution SHA包含实际执行脚本、source及依赖输入/导出receipt。
控制器仅核对root所有的overlay文件大小和导出receipt；实际worker读取并重验完整SHA，
不改变原权限，不让SSH控制器读取受限payload。CPU回归覆盖此权限边界与同size损坏拒绝。

test覆盖36实例247轨迹，各取完整历史<=64512tokens的最后合法assistant边界，
每请求1024输出、temperature0/ignore_eos、gamma5、单并发、TP8/EP8/eager。
176条原长轨迹只是较早完整前缀；不是完整长轨迹覆盖或SWE闭环成功率。
旧FP4 AR/verify greedy差异仍未定位；有限诊断矩阵和test结果分别记录，
有差异就lossless_greedy_verified=false，不声称无损加速，不用test调参。

monitor.py只读SSH、精确API并写本地snapshot；看image-import/worker.log/import.log，
eval-worker.log、phase.json及各组server.log。controller正常存活才继续等待；终态
保存证据并精确清理本job。850GBheadroom。CPUexport和训练均不重启，不删ckpt。

全部当前/后续修复须push到GitHub main并读回核验；仓库归档
acceptance/dspark-stage1-test-20261104/test-r3。保持冻结run源码、manifest和历史receipt，
发布副本可按仓库要求格式化，记录执行SHA和归档SHA及等价性。
