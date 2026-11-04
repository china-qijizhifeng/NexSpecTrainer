# dSpark test r4: EOS-aware generation

用户明确选择“按 EOS 停止、最多 1024 tokens（推荐，更贴近实际生成）”。
原r3固定长度A/A仍为失败历史，不改写；没有test输出，所有test输入与实例划分原样复用。
本地/远端根 /Users/shom/Downloads/dspark-test-20261104-r4 与
/gpfs/users/linlizhi/dspark-test-20261104-r4。

唯一新alias denovo-dspark-test-eval-20261104-r4，WeaverRSI/proj-32，8H100/node040。
复用r3已注入并验收的固定serving digest c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa，
imagePullPolicy Never，不再次导入；无init/runtime socket/privileged挂载，只GPFS/shm。
source/export权重/receipt复用r2，输入SHA不变，不重导出/采集。worker在server之前全量hash两份overlay。
独立r4 cache由worker只读复制r3编译缓存，跳过symlink/不可读项，不修改旧权限或旧cache。

本轮唯一采样语义变更：ignore_eos=False，官方EOS ID1真实服务端结束，最多1024token；
预算也是10条validation A/A与四组test的相同上限，warmup单独上限32。服务后端/TP8EP8/gamma5/
单并发/eager/禁radix不变，不添加不支持dsv4的全局deterministic flag，不称数值根因已修复。
原r3长validation输入的EOS前61token额外与冻结历史reference核对，完整10条target-only A/A及
finish reason必须相同，然后才执行test247；任一A/A/运行/计数错误立即停止，无自动重试。
官方MTP、step25（主候选由train-validation预选）、step118各自先诊断，再test247；
有差异仅记录预声明common-prefix探针，不反挑模型、不调test，不称无损或输出等价加速。

返回长度/SSE/meta必须一致，EOS只允许位于最后一个返回token，正常上限结束则必须为length1024。
吞吐量按实际返回token数，TPOT按首个有token chunk之后实际新增token平均间隔（含网络）；
单token或仅一个有token chunk则TPOT=null，不编造逐token时间。保留raw SSE及原生raw spec计数。
verify0的prefill即EOS请求允许无spec指标；tau=sum completion/sum verify；rate=sum correct/sum proposed，
gamma5 proposed=verify*5；最终verify block可能计算EOS后草稿，raw correct不等于实际返回的有效草稿数。
按instance与verify加权汇总，4*247请求、冻结inputSHA、EOS/finite指标最终验收。

test36实例247轨迹13998624input tokens，每条完整历史<=64512，176条原始长轨迹只是较早完整前缀；
不是完整长轨迹或闭环SWE解决率。训练r6已完成，2862长train仍deferred，不启动训练、不删ckpt或依赖。
850GB GPFS headroom。Northjob只能WeaverRSI/proj-32；禁止403 pods/exec/events及停止他人资源。

单次controller：--check只读；launch_controller.py独占创建launch-intent，controller-started和submit-intent
也独占创建，任何提交不确定都禁止重提。已有intent后不得重启本attempt。终态先留receipt再按精确identity
删除本job，确认alias空。monitor.py只读SSH/API并保存本地snapshot；终态controller_verified=false正常。

新源码/协议/回归发布acceptance/dspark-stage1-test-20261104/test-r4并push GitHub main，读回API SHA及祖先核验。
运行脚本先按仓库格式化后再冻结上传，发布字节与实际执行一致；不提交原始输入、reference输出或权重。
不能把CPU EOS契约回归说成GPU数值稳定或test成绩。真实镜像的A/A gate和测评分数另行取证。
