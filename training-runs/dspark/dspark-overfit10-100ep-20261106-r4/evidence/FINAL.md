# DSpark：10条样本、8卡、100 epoch 结果

**结论：能很好地拟合教师分布，但本次配置没有把原训练标签记到99%。** 100 epoch最终标签准确率97.1550%，低于预先设定的99%标准；训练中最好为第10 epoch的97.6221%。教师top-1一致率达到99.8301%，总loss下降91.41%。

| 指标 | 训练前 | 100 epoch |
|---|---:|---:|
| 标签准确率 | 92.3992% | 97.1550% |
| 教师top-1一致率 | 93.8004% | 99.8301% |
| CE loss | 0.249706 | 0.058878 |
| L1 loss | 0.119328 | 0.007207 |
| Confidence loss | 0.183497 | 0.014774 |
| Confidence绝对误差 | 0.051085 | 0.001421 |
| 离线hard-label walk长度 | 5.146853 | 5.468401 |
| 离线概率接受长度估计 | 5.214836 | 5.809537 |
| 总loss（0.1×CE + 0.9×L1 + confidence） | 0.315863 | 0.027148 |

5个draft位置的最终标签准确率：97.3159%, 97.1389%, 97.1691%, 97.0567%, 97.0867%。7065次目标token exposure中6864次预测正确；同一个token可能因重叠anchor被多次计入。

教师分布已接近拟合，而标签准确率在约97%停住，说明现有蒸馏目标与逐字记住原trace标签不能等同。仅由最终标签错误率2.8450%和draft/teacher分歧率0.1699%，可推得教师top-1与原标签至少约2.6752%的位置不同；这是集合关系推得的下界，不是额外运行的teacher-vs-label测量。要进一步检验严格的标签记忆能力，后续可分别做纯CE对照或按SpecForge示例使用目标模型temperature=0重新生成的10条数据；本次没有额外启动这些实验。

使用的是10个不同DeNovo trace各自首个不超过4096 token的缓存窗口，含1453个合法anchor；并非10条完整多调用trace，也没有本次重新greedy生成。模型从DeepSeek-V4.1-Flash-1017自带MTP权重开始，没有续训已有微调checkpoint。AdamW峰值1e-4、cosine200、warmup0.03，all anchors/chunk64；与前次大规模实验保持CE/L1/confidence权重。

验收通过：8个rank各200次参数更新与200微批；10条真实样本各全局消费100次；600个补齐微批严格零loss、零anchor、零target exposure；21次完整同样本评估；loss/grad有限、anchor零重复、冻结权重哈希不变。峰值显存44.32 GiB/卡。从rank启动到最后一次更新约43.8分钟。平台Completed、容器exit0，终态证据保存后仅删除本作业，8卡已释放。

最终checkpoint保留在SSH主机linlizhi-bp-dev：`/gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r4/training/dspark-overfit10-step200`。已校验step200、common.pt SHA256、NFC元数据以及32个非空分片；没有执行checkpoint恢复测试。离线walk与概率接受长度均不是SGLang线上实测接受长度；本次没有运行线上serving gate。

总loss使用全样本加性分子/分母汇总；原Evaluator的loss受各微批全局分母及padding布局影响，因此保留在eval/avg_loss_legacy，但不用于前后比较。teacher_top1_prob是教师最大概率均值，不能当作教师对标签准确率。

证据：[完整结果](RESULT.json)、[逐rank验收](final-verification.json)、[checkpoint校验](checkpoint-verification.json)、[平台终态](evidence/monitor-terminal.json)、[样本及参数](RUN.md)。

![训练曲线](overfit-curves.png)
