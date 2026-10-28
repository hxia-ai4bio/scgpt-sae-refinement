# Findings

2026-10-19：结果审查为 **claim_supported=no；integrity_status=warn；same-family/provisional**。

在固定K32、六供体轮换、三耦合种子、CE训练头选checkpoint后以新Ridge选beta/identity的方案中，SAE Gate相对原始表征−0.001983百分点；16/18回退identity、0/6正向供体，未超过旋转控制。预设继续门槛未达成。378候选均有记录到的梯度流，但262个选回epoch0；Gate保留残差极小。因此这是一项有完整执行证据的配方级阴性结果，不是对所有SAE优化方法的否定。

已修复或澄清：TEST前重启未绑定的部分开发运行并加入源/协议/checkpoint/输出hash；明确两类参数匹配；导出类别名称。kNN性能后端替换通过原预测与等距规则一致性检查。未发现未修复实质数值错误；矩阵乘法告警和范围限制保留于审查。

Routing：完成归档；停止本轮扩展；不推进正向增益或生物机制主张。未来如另行开展，需TRAIN/VALID预注册新优化假设及新供体/研究确认，不在这些既有TEST结果上继续调参。
