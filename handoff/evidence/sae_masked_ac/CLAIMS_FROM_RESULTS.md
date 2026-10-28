# Results to Claims

**结论：PARTIAL。存在明确的任务特定收益，但“普遍改善clean scGPT embedding或缺失输入鲁棒性”的主张不成立。**

本结论仅针对已完成的固定1000步方案。完整性状态WARN来自范围与代理指标限制；实现、数值、预测链接检查通过。语义审查属于same-family / provisional。

## 主要分类结果

下表为组内三个种子平均后，对供体/批次方向等权平均的固定八类Ridge Macro-F1（%）。

|数据|原始h clean|A clean|C clean|原始h masked|A masked|C masked|
|---|---:|---:|---:|---:|---:|---:|
|ctrl|82.754|82.721|82.669|81.398|81.885|81.739|
|stim|84.525|83.467|81.850|82.616|81.594|79.813|
|pbmc|82.676|83.145|79.130|82.420|81.994|78.324|

A的PBMC clean均值提高0.469pp，但只有1/2批次方向、2/3种子均值为正，且准确率从91.357%降至89.491%。A在Kang ctrl接近原始h，stim下降1.058pp。C的clean Ridge在三个设置中均没有平均增益。

相对相同warmup、相同续训预算的clean-MSE，A clean Ridge在ctrl/stim/PBMC分别+0.014/+0.311/+3.129pp，说明某些目标/设置之间有差异；其中一部分是恢复MSE续训造成的退化，不能自动变成超过原始h的结论。

## C的两个成立的有限结论

|数据|C表达MAE|TRAIN逐基因均值MAE|相对下降|改善组数|
|---|---:|---:|---:|---:|
|ctrl|0.350756|0.436179|19.58%|6/6|
|stim|0.361680|0.452138|20.01%|6/6|
|pbmc|0.409972|0.475337|13.75%|2/2|

这是原logCP10k尺度上的同RNA、原先被选中且阳性的遮蔽token恢复；不包括零表达、未选基因或全转录组补全。尚未用直接masked_h预测头或匹配dense-C控制证明收益必须来自SAE。

C的Kang clean 15NN Macro-F1分别提高4.907pp（ctrl）和4.381pp（stim），两条件均6/6供体组、3/3种子均值为正；masked 15NN分别提高5.379pp和3.998pp。它支持该固定近邻读出的收益。两个条件共享供体；PBMC masked 15NN变化为-0.043pp，不能据此推广为一般缺失输入鲁棒性。

## 不成立的更强解释

- **普遍鲁棒性：** A/C的masked Ridge仅在ctrl略高于原始h且低于clean-MSE；stim和PBMC平均都下降。clean-minus-masked差值缩小可以同时伴随clean性能下降，因此不是单独的成功判据。
- **A_sae成功去噪：** masked embedding归一化MSE在ctrl/stim/PBMC为0.224563/0.203776/0.272130，直接masked_h为0.140615/0.134960/0.113540；A_sae没有超过直接输入的重建保真度。
- **状态保存：** embedding-only CD14 IFN clean R²，ctrl原始h/A/C为0.324/0.116/0.045，stim为0.500/0.323/0.309。PBMC CD14仅一个合格方向，结果为负且A/C更差。其他协变量变体使用完整RNA协变量，只能作为oracle敏感性分析。
- **SAE特有机制：** A_dense初始化不同，C有额外训练头；没有匹配控制或独立功能/干预实验，不能证明稀疏性、概念因果性或新增生物学信息。
- **整个A/C思想无效：** C的42/42次运行最低监控VALID目标均在step100，A_sae为41/42次更早、A_dense为14/42次更早；固定step1000是公平预算对照，不代表最优/收敛方案。不得在本轮TEST后另挑早期checkpoint。

## 各主张判定

- **C1：no** — A/C普遍改善clean scGPT embedding的下游质量。A相对original-h clean Ridge变化为ctrl -0.033pp、stim -1.058pp、PBMC +0.469pp；PBMC仅1/2方向与2/3种子均值为正，accuracy下降。C分别-0.085、-2.675、-3.546pp。A/C的embedding-only IFN也没有普遍收益。
- **C2：partial** — A相对相同warmup和续训预算的clean-MSE存在任务特定收益。A clean Ridge相对MSE分别+0.014、+0.311、+3.129pp；masked分别-0.457、+0.510、+2.171pp。这包含从MSE退化中恢复，不能等同超过原始h；PBMC超原始h的clean均值很小且方向/种子不一致。
- **C3：yes** — C通过512D输出预测本协议中的masked-positive表达，优于TRAIN逐基因均值。原logCP10k MAE：ctrl0.350756 vs0.436179、stim0.361680 vs0.452138、PBMC0.409972 vs0.475337；组均值相对降幅19.58%、20.01%、13.75%，14/14组方向为正。仅为同RNA阳性token恢复，缺少直接masked_h线性头/容量匹配C控制以证明SAE必要性。
- **C4：yes** — C改善Kang的固定centered-cosine 15NN标签读出。clean 15NN相对h：ctrl+4.907pp、stim+4.381pp，两条件均6/6供体组、3/3种子均值为正；masked+5.379、+3.998pp。ctrl/stim共享供体，PBMC masked为-0.043pp且方向混合；不推出一般鲁棒性或Ridge/状态保真度改善。
- **C5：no** — A/C提供普遍的missing-input robustness。A/C masked Ridge在ctrl分别+0.487/+0.341pp，但都低于clean-MSE；stim为-1.022/-2.804pp，PBMC为-0.426/-4.096pp。较小clean-minus-masked差值可能伴随clean性能下降；必须同时看绝对masked性能。只有一种固定遮蔽条件。
- **C6：no** — A_sae相对直接使用masked_h改善clean-h重建。masked normalized embedding MSE A_sae vs直接masked_h：ctrl0.224563 vs0.140615、stim0.203776 vs0.134960、PBMC0.272130 vs0.113540。相对clean-MSE在Kang略低，但不是超越直接输入的去噪收益。
- **C7：no** — 这些收益证明SAE特有生物学机制、独立新信息或因果功能。没有等初始化/等训练头控制、独立功能或干预证据。A/几何是模型教师保真度，C/IFN是同RNA代理；冻结输入的确定性重映射不能凭这些指标被命名为新的因果生物学模块。

## 推荐采用的表述

在历史Kang/PBMC的固定1000步、单一25%阳性token遮蔽协议中，C降低了遮蔽表达恢复误差，并改善了Kang固定15NN读出；A/C未显示跨条件、跨读出的普遍clean-embedding质量或缺失输入鲁棒性提升。

## 后续处理

关闭本轮固定实验并保留完整结果，不继续依据这些TEST挑模型或叠加目标。若未来需要更强结论，再预注册独立生物学测试集、直接masked-h/C-head与匹配dense-C对照、多遮蔽/真实深度测试及VALID停止规则。该建议不授权本轮继续训练。

证据：`summary.csv`、`planned_contrasts.csv`、`seed_metrics.csv`、`expression_summary.csv`、`IFN_summary.csv`、`geometry_summary.csv`、`verification.json`；完整输入哈希和结构化判定见同名JSON。
