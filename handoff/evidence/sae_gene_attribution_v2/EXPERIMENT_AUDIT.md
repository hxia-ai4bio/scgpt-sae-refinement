# 实验完整性审计：F160 多参照基因归因

日期：2026-10-18。审阅者：GPT-6-Astra / ultra，fresh agent（fork_turns=none）。**same-family / provisional**；这不是跨模型家族的独立认证。

**Overall verdict：WARN；Integrity status：warn。** 当前局部归因任务可交付。未发现伪造基因真值、用自身最大值美化分数、虚构结果文件或汇总计算错误。WARN来自选择范围、参照依赖和技术状态差异，不代表发现了结果造假。

## A. 真值与参照来源：PASS

细胞类型是数据集注释；F160与对照方向由TRAIN标签筛选，不能将后续marker一致性当作独立的生物真值。基因替换值来自真实同供体、同类型细胞的分箱值。这里没有实验提供的gene-effect ground truth，评估类型为 **self_supervised_proxy / model_input_sensitivity**。

证据：`work/sae_interpretability/01_atlas.py:48–76,92–104`；`work/sae_gene_attribution_v2/run.py:24–36`；`work/sae_interpretability/02_forward.py:43–52`。

## B. 分数归一化：PASS

归因为原始坐标的“原型值−替换后值”，未除以自身最大值或范围。表达差异对照使用library-size log归一化，它是输入预处理。不同输出坐标的绝对大小不能用来排名可解释性；在同一F160输出上比较不同基因排序的响应则是定义明确的局部比较。

证据：`run.py:15–18,53–56`；`analyze.py:59–66,97–104`。上述短文件名均位于`work/sae_gene_attribution_v2/`。

## C. 文件存在与数字核验：PASS

已完成34对、13,570次有效单基因替换；组合阶段12对、1,152个输入（含空替换），其中960个非空输入。训练与测试分别覆盖1,838与1,166个不同token基因。

审阅者另写代码，未导入实验的run.py/Forward：直接从HDF5重建token和分箱，重选34对，检查来源与缓存绑定，重算原始4坐标，并独立复原全部1,152个组合输入及120条完整随机序列。全基因汇总、覆盖、条件均值、供体SD、per-prototype输出、TRAIN-top40迁移和删除汇总均一致。原始4坐标最大误差`7.15e-7`；下游汇总误差约`1e-16`。

**核验边界**：未独立重跑每一次完整Transformer扰动前向。生产脚本另做34次单基因和12次组合的快/慢后端回放，误差分别≤`1.67e-6`、`1.14e-6`。这些是数值一致性检查，不是生物验证。

证据：`occlusion_completion.json`、`analysis_completion.json`、`verification.json`；审阅者证据为`work/sae_gene_attribution_v2/review_trace/auditor_verify.py`与`auditor_verification.json`。

## D. 实际调用与保存：PASS

`run.py:45–65`执行单基因前向；`analyze.py:107–110`调用汇总与组合替换。参照敏感性、供体一致性、迁移、核糖体前缀汇总及删除曲线均有实际CSV输出，报告读取对应文件。不存在只定义未运行却作为结果引用的指标。

## E. 实验范围、选择与独立性：WARN

- 只有一个历史模型（seed42、K32）和一个预先选定特征F160。TRAIN为4供体各2个高激活B细胞，共22对；TEST为1个历史供体的4个高激活B细胞，共12对。
- TEST的3个参考细胞各重复使用4次；这些配对不是12个独立生物重复。历史TEST已经分析过，不能叫新盲测。
- 训练原型/参考的平均检出基因数约431/772；测试约444.5/1035。原型和参考的表达复杂度并未匹配。
- 随机对照匹配原始bin、改变方向与幅度的**分箱**，不是原始数值完全匹配。每40个随机基因与SAE前40重叠10–22个，均值16.69。此null只针对SAE排序；不能把它当作非核糖体排序的专属对照。

证据：`run.py:25–36`；`analyze.py:74–85,101–105`；`pairs.json`、`pair_qc_donor_summary.csv`、`matched_random_diagnostics.csv`。

## F. 结果支持到哪里：WARN（结论范围）

TRAIN前列包括YBX1、TPT1、RPS24、CD74、RPLP2；前40中22个有RPL/RPS前缀。TRAIN供体两两归因Spearman中位数为0.2823；换参照的top20 Jaccard中位数为TRAIN0.2311、TEST0.1765。因此可报告有条件的敏感性排序，不能声称获得稳定基因模块。

固定TRAIN排序后，测试替换40个基因使F160平均下降 **0.248289**，联合分箱随机为 **0.135295**，表达差异排序为 **0.139317**；4个测试原型都显示SAE排序相对所用随机对照更大的平均下降。它支持这一组选定输入上的局部排序迁移，不构成人群显著性或普遍优势。

**全部单基因替换和960个非空组合替换中，F160都没有退出TopK；SAE与pre-TopK数值完全一致。** 不能把本轮效果归因于稀疏门控跳变或声称TopK带来归因优势。

证据：`gene_rankings_train.csv`、`train_donor_consistency.csv`、`reference_sensitivity.csv`、`deletion_summary.csv`、`deletion_per_prototype.csv`、`topk_observed_diagnostics.json`。

## 交付与后续

保留报告已有的选择、参照、技术复杂度及非因果限定即可交付当前任务，无需为了完成这次局部归因而继续扩大实验。若以后要提出生物机制或一般化结论，再使用独立供体、代表性采样、技术匹配及独立干预证据。

复现加固项：当前直接loader未在入口强制绑定所有传递依赖（例如安装的scGPT源码）。本次已对已记录来源重新hash，并记录审阅的传递依赖；这未证明历史未记录源码的不可变性。后续重跑建议将这些源码及输入预检纳入入口。

完整请求、原始审阅响应、确定性验证、解析verdict、routing及输入hash见`work/sae_gene_attribution_v2/review_trace/`和伴随JSON。当前数字审计通过；语义结论始终标记same-family/provisional。
