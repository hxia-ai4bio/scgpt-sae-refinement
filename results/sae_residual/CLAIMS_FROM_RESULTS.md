# Results to claims

**claim_supported: no**（针对预设SAE Gate改善/超过旋转控制的主张）。
**integrity_status: warn · confidence: high within this fixed recipe · same-family / provisional**。

新Ridge主终点：原始表征82.753960%，SAE Gate 82.751977%，差**−0.001983百分点**，实质上无改善；六供体中0个正向、三种子均值中0个正向。旋转Gate与原始表征同分。Gate仅满足PBMC损失不超过0.5百分点这一项，未达到其余继续门槛。依据：`summary.csv`、`continuation_checks.json`；已由`independent_final_checks.json`独立重算。

| 方案 | TEST新Ridge差值（百分点） | 回退identity |
|---|---:|---:|
| SAE Gate | −0.001983 | 16/18 |
| 旋转Gate | 0.000000 | 17/18 |
| SAE lowrank | +0.003885 | 14/18 |
| Dense MLP | −0.148146 | 13/18 |
| SAE Gate + SupCon | −0.001301 | 15/18 |
| 旋转Gate + SupCon | 0.000000 | 17/18 |
| Dense MLP + SupCon | +0.015829 | 15/18 |

训练确实有梯度和更新：378个候选均有非零初始适配器梯度及第二步全部适配器参数的非零梯度；但262个最终选中epoch0，Gate为39/54。训练头VALID选择checkpoint后，新Ridge再选择beta/identity，最终Gate仅2/18运行保留非零适配器。平均TEST相对残差范数0.0001213，包含16次identity。这支持“本配方多数更新未被选择保留、主性能没有可见增益”，不支持“所有SAE残差方法都无效”。

Dense MLP + SupCon的kNN差值为+1.196807百分点，但新Ridge仅+0.015829百分点、只有1/6供体均值正向。它是次要读出上的描述性变化，不能替代预设主终点，也不能归因于SAE语义方向。Gate + SupCon仍未形成主终点收益。

建议采用的结论：**在六个历史开发供体、三个耦合种子和固定训练/选择方案下，冻结SAE残差门控未提高新Ridge分类表现；大多数运行回退至原始表征，少数保留更新很小。加入跨供体SupCon未改变主要结论。**

不支持稳定泛化提升、SAE语义字典的特异优势、盲测外部复现或生物因果机制。历史Kang/PBMC不能作为新盲测；三个种子不能当独立供体。不能从微小或零效应排除其他checkpoint目标、学习率/损失或更大的有效残差。

本轮按预设停止扩展，完整保存阴性结果，无需补实验才能交付。若以后另立新研究，先仅用TRAIN/VALID预注册新的checkpoint/优化假设，再使用新供体或新研究作一次确认；不要依据本轮TEST/PBMC继续调参。生物机制仍需独立扰动/功能验证。

Deterministic pre-check: 六项证据存在性verified，且独立重算通过；这不将同家族语义审查升级为外部accepted结论。科学文件在本次快照审查一次，后续仅允许状态/索引刷新并由最终manifest绑定交付。
