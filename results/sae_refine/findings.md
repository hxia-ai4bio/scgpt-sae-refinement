# Research Findings

## 2026-10-19 — Frozen scGPT SAE refinement

在历史已探索的 Kang ctrl 数据（9,577细胞、6供体、8类）上，冻结 scGPT 最后层512维均值表示，采用固定20轮上限的TRAIN-only SAE、VALID checkpoint/混合与Ridge选型及供体留出评估，本轮单SAE、三成员集成、同容量大SAE、AbsTopK、双小SAE平均和两级残差方法均未达到预设推进门槛。结果支持记录该有限协议下未获得稳定512维分类改进，不能推出SAE普遍无用或任何生物机制结论。

**Integrity verdict:** WARN. **claim_supported:** no. **Assurance:** same-family / provisional.

原始h六供体等权Macro-F1为82.753960%；混合512维中的双小SAE平均为82.909667%（+0.155707 pp，2/6供体改善、2/3 seed均值为正、9/18回退原h），大SAE为82.851298%（+0.097338 pp、2/6供体改善）。这些小正均值未达到预设+0.5 pp且4/6供体改善的门槛。

既有三成员集成的强制重建为83.002261%（+0.248301 pp，3/6供体改善），其混合512维为81.647229%（-1.106731 pp）。该事后差异不能用于更换原先选中的模型。

容量对照中，强制集成比大SAE均值高0.272394 pp、4/6供体为正；混合集成低1.204069 pp。AbsTopK比同尺寸raw SAE的混合结果低0.402848 pp；boost比独立bag混合低0.790605 pp。只能报告这些描述性成对差异，不能确证算法额外收益。

新训练90个SAE加1个原流程复现均确有梯度和权重变化；全部完成20轮，除2个boost第二级选epoch19，其余新模型与18个既有raw模型选epoch20。总字典宽度/K/最大字典加权训练轮数按计划匹配；有效激活、优化难度、实际墙钟不等同。

AbsTopK稀疏z读出为81.449793%（相对原h -1.304167 pp）；其2048维读出不构成512维embedding改善证据。

已选三成员集成混合相对强制重建在VALID为+1.252825 pp、TEST为-1.355031 pp。它说明此次选择的泛化结果与VALID排序不一致，不能单独证明过拟合、供体偏移或稀有类别是唯一原因。

## Interpretation and stop rule

本轮结果不是全体SAE方法的否定结论。重建目标、有限训练预算、验证供体分布和稀有类别可能参与结果，但本轮证据不能唯一归因。不得再用这些TEST结果调整alpha、挑选seed、删除类别或改写成功门槛。不能把强制重建中较好的TEST数值转为事后新操作模型。保留原问题的有限阴性结果，当前不自动新增复杂实验。

后续仅在研究问题改变且具有独立新数据/新预设方案时重开；DIAL尚未执行，不能记为失败。

## Pipeline status

Review complete; bounded negative result recorded; no automatic new training or TEST-based operational selection. Numerical verification PASS; semantic assurance provisional. See EXPERIMENT_AUDIT.json and CLAIMS_FROM_RESULTS.json.
