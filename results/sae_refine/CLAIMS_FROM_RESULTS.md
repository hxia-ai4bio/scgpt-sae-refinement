# Claims From Results

- **claim_supported**: no
- **integrity_status**: warn [INTEGRITY: WARN]
- **confidence**: high for the bounded conclusion
- **review_independence**: same-family
- **acceptance_status**: provisional

## What results support

- 原始h六供体等权Macro-F1为82.753960%；混合512维中的双小SAE平均为82.909667%（+0.155707 pp，2/6供体改善、2/3 seed均值为正、9/18回退原h），大SAE为82.851298%（+0.097338 pp、2/6供体改善）。这些小正均值未达到预设+0.5 pp且4/6供体改善的门槛。
- 既有三成员集成的强制重建为83.002261%（+0.248301 pp，3/6供体改善），其混合512维为81.647229%（-1.106731 pp）。该事后差异不能用于更换原先选中的模型。
- 容量对照中，强制集成比大SAE均值高0.272394 pp、4/6供体为正；混合集成低1.204069 pp。AbsTopK比同尺寸raw SAE的混合结果低0.402848 pp；boost比独立bag混合低0.790605 pp。只能报告这些描述性成对差异，不能确证算法额外收益。
- 新训练90个SAE加1个原流程复现均确有梯度和权重变化；全部完成20轮，除2个boost第二级选epoch19，其余新模型与18个既有raw模型选epoch20。总字典宽度/K/最大字典加权训练轮数按计划匹配；有效激活、优化难度、实际墙钟不等同。
- AbsTopK稀疏z读出为81.449793%（相对原h -1.304167 pp）；其2048维读出不构成512维embedding改善证据。
- 已选三成员集成混合相对强制重建在VALID为+1.252825 pp、TEST为-1.355031 pp。它说明此次选择的泛化结果与VALID排序不一致，不能单独证明过拟合、供体偏移或稀有类别是唯一原因。

## What results do not support

- C1：特定混合512维方法取得符合预设门槛的可重复改进。
- C2：集成、AbsTopK或boost普遍/稳定优于其容量或bagging对照。
- SAE普遍无用、已达到收敛极限、更多训练必定无效。
- 由更好MSE、Q2或z读出直接推出更好512维下游表示、真实生物模块或因果机制。
- 全流程无监督选型、新盲测、18个独立供体、PBMC或其他数据集迁移成功、DIAL阴性结果。

## Missing evidence

- 新的未参与历史探索的供体/队列及预先锁定的确认方案。
- 若要解释算法机理或失败原因，需能区分表示损失、分类正则化、VALID供体/稀有类差异的独立识别证据；本轮事后诊断不能唯一归因。
- 若要声称集成稳定性，需独立重复的成员seed组；现有每折仅一个三成员集成。
- 20轮预算下多数模型选最终epoch，不能证明优化已收敛。
- 生物解释/因果主张没有外部扰动、富集稳定性或功能验证证据。

## Suggested claim revision

在历史已探索的 Kang ctrl 数据（9,577细胞、6供体、8类）上，冻结 scGPT 最后层512维均值表示，采用固定20轮上限的TRAIN-only SAE、VALID checkpoint/混合与Ridge选型及供体留出评估，本轮单SAE、三成员集成、同容量大SAE、AbsTopK、双小SAE平均和两级残差方法均未达到预设推进门槛。结果支持记录该有限协议下未获得稳定512维分类改进，不能推出SAE普遍无用或任何生物机制结论。

## Next experiments needed

- 完成这一轮并记录有限阴性结果，无需自动新增训练或扩展搜索。
- 只有未来重新提出独立确认问题时，才另立协议并使用未参与当前探索的供体/队列；此项为有条件的证据需求，不是当前自动执行任务。

## Routing

记录有限阴性结果，结束本轮。没有方法满足预设推进条件；不根据TEST排行榜另选操作模型，不自动追加复杂训练。

Evidence: `outputs/sae_refine/summary.json:123-150,213-244,264-278,341-400`; `outputs/sae_refine/capacity_contrasts.json:1-162`; `outputs/sae_refine/selection_diagnostics.json`; `outputs/sae_refine/INTENDED_CLAIMS.json:2-12`; independent verifier artifacts listed in EXPERIMENT_AUDIT.md.
Trace: `.aris/traces/result-to-claim/2026-10-19_sae_refine`
