# 结果支持的结论

**claim_supported: yes [INTEGRITY: WARN]，仅限描述性基因敏感性与候选注释。** 审查为same-family/provisional；confidence: high只针对该限定结论。

在固定历史scGPT+K32 SAE（seed42）中，按TRAIN供体平均B-vs-other AUROC≥0.70选出的其余6个合格特征，具有可复核的、依赖所选原型与参照的输入基因值敏感性谱。39911项通路检验中普通ORA有155项全局FDR<0.05；按表达量、检出率和输入覆盖分箱匹配后，MC与exact均无条目通过全局FDR。所列通路与基因可用作探索性后续候选，尚不构成稳定功能模块或机制证据。

## 可以陈述

- 六特征筛选、分层归因、冻结TRAIN排序、实际TEST输入及通路成员映射均可复核；F1125/F1409须保留弱供体标记。
- 所选TEST原型中，按TRAIN排序的40基因替换平均drop高于匹配随机均值；该结果仅描述一个历史供体的4个所选原型。
- 本轮目标特征未退出TopK，其效应与preTopK encoder未出现超过3e-5差异；不能扩展为全字典门控不变。

## 不能陈述

- 不能把155个普通ORA行数称作155个独立机制；匹配与exact均无全局显著条目。
- 不能称为稳定跨供体功能模块：供体排名Spearman中位数仅0.038–0.171；部分高排名基因覆盖很少。
- 不能将同RNA偏相关、模型输入替换或事后TEST正向筛选称为独立生物学验证、通路激活或因果作用。
- 不能比较不同特征原型上的原始drop来宣称特征优劣或SAE全局优势。匹配阴性也不证明不存在真实功能。

## 缺少的证据与下一步

当前用户要求的描述性基因分析已完成，**不需要追加实验**。只有升级到稳定功能或机制结论时，才需未参与选择的独立供体、跨种子特征对齐，以及针对具体假设的独立扰动验证。

Routing: `confirm_narrow_claim_and_package`。完整性WARN与所有限定语随报告保留。证据：`feature_summary.csv`、`test_40gene_replacement.csv`、`all_pathway_results.csv`、`TopK_gate_diagnostics.csv`、`EXPERIMENT_AUDIT.md`及`review_trace/final_evidence_check.json`。
