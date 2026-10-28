# 固定 λ 敏感性实验独立审计

日期：2026-10-19。审查者：gpt-6-astra ultra，fork_turns=none；canonical `/root/sae_lambda_audit`；opaque id unavailable。同系列独立上下文审查，**provisional**。

## 总体结论：WARN

未发现实质数值或选择流程缺陷。确定性核验 PASS；WARN 来自历史探索性范围与已记录的继承元数据差异，不能将该核验视为语义结论的 accepted 认证。

| 检查 | 结论 | 直接证据与说明 |
|---|---|---|
| A. Ground truth | PASS | `work/sae_refine/run.py:66-79`；`work/sae_complementarity/core.py:61-66`；`phase1_check.json`：9577个真实原始H5注释、供体、条件和cell ID全部匹配。关系几何另标为自监督代理。 |
| B. Score normalization | PASS | `work/sae_relation/run.py:112-134,174-218`；`core.py:61-66`：分类F1使用混淆矩阵；固定TRAIN方差只归一化重建loss，没有按模型自身最大分数美化指标。 |
| C. Result existence | PASS | `independent_numeric.json`、`independent_aggregates.json`：126训练、132读出与1452 VALID候选全部实物复算；60复用读出TEST分数逐值一致。 |
| D. Dead code / plots | PASS | `work/sae_lambda/run.py:99-218`；`report.py:28-81`；`independent_plot_check.json`：30条线、4组区间、3张PNG逐像素对应独立核对数据。 |
| E. Scope | WARN | `PROTOCOL.json`、`INTENDED_CLAIMS.json`：同一历史Kang数据，七点grid受此前TEST结果启发；6供体且TRAIN重叠，bootstrap只作描述。 |
| F. Evaluation type | 分离标注 | 分类为 `real_gt`；重建与关系KL为 `self_supervised_proxy`，不提供生物因果真值。 |

## 确定性核验

- 65来源哈希、1074选择锁哈希；18组独立重生成批次，供体资格与64个无放回细胞/供体匹配。
- 18个初始checkpoint、126个最终checkpoint（72新增、54复用）；固定TRAIN均值、参数真实变化、decoder列范数、学习率及梯度记录核对。审查者优化器更新为0。
- 1452个VALID候选全部用Cholesky独立重拟合，核对132个被选alpha与预测。最大VALID分数误差 3.32e-12；TEST分数误差 1.78e-12；复用TEST分数误差0。
- 208146项汇总检查，覆盖8汇总行、16 contrasts、48供体行、42 seed行、96个LOO值、1056逐类行、630监控行与630梯度行。初始与最终几何监控独立重算；没有重演全部中间优化步骤。
- 91项证据存在性检查全部 verified。解析链、输入、输出和stderr均已保存；verified仅代表证据存在。

支持的选择路径只接触TRAIN/VALID，所有候选锁定后统一TEST。文件锁不能证明不存在未记录的额外人工查看，但未发现代码中TEST选择或结果驱动的预算更改。

## 必须保留的限定

`reference_arms`继承字段列出existing_raw20，实际仅作为初始化来源；`PROTOCOL_NOTES.json`已明确132读出的真实范围。旧数据元信息的excluded_cell_types仍列出Megakaryocytes，而实际包含60个；这一历史描述差异不影响本轮行和标签，冻结来源未改。

本轮不支持最佳λ、连续有益区间、独立确认、普遍SAE优势或因果生物学结论。三个种子及细胞不是新的独立供体。正式语义判断见 `CLAIMS_FROM_RESULTS.md`。

最终REPORT已包含精确审查结论，tracker为COMPLETE；1593项输入完成复核，其中1590项字节哈希一致；3张SVG仅有重渲染元数据/生成ID变化，图形数值结构完全一致。展示与审查文件已封存，见AUDIT_SEAL.json。
