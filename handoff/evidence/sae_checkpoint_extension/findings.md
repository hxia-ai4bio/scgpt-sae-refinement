# Research findings

在已历史探索的Kang stim六供体与PBMC两个批次方向上，原样迁移ctrl的七点VALID表达MSE checkpoint规则，并用相同方案重新训练冻结表征的独立线性探针，使masked-positive表达TEST MAE低于原始h与固定C1000。stim两项比较均18/18配对、6/6供体改善；PBMC均6/6配对、2/2方向改善。但改善具有任务依赖性，PBMC的clean/masked 15NN均低于两项对照。

- Stim：MAE 0.334368，较original_h/C1000降低1.28%/3.51%；18/18配对与6/6供体方向一致。
- PBMC：MAE 0.344115，降低5.13%/10.89%；6/6配对与2/2批次方向一致。
- 重要限制：PBMC clean/masked kNN较original_h下降3.86/5.10个百分点。表达选型对任务具有依赖性。
- 判定：限定表达复现yes；普遍表征提升no；独立性same-family，provisional。完整性WARN为范围限制，数值复核PASS。
- 动作：保留限定结论并停止本轮，无追加实验、调参或重新拟合。广泛泛化/稀疏性必要/生物机制仍未验证。

证据与输入哈希见EXPERIMENT_AUDIT.json和CLAIMS_FROM_RESULTS.json。
