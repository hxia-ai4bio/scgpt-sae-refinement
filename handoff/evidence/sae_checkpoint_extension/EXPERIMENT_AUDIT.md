# 实验完整性审计

结论：**WARN（范围限制）**。数值、来源、选型与独立复算通过，未发现未解决的完整性缺陷。

审查：fresh same-family / provisional；请求 gpt-6-astra + ultra，实际解析模型不可见；仍待外部审查。

| 检查 | 结论 | 证据 |
|---|---|---|
| A 目标来源 | PASS | 原始H5AD复算768个logCP10k目标精确一致，全部新kNN标签对应原始注释。`work/sae_masked_ac/prepare.py:28-36` |
| B 分数归一化 | PASS | 主MAE保留原尺度；MSE分母来自TRAIN基因标准差。`work/sae_expression_probe/run.py:163-167` |
| C 文件与数值 | PASS | 24组表达指标、48组kNN投票/F1和汇总独立重算；1002当前绑定文件、93旧ctrl文件哈希一致。`002-independent-evidence.json:5-19` |
| D 实际执行 | PASS | 24本体/24新探针完成；24个step1000逐张量等于旧模型；验证器重放168个VALID目标。`verification.json:2-23`、`run.py:147-190` |
| E 范围 | WARN | 历史数据、同一VALID复用、单mask；stim共享ctrl供体，PBMC仅两个批次方向。`report.py:60` |
| F 评价类型 | PASS | 表达为真实测量值上的masked-positive self-supervised proxy；kNN为数据集注释real_gt。无因果验证。 |

以上短文件名分别指本目录verification.json、新工作目录run.py/report.py，以及`.aris/traces/experiment-audit/2026-10-20_checkpoint_extension/002-independent-evidence.json`。完整引用及输入SHA256见JSON。

TEST前全局锁缺口已在训练前修复：`run.py:159-166`同时验证两组锁，`execute.py:23-28`先完成两组开发再评价；`full.log:1-6`记录该顺序。科学规则未改变。

必须保留任务权衡：PBMC表达MAE下降，但clean/masked 15NN比原始h下降3.86/5.10个百分点；stim的15NN仍高于原始h，但低于C1000（`knn_summary.csv:8-19`）。独立探针比较成立，不能据此证明稀疏性必要或同总上游计算预算优势。

审计边界：原始目标和独立线性预测各抽查32项/运行，共768项；所有保存的表达指标和kNN投票全量重算。完整VALID目标及有界邻居几何重放另由已审阅验证器提供；未重新拟合任何模型。本轮无需额外训练。
