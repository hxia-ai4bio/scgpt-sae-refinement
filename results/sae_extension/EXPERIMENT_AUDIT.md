# 实验完整性审查：Kang stim 与 PBMC SAE 关系重建

日期：2026-10-19。审查者：gpt-6-astra / ultra，独立上下文，同系列模型；`same-family / provisional`。确定性数值核验可接受，语义判断仍为 provisional。运行时只暴露 canonical agent `/root/sae_extension_audit`，未暴露 opaque ID。

**总体：WARN；未发现未解决的实质计算缺陷。** 警告来自历史开发数据、PBMC 批次语义、稀有类别及结论外推范围。主改进假设不成立，是实验结果，不是完整性失败。

## A. 标签与参考来源：PASS

PBMC 全部4,638个标签、CellID、batch与原H5AD逐行吻合；源obs仅有CellID、batch、celltype，不能将batch称为已验证供体。Kang stim全部9,728个标签、细胞ID、供体ID与condition对应原始数据，保留六个原开发供体，两个保留供体未使用；stim与ctrl细胞行及ID不同，但参与者相同。数据集注释属于`real_gt`评价标签，其生物准确性、预训练重叠并未独立验证。

关系教师来自冻结原始embedding，属于`self_supervised_proxy`；它不是生物学真实关系。重建MSE亦为表示自重建指标。证据：`work/sae_extension/prepare.py:30-46,64-67`；`work/sae_extension/review_trace/phase1_verification.json`；`stim_metadata_verification.json`。

## B. 分数归一化：PASS

分类Macro-F1/accuracy由真实注释和argmax预测直接计算。固定八类Macro-F1对缺失类别计0，并另存supported Macro-F1。MSE归一化分母为TRAIN原始embedding的中心化均方，不来自预测极值。KL为temperature=0.1的teacher→student逐行分布KL，teacher停止梯度。证据：`work/sae_complementarity/core.py:61-66`；`work/sae_relation/run.py:114-134,178-205`。

## C. 实际产物与数值：PASS

- 24份warmup、168份继续训练、176份实际512维embedding及176份预测完整存在。
- 独立直接checkpoint张量前向重算全部实际embedding；全部标签、行号、group一致。
- 对全部1,936个VALID正则化候选独立使用Cholesky求解并重评分，选择的alpha及早期tie规则均吻合。所有176份TEST分数与保存结果最大误差为0.0，选中VALID Cholesky重算最大误差为1.049e-11。
- 独立重算所有混淆矩阵、固定八类与supported Macro-F1、accuracy、逐类support、组/种子均值、对照差值、留一组敏感性和描述性bootstrap区间。
- 全部最终TRAIN/VALID MSE、关系KL和图表数据一致；四张PNG已人工目视核查。证据检查工具核验84项引用值全部存在。

证据：`work/sae_extension/review_trace/independent_numeric_verification.json`、`independent_metrics.json`、`evidence_precheck.json`；执行器独立核验另见`outputs/sae_extension/verification.json`。

## D. 实际调用、划分与锁定：PASS

每个报告指标均有已执行产物，未把历史未调用辅助函数当作新实验。训练均值、SAE loss及读出scaler/权重只用TRAIN；warmup按VALID MSE选checkpoint，Ridge按VALID八类Macro-F1选alpha；继续训练固定step1000。每个fold/seed七个lambda共用warmup和1,000份batch，均重置Adam。

六套stim和两套PBMC切分全部独立重建，角色互斥且完整。1,480项选择产物全部在首次新TEST结果前锁定；协议文件时间早于各自训练。原协议没有把SPLITS与上游rotation直接纳入completion binding，本审查补充前期hash快照、重建及输入hash核对，未发现漂移。证据：`work/sae_extension/run.py:38-58,118-141,192-254`；`work/sae_extension/review_trace/chronology_and_reviewer_diagnostics.json`。

## E. 范围与限制：WARN

只有六个历史Kang供体、两个PBMC批次、三个数值种子。训练折重叠；种子/细胞不是独立生物重复，stim bootstrap仅描述性。λ=0.3沿用此前ctrl探索，不能称为独立队列确认或普适最优值。

PBMC每个query使用255个同TRAIN批次非自身细胞；Kang使用192个其他TRAIN供体细胞。两者mask和锚点语义不同。PBMC反向TRAIN/VALID各仅1个Megakaryocyte，前向TEST仅2个；stim各TEST供者该类数量为[0,58,3,0,4,1]。不支持稀有类稳定泛化或PBMC跨供体结论。

继承的backbone文本“no new scGPT inference”对stim不准确；`PROTOCOL_NOTES.json`及报告已明确：stim是新的冻结前向提取，scGPT没有优化。没有改变绑定的执行代码。

## F. 评价类型与主结论

分类：`real_gt`（数据集注释）；关系几何与重建：`self_supervised_proxy`。原始h是分类基线，不能把它的关系矩阵当生物机制ground truth。

| 数据集 | 原始h Macro-F1 | λ=0 MSE | 固定λ=0.3 | λ=0.3相对h | λ=0.3相对MSE |
|---|---:|---:|---:|---:|---:|
| Kang stim | 84.5250% | 83.1558% | 83.4995% | −1.0255 pp | +0.3437 pp |
| PBMC | 82.6761% | 80.0161% | 80.4082% | −2.2679 pp | +0.3921 pp |

固定λ=0.3相对h仅1/6 stim供体改善、0/2 PBMC批次改善，两个数据集均为0/3种子均值改善。相对MSE分别为3/6供体、2/3种子及1/2批次、1/3种子改善。完整七点在两个数据集的组均值均低于原始h。详见`CLAIMS_FROM_RESULTS.md`。

无需改变本轮实验或自动追加实验；保留阴性主结果和所有网格值。最终报告嵌入本判定后，另行做文件封存。

审查器诊断：首次独立核验的NumPy向量乘法产生浮点状态warning，但所有分数有限且核对通过；将审查脚本该收缩改为显式einsum后完整复跑，无warning，所有候选指标、选择及TEST分数完全一致，仅比较误差有1.11e−15量级变化。初始日志保留，执行器代码/结果未改。

## 最终呈现封存

2026-10-19T08:29:33Z：2067项稳定输入hash全部复核一致；最终REPORT逐项数值、精确结论文本、引用及COMPLETE状态核对通过。四张最终PNG再次目视检查并建立最终字节/像素hash，SVG建立最终字节及去metadata/规范化ID的hash。未保存重渲染前图像hash，故不声称前后字节一致或已证明SVG仅metadata变化。详见`AUDIT_SEAL.json`。
