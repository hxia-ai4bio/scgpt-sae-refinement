# Experiment Audit

日期：2026-10-19。审查员：gpt-6-astra / ultra，独立新建任务 `/root/complementarity_audit`，延续同一任务完成静态与最终审查。**review_independence=same-family；acceptance_status=provisional。**

**Overall verdict: WARN；integrity_status: warn。没有未修复的实质计算错误。** WARN 来自历史数据重用、有限供体及外部独立性未确认；不代表本次发现了伪造标签、指标自归一化或虚构结果。

## A. Ground truth provenance — PASS

审查直接读取原始 `kang.h5ad` 和 `PBMC368k_raw.h5ad` 注释，全部 **9,577 个 Kang、4,638 个 PBMC** 细胞的类别和细胞 ID 与缓存逐行相符；Kang 供体 ID 也相符。来源是数据集细胞类型注释，不是当前模型输出。证据：源 `sae_topk_20261014/prepare.py:16-22,40-48`，当前 `work/sae_complementarity/core.py:105-121`，审查 `independent_static_checks.json`。

残差 `2·onehot(Y)−1−s_OOF` 含模型分数，因此属于明确披露的读出平方损失代理目标；没有被冒充为独立生物真值。

## B. Score normalization — PASS

Macro-F1 从原始 argmax 预测形成混淆矩阵，在固定 8 类上取均值；缺失类别按预设规则记 0。带截距类别平衡 Ridge 使用 ±1 目标，样本权重之和等于 TRAIN 样本数，输入标准化仅在 TRAIN 拟合。拼接先分别标准化 h、z，再乘 eta，随后禁用第二次标准化，因此分支权重保留。

MSE 是未作自归一化的目标分数平方误差。残差 Q2 为 `1−SSE_corrected/SSE_original_h`，分母是最终基模型误差，不是自身最大分数。分数不是概率，Q2 不能解释为生物信息量。残差修正的 alpha/gamma 按 VALID Macro-F1 选择；这里的 Q2 仅描述该分类选型策略，不是按 MSE 选最优模型的结果。Q2 先逐模型求比值再平均，其符号不能替代平均绝对 MSE 的变化。证据：`core.py:61-97`、`run.py:50-85,168-185`。

## C. Result files, selection and replay — PASS

- 独立重算全部 **432 份 TEST/PBMC** 保存预测的 Macro-F1、supported-class Macro-F1、accuracy、每类 F1、support、混淆矩阵、分数 MSE、纠正/损害数量及残差 Q2；分类指标最大误差为 **0**。
- 从冻结模型重新构造输入并重放全部 **432 份外层分数**，最大误差为 **0**。
- 独立检查 **7,128 组 VALID 预测数组**与 **216 次选择**，包含原 h 基线按种子重复核对；对应 **6,996 组唯一数组、204 个唯一选型目录**。198 个分支选型加 6 个共享基线。逐一核对最优值、并列顺序、identity 回退、行号及标签。
- 独立重算全部主表、供体差异、种子表、每类表、选择表、配对对照及描述性 bootstrap。全部相符。
- 所有 **1,382 项 selection-lock 哈希**一致；新评价文件时间均晚于统一锁定时间 **2026-10-18 18:43:37 UTC**。审查早期曾独立观察到开发进行中且新评价目录不存在。代码也要求先锁定全 18 组开发结果，再评分。该证据支持“本轮新评分前锁定”，不能抹去此前历史数据探索。
- 六个外层轮次的标签 crossfit 行覆盖、内层供体排除、类别数、目标恒等式和 alpha 选择均核对通过。
- 历史 `outputs/sae_residual/FINAL_MANIFEST.json` 的 **1,062 项**文件保持不变。

证据：`review_trace/independent_final_checks.{py,json}`、`independent_supplement_checks.{py,json}`、`outputs/sae_complementarity/SELECTION_LOCK.json`。

## D. Dead code / actual fits — PASS

选型和评价路径确实调用指标并写入结果；未发现只定义未执行的核心评价。独立 Cholesky 求解复算了 7 个真实拟合：原 h、稀疏 z、SAE 拼接、SAE 残差修正、random 拼接、linear 拼接及一个排除目标供体的 crossfit 基模型。最大 VALID/OOF 分数差 **2.8431×10⁻¹²**，类别预测全部相同；这是代表性独立拟合复算，不声称对所有模型重训。

两组固定混合的 108 个配对检查最大预测差 **5.6206×10⁻¹²**；22 次选中 linear 分支的 TEST/PBMC 仿射展开检查最大分数差 **2.0650×10⁻¹⁴**。`regularization_diagnostic.py:8-24` 只读取已存结果，没有新拟合或测试选参。证据：`independent_cholesky_checks.json`、`independent_final_checks.json`。

## E. Scope — WARN

1. 六个历史 Kang 供体、三个 SAE 种子和一个历史 PBMC 外部文件，仅支持当前设置的探索性描述。训练集重叠；18 个 PBMC 拟合不是 18 个独立数据集。PBMC 研究来源、供体与预训练重叠未确认。
2. SAE 外层只用四个 TRAIN 供体拟合，VALID 重构 MSE 选 checkpoint；内层 label crossfit 没有重新训练 SAE。协议正确只声称标签/读出交叉拟合。
3. random 对照只匹配名义维数、TopK 数与搜索预算，没有匹配激活频率、相关性或语义。linear 对照仍是 h 的仿射函数，只改变有效正则化。
4. 两个固定混合网格大部分候选在标准化与截距下仿射等价，不能算两份独立证据。
5. 六供体 bootstrap 是描述性区间。稀有 Megakaryocytes 的各供体支持数为 1、1、52、0、2、4，F1 高度离散；局部类别算术不能推翻预设八类终点。
6. z 是 h 的确定性函数；即使读出改善，也不能证明新增观测信息、普遍生物语义或因果机制。当前阴性结果同样不能证明 SAE 普遍无用。

原报告的“1 个独立外部数据集”字段已改为“1 个外部数据文件”，并补充独立性未确认；该问题已解决。证据：`PROTOCOL.json`、`DATA_INVENTORY.json`、`report.py:84-93`。

## F. Evaluation type

- 分类评价：**real_gt**，使用数据集细胞类型注释；不等同于实验验证的因果真值。
- 分类分数残差与 Q2：**显式标识的 model-dependent squared-loss proxy**；按技能分类归入 synthetic_proxy，因为目标包含基模型预测，但真实标签仍来自原数据。
- 本轮没有生物干预、救援或因果评价。

## 补充诊断和结论影响

独立核对 90 个新旧重叠 alpha 的 VALID 指标，最大差为 0；读取历史预测重算 36 份 TEST/PBMC 指标，并核验源 artifact manifest。稀疏 z 的 17/18 组新 alpha 超过旧上限 100。旧网格历史 TEST/PBMC 为 **77.0931%/63.9318%**，本轮为 **83.1049%/72.3927%**。这支持“读出正则化敏感”，不能把较宽搜索的收益直接归因为新生物结构。

两份 `class_effect_decomposition.json` 均是原八类终点的准确算术分解，没有删类或新增主指标。SAE 拼接总变化 −0.4354 pp，其中 Megakaryocytes 贡献 −0.6086 pp，其余七类合计 +0.1732 pp；这是局部不均一性的事后描述，不是新整体收益或已证实的机制解释。

**三个正向主张 Q1–Q3 均未获支持；详细边界见 `CLAIMS_FROM_RESULTS.md`。** 所有结果和报告已保存审查前快照，审查哈希与新增审查产物分开，避免自引用。

## 收尾

无需为本审查追加实验或调参。允许仅刷新报告状态/索引、生成最终 manifest 与 ZIP 并核验打包完整性；无实质变化时无需第三轮审查。审查不能证明从未发生任何未记录活动，也不能将 same-family 的语义判断升级为跨模型家族验收。
