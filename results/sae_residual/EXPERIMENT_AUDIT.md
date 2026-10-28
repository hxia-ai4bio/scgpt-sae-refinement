# SAE residual experiment audit

Date: 2026-10-19 · Reviewer: fresh GPT-6-Astra ultra route · **same-family / provisional**.

**Overall verdict: WARN. Integrity status: warn.** 未发现尚未修复的实质计算、标签映射或模型选择错误；WARN 主要来自历史开发数据的范围、稀有类别与保留的数值告警。数值核验通过并不代表方法有效。

本审查直接读取源码、数据元信息、checkpoint、预测及汇总；没有重训模型。原始来源目录为 `/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014`，下文 `original/` 指该目录。

## A–F 完整性检查

| 检查 | 状态 | 证据与结论 |
|---|---|---|
| A. Ground-truth provenance | PASS | 原始 H5AD 提供类别；直接核对 9,577 个 Kang、4,638 个 PBMC 细胞的标签/行号/供体或ID。没有使用预测生成标签。`original/prepare.py:16–23,40–48`；`experiment.py:64–76`。 |
| B. Score normalization | PASS | 固定八类 F1/accuracy；表中百分数及百分点差值转换正确，没有按模型自身分数最大值归一化。`experiment.py:61–63`；`report.py:23–33`。 |
| C. Result existence | PASS | 144 组结果、126 个选型、378 个候选及18个基线头完整；独立重算1,296组指标，432个得分数组有限，8行汇总与10,368行类别表吻合。`independent_final_checks.json`；`completion.json`。 |
| D. Dead metric code | PASS | Ridge、kNN、训练头均实际调用并保存；固定一轮一种子的8臂另重放176行得分/训练头及48个直接距离kNN查询，全部一致。`experiment.py:240–265`。 |
| E. Scope | WARN | 六个历史开发供体、训练集跨轮重叠；三个种子不是独立供体。PBMC是同一个历史文件，独立研究/供体/预训练重叠未知。`PROTOCOL.json`；`report.py:61–64`。 |
| F. Evaluation type | real_gt | 使用数据集注释作为分类参考；不代表扰动因果证据。 |

## 数据、训练与选择

- 每轮TRAIN/VALID/TEST供体分离，所有18个SAE均值与对应TRAIN均值完全一致。792个独立抽样缓存行重放通过；原始TopK按raw数值排序并保留符号。scGPT历史嵌入未在本审查中逐细胞重新推理。见 `independent_static_checks.json`、`original/sae.py:12–17`。
- 原始512维h有完整skip连接；这仅说明代码保留输入路径，不保证变换在信息论上可逆。零残差初始化、第二步梯度流、旋转字典范数/Gram及SupCon重复行与跨供体掩码均通过独立检查。见 `experiment.py:78–131`。
- checkpoint/早停由训练头VALID F1选择；beta和新Ridge alpha随后由VALID新探针选择，identity赢平局。全部126选型在TEST评分前锁定，当前源文件/协议/checkpoint/metadata/输出hash均匹配。初始未绑定部分运行在TEST前废弃并保留记录。见 `experiment.py:45–60,204–232,240–265`。
- Gate与旋转Gate各2,048参数；lowrank与MLP各40,960参数。Gate对MLP并非精确容量匹配。所有适配器共享3个beta候选、相同种子采样、基线头及最大训练轮数，实际早停预算不同。SupCon各臂从零残差独立训练。该比较不证明各架构均已优化到最佳。
- 378个候选均记录到非零初始适配器梯度和第二步全部适配器参数的非零梯度；262/378个候选最终选回epoch0，其中Gate为39/54。最终Gate 16/18次回退identity，其平均TEST相对残差范数仅0.0001213（约0.0121%，含identity）。因此这是**发生了训练但多数更新未被保留**的阴性结果，不能说梯度未连通，也不能排除其他checkpoint目标或优化设置。见 `update_diagnostics.json`、`selected_models.csv`、`summary.csv`。

## 数值、范围与归档边界

NumPy矩阵乘法的divide/overflow/invalid告警被保留。保存数组全部有限；执行器全量独立重放最大得分误差0、法方程相对误差7.46e−14；本审查新Ridge与sklearn差异1.88e−12以内，抽样最终得分和kNN一致。没有观察到结果损坏，但不把告警原因当作已证明。见 `verification.json`、`numerical_warning_summary.json`、review trace日志。

评分过程中仅替换kNN计算后端：原有34,105个查询预测完全一致，并通过重复/等距边界测试；选择锁与训练源未改变。`fast_evaluate.py:12–38`、`EVALUATION_BACKEND.json`记录该性能修订。

缺失类别计0的固定八类主指标按协议执行，同时保留supported-type指标。Megakaryocyte在一轮TEST、另一轮VALID缺失，部分供体只有1–4个该类细胞；不支持稳健的稀有类别改善主张。三种子先在供体内平均，六供体bootstrap仅作描述，不能当独立折显著性或总体保证。原始Ridge在每轮跨种子复用，18条基线记录不是18次独立随机拟合。

本地完整工作区包含所有候选；分享ZIP按说明不包含原始数据/SAE缓存及未选候选张量。审查hash绑定本次快照，报告状态/索引可随后刷新；最终交付字节由执行器 `FINAL_MANIFEST.json` 绑定。该审查是同模型家族的暂定结论，不是外部复审或无错误保证。

## Claim impact / action

主Gate改善主张不支持：新Ridge 82.751977%，原始82.753960%，差−0.001983百分点；0/6正向供体、0/3正向种子均值，未优于旋转控制，预设继续门槛未达成。保持完整阴性结果与身份回退频率；无需为得到正面结论再修改本轮结果。细节见 `CLAIMS_FROM_RESULTS.md`。
