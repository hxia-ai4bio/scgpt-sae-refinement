# Experiment Audit Report

**Date**: 2026-10-19
**Auditor**: GPT-6-Astra ultra (fresh same-family agent, provisional)
**Project**: Frozen scGPT SAE refinement / Kang ctrl

## Overall Verdict: WARN

## Integrity Status: warn

未发现未修复的实质计算错误。独立确定性核验通过；WARN来自探索性历史TEST复用、六供体依赖与尚未定位原因的参考数值警告。语义结论为same-family/provisional。

相对路径以 `/Users/apple/Documents/Codex/2026-10-18/wo` 为根。

## Checks

### A. Ground Truth Provenance: PASS

八类目标来自原始 Kang H5 cell_type 注释，非模型预测。9,577 行 labels、donors、cell_ids、condition 与原始 H5 逐行一致，且均为 ctrl。缓存源及 checkpoint 哈希核对。没有独立重跑全量 scGPT 特征提取，故冻结 backbone 身份/层位依赖已核对的提取代码和缓存谱系。

Evidence: `/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014/prepare.py:19`; `/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014/prepare.py:38-46`; `work/sae_refine/run.py:66-79`; `work/sae_refine/review_trace/provenance_verification.json:1-19`

### B. Score Normalization: PASS

Macro-F1 由原始预测 argmax 与数据标签的固定八类混淆计数得到；无按自身预测最大值归一化。TRAIN 均值/方差用于输入标准化；balanced 权重之和为训练细胞数；带截距、2*onehot-1 目标的 Ridge 正规方程正确。重建 Q2/相对位移是明确的 self_supervised_proxy，不冒充分类准确率。全部 222 个选中模型已用独立 Cholesky 与显式 einsum 核查。

Evidence: `work/sae_complementarity/core.py:61-100`; `work/sae_refine/run.py:245-249`; `work/sae_refine/verify.py:69-82`; `work/sae_refine/review_trace/independent_verify.py:22-28`; `work/sae_refine/review_trace/independent_verify.py:126-166`

### C. Result File Existence: PASS

全部 222 份 TEST 预测、222 份选中 VALID 预测、5,808 个 VALID trials、91 份新训练记录（含1份复现）和102份512维导出已独立读回并复算。15组汇总、90行供体汇总、1,776行类别指标、120行类别汇总、10项容量对比一致。43项数值证据预检均 verified。最终 Tracker 为 COMPLETE；没有把待执行实验列为结果。上述数量为保存文件/变体数，不是相互独立的实验次数。

Evidence: `work/sae_refine/run.py:365-431`; `outputs/sae_refine/verification.json:2-5`; `outputs/sae_refine/EXPERIMENT_TRACKER.json:2-8`; `work/sae_refine/review_trace/independent_verification.json:1-33`; `work/sae_refine/review_trace/evidence_precheck.json`

### D. Dead Code Detection: PASS

训练、推断、VALID选型、TEST指标与聚合均在实际日志和产物中可追踪。91份checkpoint均不同于初始化（3个可训练张量改变），初始梯度独立重算一致；无新增训练。AbsTopK按绝对值排序并保留原符号，bag平均与boost残差求和实现符合协议。core.py 中本轮未用的其他实验工具不用于本轮结果。DIAL明确未执行，无虚构DIAL结果。

Evidence: `work/sae_refine/run.py:101-107`; `work/sae_refine/run.py:134-191`; `work/sae_refine/run.py:255-281`; `work/sae_refine/pipeline.log:217-242`; `outputs/sae_refine/PROTOCOL.json:67`

### E. Scope Assessment: WARN

只有历史已探索的 Kang ctrl 1数据集、6供体；四TRAIN/一VALID/一TEST按供体分离，但各折训练集重叠。18个seed/fold fits不是18个供体，既有三成员集成每折只有一个结果。相同VALID供体兼用于SAE MSE checkpoint与带标签的混合/Ridge选择；当前锁定先于本轮TEST，不能消除历史TEST复用。描述性bootstrap不支持独立供体显著性或普适性。罕见巨核细胞仅60个，其中52个在一个供体，另一个供体缺失；固定八类含缺失类0分且另存present-class诊断。

Evidence: `outputs/sae_refine/PROTOCOL.json:45-67`; `work/sae_refine/run.py:206-239`; `work/sae_refine/run.py:331-342`; `work/sae_refine/run.py:401-430`; `work/sae_refine/report.py:92-99`; `work/sae_refine/review_trace/provenance_verification.json`

### F. Evaluation Type: PASS

分类使用真实数据集注释；重建目标为冻结模型的 embedding，只能说明重建/几何，不说明生物机制。没有模拟、人工评测或扰动验证。

Evidence: `work/sae_complementarity/core.py:61-66`; `work/sae_refine/run.py:151-168`; `work/sae_refine/run.py:245-249`; `outputs/sae_refine/INTENDED_CLAIMS.json:5-12`

分类指标：`real_gt`。重建、几何与稀疏诊断：`self_supervised_proxy`。

## Independent numerical verification

完整复算 5,808 个 VALID trial、222 个选中读出、222 份 TEST 预测；复核 91 个新训练记录与 102 份512维导出。最大TEST得分差 1.243e-14，独立Cholesky参考差 6.405e-10，正规方程相对残差 1.204e-14。

sklearn参考路径出现NumPy matmul警告；不能仅据警告判定数值失效，也不声称原因已经找到。当前有限值、全量F1/选型重算、独立Cholesky得分、正规方程与显式einsum均通过，未发现对本轮结果的数值影响。

验证文件：`work/sae_refine/review_trace/independent_verification.json`、`supplemental_verification.json`、`provenance_verification.json`、`evidence_precheck.json`。

## Selection and budget boundaries

TRAIN均值/标准差和所有拟合仅用四个TRAIN供体；选型用独立VALID供体；源码的锁定与日志先于本轮TEST。混合选择包含alpha=0且平局优先原h；强制重建固定alpha=1单独选Ridge。SAE目标无标签，但整个表示选择流程使用VALID标签，不能称全程无监督。

三成员集成每折只算1个集成；bag/boost共享第一级并用相同第二级初始化作配对，目标分别为h与h−一级重建。实际90个新方法训练与18个既有raw训练均跑满20轮；两个boost二级选择epoch19，其他选择epoch20。容量比较仅匹配总字典宽度/K/最大字典加权训练轮数，不证明优化难度或墙钟相等，也不证明收敛。

## Action Items

- 保留探索性历史数据复用、六供体描述性统计与VALID标签选型限定。
- 保持固定八类与逐类/逐供体报告；不得隐藏巨核细胞不稳定性或缺失类规则。
- 记录本轮无方法达到推进门槛；不据TEST另选操作模型，不自动新增复杂训练。
- 保留NumPy参考路径警告及独立复核证据；不要将未查明原因描述为已修复。

## Claim Impact

- C1: unsupported under the advancement rule.
- C2: descriptive contrasts only; stable algorithm-specific improvement unsupported.
- C3: supported as an interpretation boundary; z-only does not establish512D improvement.

## Review limitations

- 未独立重跑全部12层scGPT特征提取；直接核验原始H5行标签、缓存谱系与文件哈希。
- 文件锁和日志支持本轮执行顺序，并非外部预注册时间戳或防篡改证明。
- 语义审查为same-family/provisional，确定性核验通过不升级为跨模型独立接受。

Trace: `.aris/traces/experiment-audit/2026-10-19_sae_refine`
Agent: `/root/sae_refine_audit`; reviewer_model=`gpt-6-astra`; reasoning=`ultra`.

Final report/Tracker evidence sealed after the display-only refresh. All numeric/model/protocol hashes unchanged. See 002-final-seal in the review trace.
