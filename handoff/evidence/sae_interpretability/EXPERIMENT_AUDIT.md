# SAE 解释性实验完整性审计

日期：2026-10-18。审计者：fresh GPT-6-Astra / ultra，独立读取文件并复算；`review_independence: same-family`，`acceptance_status: provisional`。

**Overall verdict: WARN。当前确定性证据通过；生物学解释仍是探索性、部分支持。未发现伪造标签、用自身最大值美化分数、引用不存在的已完成结果或当前数据错位。** 此结论不是独立实验室复核，也不是生物机制验证。

审计范围：当前目录 `outputs/sae_interpretability/`，执行代码 `work/sae_interpretability/`，以及历史 SAE 数据准备、训练、tokenizer 和 checkpoint 来源。下文 `W/` 指当前 `work/sae_interpretability/`，`O/` 指当前输出目录，`H/` 指 `/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014/`。机器报告保存本轮审计输入的 SHA-256。

## A. 标签、数据划分与特征选择：PASS，带来源范围限制

- Kang 和 PBMC 的 cell-type 标签直接来自原始 H5 元数据，不是 SAE/scGPT 预测。证据：`H/prepare.py:16-23,40-48`。审计者直接读取 H5 注释、细胞 ID、donor/batch，均与 embedding cache 对齐。
- 9,577 个 Kang 细胞，6 个 donor；每轮 4 TRAIN、1 VALID、1 TEST。审计者逐轮验证索引无交集且完整覆盖，donor 身份与角色吻合。历史训练只用 TRAIN 重建、VALID MSE 选 epoch：`H/run.py:30-46`；固定 TRAIN mean：`:33`。该留出设计有效，但这些 donor 历史上已经被探索，不能称盲测。
- 主模型是历史第一轮 patient_1016 TEST / patient_1039 VALID、seed42、K32。候选 440/160/120 由 TRAIN 内最差 donor 的 one-v-rest AUC 选择；1724 由经过加性 cell-type 与 donor 调整后的 QC 相关选择。审计者从 atlas 独立重做选择，得到相同四个 ID。证据：`W/01_atlas.py:52-80`。
- PBMC 共 4,638 个细胞、两个批次；raw 标签确实来自文件。研究/donor 独立性及预训练重叠未知；两个 batch 不应直接当作两个独立 donor。证据：`H/prepare.py:47-53`。
- `technical_candidate` 原先“within cell-type-and-donor”的表述已改为准确的“additive cell-type and donor adjustment”。没有拟合 donor×type 交互，不能声称所有群体内技术混杂均被消除。`W/01_atlas.py:63-69`。

## B. 指标、归一化与数值：PASS

- AUC 使用正确的 tie-aware rank 公式：`W/common.py:56-61`。审计者以 sklearn 独立复算 24 个有类别目标的 candidate/group AUC，最大误差 2.22e-16。
- 六轮×三种子×三目标×三表示共 162 行；先在每个 TEST donor 内平均种子，再汇总六个 donor。独立复算 `donor_equal_single_feature_readouts.csv` 最大误差 3.33e-16。三个种子不是三个独立生物样本。`W/05_all_folds.py:41-56`。
- 遮挡输出保留原始 activation、baseline 和 drop，`drop = baseline - activation`，没有除以本模型的最大分数或重新缩放至接近 1。`W/executed_v1/03_occlusion.py:124-131`。不同 SAE/原始坐标/PCA 方向的原始数值单位仍不能直接作跨表示优劣比较。
- 表达关联是同一输入 RNA 的 log1pCP10K partial Pearson correlation。审计者用直接最小二乘对 40 个 TEST 基因关联复算，最大差约 1.93e-5；稀疏 float32 平方和造成很小的数值差异，未改变当前解释。`W/04_expression.py:14-32,54-74`。低残差方差场景建议使用 float64 稀疏平方和，当前没有发现影响结论的数值异常。
- ORA hypergeometric p 独立复算误差 1.11e-16；用 statsmodels 独立复算所有 BH family，误差 1.10e-14。`W/06_enrichment.py:40-49,69-75`。

## C. 产物存在、执行及缓存：PASS（本轮），复用仍需完整来源校验

- 24 个 prototype/reference pair 已完成；10,781 次发生数值改变的单基因 forward；576 条删除曲线。`O/occlusion_completion.json:2-10`。不是只启动了计算便宣告完成。
- 审计者直接从 24 个 NPZ 重建 8 张 gene-attribution 表，最大误差约 1e-16；基因集合、exposure count、原始/reference bins 对应的 changed index 均吻合；未改变 bin 的 attribution 全为 0。
- 原始执行版本保存在 `W/executed_v1/03_occlusion.py`。最初只按文件存在复用缓存（该文件 `:58-60`）是实际复用缺陷；当前 `W/03_occlusion.py:49-74,84-90` 已锁定 pair 身份并检查绑定 hash。
- 对当前既有缓存，`W/09_verify.py:25-43` 检验每个 pair 的 token column、position、原始/reference bins、baseline，并对每个 pair 最大效应基因重做完整 score-vector forward，然后附加绑定。`O/deterministic_verification.json:4-11` 记录 32 次慢路径重放，最大误差 1.13e-6，24 个 cache 已绑定。审计者再次核验全部 24 个 cache hash 与 binding 一致。
- 审计者核对 6 个当前 input hash，以及历史 data-manifest 12 个源文件和历史 protocol 7 个源文件 hash，全部一致。当前 checkpoint、tokenizer、数据和训练代码的来源链可追溯。
- 边界：不是逐个重算全部 10,781 个 forward。绑定对当前证据有效；未来复用时仍应重新核对 manifest 所引用的实际 raw/checkpoint/vocab/PCA/基线配置文件，不能只核对 manifest 文本的 hash。

## D. 死代码与覆盖：PASS（已审计范围）

候选/atlas、forward preflight、occlusion、表达关联、全 fold 单特征读出、富集、TopK 诊断、初始化对照和验证均有对应输出。当前在声明中使用的指标能追溯到实际 CSV/JSON；没有发现仅定义函数但把结果当已执行的情况。forward preflight 只抽样 8 个原始细胞；后续验证扩大到所有 24 个选定 prototype 的输入检查与抽样 effect 重放，仍不是全量 forward 独立重现。

## E. 实验范围、公平基线与空结果：WARN

### 已完成结果支持什么

六个 donor 等权、先平均三个 seed 的 one-feature TEST AUC：

| 目标 | SAE | 原始 embedding 坐标 | TRAIN PCA 坐标 |
|---|---:|---:|---:|
| B cells | 0.97561 | 0.95322 | 0.91273 |
| CD14+ Monocytes | 0.97049 | 0.97307 | 0.97893 |
| NK cells | 0.93126 | 0.93046 | 0.85927 |

来源：`O/donor_equal_single_feature_readouts.csv`。这说明所选标签的信息能集中在单坐标中，尤其 B 细胞。SAE 有 2048 个候选方向，原始/PCA 只有 512，且 SAE 设覆盖筛选、其他表示可选正负方向；并非公平隔离“稀疏性导致可解释性提高”的试验。`W/05_all_folds.py:13-29,43-48`。

外部 PBMC 的结果必须同等呈现：monocyte SAE 两批次平均 AUC 为 0.50411 / 0.52711，PCA 为 0.95497 / 0.98253；NK SAE 为 0.86385 / 0.86762，原始坐标为 0.97962 / 0.97148；B SAE 为 0.98423 / 0.96659。这不支持整体跨数据集优势。TopK 前后诊断是 post-hoc 机制线索，不能把读取 raw encoder 的诊断当成已经验证的新模型；`W/07_gate_diagnostics.py:13-29`。

### 遮挡与删除的边界

- 四个特征每个用 4 个 TRAIN donor 各 1 个高激活 prototype，以及主 TEST donor 内 2 个最高正激活 prototype。每个只有一个 inactive reference；一个 NK TRAIN reference 属于其他细胞类型。`W/executed_v1/03_occlusion.py:19-44`、`O/prototype_pairs.json`。因此涉及选定高激活细胞与参考选择，不代表全部细胞。
- 删除排序仅由 TRAIN 确定，然后用于 TEST。预算 40 时平均 raw activation drop（两个 TEST 细胞）为：440 0.280899、160 0.303669、120 0.121938、1724 0.262390；随机对照均值相应为 0.071535、0.080636、0.046565、0.128479。完整预算是 0/1/5/10/20/40，应保留全部曲线。来源：`O/heldout_deletion_curves.csv`。
- 这支持当前协议下的局部排序迁移。只有一个 TEST donor；10 个随机排列与多级嵌套预算不能当作独立生物重复，也不足以主张 donor-level 显著性。
- 随机基因按原始 bin 分组匹配，未匹配 reference bin、替换方向或替换幅度：`W/executed_v1/03_occlusion.py:111-117`。本轮预算40的 attribution 平均绝对 bin 改变量多数反而低于 random，因此没有证据把结果归因于更大的 attribution 扰动；严格的基因身份特异性对照仍缺失。
- 单基因 attribution 一致性有限：TRAIN prototype 间平均 Spearman 为 B 0.061、monocyte 0.184、NK 0.255、technical 0.222；主 TEST 两细胞之间分别为 0.366、0.102、0.319、0.272。`O/attribution_consistency.csv`。不应称稳定的通用基因模块。
- 干预是固定 token/padding 后替换离散 expression bin，没有重归一化/重采样 token。`W/02_forward.py:43-57`、`W/executed_v1/03_occlusion.py:61-71`。它度量 input→scGPT embedding→post-hoc SAE coordinate 的敏感性；SAE 坐标并非原始 scGPT 的内部必需组件。因此既不证明 scGPT 依赖该 SAE feature，也不等同真实基因干预。

### 富集、稳定性及混杂

- 19,577 个 pathway tests，BH 按 source×split 对四个 feature/pathway 全家族校正；未只校正展示出来的 hit。TRAIN 使用 2000 次联合 mean-expression/detection quintile 匹配随机基因集；这是条件注释的 null，不是生物因果 null。`W/06_enrichment.py:26-32,50-75`。
- TRAIN attribution 的 matched-null q<0.05 pathways：440=13、160=14、120=18、1724=18。技术 comparator 同样富集；通路名称可能由核糖体/翻译等共享基因驱动。表达关联的 matched-null q<0.05 为 0，最小 q=0.07026，不能只展示 nominal/hypergeometric positives。
- TEST/PBMC 没有运行匹配随机 null；其 q_matched 缺失，不能说“没有显著 matched hit”，应写“未评估”。各 split ORA 独立重取 top40，因此不是固定 TRAIN gene-list 的验证。`W/06_enrichment.py:33-38,50`。只有表达 summary 的 TRAIN top40 被固定后向 TEST/PBMC 传递：`W/04_expression.py:66-73`。
- 同一 seed 跨 fold decoder cosine 约 0.93–0.99，但不同 seed 只有 0.26–0.59；跨 fold 同时共享训练 donor 和初始化。初始随机匹配约 0.136–0.162；最终与自身初始方向 cosine 约 0.547–0.827。`O/dictionary_stability.csv`、`O/dictionary_initialization_controls.csv`、`W/08_dictionary_controls.py:13-21`。可称比随机初始化更一致的部分标签结构，不能称 seed-independent biological modules。
- Kang 文件没有 MT- 基因，因而不能评估线粒体混杂。全体样本 r_mt=0 是缺失可变测量的数学结果，不是排除线粒体效应。`O/expression_analysis_protocol.json`、`W/common.py:92-96`。

## F. 评估类型

| 证据 | 分类 | 允许的解释 |
|---|---|---|
| 原始注释 cell-type AUC | real_gt（数据集注释） | 在这些历史数据/划分上的标签可读出性 |
| 重建、decoder 匹配、TopK 诊断 | self_supervised_proxy | 表示与算法性质 |
| 表达相关、基因集注释 | self_supervised_proxy / same-RNA association | 候选关联；不独立于模型输入 |
| 固定-bin occlusion/deletion | synthetic_proxy，明确标为 model sensitivity | 合成输入替换下的局部响应 |

## Result-to-claim 判定：partial，置信度 high（针对受限结论）

支持：特定细胞类型标签的单坐标可读出性；B 细胞标签在所用 PBMC 文件中保留；部分 TRAIN attribution 排序在同一 TEST donor 的所选 prototype 上迁移；数值与来源链通过审计。

不支持：发现新的独立生物功能模块、机制或因果基因；原始模型依赖这些后置 SAE 坐标；全面优于 PCA/原始 embedding；稳定且普适的跨数据集功能解释；独立外部研究验证；论文原样复现。

建议工作表述：“在历史 Kang 数据及一个已使用的 PBMC 文件上，K32 SAE 的部分坐标集中表达细胞类型相关信息，并显示有限的局部输入敏感性排序迁移；跨数据集保留、跨种子字典一致性与基因解释一致性均有明显限制。结果支持探索性表示分析，不构成生物机制证据。”

后续若要升级结论：在新 donor/study 上冻结 feature 与 TRAIN gene-list；随机抽取/分层抽取测试细胞及多个同类型 reference；把本轮已补充的替换方向/幅度匹配 null 扩展到上述评估；用匹配字典规模和随机/未训练 SAE 控制分离架构效应；最后用独立实验干预或正交测量验证生物解释。当前不需要为了发布探索性报告而隐去空结果或补造显著性。

## 最终补充：严格替换匹配随机对照

这是原协议完成后、审计驱动的事后对照；候选、TRAIN 基因排序、TEST prototype/reference 均未改变。代码 `W/11_stronger_null.py:17-42` 同时匹配原始 expression bin//10、reference-original 的符号、绝对变化量//5，仍是分箱匹配而非逐基因完全等幅匹配。十个无放回随机顺序、八个原型、六个预算，共新增 480 次 forward。原始 576 条组合替换结果保留，总计 1056 条。

审计者根据缓存与固定 TRAIN 排序独立重建全部 480 个确切输入，和 `W/strong_null/pair_*.npz` 逐项完全一致；80 个预算40重叠诊断均吻合；score 到 CSV 最大差 2.22e-16；预算40汇总表独立复算误差 2.22e-16。执行方慢路径额外重放八个预算40输入，最大差 1.19e-6：`W/09_verify.py:73-104`、`O/deterministic_verification.json`。没有重新计算全部 480 个模型输出，审计范围仍是全量输入核对与抽样 forward。

| Feature | Attribution 平均 drop | 原始 random | 严格匹配 random | 严格随机复用 attribution 基因数均值/40 |
|---|---:|---:|---:|---:|
| 440 / CD14 | 0.280899 | 0.071535 | 0.142425 | 16.25 |
| 160 / B | 0.303669 | 0.080636 | 0.146679 | 19.70 |
| 120 / NK | 0.121938 | 0.046565 | 0.082429 | 21.40 |
| 1724 / technical | 0.262390 | 0.128479 | 0.193059 | 18.30 |

以上均为预算40；attribution 每项两个选定 TEST 细胞，random 每细胞十个顺序。严格对照共复用 9–25/40 个 attribution genes，必须保留该覆盖诊断。相比原始 random，严格 random 的平均 drop 提高且缩小差距；attribution 仍有较大平均 drop。技术候选同样得到正向差异，因此该补充加强局部输入排序迁移的证据，并不建立生物特异性或因果机制。整体 **WARN / partial** 不变。

读者报告核对包括 `O/REPORT.html` 与 `O/RESULTS_ZH.txt`：六 donor AUC 表、预算40 raw baseline/activation/drop 表、计算次数均与完成产物吻合。HTML 的曲线阴影明确不是置信区间。四项报告修订已经重新生成并核验：文本表格与 F440 卡片间的换行；F1724 明确为跨类型加性控制；TRAIN 表达关联 matched-null 无 q<0.05、最小 q≈0.0703 的空结果进入主文；训练表述为 20 轮预算、未验证充分收敛。审计者另查到 18 个 K32 模型的 best_epoch 全为 20 且最后一轮 VALID MSE 仍下降。读者报告的数值和科学范围符合上述审计结论。本轮未进行 HTML 视觉排版验收。
