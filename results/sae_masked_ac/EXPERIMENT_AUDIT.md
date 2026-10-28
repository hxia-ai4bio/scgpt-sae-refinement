# Experiment Audit Report

**总体结论：WARN；实现与数值核验通过，保留研究范围和代理指标限制。**

审查者为新上下文Codex代理 `/root/masked_ac_audit`，请求路线为gpt-6-astra / ultra；工具未暴露实际解析后的模型ID。与执行者属同模型家族，因此语义审查为 **same-family / provisional**，不是跨家族正式接受。

没有发现需要停止报告的泄漏、虚构结果或预测链断裂。新模型126个，既有MSE42个，固定配置182个，clean/masked视图364组；23,943个源细胞，其中Kang ctrl9,577、stim9,728，PBMC4,638。

## A. 真值来源：PASS

细胞类型来自原始数据标签；C目标来自原始RNA的logCP10k阳性token；IFN来自同一RNA的固定八基因分数。A的clean-h目标和几何指标来自冻结scGPT，已明确为模型保真度代理。没有把模型输出冒充独立生物学真值。

证据：`work/sae_masked_ac/prepare.py:24`；`work/sae_masked_ac/prepare.py:69`；`work/sae_masked_ac/pipeline.py:32`；`work/sae_masked_ac/pipeline.py:68`；`work/sae_evaluation/run.py:78`；`reviewer_prediction_checks.json`。

## B. 分数归一化：PASS

A训练尺度由clean TRAIN中心化均方确定；C均值/标准差只用TRAIN阳性表达，原logCP10k MAE与归一化MSE同时报告；几何误差以固定教师h相对TRAIN均值的能量作分母。没有用预测最大值或预测误差自身缩放成绩。cosine的逐向量L2归一化是定义本身，不是成绩夸大。

证据：`work/sae_masked_ac/pipeline.py:36`；`work/sae_masked_ac/pipeline.py:80`；`work/sae_masked_ac/pipeline.py:159`；`work/sae_masked_ac/pipeline.py:181`。

## C. 结果存在性：PASS

126个新模型、42个既有MSE模型、182个配置、364个clean/masked读出均有实际文件；36条已提交数值存在性检查均verified。独立复算汇总及对照差值；文件存在性不替代科学结论。

证据：`outputs/sae_masked_ac/verification.json`；`outputs/sae_masked_ac/EVIDENCE_PRECHECK.json`；`outputs/sae_masked_ac/SELECTION_LOCK.json`；`reviewer_final_dependency_hashes.json`。

## D. 评价代码执行：PASS

分类、15NN、IFN、几何和C表达函数均由evaluate执行并写入结果；全部364视图、2028个IFN指标和42个表达指标文件已复算。未执行的few-shot、聚类和B目标明确列为not_run。

证据：`work/sae_masked_ac/pipeline.py:161`；`work/sae_masked_ac/pipeline.py:191`；`outputs/sae_masked_ac/verification.json`。

## E. 实验范围：WARN

六个历史Kang供体在ctrl/stim中重复使用；PBMC仅两个批次方向且供体/独立研究/预训练重合未确认。一个固定25%已表达token遮蔽视图不是任意缺失或UMI下采样；C只评估阳性且原本被选中的基因。结论限于固定1000步方案，不能称普遍、独立盲测或SAE机制验证。

证据：`outputs/sae_masked_ac/PROTOCOL.json:48`；`outputs/sae_masked_ac/PROTOCOL.json:50`；`outputs/sae_masked_ac/PROTOCOL_NOTES.json`；`work/sae_masked_ac/report.py:55`。

## F. 数据隔离与冻结：PASS

独立核查14个划分及原始ID/标签/组对应；所有TRAIN/VALID/TEST源行不重叠，Kang供体不重叠，PBMC目标批次与源批次分离。scGPT冻结；PBMC源批次中的TRAIN/VALID细胞划分有标签分层，但标签未进入神经目标。

证据：`reviewer_data_precheck.json`；`work/sae_masked_ac/prepare.py:39`；`work/sae_masked_ac/prepare.py:56`；`work/sae_masked_ac/prepare.py:78`。

## G. 遮蔽及梯度路径：PASS

隐藏基因计数从可见library total和分位数构造中移除，隐藏位置为padding，池化排除隐藏token。原始表达存在/选择条件仍保留并已披露。C的线性头仅接收实际512D SAE输出。126个模型所有可训练参数张量均改变，固定mean/sd不变，step2梯度均为正且有限。

证据：`work/sae_masked_ac/prepare.py:31`；`work/sae_masked_ac/prepare.py:34`；`work/sae_masked_ac/prepare.py:58`；`work/sae_masked_ac/pipeline.py:51`；`work/sae_masked_ac/pipeline.py:71`；`reviewer_extra_integrity.json`。

## H. 对照与选择：PASS_WITH_QUALIFIERS

A/C与clean-MSE逐fold/seed共享warmup、1000个批次及优化器/学习率预算；C额外头增大总参数/FLOPs，不能称计算量完全匹配。A_dense参数数目接近但恒等初始化/激活不同。Ridge各方法同一网格由clean VALID选alpha；每个已拟合读出共同评估clean/masked TEST，固定最终step1000，无新TEST checkpoint/seed选择证据。

证据：`reviewer_data_precheck.json`；`work/sae_masked_ac/pipeline.py:25`；`work/sae_masked_ac/pipeline.py:63`；`work/sae_masked_ac/pipeline.py:117`；`work/sae_masked_ac/pipeline.py:151`；`outputs/sae_masked_ac/PROTOCOL_NOTES.json`。

## I. 数值和预测链：PASS

182个TRAIN标准化/VALID网格/加权正规方程、364组分数和有限embedding检查通过。生产验证的checkpoint replay为每视图前至多512细胞，NN为8个query，C头为有界entry；不声称完整原始RNA逐细胞scGPT重跑。审查者独立代码补充180个IFN拟合预测与相应完整TEST forward，最大IFN误差2.5868e-13，forward误差0；全部42个C均值/标准差基线和六个表达指标通过。364视图手工混淆矩阵指标最大误差1.1102e-16。

证据：`outputs/sae_masked_ac/verification.json`；`reviewer_prediction_checks.json`；`reviewer_extra_integrity.json`。

## 评价类型

|端点|类型|可以支持的解释|
|---|---|---|
|细胞类型Ridge/15NN|real_gt，原数据注释|该划分上的标签读出性能|
|C表达恢复、IFN|self_supervised_proxy，同RNA|目标恢复与代理读出，非独立生物学验证|
|A重建、教师几何|synthetic_proxy，冻结模型教师|表示保真度，不等同生物学改善|

## 额外独立核验

- 14个划分、42对warmup/clean-MSE、1091个已绑定输入/模型/读出依赖哈希重核。
- 180个独立拟合的IFN预测检查，最大绝对误差2.5868e-13；这些块的完整TEST手写forward与保存embedding一致，最大误差0。
- 原RNA IFN目标/协变量对23,943行重算；float32原计算与float64重算的最大差4.8273e-7，预期浮点精度范围内。
- 42个C预测文件的TRAIN逐基因均值、标准差及全部六个表达指标通过。
- 364个视图用手工混淆矩阵复算Macro-F1、supported-Macro-F1及Ridge准确率，最大差1.1102e-16。
- 126个模型所有可训练参数张量均改变，mean/sd固定，第二步各参数梯度均正且有限。

上述检查有明确覆盖边界：生产脚本checkpoint replay每视图至多512个细胞、NN每视图8个query、C head为有界entry；审查者IFN额外检查使用每数据集/类型的第一个满足阈值划分、seed42以及确定性原始基线。没有声称独立重跑整个原始RNA到scGPT提取过程。

## 保留限制

- Historically explored data; no independent new-study confirmation or verified absence of scGPT pretraining overlap.
- ctrl and stim share six donors; overlapping training folds, three seeds and many cells do not create additional biological replicates.
- PBMC has two batch directions, not verified donor holdout; CD14 IFN passes sample thresholds in only one direction.
- One fixed 25% mask over originally selected positive tokens; original presence/token selection is conditioned on; C excludes zero-expression and unselected genes.
- Neural refinement is trained on masked inputs, while readouts are fitted on clean refined inputs; this distribution shift is part of the fixed protocol.
- Full-RNA covariates in masked IFN sensitivity variants are oracle covariates; strict masked-input claims use embedding-only.
- C adds a training head; A_dense differs in initialization/activation. Neither comparison isolates sparse/SAE-specific biological mechanism.
- All42 C runs,41/42 A_sae runs and14/42 A_dense runs had a lower monitored VALID objective before step1000. Fixed-budget conclusions do not establish convergence or rule out other prespecified training recipes.
- No p-value or independent-study statistical confirmation; group/seed consistency is descriptive. Fixed-eight-class Macro-F1 and supported-class metrics differ when classes are absent.
- Forward/head/NN replays are bounded as stated; original scGPT mask extraction has source binding and16-cell clean replay plus hidden-magnitude invariance, not an independent full-cohort extraction replay.

## 处理

当前实验可以按固定方案的探索性结果完整报告；没有要求改模型或追加训练。必须保留全部方法、组、种子与clean/masked端点，并使用 `CLAIMS_FROM_RESULTS.md` 中的收窄结论。强结论需要另行预注册新证据。

审查输入的SHA256位于同名JSON；完整取证记录位于 `.aris/traces/experiment-audit/2026-10-19_sae_masked_ac/`。
