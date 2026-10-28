# SAE 评估实验审计

日期：2026-10-19。结论：**WARN；已完成结果可用于限定范围的描述，广泛改善结论不成立。**

审查路线：fresh `gpt-6-astra` / `ultra` 请求，`fork_turns=none`，代理 `/root/sae_eval_audit`。实际解析后的模型 ID 未由工具公开。`review_independence=same-family`，`acceptance_status=provisional`。确定性文件与数值核查通过，不将此等同于跨模型家族的语义认可。

## A. 真值来源：PASS

Cell-type labels, donor/batch IDs and cell IDs are loaded from raw H5AD annotations and matched by source row to cached embeddings. IFN targets are an eight-gene mean of log1p(count/library*10000), derived from the same RNA and explicitly called a proxy. Raw sparse matrices are canonical nonnegative integer counts. No independent perturbation/protein/functional ground truth is supplied; provenance is verified to local dataset annotations, not an external reannotation.

证据：`work/sae_evaluation/run.py:62-88`；`work/sae_extension/prepare.py:30-46`；`work/sae_evaluation/verify.py:17-20,59-62`。

## B. 分数归一化：PASS

Raw fixed-eight-class Macro-F1, R2, Spearman, MAE, ARI/NMI and cosine silhouette are reported. AULC divides by a common within-fold log-label-count range, not by predictions. Reconstruction R2 uses original TRAIN-centered h variation. Geometry KL is original-h teacher fidelity, not biological ground truth. Negative R2 and performance losses remain visible.

证据：`work/sae_evaluation/run.py:94-95,108-121,124-140`；`work/sae_evaluation/report.py:24-27`。

## C. 结果存在与完成状态：PASS

All 98 expected blocks exist; reviewer independently checked the same run.py SHA in all 98 completions and all 686 bound files. Aggregate row counts: kNN392, per-class3136, few-shot2940, IFN560 including14 predeclared skips, clustering294, geometry196. Standard evidence precheck reports67 verified values; these verify existence, not semantic claims.

证据：`outputs/sae_evaluation/verification.json`；`outputs/sae_evaluation/EVIDENCE_PRECHECK.json`；`outputs/sae_evaluation/blocks/**/completion.json`；`work/sae_evaluation/run.py:155-178`。

## D. 实际执行：PASS

The endpoint routines are called inside evaluate_block, and their predictions, metrics and completion hashes are present. Geometry runs for TRAIN and TEST; readout, IFN, clustering and silhouette have actual outputs. This is frozen-cache evaluation, not new SAE/scGPT training.

证据：`work/sae_evaluation/run.py:97-178`；`work/sae_evaluation/verify.py:31-67`。

## E. 范围与重复单位：WARN

Kang ctrl9577 and stim9728 cells share six historical donors; PBMC4638 cells supply two batch directions, not verified donors. Seeds and cell/subset replicates are not biological replication. IFN uses five ctrl B-cell donors, six ctrl CD14 donors, six stim donors per type, two PBMC B-cell directions, and only one PBMC CD14 direction. Historical endpoint exploration makes results exploratory. Unified ctrl/stim, UMI downsampling, new independent cohorts and the full seven-lambda grid on these endpoints were not run.

证据：`outputs/sae_evaluation/PROTOCOL.json`；`outputs/sae_evaluation/PREFLIGHT.json`；`work/sae_extension/run.py:70-85`；`work/sae_evaluation/report.py`。

## 数据划分与模型一致性：PASS

TRAIN/VALID/TEST rows are disjoint. TRAIN and TEST groups are disjoint; PBMC VALID shares the SOURCE batch by design. Every reconstruction arm uses one frozen dataset/fold/seed SAE for TRAIN references and TEST queries, with its TRAIN mean checked. Final verification matches all84 training/VALID input hashes and frozen split rows. No TEST-label parameter selection occurs in this evaluation.

证据：`work/sae_evaluation/run.py:48-60,81-86,103-104`；`work/sae_evaluation/verify.py:31-42`；`outputs/sae_evaluation/PREFLIGHT.json`。

## 固定选择与标签预算：WARN

Fixed lambda0.3, Ridge alpha10, k15 primary, Leiden resolution1 primary; sensitivities are declared. Few-shot rows are nested without replacement and identical across methods and SAE seeds; weights are class-balanced from selected labels. All TRAIN unlabeled representations fit scalers. Full TRAIN labels are used to construct stratified subsets, so labeled_n is readout-fit label count, not prospective annotation cost. Inherited VALID-MSE SAE selection is disclosed and uses no cell-type labels itself. Prior research choices remain historically explored. Fixed-eight-class F1 includes zero for missing TEST classes; rare PBMC classes can have only one TRAIN cell.

证据：`work/sae_evaluation/run.py:22-24,91-93,123-140`；`work/sae_complementarity/core.py:68-69`；`work/sae_extension/run.py:118-140`；`work/sae_evaluation/report.py`。

## 汇总方式：PASS

Within-fold SAE seed and subset repeats are averaged before equal group weighting. Original h is evaluated once per fold. Contrasts use paired heldout groups, and positive-group counts are descriptive, not significance. No independent-cell/seed inference or pooled ctrl+stim donor count is claimed. PBMC two directions are dependent evaluations of the same two batches.

证据：`work/sae_evaluation/report.py:12-37`；`outputs/sae_evaluation/*_paired_groups.csv`；`outputs/sae_evaluation/planned_contrasts.csv`。

## 数值复算：PASS

Custom weighted Ridge agrees with sklearn on primal/dual fixtures. Reviewer independently refit all30 few-shot classifiers for PBMC fold0 relation0.3 seed42 using sklearn: zero class-prediction disagreements; two within-type IFN embedding readouts agree within1.2284617767477357e-13. sklearn emitted repeated divide/overflow/invalid matmul warnings during this independent replay, despite matching finite outputs. Final verification replays all98 silhouettes from explicit SciPy-BLAS precomputed distances within1e-10, first8 neighbor queries per block, all saved classification/IFN/clustering metric arithmetic and raw target linkage. This does not mean every model fit or every neighbor was independently refit.

证据：`outputs/sae_evaluation/READOUT_VERIFICATION.json`；`outputs/sae_evaluation/BACKEND_VERIFICATION.json`；`outputs/sae_evaluation/verification.json`；`work/sae_evaluation/check_readout.py:5-8`；`work/sae_evaluation/verify.py:40-65`。

## 证据绑定与续跑：PASS

Final snapshot verifies90 source,95 dependency and3 target hashes. All98 blocks carry identical run.py hash e842231a61d2483003ccc1724cc86f88d340340e6b675f9ca808e8d5e282af9f. Initial run.py did not check dependencies/derived-target caches; those manifests were augmented during execution. Current data/checkpoints/splits/targets are subsequently validated, and future resume must use run_checked.py. Initial backend note wrongly described an unimplemented replacement; it was corrected transparently without changing runtime or results. Environment versions are recorded.

证据：`outputs/sae_evaluation/SOURCE_BINDINGS.json`；`outputs/sae_evaluation/DEPENDENCY_BINDINGS.json`；`outputs/sae_evaluation/TARGET_BINDINGS.json`；`work/sae_evaluation/run_checked.py:5-7`；`work/sae_evaluation/verify.py:12-20,31-39`；`outputs/sae_evaluation/REPRODUCIBILITY.txt`。

## F. 评估类型

- 分类及 ARI/NMI：数据提供的细胞类型注释（real_gt）；未重新独立验证注释。
- IFN：同一 RNA 派生的八基因分数（self_supervised_proxy），不是独立生物学真值。
- 几何：原始 scGPT h 作为教师（synthetic_proxy），只能说明几何保真。
- silhouette：基于同一细胞类型标签的几何描述，不能作为另一份独立生物学验证。

## 已解决的问题与剩余边界

最初后端说明误写为已替换成预计算距离，实际运行一直是 sklearn cosine silhouette；现已透明更正，并以全部98块显式BLAS距离复算核对。最初依赖和目标缓存未进入续跑检查，现已补齐实际协议、84个训练记录和目标缓存绑定，并核对训练输入与划分。它们是本次审计发现后修正的可复现性问题，没有据此宣称早先就已冻结这些增补文件。

保留历史数据复用、条件间供体共享、PBMC缺少供体、IFN同源代理、少标签抽样需要完整类别标签等限制。不存在从当前结果推出独立生物学改善、普适收益或因果机制的依据。后续续跑使用 `run_checked.py`。

完整哈希快照、逐项证据与独立复算范围见 `EXPERIMENT_AUDIT.json`；结论判断见 `CLAIMS_FROM_RESULTS.md`。
