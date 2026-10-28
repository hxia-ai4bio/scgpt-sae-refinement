# Results → claims

**verdict: partial [INTEGRITY: WARN]**

`review_independence: same-family`; `acceptance_status: provisional`; confidence: high for the constrained judgment below. Deterministic checks passed; this does not upgrade biological claims to accepted.

## Supported, with precise scope

1. TRAIN-selected K32 SAE coordinates recover three chosen dataset labels in six historical Kang donor rotations. B-cell single-coordinate mean held-out AUC is 0.97561 versus original-coordinate 0.95322 and PCA 0.91273. Unequal dictionary sizes and selection constraints prevent attributing this difference to interpretability or sparsity alone.
2. B-cell readout persists in the examined PBMC batches (0.98423 / 0.96659 mean across trained dictionaries). Monocyte and NK external findings are weaker and must accompany it.
3. TRAIN gene attribution ordering transfers locally to two high-activation TEST cells per feature from one donor under fixed-token bin substitution. This is model-input sensitivity of a post-hoc coordinate, not gene causality or original-model reliance on that feature.
4. Source identities, saved numerical metrics, aggregation and tested forward replays are consistent. See EXPERIMENT_AUDIT.md/.json and deterministic_verification.json.

## Unsupported claims

- “We discovered three biological functional modules / causal gene programs.” Same-RNA association, technical-feature positives, modest gene-attribution agreement and one-donor occlusion do not establish this.
- “SAE is generally more interpretable or transfers better than embedding/PCA.” Monocyte PBMC SAE AUC is 0.50411 / 0.52711 versus PCA 0.95497 / 0.98253; NK SAE 0.86385 / 0.86762 versus original coordinate 0.97962 / 0.97148. Candidate dictionaries have unequal size.
- “Features are stable across seeds / independently replicated across 18 runs.” There are six donors and three seeds; crossseed decoder cosines are only 0.26–0.59, and crossfold training and initialization overlap.
- “Enrichment validates biological specificity.” TRAIN matched-null expression-association ORA has no q<0.05 result; technical comparator 1724 has 18 matched-null attribution hits. TEST/PBMC matched-null ORA was not evaluated. Each split independently selects its own top40 ORA genes.
- “scGPT depends causally on these SAE features.” SAE is a post-hoc map of the pooled embedding; no feature-ablation/patching experiment measures original-model dependence.
- “Independent external validation / faithful paper reproduction.” Donor/study/pretraining overlap in PBMC is not established, and this uses a limited fixed-bin occlusion/ORA protocol.

## Suggested claim revision

在这些历史数据上，部分 SAE 坐标可读出细胞类型相关信息，并表现出有限的局部输入敏感性排序迁移；B 细胞相关坐标的保留较好，单核细胞的外部迁移失败。当前证据支持探索性表示解释，不支持生物机制、普遍优势或稳定独立功能模块。

## Routing

Continue as a qualified exploratory analysis. For a stronger scientific claim, freeze candidates and gene lists before a new study/donor evaluation; use multiple references and representative test cells; extend the present perturbation-size/direction-matched null to representative independent donors and add dictionary-size/random-SAE controls. Biological mechanism claims additionally require orthogonal/experimental evidence. Do not turn three seeds or 10 random deletion orders into biological sample size.

## Supplemental matched-null review

The audit-motivated post-hoc stronger null is complete: 480 exact numerical inputs independently reproduced; saved scores and summary arithmetic agree within 2.22e-16. At 40 substitutions, strict-null mean drops are 0.142425 (F440), 0.146679 (F160), 0.082429 (F120), and 0.193059 (F1724), below the corresponding attribution-order means. The strict random sets reuse 9–25 of the 40 attribution genes. This improves local ranking-transfer evidence and narrows the gap relative to the original null, while leaving the same one-donor selected-prototype scope. **The claim verdict remains partial; biological specificity and causality remain unsupported.**
