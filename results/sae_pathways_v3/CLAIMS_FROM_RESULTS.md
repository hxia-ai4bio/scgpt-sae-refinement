# Claims from Results

Date: 2026-10-18 · Reviewer: GPT-6-Astra ultra, fresh Codex agent.
review_independence: **same-family** · acceptance_status: **provisional**.

**claim_supported: yes — only for the requested descriptive annotation task.**
**Integrity: WARN. Confidence: high in this limited conclusion.**

The results support a traceable mapping from the F160 positive TRAIN attribution lists to fixed-version GO BP, Reactome and Hallmark terms, including their actual overlapping genes. The declared 6,462-test global family contains 42 ordinary-ORA findings and **zero matched-null global-FDR findings**; minimum matched q is 0.3230676932306769.

The full list has translation-related annotations. The non-RPL/RPS list has ordinary-global-ORA mRNA-catabolism and post-transcriptional candidates. Antigen-processing and IFN-γ examples are useful gene annotations but fail ordinary as well as matched global FDR. These are candidate biological labels, not validated functional modules.

The results do **not** establish pathway activation or inhibition, an independent/stable program, biological causality, independent TEST replication, infection, or absence of biological meaning. Positive attribution describes a frozen model's response to the stated input replacement, not a biological intervention. Coarse matching leaves residual covariate differences; gene selection/background and overlapping annotations constrain interpretation.

## Permitted working claim

F160 的正向 TRAIN 归因基因可映射到翻译、mRNA 分解及转录后调控等候选注释；在当前覆盖基因背景、联合分箱匹配与 6,462 项全局 BH 校正下，没有通路通过匹配随机 FDR 0.05。对应基因可用于后续核查，不代表已确认或被激活的功能程序。

## Route

**Confirm the narrow claim and deliver with WARN.** No supplementary experiment is needed for this user's annotation request, and none was launched. Stronger claims would require independent selection/donor validation, robustness to cutoff/background/matching, and independent biological perturbation evidence. Do not expand this completed task merely to obtain significance.

Evidence: `all_pathway_results.csv`, `candidate_pathway_summary.csv`, `candidate_pathway_genes.csv`, `EXPERIMENT_AUDIT.md`; independent numerical and lineage checks in `work/review_trace/sae_pathways_v3/`.
