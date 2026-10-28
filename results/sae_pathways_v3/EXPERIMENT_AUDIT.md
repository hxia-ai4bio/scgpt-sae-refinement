# Experiment Audit Report

Date: 2026-10-18 · Auditor: GPT-6-Astra ultra, fresh agent `/root/pathways_v3_audit`.
Backend: Codex · review_independence: **same-family** · acceptance_status: **provisional**.
Executor exact model: unknown (GPT-6 family reported). No model inference or scientific-output edits were performed by the reviewer.
Paths below are relative to `/Users/apple/Documents/Codex/2026-10-18/wo`.

## Overall verdict: WARN

**Deterministic checks: PASS. Integrity status: warn.** No fabricated ground truth, self-normalized performance score, nonexistent result, or material arithmetic/membership error was found. WARN reflects inference limits and null-model assumptions, not a failed numerical calculation.

| Check | Status | Finding and evidence |
|---|---|---|
| A. Ground-truth provenance | PASS | This is annotation association, not pathway ground truth. Fixed GMT definitions and all 7,583 GO JSON membership/ID mappings agree. `work/sae_pathways_v3/run.py:13-26`; `annotate_go.py:5-10`. |
| B. Score normalization | PASS | Raw overlap, expected overlap, fold enrichment and `(tail+1)/10001` are reported; no metric is divided by the model's own maximum. Raw attribution is retained. `run.py:37-50`; upstream `work/sae_gene_attribution_v2/run.py:53-61`. |
| C. Result existence | PASS | 6,462 term tests, 42 ordinary global-BH findings, 0 matched findings, minimum matched q 0.3230676932306769. All 5,500 pathway/gene links and 51 highlighted gene rows match source membership/attribution. `outputs/sae_pathways_v3/summary.csv:1-7`; `work/review_trace/sae_pathways_v3/lineage_membership_checks.json`. |
| D. Executed code | PASS | Main calls, logs, saved draws, completed tracker and independent seed replay agree. `run.py:68`; `verify.py:38`; `run.log:1-9`. |
| E. Scope | WARN | One feature, two top40 lists, 8 TRAIN prototypes/4 TRAIN donors, one historical TEST donor. Candidate annotations cannot establish an independent program or biological cause. `outputs/sae_gene_attribution_v2/protocol.json:5-17`; `work/sae_pathways_v3/report.py:44-47`. |
| F. Evaluation type | self_supervised_proxy | Closest taxonomy: annotation proxy of model sensitivity. External gene sets are not ground-truth mechanism labels. |

Independent reconstruction verified the 1,838-gene input-covered universe, source ranking, prototype coverage and raw-count/detection metrics from the correct 941 TRAIN B ctrl cells. All ordinary/Fisher p values, global and within-analysis BH values, term filters, 20,000 random sets, matched tails/means and seed replay agree (maximum numerical error below 2.7e-14). See `work/review_trace/sae_pathways_v3/statistical_checks.json`.

## Limits and reproducibility notes

- Matching preserves joint bins, not exact covariate values. Selected detection-rate means remain 3.16 and 2.79 null SD above their respective random-set means. JUN is fixed in every all-input random set because its stratum has one eligible gene. Conditional sampling is implemented correctly; full confounder removal is not established.
- Global BH includes both lists and all collections. Keep **global** in every enrichment statement: non-RPL/RPS has 8 ordinary global-BH findings, but 0 under its own within-analysis BH. Overlapping terms are not independent mechanisms.
- CSV-only rebinning moves 8 + 5 genes at floating-point tie boundaries; none is selected or changes a used sampling pool. This does not affect current results. Saved NPZ `strata/indices` are authoritative for exact null replay; future rebinning should use pinned raw inputs or deterministic tie handling.
- The official Monte Carlo floor is 1/10001. Reviewer-only exact conditional convolution also finds zero global-FDR discoveries (minimum q 0.262121), so this run's lack of findings is not solely a rounding-to-zero artifact. Neither analysis proves absence of biology.
- Highlighted antigen-processing genes are CD74/HLA-B/HLA-A; IFN-γ genes add PSME2/BTG1. Both entries fail ordinary global FDR. CD79A is correctly kept outside these overlap lists. GO IDs and all highlighted memberships match the source tables (`candidate_pathway_summary.csv:2-6`).

## Action and claim impact

Deliver the requested candidate annotations and genes with WARN, retaining current no-activation/no-causality language. Rerender the audit label and document saved-null replay. No scientific rerun is required for this task. Stronger functional-program, activation, mechanism, or absence claims are unsupported.

Input hashes and full request/response/parsed/routing records are recorded in the JSON and trace directory. Final report presents WARN and the saved-null replay note; final input hashes are sealed in the JSON. Scientific tables and caches are unchanged since review.
