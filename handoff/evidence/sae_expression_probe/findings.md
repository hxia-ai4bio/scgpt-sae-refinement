# Research Findings

2026-10-19 — matched frozen expression probe: **NO** for C superiority; audit **WARN** due scope, implementation/numerics PASS. Same-family/provisional review.

Tested84 fresh linear heads on fixed masked original_h/C_sae embeddings. Independent seed namespace was corrected before TEST; no old trained C head or v1 probe enters results. C primary token MAE is worse in42/42 pairs: ctrl+2.66%,stim+2.32%,PBMC+6.47%. Both arms beat TRAIN gene means. Kang gene-macro MAE has tiny mixed-direction improvements, which do not overturn the primary result.

Interpretation: current C expression supervision did not provide a primary masked-expression readout advantage under this probe recipe. Optimization/representation causes were not isolated; do not invent a causal explanation, call this irreversible information loss, or generalize to all SAE methods. All heads select step100; best-achievable linear decoding was not established.

Decision: report the qualified negative result and close/seal the fixed experiment. No test-driven tuning, checkpoint reselection, retraining, or new architecture is authorized. For evidence and exact bounds see EXPERIMENT_AUDIT.json and CLAIMS_FROM_RESULTS.json.
