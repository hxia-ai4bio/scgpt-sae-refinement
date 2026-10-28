# Code map

| Study component | Original scripts | Report evidence |
|---|---|---|
| Frozen scGPT preprocessing / tokenization | `archive/historical_project/experiments/kang_error_atlas_20261012/run_error_atlas.py`, `sae_topk_20261014/prepare.py`, `archive/upstream/train_sparse_modules.py` | `handoff/evidence/original/` |
| TopK SAE warmup / basic probes | `archive/historical_project/experiments/sae_topk_20261014/` | historical protocols and data manifest in the same directory |
| Cell-type-associated features | `archive/work/sae_interpretability/`, `sae_figures/compute_all_celltypes.py` | `handoff/data/figure3/` |
| Gene substitution / B-cell features | `archive/work/sae_gene_attribution_v2/`, `sae_B898_1387/`, `sae_B_remaining/` | `handoff/data/figure4/`, `results/` |
| GO/pathway candidate annotation | `archive/work/sae_pathways_v3/`, B-cell `pathways.py` | `results/sae_pathways_v3/` |
| Gate / low-rank residuals | `archive/work/sae_residual/`, `sae_residual_diagnosis/` | matching `results/` folders |
| Concatenation / residual complementarity | `archive/work/sae_complementarity/` | matching `results/` folder |
| SAE capacity / reconstruction variants | `archive/work/sae_refine/` | matching `results/` folder |
| MSE + relation KL | `archive/work/sae_relation/` | `results/sae_relation/` |
| Lambda sensitivity, Kang ctrl | `archive/work/sae_lambda/` | `results/sae_lambda/` |
| Kang stim / PBMC refinement | `archive/work/sae_extension/` | `results/sae_extension/` |
| kNN, few-shot, state, geometry, clustering | `archive/work/sae_evaluation/`, `sae_evaluation_seed_summary/` | matching `results/` folders |
| Masked input / expression objectives | `archive/work/sae_masked_ac/`, `sae_expression_probe/` | matching `results/` folders |
| Expression checkpoint selection and extension | `archive/work/sae_checkpoint_selection/`, `sae_checkpoint_extension/` | matching `results/` folders |
| Figures 3–4 | `archive/work/sae_figures/` | `handoff/figures/`, `handoff/data/` |
| Table 1 / Figures 5–6 | `figures/plot_results.py`, `figures/plot_secondary.py` | `figures/data/` |

Python code under `archive/` is preserved, including exploratory arms that did
not show gains. `src/sae_refinement/core.py` copies the key numerical definitions
verbatim; `provenance.json` identifies their original source. `train.py` is the
new portable continuation interface, distinct from historical run orchestration.
