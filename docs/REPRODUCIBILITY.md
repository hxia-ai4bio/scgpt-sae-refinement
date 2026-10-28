# Reproducibility / 运行说明

## 1. What works without raw data

1. `python scripts/verify_release.py` checks the release manifest and Python syntax.
2. `python handoff/verify_handoff.py` recomputes the reported table, lambda curve,
   donor AUROC aggregation and downstream means from included records.
3. `python figures/plot_results.py` regenerates Table 1 and the lambda figure.
4. `python figures/plot_secondary.py` regenerates the three downstream panels.
5. `python -m unittest discover -s tests -v` checks numerical definitions and splits.

These checks do not rerun neural training, establish biological causality, or
prove that independent users can reproduce every paper number from raw data.
The delivered paper is v8: Code Availability was added after the conclusion.
The experimental results are unchanged.

## 2. Portable continuation (from an existing SAE)

Install `python -m pip install -e '.[figures]'`. For the recorded CPU environment,
use Python 3.9 and `requirements.txt`; other platforms may require a matching
PyTorch wheel. The installed scGPT environment used scgpt 0.2.4. Installing its
full raw-input pipeline also requires a compatible torchtext / PyTorch combination;
it is not necessary for plots or training from cached embeddings.

Cache schema (NumPy NPZ, `allow_pickle=False`):

- `embedding`: finite float array `[N,512]`.
- `labels`: integer class IDs `[N]`, 0..7, same class order as the protocol.
- `groups` (or `donors` / `batches`): string IDs `[N]`.

Split JSON schema: `{"train": [0,1,...], "valid": [...], "test": [...]}`.
Arrays are positions in the NPZ cache, **not raw h5ad row IDs**. No overlapping
rows are allowed. TEST groups must be unseen in TRAIN/VALID. For Kang,
TRAIN/VALID groups must also be disjoint. PBMC source TRAIN/VALID can share a batch.
Use `configs/kang_folds.json` for the actual six donor rotations; convert group
masks into row indices for your cache. PBMC split records are retained in
`results/sae_extension/pbmc/SPLITS.json` if available; do not invent donor IDs.

Checkpoint: original raw-value TopK SAE state dictionary with `mean`,
`encoder.weight`, `encoder.bias`, `decoder.weight`. Width 2048, k=32.
It must have been fitted on TRAIN and selected on VALID MSE; the CLI checks
its stored mean against TRAIN but cannot infer whether the original training
used held-out data. The initial warmup training code is
`archive/historical_project/experiments/sae_topk_20261014/run.py`.

The new interface:

- continues for 1000 fixed steps with reset Adam, cosine LR 3e-4 → 1e-5;
- minimizes MSE / TRAIN centered mean-square + lambda KL;
- uses temperature 0.1, fixed TRAIN mean, float64 relation softmax;
- samples balanced TRAIN groups for cross-group mode, non-self anchors otherwise;
- selects an 11-value balanced Ridge regularization grid on VALID;
- scores TEST only after both original/reconstruction readouts are selected.

For Kang with the same checkpoint, split ordering, fold, seed, batch size and
step count, the batch generator follows the original relation experiment.
For PBMC, the convenience interface uses a general non-self sampler, **not the
historical extension's exact RNG stream**. Full historical implementation is
preserved in `archive/work/sae_extension/run.py`.
The portable interface is a newly packaged route; no new paper result is claimed.

Training SAE costs O(S(B d m + B²d)); memory O(Nd + Bm + B² + dm).
Ridge costs O(Nd² + d³). Here d=512, m=2048. Default is CPU with four threads.
Apple MPS is not supported by this runner because relation calculations use float64.

## 3. Full historical workflows

The original scripts are byte-identical copies (see `SOURCE_MANIFEST.json`).
`archive/work/` mirrors the project work directory; `archive/historical_project/`
contains earlier dependencies. Original absolute paths remain in the archive.

Historical execution chain:

1. `kang_error_atlas_20261012/run_error_atlas.py`: cohort/tokenizer and scGPT extraction;
   imports `train_sparse_modules.load_dense` (copied under `archive/upstream/`).
2. `sae_topk_20261014/prepare.py` and `run.py`: eight-class caches, warmup TopK SAE.
3. `work/sae_interpretability`, `sae_gene_attribution_v2`, `sae_pathways_v3`,
   `sae_B898_1387`, `sae_B_remaining`: feature selection, substitution and annotation.
4. `work/sae_relation`, `sae_lambda`, `sae_extension`: MSE/KL continuation and scans.
5. `work/sae_evaluation`: kNN, state, few-shot, clustering and geometry diagnostics.
6. `work/sae_masked_ac`, `sae_expression_probe`, `sae_checkpoint_selection`,
   `sae_checkpoint_extension`: separate expression-supervised pipeline.

To rerun: restore the input assets listed in DATA.md, adapt paths in a **working
copy**, and initialize fresh protocols / source bindings for a new experiment.
Do not silently disable hash checks or relabel old audit seals as validating
new code. Many historical run stages require predecessor checkpoints/caches and
source manifests. The archive is complete source provenance, not a one-command
fresh-machine pipeline. Exact historical end-to-end reruns were not performed
for this code release.

The original handoff's local paths document the source machine; they are not
required to read the paper or regenerate the portable summary plots.

## 4. LaTeX / Overleaf

Upload `paper/overleaf.zip` to Overleaf; set `main.tex` as the main file and use
pdfLaTeX + BibTeX. Or run `latexmk -pdf main.tex` from `paper/latex/`.
The NeurIPS preprint style is formatting only, not an acceptance claim.
