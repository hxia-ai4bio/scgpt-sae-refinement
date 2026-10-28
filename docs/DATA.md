# Data and model assets

| Dataset | Cells in final analysis | Group units |
|---|---:|---|
| Kang ctrl | 9,577 | six development donors |
| Kang IFN-stimulated | 9,728 | the same six donors |
| PBMC | 4,638 | two batches, not verified independent donors |

Kang source identifier: **GSE96583**; original local input `kang.h5ad`.
The historical PBMC input is `PBMC368k_raw.h5ad` with batches `Batch1_3k` and
`Batch2_68k_r`. Its complete download/version provenance remains unresolved;
do not identify it as a different public PBMC cohort based only on its name.
Dataset row order, labels and preprocessing must match the recorded manifests.

Assets not redistributed:

- RNA count matrices (`.h5ad`), per-cell caches (`.npz`) and prediction arrays.
- scGPT whole-human checkpoint `best_model.pt`, `args.json`, `vocab.json`.
- Initial and fine-tuned SAE checkpoints; expression heads and masked-input caches.
- Third-party gene-set databases (e.g. MSigDB); obtain them under the provider's terms.

Official sources:
- Kang GEO accession: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583
- scGPT code / model instructions: https://github.com/bowang-lab/scGPT
- MSigDB: https://www.gsea-msigdb.org/gsea/msigdb/

Input fingerprints, class order and preprocessing are preserved in
`archive/historical_project/experiments/sae_topk_20261014/data_manifest.json`
and `results/sae_extension/DATA_MANIFEST.json`.
Do not change tokenization, pooling, split membership or label order while
claiming reproduction of the existing numbers.

Original extraction uses final layer 12 signed masked mean pooling over valid
gene tokens, max sequence length 512, 51 expression bins and at least 16
vocabulary-matched expressed genes. No CLS or absolute-value pooling.

Six-donor Kang splits use four TRAIN / one VALID / one TEST donor.
PBMC uses one source batch (with about 20% held out internally for VALID) and
the other batch as TEST, then swaps the direction. Raw data and checkpoint
availability are therefore prerequisites to full experimental reproduction.
