from pathlib import Path
import importlib.metadata as md
import json,sys,hashlib,zipfile
root=Path.cwd(); out=root/'outputs/sae_interpretability'; work=root/'work/sae_interpretability'
versions={'python':sys.version,'packages':{}}
for name in ['numpy','pandas','scipy','torch','scikit-learn','anndata','scanpy','matplotlib','numba','h5py','scgpt','torchtext']:
    try:versions['packages'][name]=md.version(name)
    except md.PackageNotFoundError:versions['packages'][name]='not registered; may be source import'
(out/'ENVIRONMENT.json').write_text(json.dumps(versions,indent=2))
(out/'REPRODUCIBILITY.txt').write_text('''SAE biological interpretability — 2026-10-18 exploratory run

Start with REPORT.html or RESULTS_ZH.txt. All CSV/JSON tables are included.
Scope: existing frozen K32 SAE models, Kang control cells and PBMC; no retraining or stimulation analysis.

The archive preserves work/sae_interpretability and outputs/sae_interpretability relative layout.
It includes analysis scripts, cached activations and perturbation inputs/scores, execution logs, deterministic checks and reviewer records.
Original scGPT/SAE checkpoints, original RNA data, historical project code and gene-set files are external prerequisites, NOT included. This is an analysis archive, not a self-contained data/model distribution.

Original local prerequisites:
- Project: /Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7
- Historical SAE experiment: experiments/sae_topk_20261014 under that project
- Gene sets: experiments/dense_validation_20261006/gene_sets under that project
- RNA: /Users/apple/Desktop/Llava-Cell/dataset/kang.h5ad and PBMC368k_raw.h5ad
- scGPT: /Users/apple/Desktop/scGPT/scGPT_human
Exact inputs/hashes: input_manifest.json, all_fold_model_hashes.json, gene_set_provenance.json.
Environment: ENVIRONMENT.json. Original interpreter: /opt/anaconda3/envs/scgpt/bin/python.

On the original machine, analysis scripts take no command-line arguments. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=2, VECLIB_MAXIMUM_THREADS=1 and MPLCONFIGDIR to a writable directory before starting.
Use a separate extracted copy to preserve this result snapshot. Ordered fresh reconstruction: 01_atlas.py, 02_forward.py, 03_occlusion.py, 04_expression.py, 05_all_folds.py, 06_enrichment.py, 07_gate_diagnostics.py, 08_dictionary_controls.py, 11_stronger_null.py, 09_verify.py, 10_report.py.
Do not regenerate the scientific reviewer verdict merely by rerunning scripts; it is a separate review.
Caches are input/code-bound. If intentionally changing a scientific input or producer, use a new output/cache directory rather than deleting validation guards.
The executed_v1 directory preserves the initial producer version before review-driven cache validation improvements; current cached results were independently replayed and bound by 09_verify.py.

Arithmetic and replay checks PASS; interpretation review WARN/partial, same-family provisional. Local perturbations cover one test donor and two selected cells per feature, not population causal validation. The historical 20-epoch model budget does not establish convergence.
The report generator was edited after the deterministic check for presentation wording only. FINAL_MANIFEST.json records the final package hashes; deterministic_verification.json preserves the earlier check-time hashes.
''')
if '--archive' in sys.argv:
    paths=[]
    for base in [work,out]:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or any(v in p.parts for v in ['__pycache__','numba_cache','matplotlib_cache']):continue
            if p.name in ['finalize_report.py','FINAL_MANIFEST.json']:continue
            paths.append(p)
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'FINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    paths.append(out/'FINAL_MANIFEST.json')
    target=root/'outputs/SAE_interpretability_20261018.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(root))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        print(json.dumps({'archive':str(target),'files':len(z.namelist()),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
