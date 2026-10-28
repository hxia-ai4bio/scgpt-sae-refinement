from pathlib import Path
import sys,json,hashlib,zipfile,platform,importlib.metadata
import numpy as np,torch
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_residual_diagnosis'
(O/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','scikit-learn','matplotlib']}},indent=2))
(O/'REPRODUCIBILITY.txt').write_text('''Posthoc diagnosis of discarded SAE residuals. No TEST/PBMC scoring.
Read REPORT.html, diagnosis_facts.json, trajectory_summary.csv, conditional_readout_summary.csv and audit/claim files first. VALID maxima are exploratory after selection, not new generalization estimates. Original experiment and its archive remain unchanged.

External prerequisites: original work/sae_residual full workspace (including all candidate checkpoint tensors and metadata) and original sae_topk_20261014 datasets/SAE checkpoints/caches at the exact source paths in diagnose.py/experiment.py. These are not bundled. Original Python helper experiment.py and manifests are included as dependency/provenance references. Full numeric/checkpoint verification requires the original host files. Outputs are readable without the datasets.

From the original task root, using the interpreter/packages in ENVIRONMENT.json:
 export OPENBLAS_NUM_THREADS=1
 export OMP_NUM_THREADS=4
 python work/sae_residual_diagnosis/diagnose.py preflight
 python work/sae_residual_diagnosis/historical.py
 python work/sae_residual_diagnosis/diagnose.py trajectories
 python work/sae_residual_diagnosis/diagnose.py branches
 python work/sae_residual_diagnosis/summarize.py
All sources/results are hash-locked. New settings require a new experiment directory, not overwriting this one. Same historical six TRAIN/VALID splits; followup seed42 only, CPU4 threads. The source dataset contains all rows, but only TRAIN/VALID rows are extracted and scored. Existing TEST/PBMC metrics are not recomputed or used to choose these diagnostics.

ZIP contains all new scripts, metadata, saved VALID predictions, derivation tables, numerical/audit evidence and compact epoch40 parameters. Large per-epoch full checkpoints remain locally under work/sae_residual_diagnosis/trajectories and are excluded from ZIP. Compact epoch40 parameters omit frozen decoder.DT; reconstruct it from the original SAE dictionary (or its seeded orthogonal rotation) and use the recorded source hashes. Epoch40 is a diagnostic endpoint, not a validation-selected deployable model. Full checkpoints and original raw sources are needed to rerun all checkpoint verifications.
Warnings are retained in stage *_warnings.json. Independent numerical checks, validation-selection diagnostics and biological/generalization conclusions are separate. Random TopK controls match nominal dimension/K/search budget, not activation frequencies, correlations, effective active columns or semantics.
''')
if '--prepare-only' in sys.argv:sys.exit(0)
compact=O/'EPOCH40_COMPACT_PARAMETERS';compact.mkdir(exist_ok=True)
for p in (W/'trajectories').glob('fold_*/*/*/epoch_40.pt'):
 state=torch.load(p,map_location='cpu',weights_only=False);parts=p.relative_to(W/'trajectories').parts;np.savez_compressed(compact/('_'.join(parts[:-1])+'.npz'),**{k:v.numpy() for k,v in state.items() if k!='residual.DT'})
paths=[p for p in O.rglob('*') if p.is_file() and p.name!='FINAL_MANIFEST.json']
paths += [p for p in W.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.csv','.log','.npz','.md','.jsonl'] and not any(x in p.parts for x in ['__pycache__','matplotlib_cache'])]
paths += [ROOT/'work/sae_residual'/n for n in ['experiment.py','frozen_sources.json']]
paths += [ROOT/'outputs/sae_residual'/n for n in ['PROTOCOL.json','selection_lock.json','CLASS_LABELS.json']]
paths=sorted(set(paths));manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};(O/'FINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));paths.append(O/'FINAL_MANIFEST.json');archive=ROOT/'outputs/SAE_residual_diagnosis.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in paths:z.write(p,p.relative_to(ROOT))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'archive':str(archive),'files':len(paths),'bytes':archive.stat().st_size,'CRC':'PASS'}),flush=True)
