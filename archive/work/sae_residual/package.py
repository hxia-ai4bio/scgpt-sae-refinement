from pathlib import Path
import hashlib,json,zipfile,platform,sys,importlib.metadata
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_residual'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
env={'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','pandas','matplotlib','psutil']}}
(O/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2))
(O/'REPRODUCIBILITY.txt').write_text('''Frozen SAE residual comparison, K32, six donor rotations, seeds42/43/44.
Read REPORT.html and summary.csv first. all_metrics.csv and paired_donor_metrics.csv retain all arms, donors, seeds, readouts, identity fallback and negative effects.

No scGPT or SAE training occurred. Full original512-dimensional embedding remains in every residual arm. Gate and rotated-gate have2048 parameters; lowrank and MLP have40960. SupCon is a separately trained variant from zero residual and common CE head. All TRAIN/VALID selections are fixed before TEST scoring.

Reproduction requires the original SAE experiment at the absolute external source path in experiment.py and all its hashed datasets, checkpoints, latent caches and historical metadata. These are not bundled. Full training reproduction is for this original host/source layout, not a portable raw-data package. All helper code, full results/predictions, development metadata, selected model checkpoints and baseline heads are bundled; unselected candidate model tensors and the abandoned unbound partial run are excluded from the ZIP. Their provenance remains in full metadata/source manifests, and they remain on the original local workspace. Re-running development regenerates unselected candidates; a fresh output directory is necessary rather than overwriting existing hash-locked results.

Commands on the original host (Python version/packages in ENVIRONMENT.json):
 export OPENBLAS_NUM_THREADS=1
 export OMP_NUM_THREADS=4
 python work/sae_residual/experiment.py preflight
 python work/sae_residual/experiment.py benchmark
 python work/sae_residual/experiment.py develop
 python work/sae_residual/experiment.py evaluate
 python work/sae_residual/verify.py
 python work/sae_residual/report.py
The commands above use the original exact evaluation backend and work without pre-existing parity references. fast_evaluate.py is an accelerated resume option once at least one original-backend result exists; it requires exact parity with all initial saved queries before continuing, plus a duplicate/equidistant tie test. EVALUATION_BACKEND.json and evaluation_backend_amendment.json record the current run's accepted acceleration. finish_pipeline.py orchestrates this current run with those existing parity references. The scripts use CPU Torch4 threads and sparse TopK computations. No Apple-GPU access is necessary for this lightweight run.

selection_lock.json and source/checkpoint/output hashes fail closed on mismatch. Source paths and source hashes refer to the original host. Raw labels are dataset-provided, not model-derived. Metric predictions are independently rechecked. NumPy/BLAS runtime warnings are retained in logs and numerical_warning_summary.json, with finite and normal-equation/independent-score checks. Audit status and biological/performance scope are separate from numerical verification. Existing Kang/PBMC are historically explored, not new blind data. Bootstrap intervals describe six observed held-out donor differences with overlapping training sets; seeds are not independent donors. No cross-seed latent feature alignment or causal claim is made.
''')
if '--prepare-only' in sys.argv:sys.exit(0)
paths=[p for p in O.rglob('*') if p.is_file() and p.name!='FINAL_MANIFEST.json']
paths += list(W.glob('*.py'))+list(W.glob('*.log'))+list(W.glob('*.json'))
paths += [p for p in (W/'review_trace').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
paths += list(W.glob('fold_*/baseline/*.json'))+list(W.glob('fold_*/baseline/selected_head.pt'))
paths += list(W.glob('fold_*/seed_*/*/*.json'))
for sp in W.glob('fold_*/seed_*/*/selection.json'):
 s=json.loads(sp.read_text())
 if s['selected_beta'] is not None:paths.append(sp.parent/f"beta_{s['selected_beta']}.pt")
am=W/'recovery/unbound_initial/amendment.json'
if am.exists():paths.append(am)
paths=sorted(set(paths));manifest={str(p.relative_to(ROOT)):sha(p) for p in paths};(O/'FINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));paths.append(O/'FINAL_MANIFEST.json');archive=ROOT/'outputs/SAE_residual_embedding_experiment.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in paths:z.write(p,p.relative_to(ROOT))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'archive':str(archive),'files':len(paths),'bytes':archive.stat().st_size,'CRC':'PASS'}),flush=True)
