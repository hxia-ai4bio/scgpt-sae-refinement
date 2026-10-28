from pathlib import Path
import json,hashlib,zipfile,sys,platform,importlib.metadata
W=Path(__file__).resolve().parent;R=W.parents[1];O=R/'outputs/sae_B898_1387'
(O/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','statsmodels','h5py','psutil','matplotlib']}},indent=2))
(O/'REPRODUCIBILITY.txt').write_text('''B-cell SAE F898 and F1387 gene contribution and pathways
Start at REPORT.html; top40_gene_contributions.csv and candidate_pathway_genes.csv give gene-level evidence.
all_pathway_results.csv retains every tested term; significant_pathways.csv contains ordinary-ORA-only rows as well as any matched discoveries. Interpret the evidence column and both adjusted q-values.

Full rerun requires external historical checkpoint and datasets at absolute paths in work/sae_interpretability/common.py and the original data_manifest.json. Original models/raw data are not bundled. Frozen sources and exact input hashes are retained. No model training occurs.
Python environment: ENVIRONMENT.json.
From extracted root (run sequentially):
  python work/sae_B898_1387/context.py
  python work/sae_B898_1387/expression_bcells.py
  for each F898 and F1387: run.py -> analyze.py -> pairs_qc.py -> verify.py
  python work/sae_B898_1387/pathways.py
  python work/sae_B898_1387/verify_pathways.py
  python work/sae_B898_1387/report.py
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=8 for model inference; model code sets Torch CPU threads8, max16perturbations perbatch. Saved pair caches bind inputs, script, checkpoint hashes and output hashes. Do not delete completion flags for a partial run without inspecting the corresponding cache state.
Full-precision null strata and indices are in each feature null_*.npz. Replay them directly for existing random tests; do not recreate strata using CSV float round trips.
The main matched test is10,000random sets plus-one p. Exact stratum-conditional convolution is a separate prespecified sensitivity analysis. Global BH spans both new features, both gene lists and allthree libraries; F160 prior q-values use a different family.
Archived local helper sources, frozen activations, PCA and prior metadata are included. The original historical project module dependencies and original checkpoint/data remain external. Prior F160 gene table is included only to reproduce descriptive cross-feature overlap. Different feature-selected prototypes/references limit paired comparisons.
Audit is a same-family provisional reviewer assessment; numerical PASS is not biological validation.
''')
if '--prepare-only' in sys.argv:sys.exit(0)
paths=[p for base in [W,O,R/'work/sae_pathways_v3/gene_sets'] for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FINAL_MANIFEST.json']
paths += [R/'work/sae_interpretability'/n for n in ['common.py','02_forward.py','04_expression.py','pca.npz','primary_activations.npz','qc.npz']]
paths += [R/'outputs/sae_interpretability'/n for n in ['candidates.json','baseline_directions.json','input_manifest.json']]
paths += [R/'outputs/sae_gene_attribution_v2/gene_evidence_combined.csv']
paths=sorted(set(paths));manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};(O/'FINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2));paths.append(O/'FINAL_MANIFEST.json')
z=R/'outputs/SAE_Bcells_F898_F1387_gene_pathway.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in paths:f.write(p,p.relative_to(R))
with zipfile.ZipFile(z) as f:assert f.testzip() is None
print(json.dumps({'archive':str(z),'files':len(paths),'bytes':z.stat().st_size,'CRC':'PASS'}),flush=True)
