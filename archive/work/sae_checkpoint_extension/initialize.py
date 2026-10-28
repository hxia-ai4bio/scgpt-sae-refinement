"""Bind the fixed two-dataset replication and all consumed context sources."""
import json,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_checkpoint_extension'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def immutable(p,a):
 if p.exists():assert json.loads(p.read_text())==a
 else:p.write_text(json.dumps(a,indent=2))
protocol=dict(datasets=['stim','pbmc'],folds={'stim':6,'pbmc':2},seeds=[42,43,44],new_body_trajectories=24,new_linear_heads=24,seven_checkpoints=[0,25,50,100,200,500,1000],
 selection='Exact ctrl rule, per-dataset same warmup/batches/training objective/optimizers/scalers/heads; all selection TRAIN/VALID only; both datasets locked before any new TEST.',
 primary='Raw-logCP10k masked-positive gene-expression token MAE; selected vs original_h and same-dataset C1000',secondary=['gene-macro MAE','normalized MSE','clean/masked15NN MacroF1'],
 parallelism='Two processes, four CPU threads each; reused cache, no scGPT extraction.',scope='Exploratory protocol replication on historical data. Stim shares ctrl donors. PBMC two batch directions, not independent donors. Single fixed mask; same VALID reused for body/head. No per-dataset rule adaptation after TEST. Ctrl context reused explicitly, not newly rerun.')
immutable(O/'GLOBAL_PROTOCOL.json',protocol)
paths=[W/n for n in ['run.py','execute.py','build_sources.py','initialize.py']]+[O/'GLOBAL_PROTOCOL.json']
for ds in ['stim','pbmc']:paths.extend([O/ds/'PROTOCOL.json',O/ds/'INPUT_BINDINGS.json'])
for name in ['all_metrics.csv','knn_metrics.csv','selected_checkpoints.csv','SUMMARY.json','FINAL_MANIFEST.json']:paths.append(ROOT/'outputs/sae_checkpoint_selection'/name)
immutable(O/'GLOBAL_INPUT_BINDINGS.json',{str(p):sha(p) for p in paths})
print('GLOBAL_PROTOCOL_BOUND',len(paths))
