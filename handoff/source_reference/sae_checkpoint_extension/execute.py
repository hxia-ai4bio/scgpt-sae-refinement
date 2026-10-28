"""Develop both datasets before any TEST; two bounded CPU workers."""
import sys,json,hashlib,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_checkpoint_extension';DATASETS=['stim','pbmc']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def immutable(p,a):
 if p.exists():assert json.loads(p.read_text())==a,str(p)
 else:p.write_text(json.dumps(a,indent=2))
def check_sources():
 for p,h in json.loads((O/'GLOBAL_INPUT_BINDINGS.json').read_text()).items():assert sha(p)==h,p
def stage(ds,name):
 check_sources()
 if name=='evaluate':
  lock=json.loads((O/'GLOBAL_SELECTION_LOCK.json').read_text())
  for key,h in lock['dataset_lock_hashes'].items():assert sha(O/key/'SELECTION_LOCK.json')==h
 with (W/f'{name}_{ds}.log').open('w') as log:subprocess.run([sys.executable,str(W/'run.py'),name,ds],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('STAGE_COMPLETE',name,ds,flush=True)
if __name__=='__main__':
 check_sources()
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda ds:stage(ds,'develop'),DATASETS))
 immutable(O/'GLOBAL_SELECTION_LOCK.json',dict(dataset_lock_hashes={ds:sha(O/ds/'SELECTION_LOCK.json') for ds in DATASETS},source_sha=sha(W/'run.py'),all_new_datasets_selected_before_TEST=True,roles_seen=['TRAIN','VALID']))
 print('GLOBAL_SELECTION_LOCKED',flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda ds:stage(ds,'evaluate'),DATASETS))
 print('ALL_DATASETS_COMPLETE',flush=True)
