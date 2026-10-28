"""Secondary same-RNA partial expression association, not attribution or independent truth."""
from pathlib import Path
import json,sys,importlib.util
import h5py,numpy as np,pandas as pd
from scipy import sparse
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B_remaining';OLD=W.parent/'sae_interpretability';sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('old_expression',OLD/'04_expression.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
k,p,manifest=m.dataset();z=m.load_npz(OLD/'primary_activations.npz')['kang'];qc=m.load_npz(OLD/'qc.npz')['kang'];ci=manifest['classes'].index('B cells');cfg=manifest['preprocess'];vocab=json.loads((Path(cfg['checkpoint_dir'])/'vocab.json').read_text())
with h5py.File(cfg['dataset'],'r') as h:
 genes=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in h['var'][:]['name']]);x=sparse.csc_matrix((h['X/data'][:],h['X/indices'][:],h['X/indptr'][:]),shape=tuple(h['X'].attrs['h5sparse_shape'])).tocsr();x.eliminate_zeros()
ix=np.array(manifest['splits'][0]['indices']['train']);ix=ix[k['labels'][ix]==ci];det=np.asarray((x[k['rows'][ix]]>0).sum(0)).ravel();cols=np.flatnonzero((det>=20)&np.array([g in vocab for g in genes]));summary=[]
for role in ['train','test']:
 ix=np.array(manifest['splits'][0]['indices'][role]);ix=ix[k['labels'][ix]==ci];xx=m.norm(x[k['rows'][ix]])[:,cols]
 for fid in [137,370,146,1373,1125,1409]:
  (O/f'F{fid}').mkdir(parents=True,exist_ok=True)
  scores,diag=m.associations(xx,z[ix,fid],qc[ix],k['donors'][ix]);pd.DataFrame({'gene':genes[cols],'partial_r':scores}).sort_values('partial_r',ascending=False).to_csv(O/f'F{fid}/expression_association_{role}.csv',index=False);summary.append({'feature':fid,'role':role,**diag})
(O/'expression_association_protocol.json').write_text(json.dumps({'type':'secondary exploratory same-RNA association; not input attribution or independent validation','score':'partial Pearson log1pCP10k gene expression vs feature within B cells controlling logtotalcounts,loggenes,donor; absent MT in Kang not assessable','genes':'detected>=20 TRAIN B cells and in model vocab; same gene set TEST','pathways':'no pathway tests on these secondary association rankings','diagnostics':summary},indent=2));print('EXPRESSION_COMPLETE',flush=True)
