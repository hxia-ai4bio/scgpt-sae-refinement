import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'outputs/sae_masked_ac/predictions';O=ROOT/'outputs/sae_embedding_mae';records=[];hashes={}
def read(p):
 hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 with np.load(p) as a:return {k:a[k] for k in ['rows','embedding']}
for ds,n in [('ctrl',6),('stim',6),('pbmc',2)]:
 for fold in range(n):
  base=P/ds/f'fold_{fold}';target=read(base/'seed_shared/original_h/clean/test.npz')
  for method in ['original_h','clean_mse','A_sae','C_sae','A_dense']:
   for seed in (['shared'] if method=='original_h' else [42,43,44]):
    a=read(base/f'seed_{seed}'/method/'masked/test.npz');np.testing.assert_array_equal(a['rows'],target['rows']);assert a['embedding'].shape==target['embedding'].shape
    err=a['embedding'].astype('float64')-target['embedding'].astype('float64');mae=np.abs(err).mean();check=np.abs(err).sum()/(len(err)*512);assert abs(mae-check)<1e-15
    records.append(dict(dataset=ds,fold=fold,method=method,seed=str(seed),MAE=mae,cells=len(err)))
d=pd.DataFrame(records);d.to_csv(O/'per_fold_seed_MAE.csv',index=False)
g=d.groupby(['dataset','fold','method']).MAE.mean().reset_index();s=g.groupby(['dataset','method']).MAE.mean().reset_index();s.to_csv(O/'summary.csv',index=False)
a=d.groupby(['dataset','method','seed']).MAE.mean().reset_index();a.to_csv(O/'seed_MAE.csv',index=False)
(O/'provenance.json').write_text(json.dumps(dict(metric='Mean absolute coordinate error to original clean frozen scGPT embedding, raw512D scale',query='25percent gene-masked TEST input',aggregation='Mean three seeds within fold, then equal mean heldout donor/batch directions',scope='Posthoc deterministic metric recomputation on existing saved predictions. Not a new training experiment, not an independent semantic review. Original sealed outputs untouched.',formula='mean(abs(h_pred_masked - h_original_clean))',input_hashes=hashes),indent=2))
print(s.pivot(index='dataset',columns='method',values='MAE').to_string())
print('A best seeds:',a[a.method=='A_sae'].sort_values('MAE').groupby('dataset').first().to_string())
