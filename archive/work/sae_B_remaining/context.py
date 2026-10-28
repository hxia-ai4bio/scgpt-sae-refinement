"""Cached activation context; no new fitting or inferential cell-level tests."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B_remaining';OLD=W.parent/'sae_interpretability';sys.path.insert(0,str(OLD));from common import dataset,load_npz,auc_columns
k,p,m=dataset();z=load_npz(OLD/'primary_activations.npz');q=load_npz(OLD/'qc.npz');rows=[];sim=[];ci=m['classes'].index('B cells');tr=np.array(m['splits'][0]['indices']['train'])
for domain,data,groupkey in [('kang',k,'donors'),('pbmc',p,'batches')]:
 for group in sorted(set(data[groupkey])):
  ix=np.flatnonzero(data[groupkey]==group);role=next((role for role in ['train','valid','test'] if len(set(ix)&set(m['splits'][0]['indices'][role]))>0),'external_exploratory') if domain=='kang' else 'external_exploratory'
  for fid in [160,898,1387,137,370,146,1373,1125,1409]:
   v=z[domain][ix,fid];b=data['labels'][ix]==ci;rows.append({'feature':fid,'domain':domain,'group':str(group),'role':role,'n':len(ix),'B_cells':int(b.sum()),'auc_B_vs_other':float(auc_columns(v[:,None],b)[0]),'B_active_fraction':float((v[b]>0).mean()),'other_active_fraction':float((v[~b]>0).mean())})
  for scope,ids in [('all_cells',ix),('B_cells',ix[data['labels'][ix]==ci])]:
   for a,b in __import__('itertools').combinations([160,898,1387,137,370,146,1373,1125,1409],2):
    x=z[domain][ids,a];y=z[domain][ids,b];sim.append({'domain':domain,'group':str(group),'role':role,'scope':scope,'feature_a':a,'feature_b':b,'n':len(ids),'spearman':float(spearmanr(x,y).statistic),'active_jaccard':float(((x>0)&(y>0)).sum()/max(((x>0)|(y>0)).sum(),1))})
pd.DataFrame(rows).to_csv(O/'feature_readouts.csv',index=False);pd.DataFrame(sim).to_csv(O/'activation_similarity.csv',index=False)
print(pd.DataFrame(rows).query('feature != 160').to_string(index=False));print('CONTEXT_COMPLETE',flush=True)
