from pathlib import Path
import json,os,platform,psutil,shutil
import numpy as np,pandas as pd,torch
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B_remaining';H=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
a=pd.read_csv(O.parent/'B_cell_feature_ranking.csv');sel=a[(a['B cells_mean_donor_auc']>=.7)&(~a.feature.isin([160,898,1387]))].copy();sel['scope_group']=np.where(sel['B cells_min_donor_auc']>=.7,'min_donor_AUC>=0.70','mean_AUC>=0.70_only');sel.to_csv(O/'selected_features.csv',index=False)
k=np.load(H/'kang_embeddings.npz');z=np.load(W.parent/'sae_interpretability/primary_activations.npz')['kang'];m=json.loads((H/'data_manifest.json').read_text());rows=[];pairsets={}
for f in sel.feature:
 pairs=[]
 for role,n in [('train',2),('test',4)]:
  ix=np.array(m['splits'][0]['indices'][role]);ix=ix[k['labels'][ix]==m['classes'].index('B cells')]
  for d in sorted(set(k['donors'][ix])):
   cells=ix[k['donors'][ix]==d];active=cells[z[cells,f]>0];inactive=cells[z[cells,f]==0];rows.append({'feature':int(f),'role':role,'donor':str(d),'n_B_cells':len(cells),'positive':len(active),'inactive':len(inactive),'eligible':len(active)>=n and len(inactive)>=1})
   if len(active)<n or len(inactive)<1:continue
   for pi in sorted(active,key=lambda i:(-float(z[i,f]),int(i)))[:n]:
    dist=np.linalg.norm(k['embedding'][inactive].astype(float)-k['embedding'][pi],axis=1)
    for j in sorted(range(len(inactive)),key=lambda j:(dist[j],int(inactive[j])))[:3]:pairs.append((int(pi),int(inactive[j])))
 pairsets[int(f)]=pairs
pd.DataFrame(rows).to_csv(O/'cell_selection_preflight.csv',index=False)
resources={'platform':platform.platform(),'CPU':platform.processor(),'memory':dict(psutil.virtual_memory()._asdict()),'disk':dict(zip(['total','used','free'],shutil.disk_usage(W))),'mps_available':torch.backends.mps.is_available(),'mps_built':torch.backends.mps.is_built(),'torch_version':torch.__version__,'swap_probe':'not attempted: psutil swap unsupported in sandbox previously'};(W/'resources.json').write_text(json.dumps(resources,indent=2))
print(sel.to_string(index=False));print(pd.DataFrame(rows).to_string(index=False));allpairs=[p for pp in pairsets.values() for p in pp];print('PAIR_COUNTS',len(allpairs),'unique',len(set(allpairs)));print(json.dumps(resources,indent=2))
