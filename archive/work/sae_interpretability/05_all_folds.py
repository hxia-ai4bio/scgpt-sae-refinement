"""Six donor rotations x three seeds: train-selected single-feature type readouts.

This evaluates recoverability of the three prespecified labels, not universal
concept stability. Dictionary sizes differ, so this is not a fair causal test
of sparsity or superiority of biological interpretation.
"""
import time,json
import numpy as np
import pandas as pd
import torch
from common import *

def select(a,tr,donors,y,train_donors,ci,signed,coverage=None):
    aucs=np.array([auc_columns(a[tr[donors[tr]==d]],y[tr[donors[tr]==d]]==ci) for d in train_donors])
    low=aucs.min(0);mean=aucs.mean(0);sign=np.ones(a.shape[1])
    if signed:
        other=(1-aucs).min(0);rev=other>low;low[rev]=other[rev];mean[rev]=1-mean[rev];sign[rev]=-1
    if coverage is None:coverage=np.ones(a.shape[1],bool)
    ids=sorted(np.flatnonzero(coverage),key=lambda j:(-low[j],-mean[j],j))
    if not ids:raise RuntimeError('No eligible feature')
    j=int(ids[0]);return j,int(sign[j]),float(low[j]),float(mean[j])

def main():
    start=time.time();torch.set_num_threads(2);k,p,m=dataset();targets=['CD14+ Monocytes','B cells','NK cells'];rows=[];hashes={}
    js(OUT/'all_folds_protocol.json',{'targets':targets,'folds':6,'seeds':[42,43,44],
      'selection':'same primary TRAIN-only eligibility and minimum donor AUC rule, separately in each fold/seed',
      'baselines':'original512 coordinates and TRAIN512 PCA directions; sign chosen in TRAIN only',
      'scope':'label-guided single-feature diagnostic; unequal candidate dimensions 2048 vs512; historical donors/PBMC; not an interpretability benchmark or 18 independent datasets',
      'summary':'average three seeds per held-out donor first; report six donor values'})
    for split in m['splits']:
      tr=np.array(split['indices']['train']);te=np.array(split['indices']['test'])
      h=torch.as_tensor(k['embedding'],dtype=torch.float64);mean=h[tr].mean(0);ctr=h[tr]-mean
      eig,d=torch.linalg.eigh(ctr.T@ctr);d=d.flip(1)
      pc=((h-mean)@d).numpy();ppc=((torch.as_tensor(p['embedding'],dtype=torch.float64)-mean)@d).numpy()
      bas={}
      for target in targets:
        ci=m['classes'].index(target)
        for name,a,pa in [('embedding',k['embedding'],p['embedding']),('pca',pc,ppc)]:
          j,s,low,avg=select(a,tr,k['donors'],k['labels'],split['train'],ci,True)
          bas[target,name]=(j,s,low,avg,a[:,j]*s,pa[:,j]*s)
      for seed in [42,43,44]:
        model,path=load_sae(split['test'],seed);hashes[str(path)]=sha(path)
        z=encode(model,k['embedding']);pz=encode(model,p['embedding']);freq=(z[tr]!=0).mean(0)
        eligible=((z[tr]!=0).sum(0)>=20)&(freq<=.5)
        for donor in split['train']:eligible &= (z[tr[k['donors'][tr]==donor]]!=0).sum(0)>=3
        for target in targets:
          ci=m['classes'].index(target);j,s,low,avg=select(z,tr,k['donors'],k['labels'],split['train'],ci,False,eligible)
          variants=[('sae',j,s,low,avg,z[:,j],pz[:,j])]+[(name,*bas[target,name]) for name in ['embedding','pca']]
          for name,j,s,low,avg,v,pv in variants:
            rec={'test_donor':split['test'],'seed':seed,'target':target,'representation':name,'coordinate':j,'sign':s,'train_min_donor_auc':low,'train_mean_donor_auc':avg,'test_auc':float(auc_columns(v[te,None],k['labels'][te]==ci)[0]),'test_target_n':int((k['labels'][te]==ci).sum())}
            for batch in np.unique(p['batches']):
              ix=np.flatnonzero(p['batches']==batch);rec[batch+'_auc']=float(auc_columns(pv[ix,None],p['labels'][ix]==ci)[0])
            rows.append(rec)
        print('FOLD',split['test'],seed,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'all_fold_single_feature_readouts.csv',index=False)
    df.groupby(['target','representation','test_donor'])[['test_auc','Batch1_3k_auc','Batch2_68k_r_auc']].mean().reset_index().to_csv(OUT/'donor_equal_single_feature_readouts.csv',index=False)
    js(OUT/'all_fold_model_hashes.json',hashes);print('ALLFOLDS_COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
