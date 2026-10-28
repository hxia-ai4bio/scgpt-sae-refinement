"""Train-only feature discovery, descriptive donor validation and dictionary stability."""
import json, time
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, rankdata
from common import *

def main():
    start=time.time(); k,p,m=dataset(); classes=m['classes']; split=m['splits'][0]
    assert split['test']=='patient_1016'
    ix={s:np.array(split['indices'][s]) for s in ['train','valid','test']}
    protocol={
      'date':'2026-10-18','status':'exploratory; historical data; no blind-study claim',
      'primary_model':{'test_donor':split['test'],'valid_donor':split['valid'],'seed':42,'k':32},
      'selection':'TRAIN only; primary model is first predefined historical rotation, seed42',
      'targets':['CD14+ Monocytes','B cells','NK cells'],
      'candidate_filter':'TRAIN nonzero>=20, frequency<=0.5, nonzero>=3 in each TRAIN donor; maximize minimum donor one-v-rest AUC, then mean AUC, then lower feature index; distinct features',
      'technical_candidate':'max absolute residual correlation after additive cell-type and donor adjustment with log total counts/log detected genes/mitochondrial fraction; same coverage filter; not already selected',
      'negative_values':'retained signed activations are preserved; nonzero means selected, positive direction is analyzed; negative fractions reported',
      'replicates':'donor-level descriptive effects, no cell-level significance for cross-donor claims',
      'occlusion':'fixed original token identities/padding; replace only gene value with reference full-cell deterministic bin (0 if absent); no retokenization or renormalization',
      'prototypes':'up to one highest positive activation target cell per TRAIN donor (4); first 2 highest positive target cells in primary TEST; no post-hoc substitution for poor results',
      'counterfactual':'one nearest embedding inactive cell in same donor/type if possible, else same donor, otherwise report no eligible control; target technical candidate uses all types',
      'budget':'benchmark CPU forward before scaling; target 4 features * (4 train+2 test) * <=512 individual perturbations, plus deletion controls',
      'deletion':'orders frozen from TRAIN attribution; DEG absolute mean-logexpression difference; expression-bin-matched random x10; evaluate at 0,1,5,10,20,40 on TEST prototypes, raw activation reported',
      'enrichment':'MSigDB2025.1 Hallmark+Reactome local verified files; top40 positive attribution ORA, hypergeometric and BH across all features/pathways within split; universe genes actually perturbed; exploratory conditional gene-set annotation, not biological causal validation',
      'stability':'same fold seeds43/44 plus other five folds seed42; Hungarian decoder matching; overlapping-data refit stability, not independent donor replication',
      'baseline':'original embedding dimension and TRAIN PCA direction selected per target by same minimum donor AUC; same prototype inputs for sensitivity comparison only; unequal dictionaries preclude global interpretability superiority claim',
      'heldout_rules':'candidate ids/signs, celltype targets and gene order locked from train; test and PBMC not used to pick candidates'
    }
    js(OUT/'protocol.json',protocol)
    prep,cfg,x,g,meta,px,pg,pm=raw_data()
    assert np.array_equal(meta['cell_ids'][k['rows']],k['cell_ids'])
    assert np.array_equal(meta['cell_type'][k['rows']],np.array(classes)[k['labels']])
    q=qc(x,g,k['rows']); pq=qc(px,pg,p['rows'])
    np.savez_compressed(HERE/'qc.npz',kang=q,pbmc=pq)
    model,path=load_sae(); z=encode(model,k['embedding']); pz=encode(model,p['embedding'])
    cache=load_npz(path.parent/'sae_outputs.npz')
    errors={}
    for s,rows in ix.items():
        ref=np.zeros((len(rows),2048),np.float32)
        np.put_along_axis(ref,cache[s+'_indices'],cache[s+'_values'],axis=1)
        assert np.array_equal(k['rows'][rows],cache[s+'_rows'])
        errors[s]=float(np.max(np.abs(z[rows]-ref))); assert errors[s]<2e-5
    np.savez_compressed(HERE/'primary_activations.npz',kang=z,pbmc=pz)
    tr=ix['train']; zd=z[tr]; donors=k['donors']; y=k['labels']
    atlas=pd.DataFrame({'feature':np.arange(2048),'train_frequency':(zd!=0).mean(0),
      'train_mean':zd.mean(0),'negative_retained':(zd<0).sum(0),
      'train_nonzero':(zd!=0).sum(0),'train_positive':(zd>0).sum(0)})
    per={}; eligible=((zd!=0).sum(0)>=20)&((zd!=0).mean(0)<=.5)
    for d in split['train']:
        di=tr[donors[tr]==d]; counts=(z[di]!=0).sum(0)
        atlas[d+'_nonzero']=counts; eligible &= counts>=3
    for ci,cl in enumerate(classes):
        aa=[]
        for d in split['train']:
            di=tr[donors[tr]==d]; aa.append(auc_columns(z[di],y[di]==ci))
        aa=np.array(aa); per[cl]=aa
        atlas[cl+'_mean_donor_auc']=np.nanmean(aa,0)
        atlas[cl+'_min_donor_auc']=np.nanmin(aa,0)
    # Residual technical correlations remove additive cell identity and donor shifts.
    design=np.column_stack([np.ones(len(tr)),*[y[tr]==i for i in range(1,8)],
                           *[donors[tr]==d for d in split['train'][1:]]])
    rz=residualize(zd,design); rq=residualize(q[tr],design)
    tech=[]
    for j,name in enumerate(['log_counts','log_genes','mt_fraction']):
        cor=corr_columns(rz,rq[:,j]); atlas['technical_partial_'+name]=cor; tech.append(cor)
    tc=np.array(tech); atlas['max_abs_technical_partial']=np.max(np.abs(tc),0)
    atlas['eligible']=eligible
    chosen=[]
    for target in protocol['targets']:
        order=sorted(np.flatnonzero(eligible),key=lambda f:(-atlas.loc[f,target+'_min_donor_auc'],-atlas.loc[f,target+'_mean_donor_auc'],f))
        f=next(int(f) for f in order if f not in [c['feature'] for c in chosen])
        chosen.append({'feature':f,'target':target,'role':'biological_candidate','train_min_donor_auc':float(atlas.loc[f,target+'_min_donor_auc'])})
    order=np.argsort(-np.max(np.abs(tc),0),kind='stable')
    f=next(int(f) for f in order if eligible[f] and f not in [c['feature'] for c in chosen])
    chosen.append({'feature':f,'target':'all','role':'technical_comparator','covariate':['log_counts','log_genes','mt_fraction'][int(np.argmax(np.abs(tc[:,f])))],'partial_r':float(tc[np.argmax(np.abs(tc[:,f])),f])})
    js(OUT/'candidates.json',chosen); atlas.to_csv(OUT/'feature_atlas_train.csv',index=False)
    # Descriptive fixed-feature replication. Do not pool cells as independent donors.
    records=[]
    for c in chosen:
      f=c['feature']; target=c['target']; ci=classes.index(target) if target!='all' else None
      for domain,zz,yy,groups,qq in [('kang',z,y,donors,q),('pbmc',pz,p['labels'],p['batches'],pq)]:
        for d in np.unique(groups):
          rows=np.flatnonzero(groups==d); vals=zz[rows,f]
          role=('train' if d in split['train'] else 'valid' if d==split['valid'] else 'test') if domain=='kang' else 'external_exploratory'
          a=float(auc_columns(vals[:,None],yy[rows]==ci)[0]) if ci is not None else None
          records.append({'feature':f,'target':target,'domain':domain,'group':d,'role':role,'n':len(rows),'nonzero':int((vals!=0).sum()),'negative':int((vals<0).sum()),'target_n':int((yy[rows]==ci).sum()) if ci is not None else None,'auc':a,'r_logcounts':float(corr_columns(vals[:,None],qq[rows,0])[0]),'r_loggenes':float(corr_columns(vals[:,None],qq[rows,1])[0]),'r_mt':float(corr_columns(vals[:,None],qq[rows,2])[0])})
    pd.DataFrame(records).to_csv(OUT/'candidate_replication.csv',index=False)
    # PCA is fitted only on primary TRAIN; tensor covariance independent verification.
    ht=torch.as_tensor(k['embedding'][tr],dtype=torch.float64); mu=ht.mean(0); centered=ht-mu
    cov=centered.T@centered/max(len(tr)-1,1); eig,vec=torch.linalg.eigh(cov); vec=vec.flip(1)
    pcs=((torch.as_tensor(k['embedding'],dtype=torch.float64)-mu)@vec).numpy()
    np.savez_compressed(HERE/'pca.npz',mean=mu.numpy(),directions=vec.numpy(),eigenvalues=eig.flip(0).numpy())
    baselines=[]
    for c in chosen:
      if c['target']=='all': continue
      ci=classes.index(c['target'])
      for name,a in [('embedding',k['embedding']),('pca',pcs)]:
        aa=np.array([auc_columns(a[tr[donors[tr]==d]],y[tr[donors[tr]==d]]==ci) for d in split['train']])
        pos=np.min(aa,0); neg=np.min(1-aa,0); score=np.maximum(pos,neg); j=int(np.argmax(score)); sign=1 if pos[j]>=neg[j] else -1
        baselines.append({'feature':c['feature'],'target':c['target'],'kind':name,'dimension':j,'sign':sign,'train_min_donor_auc':float(score[j]),'test_auc':float(auc_columns(sign*a[ix['test'],j:j+1],y[ix['test']]==ci)[0])})
    js(OUT/'baseline_directions.json',baselines)
    # Cross-seed and cross-fold dictionary matching, no activation-based cherry picking.
    stability=[]; d0=model.decoder.weight.detach().double(); d0=d0/d0.norm(dim=0)
    configs=[('patient_1016',43),('patient_1016',44)]+[(s['test'],42) for s in m['splits'][1:]]
    for fold,seed in configs:
      mm,pp=load_sae(fold,seed); dd=mm.decoder.weight.detach().double(); dd=dd/dd.norm(dim=0)
      sim=(d0.T@dd).numpy(); ri,ci=linear_sum_assignment(-sim); mapping=dict(zip(ri,ci)); zz=encode(mm,k['embedding'])
      for c in chosen:
        f=c['feature']; j=int(mapping[f]); rho=float(spearmanr(z[tr,f],zz[tr,j]).statistic)
        stability.append({'feature':f,'other_test_fold':fold,'seed':seed,'matched_feature':j,'decoder_cosine':float(sim[f,j]),'activation_spearman_primary_train':rho if np.isfinite(rho) else None,'scope':'overlapping-data refit; not independent biological replication'})
      print('MATCHED',fold,seed,flush=True)
    pd.DataFrame(stability).to_csv(OUT/'dictionary_stability.csv',index=False)
    inputs=[OLD/'data_manifest.json',OLD/'kang_embeddings.npz',OLD/'pbmc_embeddings.npz',path,Path(cfg['dataset']),Path('/Users/apple/Desktop/Llava-Cell/dataset/PBMC368k_raw.h5ad')]
    js(OUT/'input_manifest.json',{'sha256':{str(p):sha(p) for p in inputs},'classes':classes,'cache_replay_max_error':errors,'kang_n':len(y),'pbmc_n':len(p['labels']),'train_donors':split['train'],'valid_donor':split['valid'],'test_donor':split['test'],'negative_retained_fraction':float((z<0).sum()/(z!=0).sum()),'eligible_features':int(eligible.sum()),'seconds':time.time()-start})
    print('ATLAS_COMPLETE',json.dumps(chosen), 'seconds',time.time()-start,flush=True)

if __name__=='__main__': main()
