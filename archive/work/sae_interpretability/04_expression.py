"""Descriptive expression-proxy associations controlling counts, genes and donor.

This is same-RNA association, not independent biological truth or causal evidence.
Sparse matrix operations avoid materializing cells x all genes at once.
"""
import json,time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import spearmanr
from common import *

def norm(x):
    totals=np.asarray(x.sum(1)).ravel(); a=(sparse.diags(1e4/np.maximum(totals,1))@x).tocsr()
    a.data=np.log1p(a.data); return a

def associations(x,z,q,groups,labels=None):
    n=len(z); parts=[np.ones(n),q[:,0],q[:,1]]
    if np.ptp(q[:,2])>1e-12:parts.append(q[:,2])
    parts += [groups==d for d in np.unique(groups)[1:]]
    if labels is not None:parts += [labels==v for v in np.unique(labels)[1:]]
    design=np.column_stack(parts);u,s,v=torch.linalg.svd(torch.as_tensor(design,dtype=torch.float64),full_matrices=False)
    basis=u[:,s>s.max()*1e-10].numpy();rz=residualize(np.asarray(z),design)
    ss=float(np.dot(rz,rz)); scores=np.zeros(x.shape[1])
    for start in range(0,x.shape[1],512):
        xx=x[:,start:start+512]; projected=np.asarray(xx.T@basis)
        var=np.asarray(xx.power(2).sum(0)).ravel()-np.einsum('ij,ij->i',projected,projected)
        den=np.sqrt(np.maximum(var,0)*ss);cov=np.asarray(xx.T@rz).ravel()
        scores[start:start+512]=np.divide(cov,den,out=np.zeros_like(cov),where=den>1e-9)
    scores=np.clip(scores,-1,1)
    return scores, {'n':n,'design_rank':int(len(s[s>s.max()*1e-10])),'feature_residual_sum_squares':ss,'degenerate_feature':bool(ss<1e-10)}

def main():
    start=time.time();torch.set_num_threads(2)
    k,p,m=dataset();prep,cfg,x,g,meta,px,pg,pm=raw_data();act=load_npz(HERE/'primary_activations.npz');qq=load_npz(HERE/'qc.npz')
    split=m['splits'][0];classes=m['classes'];candidates=json.loads((OUT/'candidates.json').read_text())
    js(OUT/'expression_analysis_protocol.json',{
      'source':'same input RNA; exploratory proxy','score':'Pearson correlation of log1pCP10K expression and feature after residualizing intercept/log total counts/log detected genes/donor; mt if variable; within preselected target type (technical comparator adjusts all types)',
      'genes':'model vocabulary genes detected >=20 TRAIN cells in target population; same gene set mapped to TEST/PBMC',
      'outputs':'raw partial correlations; no cell-level inferential p values; train top40 carried forward descriptively',
      'mt_gene_count_kang':int(np.char.startswith(g.astype(str),'MT-').sum()),
      'mt_gene_count_pbmc':int(np.char.startswith(pg.astype(str),'MT-').sum()),
      'mt_note':'No variation/absent genes means mitochondrial confounding cannot be assessed, not zero biological effect'
    })
    vocab=json.loads((Path(cfg['checkpoint_dir'])/'vocab.json').read_text()); summary=[]
    for c in candidates:
      f=c['feature'];ci=classes.index(c['target']) if c['target']!='all' else None
      train=np.asarray(split['indices']['train'])
      if ci is not None:train=train[k['labels'][train]==ci]
      detected=np.asarray((x[k['rows'][train]]>0).sum(0)).ravel()
      keep=(detected>=20)&np.array([v in vocab for v in g]); cols=np.flatnonzero(keep); gg=g[cols]
      train_scores=None;train_gene_order=None
      for name,domain,indices in [('train','kang',np.asarray(split['indices']['train'])),('test','kang',np.asarray(split['indices']['test'])),('pbmc','pbmc',np.arange(len(p['labels'])))]:
        dd=k if domain=='kang' else p
        if ci is not None:indices=indices[dd['labels'][indices]==ci]
        if domain=='kang':xx=norm(x[dd['rows'][indices]])[:,cols]; genes=gg;groups=dd['donors'][indices]
        else:
          mapg={v:i for i,v in enumerate(pg)};inboth=np.array([v in mapg for v in gg]);genes=gg[inboth]
          cc=np.array([mapg[v] for v in genes]);xx=norm(px[dd['rows'][indices]])[:,cc];groups=dd['batches'][indices]
        if len(indices)<10:continue
        vals=act[domain][indices,f];q=qq[domain][indices]
        score,diag=associations(xx,vals,q,groups,None if ci is not None else dd['labels'][indices])
        df=pd.DataFrame({'feature':f,'domain':name,'gene':genes,'partial_r':score,'detected_cells':np.asarray((xx>0).sum(0)).ravel()})
        df.sort_values('partial_r',ascending=False,kind='stable').to_csv(OUT/f'gene_association_{f}_{name}.csv',index=False)
        if name=='train':train_scores=dict(zip(genes,score));train_gene_order=genes[np.argsort(-score,kind='stable')[:40]]
        else:
          common=np.array([v in train_scores for v in genes]); other=np.array([train_scores[v] for v in genes[common]])
          rho=spearmanr(other,score[common]).statistic if not diag['degenerate_feature'] else np.nan
          diag['train_gene_ranking_spearman']=float(rho) if np.isfinite(rho) else None
          hit=np.isin(genes,train_gene_order);diag['train_top40_present']=int(hit.sum())
          diag['train_top40_mean_partial_r']=float(score[hit].mean()) if hit.any() else None
          diag['train_top40_positive_fraction']=float((score[hit]>0).mean()) if hit.any() else None
        summary.append({'feature':f,'target':c['target'],'split':name,**diag})
      print('EXPRESSION',f,flush=True)
    js(OUT/'expression_association_summary.json',summary)
    print('EXPRESSION_COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
