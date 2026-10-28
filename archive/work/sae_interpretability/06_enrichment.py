"""Conditional ORA with complete tested-gene backgrounds and matched-gene null.

Not a replication of paper GSEA. No enrichment result is independent biological
validation. Top genes and expression measurements originate from the same RNA.
"""
import json,time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom,rankdata
from common import *

def main():
    start=time.time();torch.set_num_threads(2)
    setsdir=ROOT/'experiments/dense_validation_20261006/gene_sets'
    provenance=json.loads((setsdir/'provenance.json').read_text());sets={}
    for item in provenance:
      path=setsdir/item['file'];assert sha(path)==item['sha256']
      for line in path.read_text().splitlines():
        a=line.split('\t');sets[a[0]]=set(a[2:])
    js(OUT/'gene_set_provenance.json',provenance)
    k,p,m=dataset();prep,cfg,x,g,meta,px,pg,pm=raw_data();split=m['splits'][0]
    candidates=json.loads((OUT/'candidates.json').read_text());gc={v:i for i,v in enumerate(g)}
    tables=[]
    files=sorted(OUT.glob('gene_association_*.csv'))+sorted(OUT.glob('gene_attribution_*.csv'))
    js(OUT/'enrichment_protocol.json',{'method':'top40 positive score ORA (not paper GSEA)',
      'background':'all genes actually tested in each attribution/association table',
      'pathways':'Hallmark50 + Reactome1787, local2025.1 version hashes verified',
      'eligibility':'5 to500 tested genes in a pathway; whole-pathway coverage reported',
      'multiplicity':'BH across all four feature/pathway tests for each analysis source and split; empirical matched-null q in separate family',
      'matched_null':'TRAIN only; 2000 random sets, without replacement, preserve joint quintile bins of TRAIN target-population gene mean expression and detection rate',
      'inference':'conditional exploratory annotation; nominal gene-exchangeability null; same RNA; not causal evidence'})
    for file in files:
      df=pd.read_csv(file);source='attribution' if 'attribution' in file.name else 'association'
      feature=int(df['feature'].iloc[0]);role=str(df['role' if source=='attribution' else 'domain'].iloc[0])
      score='mean_attribution' if source=='attribution' else 'partial_r'
      df=df.sort_values(score,ascending=False,kind='stable');universe=df['gene'].astype(str).tolist();umap={v:i for i,v in enumerate(universe)}
      top=df.loc[df[score]>0,'gene'].astype(str).tolist()[:40];topset=set(top)
      paths=[];rr=[];cc=[];records=[]
      for name,genes in sets.items():
        overlap=genes&set(universe);size=len(overlap)
        if not 5<=size<=500:continue
        j=len(paths);paths.append(name);rr += [umap[v] for v in sorted(overlap)];cc += [j]*size
        hits=genes&topset;n=len(top);pval=float(hypergeom.sf(len(hits)-1,len(universe),size,n))
        records.append({'source':source,'split':role,'feature':feature,'pathway':name,
          'universe_size':len(universe),'selected_genes':n,'pathway_genes_tested':size,
          'pathway_genes_total':len(genes),'coverage':size/len(genes),'overlap':len(hits),
          'expected_overlap':n*size/len(universe),'p_hypergeom':pval,
          'overlap_genes':';'.join(sorted(hits)),'p_matched':np.nan})
      if role=='train' and records and top:
        c=next(c for c in candidates if c['feature']==feature);tr=np.array(split['indices']['train'])
        if c['target']!='all':tr=tr[k['labels'][tr]==m['classes'].index(c['target'])]
        xx=x[k['rows'][tr]][:,[gc[v] for v in universe]].astype(np.float64).tocsr()
        # Counts are used only to match nuisance distribution; no outcome labels.
        means=np.asarray(xx.mean(0)).ravel();detect=np.asarray((xx>0).mean(0)).ravel()
        mb=np.minimum(4,(rankdata(means,method='average')/len(means)*5).astype(int))
        db=np.minimum(4,(rankdata(detect,method='average')/len(detect)*5).astype(int));strata=mb*5+db
        topidx=np.array([umap[v] for v in top]);stratacount={int(s):int((strata[topidx]==s).sum()) for s in np.unique(strata[topidx])}
        rng=np.random.default_rng([20261018,feature,0 if source=='association' else 1]);rows=[];cols=[]
        for b in range(2000):
          chosen=np.concatenate([rng.choice(np.flatnonzero(strata==s),n,replace=False) for s,n in stratacount.items()])
          rows.extend([b]*len(chosen));cols.extend(chosen.tolist())
        draws=sparse.csr_matrix((np.ones(len(rows),np.int32),(rows,cols)),shape=(2000,len(universe)))
        membership=sparse.csr_matrix((np.ones(len(rr),np.int32),(rr,cc)),shape=(len(universe),len(paths)))
        null=(draws@membership).toarray();observed=np.array([v['overlap'] for v in records])
        emp=(1+(null>=observed).sum(0))/2001
        for rec,pv in zip(records,emp):rec['p_matched']=float(pv)
      tables.extend(records);print('ENRICHMENT',source,feature,role,len(records),flush=True)
    all_df=pd.DataFrame(tables);all_df['q_hypergeom']=np.nan;all_df['q_matched']=np.nan
    for (source,role),group in all_df.groupby(['source','split']):
      all_df.loc[group.index,'q_hypergeom']=bh(group['p_hypergeom'].values)
      valid=group[group['p_matched'].notna()]
      if len(valid):all_df.loc[valid.index,'q_matched']=bh(valid['p_matched'].values)
    all_df.to_csv(OUT/'pathway_enrichment_all.csv',index=False)
    all_df[(all_df['q_hypergeom']<.05)|(all_df['q_matched']<.05)].sort_values(['source','split','feature','q_hypergeom']).to_csv(OUT/'pathway_enrichment_hits.csv',index=False)
    js(OUT/'enrichment_completion.json',{'status':'complete','tables':len(files),'pathway_tests':len(all_df),'seconds':time.time()-start})

if __name__=='__main__':main()
