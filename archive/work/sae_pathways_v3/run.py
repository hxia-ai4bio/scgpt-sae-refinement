"""Positive-attribution ORA and matched-gene null; descriptive pathways, not causal truth."""
from pathlib import Path
import json,time,hashlib,sys
import numpy as np,pandas as pd,h5py,psutil
from scipy import sparse
from scipy.stats import hypergeom,rankdata
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];OUT=ROOT/'outputs/sae_pathways_v3';V2=ROOT/'outputs/sae_gene_attribution_v2';OLD=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def bh(p):
 p=np.array(p);o=np.argsort(p);q=np.empty(len(p));q[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)
def main():
 start=time.time();OUT.mkdir(exist_ok=True);df=pd.read_csv(V2/'gene_evidence_combined.csv');score='sae_F160_train';df=df.sort_values([score,'gene'],ascending=[False,True]);assert df.gene.is_unique
 protocol={'scope':'frozen F160 TRAIN positive attribution; historical TEST gene effects only descriptive; no test-based gene selection','analyses':['all_input_top40','nonribosomal_top40'],'selection':'top40 positive TRAIN attribution; second analysis removes RPL/RPS prefixes from both background and selected genes before ranking','universe':'all 1838 genes actually covered by TRAIN prototypes; remove RPL/RPS only in nonribosomal analysis; unannotated genes retained','libraries':'MSigDB 2025.1.Hs GO BP, Reactome, Hallmark','pathway_filter':'5<=number of pathway genes in actual background<=500','statistic':'upper-tail hypergeometric overrepresentation and observed/expected overlap','matched_null':'10000 sets per analysis; without replacement within each set; joint quintiles of mean raw count/detection rate across all TRAIN B cells, plus prototype-token coverage bins1-2/3-5/6-8; selected genes remain eligible','multiplicity':'BH across all tested terms in both analyses and all3 collections, separately for hypergeometric and matched p; also within-analysis q as explicitly secondary','seed':20261018,'claims':'pathway association of positive model sensitivity; no pathway activation direction or biological causality; overlapping GO/Reactome entries are not independent mechanisms'}
 js(OUT/'protocol.json',protocol);js(BASE/'resources.json',{'memory':dict(psutil.virtual_memory()._asdict()),'compute':'CPU sparse ORA; matched null in batches of1000; no model inference'})
 files={'Hallmark':'h.all.v2025.1.Hs.symbols.gmt','Reactome':'c2.cp.reactome.v2025.1.Hs.symbols.gmt','GO_BP':'c5.go.bp.v2025.1.Hs.symbols.gmt'};sets=[];prov=[]
 for lib,name in files.items():
  path=BASE/'gene_sets'/name;lines=path.read_text().splitlines();assert all(len(l.split('\t'))>=3 for l in lines)
  for line in lines:
   a=line.split('\t');sets.append((lib,a[0],a[1],set(a[2:])))
  prov.append({'collection':lib,'file':name,'url':'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/'+name,'sha256':sha(path),'bytes':path.stat().st_size,'terms':len(lines)})
 js(OUT/'gene_set_provenance.json',prov)
 m=json.loads((OLD/'data_manifest.json').read_text());k=np.load(OLD/'kang_embeddings.npz');ix=np.array(m['splits'][0]['indices']['train']);ix=ix[k['labels'][ix]==m['classes'].index('B cells')];rawrows=k['rows'][ix]
 with h5py.File(m['preprocess']['dataset'],'r') as h:
  genes=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in h['var'][:]['name']]);X=sparse.csc_matrix((h['X/data'][:],h['X/indices'][:],h['X/indptr'][:]),shape=tuple(h['X'].attrs['h5sparse_shape'])).tocsr();X.eliminate_zeros()
 gm={g:i for i,g in enumerate(genes)};xx=X[rawrows][:,[gm[g] for g in df.gene]].astype(float);df['mean_raw_count']=np.asarray(xx.mean(0)).ravel();df['detection_rate']=np.asarray((xx>0).mean(0)).ravel();df.to_csv(OUT/'gene_background_metrics.csv',index=False)
 records=[];selecteds=[];diagnostics=[]
 for sid,analysis in enumerate(protocol['analyses']):
  a=df.copy() if sid==0 else df[~df.gene.str.startswith(('RPL','RPS'))].copy();a=a.reset_index(drop=True);universe=a.gene.tolist();lookup={g:i for i,g in enumerate(universe)};N=len(a);sel=a[a[score]>0].head(40).copy();assert len(sel)==40;selected=set(sel.gene);n=len(sel);sel['analysis']=analysis;selecteds.append(sel)
  mb=np.minimum(4,(rankdata(a.mean_raw_count,method='average')/N*5).astype(int));db=np.minimum(4,(rankdata(a.detection_rate,method='average')/N*5).astype(int));cb=np.where(a.input_prototypes_train<=2,0,np.where(a.input_prototypes_train<=5,1,2));strata=mb*15+db*3+cb
  a['stratum']=strata;a.to_csv(OUT/f'background_{analysis}.csv',index=False);si=np.array([lookup[g] for g in sel.gene]);counts={int(s):int((strata[si]==s).sum()) for s in np.unique(strata[si])}
  rng=np.random.default_rng([20261018,sid]);draws=np.empty((10000,n),np.int32)
  for s,num in counts.items():diagnostics.append({'analysis':analysis,'stratum':s,'eligible_genes':int((strata==s).sum()),'selected_genes':num,'sampling_fraction':num/int((strata==s).sum())})
  for b in range(10000):draws[b]=np.concatenate([rng.choice(np.flatnonzero(strata==s),num,replace=False) for s,num in counts.items()])
  np.savez_compressed(BASE/f'null_{analysis}.npz',indices=draws,genes=np.array(universe),strata=strata,selected_indices=si)
  rr=[];cc=[];current=[]
  for lib,term,url,gg in sets:
   tested=gg&set(universe);K=len(tested)
   if not 5<=K<=500:continue
   hits=[g for g in sel.gene if g in gg];j=len(current);rr.extend(lookup[g] for g in sorted(tested));cc.extend([j]*K);exp=n*K/N
   current.append({'analysis':analysis,'collection':lib,'term':term,'url':url,'background_n':N,'selected_n':n,'pathway_background_n':K,'pathway_full_n':len(gg),'overlap':len(hits),'expected_overlap':exp,'fold_enrichment':len(hits)/exp,'p_ora':float(hypergeom.sf(len(hits)-1,N,K,n)),'overlap_genes':';'.join(hits),'pathway_tested_genes':';'.join(sorted(tested))})
  membership=sparse.csr_matrix((np.ones(len(rr),np.int32),(rr,cc)),shape=(N,len(current)));observed=np.array([x['overlap'] for x in current]);tail=np.zeros(len(current),np.int64);sums=np.zeros(len(current),np.int64)
  for lo in range(0,10000,1000):
   sub=draws[lo:lo+1000];D=sparse.csr_matrix((np.ones(sub.size,np.int32),(np.repeat(np.arange(len(sub)),n),sub.ravel())),shape=(len(sub),N));v=(D@membership).toarray();tail+=(v>=observed).sum(0);sums+=v.sum(0)
  for j,row in enumerate(current):row.update(p_matched=float((1+tail[j])/10001),matched_tail_count=int(tail[j]),matched_expected_overlap=float(sums[j]/10000))
  records.extend(current);print(analysis,'terms',len(current),'elapsed_s',round(time.time()-start,1),flush=True)
 res=pd.DataFrame(records);res['q_ora_global']=bh(res.p_ora);res['q_matched_global']=bh(res.p_matched)
 for analysis,part in res.groupby('analysis'):
  res.loc[part.index,'q_ora_within_analysis']=bh(part.p_ora);res.loc[part.index,'q_matched_within_analysis']=bh(part.p_matched)
 res['evidence']=np.where((res.q_ora_global<.05)&(res.q_matched_global<.05),'ORA_and_matched_FDR',np.where(res.q_ora_global<.05,'ORA_only_FDR','not_global_FDR'))
 res.sort_values(['q_matched_global','q_ora_global','term']).to_csv(OUT/'all_pathway_results.csv',index=False)
 res[(res.q_ora_global<.05)|(res.q_matched_global<.05)].sort_values(['analysis','q_matched_global','q_ora_global']).to_csv(OUT/'significant_pathways.csv',index=False)
 selected_df=pd.concat(selecteds);selected_df.to_csv(OUT/'selected_genes.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'matched_null_strata.csv',index=False)
 # Explicit pathway->gene evidence; both significant and nonsignificant annotations retained.
 gene_rows=[]
 for row in records:
  for g in row['overlap_genes'].split(';'):
   if not g:continue
   source=a if False else df
   v=source[source.gene==g].iloc[0]
   gene_rows.append({'analysis':row['analysis'],'collection':row['collection'],'term':row['term'],'gene':g,'train_attribution':v[score],'test_attribution':v['sae_F160_test'],'positive_train_donors':int(v['sae_F160_positive_donors']),'input_train_donors':int(v['input_donors'])})
 links=pd.DataFrame(gene_rows).merge(res[['analysis','collection','term','q_ora_global','q_matched_global','evidence','url']],on=['analysis','collection','term']);links.to_csv(OUT/'pathway_gene_links.csv',index=False)
 summary=res.groupby(['analysis','collection']).agg(tested=('term','size'),ora_fdr=('q_ora_global',lambda v:int((v<.05).sum())),matched_fdr=('q_matched_global',lambda v:int((v<.05).sum())),min_matched_q=('q_matched_global','min')).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 # Descriptive annotation lookup for previously identified CD74, irrespective of enrichment.
 res[res.pathway_tested_genes.str.split(';').apply(lambda v:'CD74' in v)].to_csv(OUT/'CD74_pathway_annotations.csv',index=False)
 sources=[V2/'gene_evidence_combined.csv',OLD/'data_manifest.json',OLD/'kang_embeddings.npz',Path(m['preprocess']['dataset']),Path(__file__)]+[BASE/'gene_sets'/v for v in files.values()]
 js(OUT/'input_manifest.json',{str(p):sha(p) for p in sources});js(OUT/'completion.json',{'status':'complete','seconds':time.time()-start,'train_B_cells_for_matching':len(ix),'pathway_tests':len(res),'random_sets_per_analysis':10000,'pathway_gene_links':len(links),'no_model_retraining':True});print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
