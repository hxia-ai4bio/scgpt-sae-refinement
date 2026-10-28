"""Prespecified positive attribution ORA, conditional matched null, global BH.
Sparse membership O(genes*terms nonzeros); random draws processed 1000 at a time.
Exact conditional sensitivity uses convolution of independent stratum hypergeometrics.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd,h5py
from scipy import sparse
from scipy.stats import rankdata,hypergeom
from statsmodels.stats.multitest import multipletests
W=Path(__file__).resolve().parent;R=W.parents[1];O=R/'outputs/sae_B_remaining';GS=W.parent/'sae_pathways_v3/gene_sets';HIST=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def bh(p):return multipletests(p,method='fdr_bh')[1]
def exact_tail(strata,si,member,k):
 distribution=np.array([1.])
 for st in np.unique(strata[si]):
  ids=np.flatnonzero(strata==st);n=int((strata[si]==st).sum());K=int(member[ids].sum());pmf=hypergeom.pmf(np.arange(n+1),len(ids),K,n)
  distribution=np.convolve(distribution,pmf)
 return float(np.clip(distribution[k:].sum(),0,1))
def main():
 start=time.time();sets=[];sources={};files={'GO_BP':'c5.go.bp.v2025.1.Hs.symbols.gmt','Reactome':'c2.cp.reactome.v2025.1.Hs.symbols.gmt','Hallmark':'h.all.v2025.1.Hs.symbols.gmt'}
 for lib,name in files.items():
  p=GS/name;sources[str(p)]=sha(p)
  for line in p.read_text().splitlines():
   a=line.split('\t');sets.append((lib,a[0],a[1],set(a[2:])))
 go=json.loads((GS/'c5.go.bp.v2025.1.Hs.json').read_text());sources[str(GS/'c5.go.bp.v2025.1.Hs.json')]=sha(GS/'c5.go.bp.v2025.1.Hs.json')
 for lib,term,url,g in sets:
  if lib=='GO_BP':assert set(go[term]['geneSymbols'])==g
 m=json.loads((HIST/'data_manifest.json').read_text());k=np.load(HIST/'kang_embeddings.npz');ix=np.array(m['splits'][0]['indices']['train']);ix=ix[k['labels'][ix]==m['classes'].index('B cells')];rawrows=k['rows'][ix]
 with h5py.File(m['preprocess']['dataset'],'r') as h:
  genes=np.array([v.decode() if isinstance(v,bytes) else str(v) for v in h['var'][:]['name']]);X=sparse.csc_matrix((h['X/data'][:],h['X/indices'][:],h['X/indptr'][:]),shape=tuple(h['X'].attrs['h5sparse_shape'])).tocsr();X.eliminate_zeros()
 assert len(np.unique(genes))==len(genes);gm={g:i for i,g in enumerate(genes)};raw=X[rawrows].astype(float);mean=np.asarray(raw.mean(0)).ravel();detect=np.asarray((raw>0).mean(0)).ravel()
 records=[];selecteds=[];stratum_rows=[]
 for fid in [137,370,146,1373,1125,1409]:
  FO=O/f'F{fid}';FW=W/f'F{fid}';df=pd.read_csv(FO/'gene_rankings_train.csv').rename(columns={'input_prototypes':'input_prototypes_train'});score=f'sae_F{fid}';df=df.sort_values([score,'gene'],ascending=[False,True]).reset_index(drop=True)
  test=pd.read_csv(FO/'gene_rankings_test.csv');df=df.rename(columns={score:score+'_train'});score+='_train';df=df.merge(test[['gene',f'sae_F{fid}','input_prototypes']].rename(columns={f'sae_F{fid}':f'sae_F{fid}_test','input_prototypes':'input_prototypes_test'}),on='gene',how='left');df['train_rank']=np.arange(1,len(df)+1);df['mean_raw_count']=[mean[gm[g]] for g in df.gene];df['detection_rate']=[detect[gm[g]] for g in df.gene];df.to_csv(FO/'gene_evidence_combined.csv',index=False)
  for sid,analysis in enumerate(['all_input_top40','nonribosomal_top40']):
   a=df.copy() if sid==0 else df[~df.gene.str.startswith(('RPL','RPS'))].copy();a=a.reset_index(drop=True);N=len(a);sel=a[a[score]>0].head(40);assert len(sel)==40;lookup={g:i for i,g in enumerate(a.gene)};si=np.array([lookup[g] for g in sel.gene]);n=len(si)
   mb=np.minimum(4,(rankdata(a.mean_raw_count,method='average')/N*5).astype(int));db=np.minimum(4,(rankdata(a.detection_rate,method='average')/N*5).astype(int));cb=np.where(a.input_prototypes_train<=2,0,np.where(a.input_prototypes_train<=5,1,2));strata=mb*15+db*3+cb;a['stratum']=strata;a.to_csv(FO/f'background_{analysis}.csv',index=False)
   s=sel.copy();s['feature']=fid;s['analysis']=analysis;selecteds.append(s)
   counts={int(st):int((strata[si]==st).sum()) for st in np.unique(strata[si])};rng=np.random.default_rng([20261018,fid,sid]);draws=np.empty((10000,n),np.int32)
   for st,num in counts.items():stratum_rows.append({'feature':fid,'analysis':analysis,'stratum':st,'eligible':int((strata==st).sum()),'selected':num})
   for b in range(len(draws)):draws[b]=np.concatenate([rng.choice(np.flatnonzero(strata==st),num,replace=False) for st,num in counts.items()])
   np.savez_compressed(FW/f'null_{analysis}.npz',indices=draws,genes=a.gene.values.astype(str),strata=strata,selected_indices=si)
   rr=[];cc=[];current=[];gene_set=set(a.gene)
   for lib,term,url,gg in sets:
    tested=gg&gene_set;K=len(tested)
    if not 5<=K<=500:continue
    hit=[g for g in sel.gene if g in gg];j=len(current);cols=[lookup[g] for g in sorted(tested)];rr.extend(cols);cc.extend([j]*K);member=np.zeros(N,bool);member[cols]=True
    current.append({'feature':fid,'analysis':analysis,'collection':lib,'term':term,'GO_ID':go[term]['exactSource'] if lib=='GO_BP' else '', 'url':url,'background_n':N,'selected_n':n,'pathway_background_n':K,'pathway_full_n':len(gg),'overlap':len(hit),'expected_overlap':n*K/N,'fold_enrichment':len(hit)/(n*K/N),'p_ora':float(hypergeom.sf(len(hit)-1,N,K,n)),'p_matched_exact':exact_tail(strata,si,member,len(hit)),'overlap_genes':';'.join(hit),'pathway_tested_genes':';'.join(sorted(tested))})
   mat=sparse.csr_matrix((np.ones(len(rr),np.int32),(rr,cc)),shape=(N,len(current)));observed=np.array([v['overlap'] for v in current]);tail=np.zeros(len(current),int);sums=np.zeros(len(current),int)
   for lo in range(0,len(draws),1000):
    sub=draws[lo:lo+1000];D=sparse.csr_matrix((np.ones(sub.size,np.int32),(np.repeat(np.arange(len(sub)),n),sub.ravel())),shape=(len(sub),N));over=(D@mat).toarray();tail+=(over>=observed).sum(0);sums+=over.sum(0)
   for j,v in enumerate(current):v.update(p_matched=(1+int(tail[j]))/10001,matched_tail_count=int(tail[j]),matched_expected_overlap=float(sums[j]/10000))
   records.extend(current)
   for metric in ['mean_raw_count','detection_rate','input_prototypes_train']:
    values=a[metric].values;random=values[draws].mean(1);stratum_rows.append({'feature':fid,'analysis':analysis,'metric':metric,'selected_mean':float(values[si].mean()),'null_mean':float(random.mean()),'null_sd':float(random.std()),'z_balance':float((values[si].mean()-random.mean())/random.std()) if random.std()>1e-12 else None})
   print('PATHWAYS',fid,analysis,len(current),'elapsed_s',round(time.time()-start),flush=True)
 result=pd.DataFrame(records)
 for p,q in [('p_ora','q_ora_global'),('p_matched','q_matched_global'),('p_matched_exact','q_matched_exact_global')]:result[q]=bh(result[p])
 # Primary family spans all SIX features, BOTH lists, ALL three libraries. Within-feature q is secondary.
 for fid,part in result.groupby('feature'):
  for p,q in [('p_ora','q_ora_within_feature'),('p_matched','q_matched_within_feature'),('p_matched_exact','q_matched_exact_within_feature')]:result.loc[part.index,q]=bh(part[p])
 result['evidence']=np.where((result.q_ora_global<.05)&(result.q_matched_global<.05),'ORA_and_matched_FDR',np.where(result.q_ora_global<.05,'ORA_only_FDR','not_global_FDR'))
 result=result.sort_values(['feature','analysis','q_matched_global','q_ora_global','term']);result.to_csv(O/'all_pathway_results.csv',index=False);result[(result.q_ora_global<.05)|(result.q_matched_global<.05)].to_csv(O/'significant_pathways.csv',index=False);pd.concat(selecteds).to_csv(O/'selected_genes.csv',index=False);pd.DataFrame(stratum_rows).to_csv(O/'matching_diagnostics.csv',index=False)
 summary=result.groupby(['feature','analysis','collection']).agg(tested=('term','size'),ora_fdr=('q_ora_global',lambda x:int((x<.05).sum())),matched_fdr=('q_matched_global',lambda x:int((x<.05).sum())),exact_sensitivity_fdr=('q_matched_exact_global',lambda x:int((x<.05).sum())),min_matched_q=('q_matched_global','min')).reset_index();summary.to_csv(O/'pathway_summary.csv',index=False)
 links=[]
 for row in result.to_dict('records'):
  for gene in str(row['overlap_genes']).split(';'):
   if gene:links.append({key:row[key] for key in ['feature','analysis','collection','term','GO_ID','q_ora_global','q_matched_global','q_matched_exact_global','evidence']}|{'gene':gene})
 # Python3.9 dictionary unions supported.
 pd.DataFrame(links).to_csv(O/'pathway_gene_links.csv',index=False)
 sources.update({str(p):sha(p) for p in [Path(__file__),HIST/'data_manifest.json',HIST/'kang_embeddings.npz',Path(m['preprocess']['dataset'])]+[O/f'F{fid}/gene_rankings_train.csv' for fid in [137,370,146,1373,1125,1409]]})
 save(O/'pathway_source_hashes.json',sources);save(O/'pathway_completion.json',{'status':'complete','seconds':time.time()-start,'tests':len(result),'TRAIN_B_cells_for_matching':len(ix),'random_sets':120000,'global_family':'all SIX features, both gene lists, all 3 databases','primary_matched_test':'MonteCarlo10000 with add-one correction; global BH','exact_sensitivity':'convolution of stratum-conditioned hypergeometric distributions; separate global BH; same coarse-bin assumptions'})
 print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
