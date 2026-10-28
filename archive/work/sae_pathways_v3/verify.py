from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy import sparse
from scipy.stats import hypergeom,fisher_exact
from statsmodels.stats.multitest import multipletests
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_pathways_v3'
def main():
 d=pd.read_csv(O/'all_pathway_results.csv');p=hypergeom.sf(d.overlap.values-1,d.background_n.values,d.pathway_background_n.values,d.selected_n.values);assert np.max(np.abs(p-d.p_ora))<1e-12
 for col,q in [('p_ora','q_ora_global'),('p_matched','q_matched_global')]:assert np.max(np.abs(multipletests(d[col],method='fdr_bh')[1]-d[q]))<1e-12
 sets={}
 for file in (W/'gene_sets').glob('*.gmt'):
  for line in file.read_text().splitlines():
   a=line.split('\t');sets[a[0]]=set(a[2:])
 checks=0;maxerr=0.;drawsets=0
 for analysis,g in d.groupby('analysis',sort=False):
  z=np.load(W/f'null_{analysis}.npz');genes=z['genes'];idx=z['indices'];si=z['selected_indices'];st=z['strata'];N=len(genes);n=len(si);lookup={v:i for i,v in enumerate(genes)}
  assert np.all(np.diff(np.sort(idx,axis=1),axis=1)>0);assert np.array_equal(np.sort(st[idx],axis=1),np.repeat(np.sort(st[si])[None,:],len(idx),axis=0));drawsets+=len(idx)
  bg=pd.read_csv(O/f'background_{analysis}.csv');assert np.array_equal(bg.gene.values,genes)
  selected=bg[bg.sae_F160_train>0].head(40).gene.tolist();assert set(selected)==set(genes[si])
  if analysis=='nonribosomal_top40':assert not any(v.startswith(('RPL','RPS')) for v in genes)
  rr=[];cc=[]
  for j,(_,row) in enumerate(g.iterrows()):
   gg=sets[row.term];tested=gg&set(genes);hits=set(selected)&gg
   assert len(tested)==row.pathway_background_n and len(hits)==row.overlap
   assert set(str(row.overlap_genes).split(';'))==hits if hits else pd.isna(row.overlap_genes)
   rr.extend(lookup[v] for v in tested);cc.extend([j]*len(tested));checks+=1
  M=sparse.csr_matrix((np.ones(len(rr),int),(rr,cc)),shape=(N,len(g)));tail=np.zeros(len(g),int);sums=np.zeros(len(g),int);obs=g.overlap.values
  for start in range(0,len(idx),1000):
   sub=idx[start:start+1000];S=sparse.csr_matrix((np.ones(sub.size,int),(np.repeat(np.arange(len(sub)),n),sub.ravel())),shape=(len(sub),N));counts=(S@M).toarray();tail+=(counts>=obs).sum(0);sums+=counts.sum(0)
  assert np.array_equal(tail,g.matched_tail_count);maxerr=max(maxerr,float(np.max(np.abs((tail+1)/(len(idx)+1)-g.p_matched))))
  assert np.max(np.abs(sums/len(idx)-g.matched_expected_overlap))<1e-12
  for _,row in g.sample(min(5,len(g)),random_state=42).iterrows():
   K,k=int(row.pathway_background_n),int(row.overlap);fp=fisher_exact([[k,n-k],[K-k,N-K-n+k]],alternative='greater')[1];assert abs(fp-row.p_ora)<1e-10
 for file,h in json.loads((O/'input_manifest.json').read_text()).items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h,file
 result={'status':'PASS','pathway_hypotheses_recomputed':checks,'matched_random_sets_checked':drawsets,'matched_p_max_error':maxerr,'hypergeom_fisher_crosscheck':True,'BH_statsmodels_crosscheck':True,'input_hashes_unchanged':True,'scope':'arithmetic, membership, matched strata and provenance; not biological validation'}
 (O/'verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
