"""Independent recomputation of result membership/selection/null/BH, no production imports."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy import sparse
from scipy.stats import hypergeom,fisher_exact
from statsmodels.stats.multitest import multipletests
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B_remaining';GS=W.parent/'sae_pathways_v3/gene_sets'
def main():
 d=pd.read_csv(O/'all_pathway_results.csv');sets={}
 for p in GS.glob('*.gmt'):
  for line in p.read_text().splitlines():a=line.split('\t');sets[a[0]]=set(a[2:])
 expected=hypergeom.sf(d.overlap.values-1,d.background_n.values,d.pathway_background_n.values,d.selected_n.values);assert np.max(np.abs(expected-d.p_ora))<1e-12
 for p,q in [('p_ora','q_ora_global'),('p_matched','q_matched_global'),('p_matched_exact','q_matched_exact_global')]:assert np.max(np.abs(multipletests(d[p],method='fdr_bh')[1]-d[q]))<1e-12
 checked=0;maxerr=0.;exact_errors=[]
 for (fid,analysis),group in d.groupby(['feature','analysis'],sort=False):
  a=pd.read_csv(O/f'F{fid}/background_{analysis}.csv');z=np.load(W/f'F{fid}/null_{analysis}.npz');genes=z['genes'];draw=z['indices'];si=z['selected_indices'];st=z['strata'];N=len(genes);n=len(si)
  assert np.array_equal(genes,a.gene.values);assert np.array_equal(a.stratum.values,st);assert np.all(np.diff(np.sort(draw,axis=1),axis=1)>0);assert np.array_equal(np.sort(st[draw],axis=1),np.tile(np.sort(st[si]),(len(draw),1)))
  source=pd.read_csv(O/f'F{fid}/gene_rankings_train.csv').sort_values([f'sae_F{fid}','gene'],ascending=[False,True]);bg=source if analysis=='all_input_top40' else source[~source.gene.str.startswith(('RPL','RPS'))];selected=bg[bg[f'sae_F{fid}']>0].head(40).gene.tolist();assert set(genes)==set(bg.gene);assert list(genes[si])==selected
  valid={term for term,g in sets.items() if 5<=len(set(genes)&g)<=500};assert valid==set(group.term)
  lookup={g:i for i,g in enumerate(genes)};rr=[];cc=[]
  for j,(_,row) in enumerate(group.iterrows()):
   members=sets[row.term]&set(genes);hits=set(selected)&members;assert len(members)==row.pathway_background_n and set(str(row.pathway_tested_genes).split(';'))==members;assert len(hits)==row.overlap
   observed=set(str(row.overlap_genes).split(';')) if pd.notna(row.overlap_genes) else set();assert hits==observed
   rr.extend(lookup[g] for g in members);cc.extend([j]*len(members))
  mat=sparse.csr_matrix((np.ones(len(rr),np.int32),(rr,cc)),shape=(N,len(group)));tails=np.zeros(len(group),int);sums=np.zeros(len(group),int)
  for lo in range(0,len(draw),1000):
   sub=draw[lo:lo+1000];D=sparse.csr_matrix((np.ones(sub.size,np.int32),(np.repeat(np.arange(len(sub)),n),sub.ravel())),shape=(len(sub),N));over=(D@mat).toarray();tails+=(over>=group.overlap.values).sum(0);sums+=over.sum(0)
  maxerr=max(maxerr,float(np.max(np.abs((1+tails)/10001-group.p_matched))));assert np.array_equal(tails,group.matched_tail_count.values);assert np.max(np.abs(sums/10000-group.matched_expected_overlap))<1e-12
  # Independent DP on stratum overlap counts for selected strongest and random terms.
  rng=np.random.default_rng(898+int(fid));ids=set(group.nsmallest(15,'p_matched_exact').index)|set(rng.choice(group.index,min(20,len(group)),replace=False))
  for ix in ids:
   row=group.loc[ix];pm=np.array([1.]);member=set(sets[row.term])
   for layer in np.unique(st[si]):
    pool=genes[st==layer];nn=int((st[si]==layer).sum());kk=sum(g in member for g in pool);part=hypergeom.pmf(np.arange(nn+1),len(pool),kk,nn);result=np.zeros(len(pm)+nn)
    for s in range(len(pm)):
     for t in range(nn+1):result[s+t]+=pm[s]*part[t]
    pm=result
   exact_errors.append(abs(float(pm[int(row.overlap):].sum())-row.p_matched_exact))
  checked+=len(draw)
 for row in d.sample(min(100,len(d)),random_state=2026).itertuples():
  k=int(row.overlap);n=int(row.selected_n);K=int(row.pathway_background_n);N=int(row.background_n);p=fisher_exact([[k,n-k],[K-k,N-K-n+k]],alternative='greater').pvalue;assert abs(p-row.p_ora)<1e-12
 assert maxerr<1e-12 and max(exact_errors)<1e-12
 for path,h in json.loads((O/'pathway_source_hashes.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 for path,h in json.loads((W/'frozen_sources.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 result={'status':'PASS','pathway_hypotheses':len(d),'null_sets_checked':checked,'matched_p_max_error':maxerr,'exact_DP_sample_checked':len(exact_errors),'exact_DP_max_error':max(exact_errors),'Fisher_crosscheck':True,'BH_global_crosscheck':True,'membership_family_and_source_genes_checked':True,'hashes_unchanged':True,'scope':'numerical/provenance/membership; not biological validation'};(O/'pathway_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
