from pathlib import Path
import pandas as pd,json,hashlib
root=Path(__file__).resolve().parents[2];s=root/'outputs/sae_evaluation';o=root/'outputs/sae_evaluation_seed_summary';a=pd.read_csv(o/'all_seed_metrics.csv');n=0;maxerr=0
spec=[('knn','knn_summary.csv',{'k':15},[]),('IFN','IFN_summary.csv',{},['cell_type','variant']),('fewshot','fewshot_summary.csv',{},['fraction']),('fewshot_AULC','fewshot_AULC_summary.csv',{},[]),('clustering','clustering_summary.csv',{'resolution':1.},[]),('geometry','geometry_summary.csv',{},['role'])]
for task,fn,filt,keys in spec:
 df=pd.read_csv(s/fn);q=a[a.task==task]
 for key,v in filt.items():q=q[q[key]==v]
 for _,r in q.drop_duplicates(['dataset','method','metric']+keys).iterrows():
  if r.metric not in df.columns:continue
  b=df[(df.dataset==r.dataset)&(df.method==r.method)]
  for k in keys:b=b[b[k]==r[k]]
  assert len(b)==1;err=abs(float(b.iloc[0][r.metric])-r.method_mean);assert err<1e-12;maxerr=max(maxerr,err);n+=1
bindings=json.loads((o/'SOURCE_BINDINGS.json').read_text());sealed=json.loads((s/'FINAL_MANIFEST.json').read_text())['files']
for k,v in bindings.items():assert hashlib.sha256(Path(k).read_bytes()).hexdigest()==v==sealed[str(Path(k).relative_to(s))]
b=pd.read_csv(o/'best_seeds_all_configurations.csv');p=pd.read_csv(o/'primary_metrics_best_seeds.csv')
check=dict(status='PASS',mean_entries_verified_against_sealed_summaries=n,max_mean_error=maxerr,total_seed_metric_rows=len(a),configuration_metric_method_rows=len(b),primary_dataset_metric_rows=len(p),source_hashes_match_sealed_manifest=True,scope='Descriptive aggregation only; no new model training or checkpoint replacement')
(o/'verification.json').write_text(json.dumps(check,indent=2));print(check)
manifest={str(f.relative_to(o)):hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file() and f.name!='FINAL_MANIFEST.json'};(o/'FINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
