"""Read saved rows/predictions independently and replay endpoint arithmetic."""
from pathlib import Path
import json,hashlib,sys,importlib.util,os
for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:os.environ[name]="4"
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score,r2_score,adjusted_rand_score,normalized_mutual_info_score,silhouette_score,mean_absolute_error
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/sae_evaluation'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 for fn in ['SOURCE_BINDINGS.json','DEPENDENCY_BINDINGS.json','TARGET_BINDINGS.json']:
  for p,h in read(O/fn).items():assert sha(p)==h,(fn,p)
 spec=importlib.util.spec_from_file_location('verification_runtime',ROOT/'work/sae_evaluation/run.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.torch.set_num_threads(4);runtime.threadpool_limits(limits=4)
 target_maps={}
 # Reconstruct scores/covariates from raw RNA, not from evaluation predictions.
 for ds in ['ctrl','stim','pbmc']:
  _,xx,genes,ids,labels,gg,cc=runtime.raw(ds)
  with np.load(runtime.W/f'{ds}_targets.npz') as cached:
   rows=cached['rows'];x=xx[rows];total=np.asarray(x.sum(1)).ravel();cols=[np.flatnonzero(genes==g)[0] for g in runtime.PANEL];score=np.log1p(x[:,cols].toarray()/total[:,None]*1e4).mean(1);cov=np.column_stack([np.log1p(total),np.log1p(np.diff(x.indptr)),np.asarray(x[:,np.char.startswith(genes.astype(str),'MT-')].sum(1)).ravel()/total]);np.testing.assert_allclose(cached['score'],score,rtol=0,atol=1e-12);np.testing.assert_allclose(cached['covariates'],cov,rtol=0,atol=1e-12);np.testing.assert_array_equal(cached['cell_ids'],ids[rows]);target_maps[ds]=dict(zip(rows.tolist(),score.tolist()))
 paths=sorted((O/'blocks').glob('*/*/*/completion.json'));assert len(paths)==98
 kn=fs=ifs=cls=maskchecks=0;canonical={};masks={};seen=set();maxerr=0.
 for p in paths:
  dest=p.parent;v=read(p)
  for q,h in v['hashes'].items():assert sha(q)==h
  met=read(dest/'metrics.json');common=met['knn'][0];ds=common['dataset'];fi=common['fold'];method=common['method'];seed=common['seed'];seen.add((ds,fi,method,str(seed)))
  if ds not in canonical:
   cp=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014/kang_embeddings.npz') if ds=='ctrl' else ROOT/'work/sae_extension/data'/f'{ds}.npz'
   with np.load(cp) as a:canonical[ds]=dict(zip(a['rows'].tolist(),a['labels'].tolist()))
  labels=canonical[ds]
  data,_=runtime.getdata(ds,fi);features=runtime.features(ds,fi,seed,method,data);
  if method!='original_h':
   training=read(runtime.ckpt(ds,fi,seed,method).parent/'training.json');assert runtime.base.array_sha(data['train']['h'])==training['train_input_sha'];assert runtime.base.array_sha(data['valid']['h'])==training['valid_input_sha']
  if ds=='ctrl':
   splits=read(ROOT/'outputs/sae_relation/SPLITS.json')[str(fi)]['rows']
   for role in ['train','valid','test']:np.testing.assert_array_equal(data[role]['rows'],splits[role])
  else:
   splits=read(ROOT/'outputs/sae_extension'/ds/'SPLITS.json')[fi]['roles'];canon=runtime.canonical(ds)
   for role in ['train','valid','test']:np.testing.assert_array_equal(data[role]['rows'],canon['rows'][splits[role]])
  mu=data['train']['h'].astype('float64').mean(0);ut=runtime.norm(features['test']-mu);ur=runtime.norm(features['train']-mu);distance=np.clip(1-runtime.mm(ut,ut),0,2);np.fill_diagonal(distance,0);asw=float(silhouette_score(distance,data['test']['y'],metric='precomputed'));assert abs(asw-met['clustering'][0]['cell_type_silhouette'])<1e-10;ni,_=runtime.top(ut[:8],ur,50)
  with np.load(dest/'knn_predictions.npz') as a:
   np.testing.assert_array_equal(a['rows'],data['test']['rows']);np.testing.assert_array_equal(data['train']['rows'][ni],a['neighbors_train_rows'][:8]);yy=np.array([labels[int(r)] for r in a['rows']]);np.testing.assert_array_equal(yy,a['labels']);assert not set(a['rows'])&set(a['neighbors_train_rows'].ravel())
   nl=np.array([[labels[int(r)] for r in rr] for rr in a['neighbors_train_rows']])
   for row in met['knn']:
    k=row['k'];vote=np.stack([(nl[:,:k]==j).sum(1) for j in range(8)],1).argmax(1);np.testing.assert_array_equal(vote,a[f'knn_{k}']);f=f1_score(yy,vote,labels=np.arange(8),average='macro',zero_division=0);err=abs(f-row['macro_f1']);assert err<1e-12;maxerr=max(maxerr,err)
    pur=(nl[:,:k]==yy[:,None]).mean(1);pm=np.mean([pur[yy==c].mean() for c in np.unique(yy)]);assert abs(pm-row['purity_macro_supported'])<1e-12;kn+=1
  with np.load(dest/'fewshot_rows.npz') as a,np.load(dest/'fewshot_predictions.npz') as b:
   yy=np.array([labels[int(r)] for r in b['test_rows']])
   for row in met['fewshot']:
    key=f'sub{row["subset_seed"]}_f{str(row["fraction"]).replace(".","p")}';rr=a[key];assert len(rr)==len(set(rr))==row['labeled_n'];assert not set(rr)&set(b['test_rows']);np.testing.assert_array_equal(np.bincount([labels[int(r)] for r in rr],minlength=8),row['class_counts'])
    mk=(ds,fi,key)
    if mk in masks:np.testing.assert_array_equal(masks[mk],rr);maskchecks+=1
    else:masks[mk]=rr.copy()
    f=f1_score(yy,b[key],labels=np.arange(8),average='macro',zero_division=0);assert abs(f-row['macro_f1'])<1e-12;fs+=1
   for sub in range(5):
    last=set()
    for frac in [.01,.05,.1,.25,.5,1.]:
     rr=set(a[f'sub{sub}_f{str(frac).replace(".","p")}']);assert last<=rr;last=rr
  with np.load(dest/'IFN_predictions.npz') as a:
   for row in met['IFN']:
    if row['status']!='COMPLETE':continue
    c=['B cells','CD14+ Monocytes','CD4 T cells','CD8 T cells','Dendritic cells','FCGR3A+ Monocytes','Megakaryocytes','NK cells'].index(row['cell_type']);yy=a[f'{c}_target'];pp=a[f'{c}_{row["variant"]}'];np.testing.assert_allclose(yy,[target_maps[ds][int(r)] for r in a[f'{c}_rows']],rtol=0,atol=1e-12);assert np.isfinite(pp).all();v=float(r2_score(yy,pp));assert abs(v-row['r2'])<1e-12;assert abs(mean_absolute_error(yy,pp)-row['mae'])<1e-12;assert abs(spearmanr(yy,pp).statistic-row['spearman'])<1e-12;ifs+=1
  with np.load(dest/'clusters.npz') as a:
   for row in met['clustering']:
    yy=a['labels'];pp=a[str(row['resolution'])];assert abs(adjusted_rand_score(yy,pp)-row['ARI'])<1e-12;assert abs(normalized_mutual_info_score(yy,pp)-row['NMI'])<1e-12;cls+=1
 expected={(ds,fi,m,str(s)) for ds in ['ctrl','stim','pbmc'] for fi in range(2 if ds=='pbmc' else 6) for m in ['original_h','mse','relation_0p3'] for s in (['shared'] if m=='original_h' else [42,43,44])};assert seen==expected
 out=dict(status='PASS',blocks=len(paths),knn_metric_replays=kn,fewshot_metric_replays=fs,IFN_metric_replays=ifs,clustering_metric_replays=cls,identical_cross_method_label_masks=maskchecks,max_macro_f1_replay_error=maxerr,explicit_BLAS_silhouette_replays=98,nearest_neighbor_replays=98,raw_RNA_target_replays=3,all_expected_methods_folds_seeds=True,source_and_completion_hashes_verified=True)
 (O/'verification.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
