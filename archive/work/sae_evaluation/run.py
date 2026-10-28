"""Frozen SAE evaluation. TRAIN-only fit; fixed endpoints; restartable per arm.
Neighbor cost O(Ntest*Ntrain*d), memory bounded by query blocks. Ridge cost
O(min(n,d)^3+n*d*min(n,d)); sparse RNA scoring never densifies all genes.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[k]='4'
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(__file__).resolve().parent/'mpl')
import json,time,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,h5py
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
from scipy.special import log_softmax
from scipy.stats import spearmanr
from sklearn.metrics import f1_score,r2_score,mean_absolute_error,adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from threadpoolctl import threadpool_limits
import igraph as ig,leidenalg
W=Path(__file__).resolve().parent; ROOT=W.parents[1]; O=ROOT/'outputs/sae_evaluation'; O.mkdir(exist_ok=True)
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ext=module('eval_extension',ROOT/'work/sae_extension/run.py');base=ext.base; core=ext.old
CLASSES=core.CFG['classes']; SEEDS=[42,43,44]; FRACTIONS=[.01,.05,.1,.25,.5,1.]; SUBSEEDS=[0,1,2,3,4]
PANEL=['IFI6','IFIT1','IFIT2','IFIT3','ISG15','MX1','OAS1','RSAD2']
CONFIG=dict(datasets=['ctrl','stim','pbmc'],methods=['original_h','mse','relation_0p3'],seeds=SEEDS,knn_k=[5,15,30,50],knn_primary=15,knn='cosine after common original TRAIN mean; TRAIN references only; uniform voting; ties smallest class id; distance ties TRAIN position',IFN_genes=PANEL,IFN_source=str(core.R.parent/'supcon_tail_20261013/extract.py'),IFN_target='same-RNA proxy: mean log1p(raw counts/library total*10000), no gene-wise global fitting',IFN_types=['B cells','CD14+ Monocytes'],IFN_min_train=100,IFN_min_test=30,IFN_alpha=10.,IFN_technical_sensitivity='TRAIN fitted covariate-only and embedding+covariates; log1p library size, log1p detected genes, mitochondrial fraction',fewshot_fractions=FRACTIONS,fewshot_subseeds=SUBSEEDS,fewshot='fixed alpha10; original TRAIN unlabeled scaler per representation; nested class-stratified ceil(fraction*n_class) without replacement, min1 class; identical rows across methods; all TRAIN unlabeled h allowed; no VALID labels used by downstream readouts; SAE inherited VALID-MSE selection disclosed',fewshot_primary='area under MacroF1 curve vs log(actual total labeled n), normalized within fold',clustering='TEST-only within each fold, common original TRAIN mean, cosine 15NN union undirected graph weighted max(similarity,0); Leiden RBConfiguration resolution1 primary;0.5/2 sensitivity; fixed seed20261019',geometry='TRAIN and TEST per-cell centered reconstruction R2 relative to original TRAIN mean; relative displacement; teacher||student KL tau0.1 on up to256 fixed TRAIN anchors; 15NN retention',statistics='seed means within heldout group, equal group mean; PBMC two batch directions separate; same Kang donors across conditions; no cell or seed pseudo-replication; historically explored endpoints exploratory',scope='cached evaluations only; no new SAE/scGPT training; unified ctrl+stim and UMI downsampling phase2 not run',alpha=10.,version=1)
def save(p,v): core.save(p,v)
def sha(p): return core.sha(p)
def immutable(p,v):core.immutable(p,v)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def mm(a,b):return core.mm(a,b,tb=True)
def top(q,r,k,exclude=None):
 out=[];vals=[]
 for j in range(0,len(q),256):
  a=mm(q[j:j+256],r)
  if exclude is not None:a[np.arange(len(a)),exclude[j:j+256]]=-np.inf
  ix=np.argsort(-a,axis=1,kind='stable')[:,:k];out.append(ix);vals.append(np.take_along_axis(a,ix,axis=1))
 return np.concatenate(out),np.concatenate(vals)
def fit(x,y,xt,weights=None,alpha=10.):
 x=np.asarray(x,dtype='float64');xt=np.asarray(xt,dtype='float64');y=np.asarray(y,dtype='float64');y=y[:,None] if y.ndim==1 else y
 w=np.ones(len(x)) if weights is None else weights
 xm=np.average(x,axis=0,weights=w);ym=np.average(y,axis=0,weights=w)
 a=(x-xm)*np.sqrt(w[:,None]);b=(y-ym)*np.sqrt(w[:,None])
 if len(a)<a.shape[1]:
  g=core.mm(a,a,tb=True);g.flat[::len(g)+1]+=alpha;coef=core.mm(a,cho_solve(cho_factor(g,lower=True,check_finite=False),b,check_finite=False),ta=True)
 else:
  g=core.mm(a,a,ta=True);g.flat[::len(g)+1]+=alpha;coef=cho_solve(cho_factor(g,lower=True,check_finite=False),core.mm(a,b,ta=True),check_finite=False)
 pred=core.mm(xt-xm,coef)+ym;assert np.isfinite(pred).all();return pred

def canonical(ds):
 if ds!='ctrl':return ext.canonical(ds)
 with np.load(core.R/'kang_embeddings.npz') as a:return {**{k:a[k].copy() for k in a.files},'groups':a['donors'].copy()}
def getdata(ds,fi):return base.data(fi,('train','valid','test')) if ds=='ctrl' else ext.data(ds,fi,('train','valid','test'))
def ckpt(ds,fi,seed,arm):
 name={'mse':'continue_mse','relation_0p3':'relation_strong'}[arm] if ds=='ctrl' else {'mse':'lambda_0','relation_0p3':'lambda_0p3'}[arm]
 return (base.W if ds=='ctrl' else ext.dswork(ds))/'networks'/f'fold_{fi}'/f'seed_{seed}'/name/'sae.pt'
def features(ds,fi,seed,arm,d):
 if arm=='original_h':return {r:b['h'].astype('float64') for r,b in d.items()}
 state=torch.load(ckpt(ds,fi,seed,arm),map_location='cpu',weights_only=False)
 model=base.refine.SAE(state['mean'],2048,32,seed);model.load_state_dict(state)
 np.testing.assert_allclose(state['mean'].numpy(),d['train']['h'].astype('float64').mean(0).astype('float32'),rtol=0,atol=1e-7)
 return {r:base.refine.infer(model,b['h']).astype('float64') for r,b in d.items()}
def decode(a):return np.asarray([v.decode() if isinstance(v,bytes) else str(v) for v in a])
def raw(ds):
 p=Path('/Users/apple/Desktop/Llava-Cell/dataset')/('PBMC368k_raw.h5ad' if ds=='pbmc' else 'kang.h5ad')
 with h5py.File(p) as f:
  if ds=='pbmc':
   x=sparse.csr_matrix((f['X/data'][:],f['X/indices'][:],f['X/indptr'][:]),shape=tuple(f['X'].attrs['shape']));genes=decode(f['var/Gene'][:]);ids=decode(f['obs/CellID'][:]);cat=lambda k:decode(f['obs'][k]['categories'][:])[f['obs'][k]['codes'][:]];labels=cat('celltype');groups=cat('batch');cond=np.repeat('unannotated',len(ids))
  else:
   obs=f['obs'][:];x=sparse.csc_matrix((f['X/data'][:],f['X/indices'][:],f['X/indptr'][:]),shape=tuple(f['X'].attrs['h5sparse_shape'])).tocsr();genes=decode(f['var'][:]['name']);ids=decode(obs['index']);cat=lambda k:decode(f['uns'][k+'_categories'][:])[obs[k]];labels=cat('cell_type');groups=cat('replicate');cond=cat('label')
 x.eliminate_zeros();assert np.isfinite(x.data).all() and (x.data>=0).all() and np.allclose(x.data,np.rint(x.data))
 return p,x,genes,ids,labels,groups,cond

def preflight():
 immutable(O/'PROTOCOL.json',CONFIG)
 sources={str(Path(__file__)):sha(__file__)};checks=[]
 for ds in CONFIG['datasets']:
  a=canonical(ds);p,x,genes,ids,labels,groups,cond=raw(ds);ix=a['rows'];np.testing.assert_array_equal(ids[ix],a['cell_ids']);np.testing.assert_array_equal(labels[ix],np.array(CLASSES)[a['labels']]);np.testing.assert_array_equal(groups[ix],a['groups']);np.testing.assert_array_equal(cond[ix],a['condition'])
  cols=[np.flatnonzero(genes==g) for g in PANEL];assert all(len(c)==1 for c in cols)
  xx=x[ix];total=np.asarray(xx.sum(1)).ravel();assert (total>0).all();ng=np.diff(xx.indptr);mito=np.asarray(xx[:,np.char.startswith(genes.astype(str),'MT-')].sum(1)).ravel()/total
  score=np.log1p(xx[:,np.concatenate(cols)].toarray()/total[:,None]*10000).mean(1);cov=np.column_stack([np.log1p(total),np.log1p(ng),mito]);np.savez_compressed(W/f'{ds}_targets.npz',rows=ix,score=score,covariates=cov,cell_ids=a['cell_ids'],genes=np.array(PANEL))
  sources[str(p)]=sha(p);cache=core.R/'kang_embeddings.npz' if ds=='ctrl' else ext.W/'data'/f'{ds}.npz';sources[str(cache)]=sha(cache)
  for fi in range(2 if ds=='pbmc' else 6):
   d,f=getdata(ds,fi);sets={r:set(b['rows']) for r,b in d.items()};assert not sets['train']&sets['test'] and not sets['valid']&sets['test'] and not sets['train']&sets['valid'];assert not set(d['train']['donor'])&set(d['test']['donor'])
   for seed in SEEDS:
    for arm in ['mse','relation_0p3']:
     path=ckpt(ds,fi,seed,arm);sources[str(path)]=sha(path)
   checks.append(dict(dataset=ds,fold=fi,counts={r:len(b['rows']) for r,b in d.items()},groups={r:np.unique(b['donor']).tolist() for r,b in d.items()}))
 immutable(O/'SOURCE_BINDINGS.json',sources);save(O/'PREFLIGHT.json',dict(status='PASS',splits=checks,IFN_genes=PANEL,total_blocks=98))
 save(O/'INTENDED_CLAIMS.json',dict(question='Do fixed dataset-specific SAE reconstructions improve cross-group neighbor tasks, within-type IFN readout, or fixed-readout label efficiency?',not_supported=['Independent biological or causal validation','General improvements from geometry alone','Cross-donor PBMC validation','Blind external confirmation','Unified ctrl/stim state preservation']))
 print('PREFLIGHT_PASS',len(sources),flush=True)

def few_indices(y,fi,sub):
 rng=np.random.default_rng(np.random.SeedSequence([20261019,fi,sub]));orders={c:rng.permutation(np.flatnonzero(y==c)) for c in np.unique(y)}
 return [np.sort(np.concatenate([v[:max(1,int(np.ceil(f*len(v))))] for v in orders.values()])) for f in FRACTIONS]
def regmetric(y,p):
 return dict(r2=float(r2_score(y,p)) if np.var(y)>1e-15 else None,spearman=float(spearmanr(y,p).statistic) if np.std(y)>1e-12 and np.std(p)>1e-12 else None,mae=float(mean_absolute_error(y,p)),target_variance=float(np.var(y)))

def evaluate_block(ds,fi,seed,arm,d,f,target):
 dest=O/'blocks'/ds/f'fold_{fi}'/f'{arm}_seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completion.json').exists():
  c=core.read(dest/'completion.json');assert c['source_sha']==sha(__file__)
  for p,h in c['hashes'].items():assert sha(p)==h
  return
 start=time.time();x=features(ds,fi,seed,arm,d);tr=d['train'];te=d['test'];mu=tr['h'].astype('float64').mean(0);u={r:norm(a-mu) for r,a in x.items()};orig={r:norm(b['h'].astype('float64')-mu) for r,b in d.items()}
 ix,sim=top(u['test'],u['train'],50);oi,_=top(orig['test'],orig['train'],50)
 records=[];pc=[];preds={}
 common=dict(dataset=ds,fold=fi,group=str(te['donor'][0]),seed=seed,method=arm)
 for k in CONFIG['knn_k']:
  labs=tr['y'][ix[:,:k]];counts=np.stack([(labs==j).sum(1) for j in range(8)],1);pred=counts.argmax(1);pur=(labs==te['y'][:,None]).mean(1);per=f1_score(te['y'],pred,labels=np.arange(8),average=None,zero_division=0);supported=np.unique(te['y']);purmacro=float(np.mean([pur[te['y']==c].mean() for c in supported]));ret=np.mean([len(set(a)&set(b))/k for a,b in zip(ix[:,:k],oi[:,:k])])
  records.append(dict(**common,task='knn',k=k,macro_f1=float(per.mean()),supported_macro_f1=float(per[supported].mean()),purity_macro_supported=purmacro,neighbor_retention=float(ret)))
  for c in range(8):pc.append(dict(**common,k=k,cell_type=CLASSES[c],f1=float(per[c]),support=int((te['y']==c).sum()),purity=float(pur[te['y']==c].mean()) if c in supported else None))
  preds[f'knn_{k}']=pred
 np.savez_compressed(dest/'knn_predictions.npz',rows=te['rows'],labels=te['y'],neighbors_train_rows=tr['rows'][ix],**preds)
 # Geometry on all TEST queries and fixed TRAIN query/anchor positions.
 rng=np.random.default_rng(np.random.SeedSequence([20261019,fi,90]));an=rng.choice(len(tr['h']),min(256,len(tr['h'])),replace=False);q=rng.choice(len(tr['h']),min(512,len(tr['h'])),replace=False)
 geo=[]
 for role,query in [('test',np.arange(len(te['h']))),('train',q)]:
  a=mm(orig[role][query],orig['train'][an])/.1;b=mm(u[role][query],u['train'][an])/.1
  if role=='train':
   mask=query[:,None]==an[None,:];a[mask]=-1e10;b[mask]=-1e10
  la=log_softmax(a,axis=1);lb=log_softmax(b,axis=1);kl=float(np.sum(np.exp(la)*(la-lb),axis=1).mean());h=d[role]['h'].astype('float64');rec=x[role];err=np.sum((rec-h)**2);den=np.sum((h-mu)**2)
  geo.append(dict(**common,role=role,reconstruction_r2=float(1-err/den),relative_displacement=float(np.linalg.norm(rec-h)/np.linalg.norm(h)),centered_relative_error=float(np.sqrt(err/den)),relation_KL=kl,anchors=len(an),queries=len(query)))
 np.savez_compressed(dest/'geometry_anchors.npz',anchor_train_rows=tr['rows'][an],query_train_rows=tr['rows'][q])
 # Fixed teacher-unlabeled-TRAIN scaling and fixed alpha; no downstream VALID labels.
 sd=x['train'].std(0);sd[sd<1e-12]=1;mx=x['train'].mean(0);st=(x['train']-mx)/sd;sq=(x['test']-mx)/sd
 few=[];masks={};fscores={}
 for sub in SUBSEEDS:
  for frac,sel in zip(FRACTIONS,few_indices(tr['y'],fi,sub)):
   y=tr['y'][sel];scores=fit(st[sel],2*np.eye(8)[y]-1,sq,core.weights(y));pred=scores.argmax(1);v=f1_score(te['y'],pred,labels=np.arange(8),average=None,zero_division=0)
   few.append(dict(**common,subset_seed=sub,fraction=frac,labeled_n=len(sel),macro_f1=float(v.mean()),per_class_f1=v.tolist(),class_counts=np.bincount(y,minlength=8).tolist()));key=f'sub{sub}_f{str(frac).replace(".","p")}';masks[key]=tr['rows'][sel];fscores[key]=pred
 np.savez_compressed(dest/'fewshot_rows.npz',**masks);np.savez_compressed(dest/'fewshot_predictions.npz',test_rows=te['rows'],**fscores)
 # Within-type same-RNA IFN proxy, fixed gene panel, fixed Ridge alpha.
 pos={int(r):i for i,r in enumerate(target['rows'])};ti=np.array([pos[int(r)] for r in tr['rows']]);qi=np.array([pos[int(r)] for r in te['rows']]);state=[];sp={}
 for name in CONFIG['IFN_types']:
  c=CLASSES.index(name);i=np.flatnonzero(tr['y']==c);j=np.flatnonzero(te['y']==c);z=dict(**common,cell_type=name,n_train=len(i),n_test=len(j))
  if len(i)<100 or len(j)<30:state.append(dict(**z,status='SKIPPED_PREDECLARED_COUNTS'));continue
  a=x['train'][i];b=x['test'][j];m=a.mean(0);s=a.std(0);s[s<1e-12]=1;a=(a-m)/s;b=(b-m)/s
  cov=target['covariates'][ti[i]];ct=target['covariates'][qi[j]];cm=cov.mean(0);cs=cov.std(0);cs[cs<1e-12]=1;cov=(cov-cm)/cs;ct=(ct-cm)/cs
  yt=target['score'][ti[i]];yq=target['score'][qi[j]];ym=yt.mean();ys=max(yt.std(),1e-12);yt=(yt-ym)/ys
  for variant,aa,bb in [('embedding',a,b),('covariates',cov,ct),('embedding_plus_covariates',np.column_stack([a,cov]),np.column_stack([b,ct]))]:
   pp=fit(aa,yt,bb).ravel()*ys+ym;state.append(dict(**z,status='COMPLETE',variant=variant,train_target_variance=float(ys**2),**regmetric(yq,pp)));sp[f'{c}_{variant}']=pp
  sp[f'{c}_rows']=te['rows'][j];sp[f'{c}_target']=yq
 np.savez_compressed(dest/'IFN_predictions.npz',**sp)
 # TEST-only geometry clustering. No label resolution search.
 nn,ss=top(u['test'],u['test'],15,exclude=np.arange(len(te['h'])));edges={}
 for i in range(len(nn)):
  for j,v in zip(nn[i],ss[i]):
   if v>0:edges[tuple(sorted([i,int(j)]))]=float(v)
 g=ig.Graph(n=len(nn),edges=list(edges),directed=False);weights=list(edges.values());clusters={};cl=[]
 for res in [.5,1.,2.]:
  part=leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,weights=weights,resolution_parameter=res,seed=20261019,n_iterations=-1);pred=np.array(part.membership);clusters[str(res)]=pred
  cl.append(dict(**common,resolution=res,n_clusters=len(set(pred)),ARI=float(adjusted_rand_score(te['y'],pred)),NMI=float(normalized_mutual_info_score(te['y'],pred))))
 asw=float(silhouette_score(u['test'],te['y'],metric='cosine'))
 for r in cl:r['cell_type_silhouette']=asw
 np.savez_compressed(dest/'clusters.npz',rows=te['rows'],labels=te['y'],**clusters)
 save(dest/'metrics.json',dict(knn=records,knn_per_class=pc,geometry=geo,fewshot=few,IFN=state,clustering=cl,seconds=time.time()-start))
 hashes={str(p):sha(p) for p in dest.iterdir() if p.is_file() and p.name!='completion.json'};save(dest/'completion.json',dict(source_sha=sha(__file__),hashes=hashes,seconds=time.time()-start))
 print('BLOCK_COMPLETE',ds,fi,seed,arm,'seconds',round(time.time()-start,2),flush=True)

def run(pilot=False):
 assert core.read(O/'PROTOCOL.json')==CONFIG
 bindings=core.read(O/'SOURCE_BINDINGS.json')
 for p,h in bindings.items():assert sha(p)==h,p
 for ds in (['pbmc'] if pilot else CONFIG['datasets']):
  with np.load(W/f'{ds}_targets.npz') as a:target={k:a[k].copy() for k in a.files}
  for fi in (range(1) if pilot else range(2 if ds=='pbmc' else 6)):
   d,f=getdata(ds,fi)
   for arm in CONFIG['methods']:
    for seed in (['shared'] if arm=='original_h' else ([42] if pilot else SEEDS)):
     evaluate_block(ds,fi,seed,arm,d,f,target)
     done=len(list((O/'blocks').glob('*/*/*/completion.json')));save(O/'EXPERIMENT_TRACKER.json',dict(status='RUNNING',completed=done,total=98,dataset=ds,fold=fi,method=arm,seed=seed))
 if not pilot:summarize()

def summarize():
 paths=sorted((O/'blocks').glob('*/*/*/metrics.json'));assert len(paths)==98,len(paths)
 for task in ['knn','knn_per_class','geometry','fewshot','IFN','clustering']:
  df=pd.DataFrame([r for p in paths for r in core.read(p)[task]]);df.to_csv(O/f'{task}_all.csv',index=False)
 save(O/'EXPERIMENT_TRACKER.json',dict(status='EVALUATED_PENDING_AUDIT',completed=98,total=98))
 print('ALL_EVALUATIONS_COMPLETE',flush=True)
if __name__=='__main__':
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 with threadpool_limits(limits=4):
  stage=sys.argv[1]
  if stage=='preflight':preflight()
  elif stage=='pilot':run(True)
  elif stage=='run':run()
  elif stage=='summarize':summarize()
