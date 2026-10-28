"""Frozen-backbone gene-masked A/C refinement. No TEST-based fitting/selection.
SAE O(S*B*512*2048), C head additional O(S*B*512*2048).
Ridge O(N*512^2+512^3); nearest neighbors O(Ntest*Ntrain*512), block memory.
"""
import os
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_masked_ac'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
os.environ['MPLCONFIGDIR']=str(W/'mpl')
import json,importlib.util,time,math,copy,sys
import numpy as np,pandas as pd,torch
from torch import nn
from scipy import sparse
from scipy.special import log_softmax
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=module('masked_preparation',W/'prepare.py');ev=module('masked_metrics',ROOT/'work/sae_evaluation/run.py');core=p.core;base=p.base;save=core.save;read=core.read;sha=core.sha
SEEDS=[42,43,44];ARMS=['A_sae','C_sae','A_dense'];DATASETS=['ctrl','stim','pbmc']

def canonical(ds):
 with np.load(W/'data'/f'{ds}.npz') as q:return {k:q[k].copy() for k in q.files}
def source_warm(ds,fi,seed):return (core.src_for(fi,seed) if ds=='ctrl' else p.ext.initialdir(ds,fi,seed))/'sae.pt'
def source_mse(ds,fi,seed):return (base.W/'networks'/f'fold_{fi}'/f'seed_{seed}'/'continue_mse' if ds=='ctrl' else p.ext.dswork(ds)/'networks'/f'fold_{fi}'/f'seed_{seed}'/'lambda_0')/'sae.pt'
def fold_data(a,ds,fi):
 d,f=ev.getdata(ds,fi);pos={int(r):i for i,r in enumerate(a['rows'])};ix={r:np.array([pos[int(rr)] for rr in b['rows']]) for r,b in d.items()}
 for role,b in d.items():np.testing.assert_array_equal(a['embedding'][ix[role]],b['h']);np.testing.assert_array_equal(a['labels'][ix[role]],b['y'])
 return d,f,ix

def panel(ds,fi,a,ix):
 path=W/'panels'/ds/f'fold_{fi}.npz'
 if path.exists():
  with np.load(path) as q:return {k:q[k].copy() for k in q.files}
 x=sparse.load_npz(W/'data'/ds/'normalized_expression.npz')[ix['train']];s1=np.asarray(x.sum(0)).ravel();s2=np.asarray(x.power(2).sum(0)).ravel();nnz=np.asarray((x>0).sum(0)).ravel();mu=s1/np.maximum(nnz,1);sd=np.sqrt(np.maximum(s2/np.maximum(nnz,1)-mu**2,0));sd=np.maximum(sd,.1);variance=s2/len(ix['train'])-(s1/len(ix['train']))**2
 vocab=read(Path('/Users/apple/Desktop/scGPT/scGPT_human/vocab.json'));eligible=(nnz>=20)&np.array([g in vocab for g in a['gene_names']])&(variance>0);cc=np.flatnonzero(eligible);assert len(cc)>=2048;cols=np.sort(cc[np.argsort(-variance[cc],kind='stable')[:2048]]);v=dict(columns=cols,gene_names=a['gene_names'][cols],mean=mu[cols],std=sd[cols],train_positive_counts=nnz[cols],train_variance=variance[cols],train_rows=a['rows'][ix['train']]);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**v);return v

def targets(a,panel):
 lookup=np.full(len(a['gene_names']),-1,dtype=np.int64);lookup[panel['columns']]=np.arange(2048);cols=lookup[np.maximum(a['columns'],0)];cols[a['columns']<0]=-1;valid=a['hidden']&(cols>=0);safe=np.maximum(cols,0);target=((a['target_log_expression']-panel['mean'][safe])/panel['std'][safe]).astype('float32');return dict(cols=cols,valid=valid,target=target)

class Refiner(nn.Module):
 def __init__(self,ds,fi,seed,arm,train=None):
  super().__init__();state=torch.load(source_warm(ds,fi,seed),map_location='cpu',weights_only=False);self.arm=arm;self.register_buffer('mean',state['mean'].clone());self.register_buffer('sd',torch.from_numpy(train.astype('float64').std(0).clip(1e-6).astype('float32')) if train is not None else torch.ones(512));torch.manual_seed(seed+20261021)
  if arm=='A_dense':
   self.network=nn.Sequential(nn.Linear(512,2048),nn.GELU(),nn.Linear(2048,512));nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
  else:
   self.network=base.refine.SAE(state['mean'],2048,32,seed);self.network.load_state_dict(state)
  if arm=='C_sae':
   torch.manual_seed(seed+2003);self.head=nn.Linear(512,2048);nn.init.normal_(self.head.weight,std=.002);nn.init.zeros_(self.head.bias)
 def forward(self,h):return h+self.network(h-self.mean) if self.arm=='A_dense' else self.network(h)[0]
 def predict_genes(self,rec):return self.head((rec-self.mean)/self.sd)
 def normalize(self):
  if self.arm!='A_dense':self.network.normalize()

@torch.no_grad()
def inference(model,h):
 model.eval();out=[]
 for j in range(0,len(h),512):out.append(model(torch.from_numpy(h[j:j+512].astype('float32'))).numpy())
 out=np.concatenate(out);assert np.isfinite(out).all();return out.astype('float64')

def arraysha(a):return base.array_sha(a)
def batches(ds,fi,seed,d):
 home=base.W if ds=='ctrl' else p.ext.dswork(ds);path=home/'batches'/f'fold_{fi}_seed_{seed}.npz'
 with np.load(path) as q:ix=q['indices'].copy();np.testing.assert_array_equal(q['rows'],d['train']['rows'][ix])
 assert ix.shape==(1000,256);return ix

def mask_loss(model,h,clean,co,va,ta,den):
 rec=model(h)
 if model.arm=='C_sae':
  pred=model.predict_genes(rec).gather(1,co.clamp(min=0));assert va.any();return (pred[va]-ta[va]).square().mean()
 return (rec-clean).square().mean()/den

def train(ds,fi,seed,arm,a,d,ix,pn,t):
 dest=W/'networks'/ds/f'fold_{fi}'/f'seed_{seed}'/arm;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completion.json').exists():
  seal=read(dest/'completion.json');assert seal['source_sha']==sha(__file__)
  for path,h in seal['hashes'].items():assert sha(path)==h
  return
 start=time.time();model=Refiner(ds,fi,seed,arm,d['train']['h']);initial=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.0003);bi=batches(ds,fi,seed,d);idx=ix['train'];vh=ix['valid'];h=torch.from_numpy(a['masked_embedding'][idx].astype('float32'));clean=torch.from_numpy(a['embedding'][idx].astype('float32'));co=torch.from_numpy(t['cols'][idx]);va=torch.from_numpy(t['valid'][idx]);ta=torch.from_numpy(t['target'][idx]);den=float(np.mean((d['train']['h'].astype('float64')-model.mean.numpy())**2));assert den>0
 rng=np.random.default_rng(np.random.SeedSequence([20261021,fi,seed]));vi=rng.choice(vh,min(512,len(vh)),replace=False);validation=[torch.from_numpy(a['masked_embedding'][vi].astype('float32')),torch.from_numpy(a['embedding'][vi].astype('float32')),torch.from_numpy(t['cols'][vi]),torch.from_numpy(t['valid'][vi]),torch.from_numpy(t['target'][vi])];history=[];grad=[]
 for step in range(1001):
  if step in [0,100,250,500,750,1000]:
   model.eval()
   with torch.no_grad():value=float(mask_loss(model,*validation,den));recon=inference(model,a['embedding'][vi]);mse=float(np.mean((recon-a['embedding'][vi])**2)/den)
   assert np.isfinite(value);history.append(dict(step=step,valid_objective=value,valid_clean_normalized_embedding_MSE=mse))
  if step==1000:break
  model.train();ids=bi[step];lr=.00001+.5*(.0003-.00001)*(1+math.cos(math.pi*step/999))
  for group in opt.param_groups:group['lr']=lr
  opt.zero_grad(set_to_none=True);loss=mask_loss(model,h[ids],clean[ids],co[ids],va[ids],ta[ids],den);assert torch.isfinite(loss);loss.backward()
  if step in [0,1,999]:
   norms={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None};assert all(np.isfinite(v) for v in norms.values());grad.append(dict(step=step+1,norms=norms,loss=float(loss)))
  opt.step();model.normalize()
 state=model.state_dict();q=dest/'model.pt';temp=q.with_suffix('.tmp');torch.save(state,temp);temp.replace(q);changed={name:float((state[name]-initial[name]).double().norm()) for name in state if torch.is_floating_point(state[name])};assert max(changed.values())>0
 save(dest/'training.json',dict(dataset=ds,fold=fi,seed=seed,method=arm,steps=1000,selected_step=1000,parameter_count=sum(v.numel() for v in model.parameters()),warmup_sha=sha(source_warm(ds,fi,seed)),train_clean_sha=arraysha(d['train']['h']),train_masked_sha=arraysha(a['masked_embedding'][idx]),valid_clean_sha=arraysha(d['valid']['h']),batch_array_sha=arraysha(bi),train_rows=d['train']['rows'].tolist(),history=history,gradients=grad,weight_changes=changed,seconds=time.time()-start,checkpoint_sha=sha(q),protocol_sha=sha(O/'PROTOCOL.json'),panel_sha=sha(W/'panels'/ds/f'fold_{fi}.npz')))
 save(dest/'completion.json',dict(source_sha=sha(__file__),hashes={str(q):sha(q),str(dest/'training.json'):sha(dest/'training.json')}));print('FIT_COMPLETE',ds,fi,seed,arm,'seconds',round(time.time()-start,2),flush=True)

def restore(ds,fi,seed,arm):
 if arm=='original_h':return None
 if arm=='clean_mse':
  state=torch.load(source_mse(ds,fi,seed),map_location='cpu',weights_only=False);m=base.refine.SAE(state['mean'],2048,32,seed);m.load_state_dict(state)
  class Wrap(nn.Module):
   def __init__(self,m):super().__init__();self.m=m
   def forward(self,h):return self.m(h)[0]
  return Wrap(m)
 model=Refiner(ds,fi,seed,arm);model.load_state_dict(torch.load(W/'networks'/ds/f'fold_{fi}'/f'seed_{seed}'/arm/'model.pt',map_location='cpu',weights_only=False));return model

def features(model,a,ix,view):
 h=a['embedding'] if view=='clean' else a['masked_embedding'];return {role:(h[ii].astype('float64') if model is None else inference(model,h[ii])) for role,ii in ix.items()}

def develop(ds,fi,seed,arm,a,d,ix):
 dest=W/'probes'/ds/f'fold_{fi}'/f'seed_{seed}'/arm;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completion.json').exists():
  z=read(dest/'completion.json');assert z['source_sha']==sha(__file__)
  for p0,h in z['hashes'].items():assert sha(p0)==h
  return
 model=restore(ds,fi,seed,arm);x=features(model,a,{r:ix[r] for r in ['train','valid']},'clean');ridge=core.Ridge(x['train'],d['train']['y']);best=-1;rows=[]
 for alpha in core.ALPHAS:
  ridge.fit(alpha);value=core.metrics(d['valid']['y'],ridge.scores(x['valid']))['macro_f1'];rows.append(dict(alpha=alpha,valid_macro_f1=value))
  if value>best:best=value;selected=alpha;weights=copy.deepcopy(ridge.arrays())
 np.savez_compressed(dest/'readout.npz',**weights);save(dest/'selection.json',dict(alpha=selected,valid_macro_f1=best,grid=rows,fit_roles=['TRAIN_clean'],selection_roles=['VALID_clean'],no_TEST_seen=True));save(dest/'completion.json',dict(source_sha=sha(__file__),hashes={str(q):sha(q) for q in dest.iterdir() if q.is_file() and q.name!='completion.json'}))

def check_sources():
 for path,h in read(W/'sources.json').items():assert sha(path)==h,path

def preflight():
 cfg=read(O/'PROTOCOL.json');assert cfg['datasets']==DATASETS and cfg['seeds']==SEEDS
 src={str(q):sha(q) for q in [Path(__file__),W/'prepare.py',ROOT/'work/sae_evaluation/run.py',ROOT/'work/sae_complementarity/core.py',ROOT/'work/sae_refine/run.py',ROOT/'work/sae_extension/run.py',ROOT/'work/sae_relation/run.py',O/'PROTOCOL.json',core.R/'protocol.json']};checks=[]
 for ds in DATASETS:
  q=W/'data'/f'{ds}.npz';manifest=read(O/f'{ds}_DATA_MANIFEST.json');assert sha(q)==manifest['canonical_data_sha'];src[str(q)]=sha(q);src[str(W/'data'/ds/'normalized_expression.npz')]=sha(W/'data'/ds/'normalized_expression.npz');a=canonical(ds)
  for fi in range(2 if ds=='pbmc' else 6):
   d,f,ix=fold_data(a,ds,fi);assert not set(d['train']['rows'])&set(d['test']['rows']);pn=panel(ds,fi,a,ix);t=targets(a,pn);assert t['valid'][ix['train']].any();src[str(W/'panels'/ds/f'fold_{fi}.npz')]=sha(W/'panels'/ds/f'fold_{fi}.npz')
   for seed in SEEDS:
    for q0 in [source_warm(ds,fi,seed),source_mse(ds,fi,seed)]:src[str(q0)]=sha(q0)
    batches(ds,fi,seed,d)
   checks.append(dict(dataset=ds,fold=fi,roles={r:dict(rows=d[r]['rows'].tolist(),groups=np.unique(d[r]['donor']).tolist()) for r in d},gene_panel_size=len(pn['columns']),train_masked_target_entries=int(t['valid'][ix['train']].sum()),test_masked_target_entries=int(t['valid'][ix['test']].sum())))
 core.immutable(W/'sources.json',src);save(O/'PREFLIGHT.json',dict(status='PASS',checks=checks));print('PREFLIGHT_PASS',flush=True)

def pipeline(pilot=False):
 check_sources()
 for ds in (['ctrl'] if pilot else DATASETS):
  a=canonical(ds)
  for fi in (range(1) if pilot else range(2 if ds=='pbmc' else 6)):
   d,f,ix=fold_data(a,ds,fi);pn=panel(ds,fi,a,ix);t=targets(a,pn)
   develop(ds,fi,'shared','original_h',a,d,ix)
   for seed in ([42] if pilot else SEEDS):
    develop(ds,fi,seed,'clean_mse',a,d,ix)
    for arm in ARMS:train(ds,fi,seed,arm,a,d,ix,pn,t);develop(ds,fi,seed,arm,a,d,ix)
    done=len(list((W/'networks').glob('*/*/*/*/completion.json')));save(O/'EXPERIMENT_TRACKER.json',dict(status='TRAINING_AND_VALID_SELECTION',new_fits_done=done,new_fits_total=126,dataset=ds,fold=fi,seed=seed))
 if not pilot:
  files={str(q):sha(q) for home in ['networks','probes','panels'] for q in (W/home).rglob('*') if q.is_file()};core.immutable(O/'SELECTION_LOCK.json',dict(source_sha=sha(W/'sources.json'),protocol_sha=sha(O/'PROTOCOL.json'),hashes=files,roles_seen=['TRAIN','VALID'],historical_TEST_explored=True));print('SELECTIONS_LOCKED',flush=True)

def gene_metrics(model,a,pn,t,ii,dest):
 errors=[];targets0=[];preds=[];basepred=[];std=[];gene=[];rows=[]
 with torch.no_grad():
  for j in range(0,len(ii),256):
   jj=ii[j:j+256];rec=model(torch.from_numpy(a['masked_embedding'][jj].astype('float32')));out=model.predict_genes(rec).numpy();co=np.maximum(t['cols'][jj],0);valid=t['valid'][jj];pp=np.take_along_axis(out,co,axis=1);predraw=pp*pn['std'][co]+pn['mean'][co];yy=a['target_log_expression'][jj];preds.extend(predraw[valid]);targets0.extend(yy[valid]);basepred.extend(pn['mean'][co][valid]);std.extend(pn['std'][co][valid]);gene.extend(co[valid]);rows.extend(np.broadcast_to(a['rows'][jj,None],co.shape)[valid])
 pred=np.asarray(preds);yt=np.asarray(targets0);bp=np.asarray(basepred);ss=np.asarray(std);gi=np.asarray(gene,dtype='int64');count=np.bincount(gi,minlength=2048);err=np.abs(pred-yt);ge=np.bincount(gi,weights=err,minlength=2048)/np.maximum(count,1);gbe=np.bincount(gi,weights=np.abs(bp-yt),minlength=2048)/np.maximum(count,1);np.savez_compressed(dest/'masked_expression_predictions.npz',rows=np.asarray(rows),gene_names=pn['gene_names'][gi],prediction=pred,target=yt,train_gene_mean=bp,std=ss)
 return dict(entries=len(pred),genes_observed=int((count>0).sum()),MAE=float(err.mean()),normalized_MSE=float(np.mean(((pred-yt)/ss)**2)),gene_macro_MAE=float(ge[count>0].mean()),gene_mean_baseline_MAE=float(np.abs(bp-yt).mean()),gene_mean_baseline_normalized_MSE=float(np.mean(((bp-yt)/ss)**2)),gene_mean_baseline_gene_macro_MAE=float(gbe[count>0].mean()))

def evaluate():
 check_sources();lock=read(O/'SELECTION_LOCK.json')
 for p0,h in lock['hashes'].items():assert sha(p0)==h,p0
 records=[];pcs=[];state=[];geom=[];genes=[]
 for ds in DATASETS:
  a=canonical(ds)
  with np.load(ROOT/'work/sae_evaluation'/f'{ds}_targets.npz') as q:ifn={k:q[k].copy() for k in q.files}
  pos={int(r):i for i,r in enumerate(ifn['rows'])}
  for fi in range(2 if ds=='pbmc' else 6):
   d,f,ix=fold_data(a,ds,fi);pn=panel(ds,fi,a,ix);t=targets(a,pn);mu=d['train']['h'].astype('float64').mean(0);tr=d['train'];te=d['test'];rng=np.random.default_rng(np.random.SeedSequence([20261021,fi,99]));an=rng.choice(len(tr['h']),min(256,len(tr['h'])),replace=False);origtrain=ev.norm(tr['h'].astype('float64')-mu);origtest=ev.norm(te['h'].astype('float64')-mu);origix,_=ev.top(origtest,origtrain,15)
   for arm in ['original_h','clean_mse']+ARMS:
    for seed in (['shared'] if arm=='original_h' else SEEDS):
     model=restore(ds,fi,seed,arm);cleanfeat=features(model,a,ix,'clean');ref=ev.norm(cleanfeat['train']-mu);probe=W/'probes'/ds/f'fold_{fi}'/f'seed_{seed}'/arm/'readout.npz';common=dict(dataset=ds,fold=fi,group=str(te['donor'][0]),method=arm,seed=seed)
     if arm=='C_sae':
      dest=O/'predictions'/ds/f'fold_{fi}'/f'seed_{seed}'/arm;dest.mkdir(parents=True,exist_ok=True);genes.append(dict(**common,**gene_metrics(model,a,pn,t,ix['test'],dest)))
     for view in ['clean','masked']:
      x=cleanfeat if view=='clean' else features(model,a,{'test':ix['test']},view);xt=x['test'];scores=core.score_model(probe,xt);mt=core.metrics(te['y'],scores);kn,sv=ev.top(ev.norm(xt-mu),ref,15);labs=tr['y'][kn];vote=np.stack([(labs==c).sum(1) for c in range(8)],1).argmax(1);kf=f1_score(te['y'],vote,labels=np.arange(8),average=None,zero_division=0);pur=(labs==te['y'][:,None]).mean(1);purmacro=np.mean([pur[te['y']==c].mean() for c in np.unique(te['y'])]);ret=np.mean([len(set(i)&set(j))/15 for i,j in zip(kn,origix)])
      records.append(dict(**common,view=view,ridge_macro_f1=mt['macro_f1'],ridge_supported_macro_f1=mt['supported_macro_f1'],ridge_accuracy=mt['accuracy'],knn_macro_f1=float(kf.mean()),knn_supported_macro_f1=float(kf[np.unique(te['y'])].mean()),knn_purity=float(purmacro),teacher_neighbor_retention=float(ret)))
      for c,name in enumerate(core.CFG['classes']):pcs.append(dict(**common,view=view,cell_type=name,support=mt['support'][c],ridge_f1=mt['per_class_f1'][c],knn_f1=float(kf[c])))
      dest=O/'predictions'/ds/f'fold_{fi}'/f'seed_{seed}'/arm/view;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'test.npz',rows=te['rows'],labels=te['y'],ridge_scores=scores,knn_prediction=vote,neighbor_train_rows=tr['rows'][kn],embedding=xt)
      err=np.sum((xt-te['h'].astype('float64'))**2);den=np.sum((te['h'].astype('float64')-mu)**2);teacher=log_softmax(ev.mm(origtest,origtrain[an])/.1,axis=1);student=log_softmax(ev.mm(ev.norm(xt-mu),ref[an])/.1,axis=1);geom.append(dict(**common,view=view,reconstruction_r2=float(1-err/den),normalized_embedding_MSE=float(err/den),relative_displacement=float(np.linalg.norm(xt-te['h'])/np.linalg.norm(te['h'])),relation_KL=float(np.sum(np.exp(teacher)*(teacher-student),axis=1).mean())))
      ti=np.array([pos[int(r)] for r in tr['rows']]);qi=np.array([pos[int(r)] for r in te['rows']])
      for ct in ['B cells','CD14+ Monocytes']:
       c=core.CFG['classes'].index(ct);i=np.flatnonzero(tr['y']==c);j=np.flatnonzero(te['y']==c);meta=dict(**common,view=view,cell_type=ct,n_train=len(i),n_test=len(j))
       if len(i)<100 or len(j)<30:state.append(dict(**meta,status='SKIPPED_PREDECLARED_COUNTS'));continue
       aa=cleanfeat['train'][i];bb=xt[j];mx=aa.mean(0);sd=aa.std(0);sd[sd<1e-12]=1;aa=(aa-mx)/sd;bb=(bb-mx)/sd;cov=ifn['covariates'][ti[i]];ctest=ifn['covariates'][qi[j]];cm=cov.mean(0);cs=cov.std(0);cs[cs<1e-12]=1;cov=(cov-cm)/cs;ctest=(ctest-cm)/cs;yt=ifn['score'][ti[i]];yq=ifn['score'][qi[j]];ym=yt.mean();ys=max(yt.std(),1e-12)
       for variant,v1,v2 in [('embedding',aa,bb),('covariates',cov,ctest),('embedding_plus_covariates',np.column_stack([aa,cov]),np.column_stack([bb,ctest]))]:
        pp=ev.fit(v1,(yt-ym)/ys,v2).ravel()*ys+ym;state.append(dict(**meta,status='COMPLETE',variant=variant,**ev.regmetric(yq,pp)));np.savez_compressed(dest/f'IFN_{c}_{variant}.npz',rows=te['rows'][j],target=yq,prediction=pp)
     print('EVAL_COMPLETE',ds,fi,seed,arm,flush=True)
   save(O/'EXPERIMENT_TRACKER.json',dict(status='EVALUATING',dataset=ds,fold=fi,readouts_done=len(records),readouts_total=364))
 for name,rows in [('all_metrics',records),('per_class_metrics',pcs),('IFN_all',state),('geometry_all',geom),('expression_all',genes)]:save(O/f'{name}.json',rows);pd.DataFrame(rows).to_csv(O/f'{name}.csv',index=False)
 assert len(records)==364;save(O/'EXPERIMENT_TRACKER.json',dict(status='EVALUATED_PENDING_AUDIT',new_fits_done=126,new_fits_total=126,test_readouts=364));print('ALL_EVALUATIONS_COMPLETE',flush=True)

if __name__=='__main__':
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 with threadpool_limits(limits=4):
  stage=sys.argv[1]
  if stage=='preflight':preflight()
  elif stage=='pilot':pipeline(True)
  elif stage=='develop':pipeline()
  elif stage=='evaluate':evaluate()
