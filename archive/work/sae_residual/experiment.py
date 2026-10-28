"""Frozen-SAE residual experiment. TRAIN/VALID selection precedes all TEST scoring.

Sparse gate cost O(B*K*512); low-rank O(B*K*r+B*r*512).
Training memory O(B*K*512), final Ridge O(n*512^2+512^3).
"""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, math, os, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

W=Path(__file__).resolve().parent
O=W.parents[1]/'outputs/sae_residual'
R=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
SEEDS=[42,43,44]
ARMS=['gate','rotated_gate','lowrank','mlp','gate_supcon','rotated_gate_supcon','mlp_supcon']
BETAS=[0.,.1,1.]
ALPHAS=[10.**v for v in range(-4,7)]
CFG=dict(k=32,sae_seeds=SEEDS,arms=ARMS,betas=BETAS,ridge_alphas=ALPHAS,
         gate_bound=.5,lowrank_rank=16,mlp_width=40,batch_size=256,
         baseline_epochs=100,adapter_epochs=60,patience=10,baseline_lr=.01,
         adapter_lr=.001,head_weight_decays=[0.,.001,.01],supcon_lambda=.1,
         supcon_temperature=.1,knn_k=15,device='cpu',torch_threads=4,
         selection='VALID fresh Ridge macroF1, identity wins ties; lower beta then alpha ties',
         checkpoint='training head VALID macroF1; earliest tie; epoch0 eligible',
         sampler='same seeded class-uniform donor-balanced replacement sampling for every arm',
         supcon='positives same type different TRAIN donor; exclude same-type same-donor and duplicate cell identities from denominator; skip anchors without cross-donor positive',
         frozen='scGPT, SAE encoder/decoder; full original raw h preserved in skip connection',
         preprocess='raw h and signed raw TopK z; fixed TRAIN h mean/std at training head; fitted TRAIN h-prime mean/std at fresh Ridge/kNN',
         baseline='original fresh Ridge plus validation-selected CE head; primary uses fresh Ridge for every arm',
         statistics='six donor differences after averaging three SAE/training seeds; seeds not independent donors',
         primary='gate versus original and rotated_gate, donor mean fresh-Ridge fixed8-label macroF1',
         parameter_matching='gate/rotated_gate each2048; lowrank/MLP each40960; gate versus MLP is not parameter-matched',
         continuation=dict(mean_gain_pp=1.,positive_donors=4,positive_seed_means=2,pbmc_noninferiority_pp=-.5),
         scope='historically explored Kang/PBMC; development, not new blind validation; K64 optional future sensitivity not executed here')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def immutable(p,value):
 p=Path(p)
 if p.exists():assert json.loads(p.read_text())==value,f'Immutable artifact changed: {p}'
 else:save(p,value)
def bindings():
 return {'sources_sha':sha(W/'frozen_sources.json'),'protocol_sha':sha(O/'PROTOCOL.json')}
def check_sources():
 for p,h in json.loads((W/'frozen_sources.json').read_text()).items():assert sha(p)==h,f'Source changed: {p}'
 assert json.loads((O/'PROTOCOL.json').read_text())==CFG
def check_selection(p):
 a=json.loads(Path(p).read_text());assert a['bindings']==bindings()
 for t in a['trials']:
  assert sha(t['checkpoint'])==t['checkpoint_sha']
  assert sha(t['metadata'])==t['metadata_sha']
 assert sha(Path(p).parents[2]/'baseline/selected_head.pt')==a['baseline_head_sha']
 return a
def metrics(y,p):
 support=np.bincount(y,minlength=8);f=f1_score(y,p,labels=np.arange(8),average=None,zero_division=0)
 return dict(macro_f1=float(f.mean()),supported_macro_f1=float(f[support>0].mean()),accuracy=float(accuracy_score(y,p)),per_class_f1=f.tolist(),support=support.tolist(),confusion_matrix=confusion_matrix(y,p,labels=np.arange(8)).tolist())
def load_data(fi,seed):
 proto=json.loads((R/'protocol.json').read_text());fold=proto['folds'][fi]['fold']
 a=np.load(R/'kang_embeddings.npz');b=np.load(R/'pbmc_embeddings.npz');h=a['embedding'].astype('float32');y=a['labels'].astype('int64');don=a['donors'];rows=a['rows']
 ix={s:np.flatnonzero((don==fold[s]) if s!='train' else ~np.isin(don,[fold['valid'],fold['test']])) for s in ['train','valid','test']}
 src=R/f"test_{fold['test']}"/f'seed_{seed}'/'k_32';z=np.load(src/'sae_outputs.npz');state=torch.load(src/'sae.pt',map_location='cpu',weights_only=False)
 _,donor=np.unique(don,return_inverse=True);out={}
 for role in ['train','valid','test','pbmc']:
  ids=ix.get(role);hh=h[ids] if ids is not None else b['embedding'].astype('float32');yy=y[ids] if ids is not None else b['labels'].astype('int64')
  rr=rows[ids] if ids is not None else np.arange(len(hh))
  assert np.array_equal(z[role+'_rows'],rr) and np.array_equal(z[role+'_labels'],yy)
  out[role]={'h':torch.from_numpy(hh.copy()),'y':torch.from_numpy(yy.copy()),'idx':torch.from_numpy(z[role+'_indices'].astype('int64')),'val':torch.from_numpy(z[role+'_values'].astype('float32')),'row':torch.from_numpy(rr.astype('int64')),'donor':torch.from_numpy(donor[ids].astype('int64') if ids is not None else np.full(len(hh),-1,np.int64))}
 assert sum(len(ix[s]) for s in ix)==len(h) and len(np.unique(np.concatenate(list(ix.values()))))==len(h)
 return out,state['decoder.weight'].float(),fold,src,proto

class Residual(nn.Module):
 def __init__(self,arm,D,mu,scale,seed):
  super().__init__();torch.manual_seed(seed);self.arm=arm;self.register_buffer('mu',mu);self.register_buffer('scale',scale)
  if arm in ['gate','rotated_gate']:
   DD=D.clone()
   if arm=='rotated_gate':
    # Fixed orthogonal rotation preserves D.T@D and all column norms.
    q,r=torch.linalg.qr(torch.randn(512,512));q=q*torch.sign(torch.diag(r))[None,:];DD=q@DD
   self.register_buffer('DT',DD.T.contiguous());self.theta=nn.Parameter(torch.zeros(2048))
  elif arm=='lowrank':
   self.V=nn.Parameter(torch.randn(2048,16)/math.sqrt(2048));self.U=nn.Parameter(torch.zeros(16,512))
  elif arm=='mlp':
   self.V=nn.Parameter(torch.randn(512,40)/math.sqrt(512));self.U=nn.Parameter(torch.zeros(40,512))
  elif arm!='identity':raise ValueError(arm)
 def forward(self,b):
  h=b['h']
  if self.arm=='identity':return h,torch.zeros_like(h)
  if self.arm in ['gate','rotated_gate']:
   weight=CFG['gate_bound']*torch.tanh(self.theta[b['idx']])*b['val']
   delta=torch.einsum('bk,bkd->bd',weight,self.DT[b['idx']])
  elif self.arm=='lowrank':delta=(self.V[b['idx']]*b['val'][...,None]).sum(1)@self.U
  else:
   # Dense baseline receives the same centered raw units as the SAE encoder.
   delta=F.relu((h-self.mu)@self.V)@self.U
  return h+delta,delta

class Model(nn.Module):
 def __init__(self,residual):
  super().__init__();self.residual=residual;self.head=nn.Linear(512,8)
 def forward(self,b):
  h,d=self.residual(b);s=(h-self.residual.mu)/self.residual.scale
  return self.head(s),s,d

def get_stats(data):
 h=data['train']['h'];mu=h.double().mean(0).float();sd=h.double().std(0,unbiased=False).float();sd[sd<1e-8]=1.
 return mu,sd
def batches(b,order,size=256):
 for j in range(0,len(order),size):yield {k:v[order[j:j+size]] for k,v in b.items()}
def order_for(b,seed,epoch):
 rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,20261019]));y=b['y'].numpy();d=b['donor'].numpy();pools={c:{dd:np.flatnonzero((y==c)&(d==dd)) for dd in np.unique(d[y==c])} for c in range(8)};order=[]
 for _ in range(math.ceil(len(y)/CFG['batch_size'])):
  one=[]
  for c in range(8):
   ds=list(pools[c]);ds=np.resize(rng.permutation(ds),CFG['batch_size']//8);rng.shuffle(ds)
   one.extend(int(rng.choice(pools[c][dd])) for dd in ds)
  rng.shuffle(one);order.extend(one)
 return np.asarray(order)
def supcon(s,b):
 v=F.normalize(s,dim=1);logits=v@v.T/CFG['supcon_temperature'];y=b['y'];d=b['donor'];r=b['row'];distinct=r[:,None]!=r[None,:];same=y[:,None]==y[None,:]
 pos=same&(d[:,None]!=d[None,:])&distinct;den=((~same)|pos)&distinct;keep=pos.sum(1)>0
 if not keep.any():return s.sum()*0.,0.
 a=logits[keep];mask=den[keep];p=pos[keep];norm=torch.logsumexp(a.masked_fill(~mask,-torch.inf),dim=1)
 loss=-((a-norm[:,None])*p).sum(1)/p.sum(1)
 return loss.mean(),float(keep.float().mean())
def head_valid(model,b):
 model.eval();p=[]
 with torch.no_grad():
  for bb in batches(b,np.arange(len(b['y']))):p.append(model(bb)[0].argmax(1).numpy())
 return metrics(b['y'].numpy(),np.concatenate(p))['macro_f1']

def train_model(data,D,arm,seed,beta,wd,base_head=None,max_epochs=None,force_epochs=False):
 mu,sd=get_stats(data);pure=arm.replace('_supcon','');res=Residual(pure,D,mu,sd,seed);model=Model(res)
 if base_head is not None:model.head.load_state_dict(base_head)
 params=[{'params':model.head.parameters(),'weight_decay':wd}]
 rp=list(res.parameters())
 if rp:params.append({'params':rp,'weight_decay':0.})
 opt=torch.optim.Adam(params,lr=CFG['baseline_lr'] if arm=='identity' else CFG['adapter_lr'])
 epochs=max_epochs if max_epochs is not None else CFG['baseline_epochs' if arm=='identity' else 'adapter_epochs'];best=-1.;stale=0;hist=[];start=time.time();beststate=None;grad_first={};grad_later={};maxres=0.
 for epoch in range(epochs+1):
  score=head_valid(model,data['valid']);hist.append({'epoch':epoch,'valid_head_macro_f1':score})
  if score>best+1e-12:best=score;bestepoch=epoch;beststate=copy.deepcopy(model.state_dict());stale=0
  else:stale+=1
  if epoch==epochs or (not force_epochs and stale>=CFG['patience']):break
  model.train();total=0.;ce_total=0.;pen_total=0.;con_total=0.;n=0;positive=[]
  for j,b in enumerate(batches(data['train'],order_for(data['train'],seed,epoch))):
   opt.zero_grad(set_to_none=True);logits,s,delta=model(b);ce=F.cross_entropy(logits,b['y']);pen=(delta.square().sum(1)/(b['h'].square().sum(1)+1e-12)).mean();con,fraction=supcon(s,b) if arm.endswith('_supcon') else (ce*0.,0.)
   loss=ce+beta*pen+CFG['supcon_lambda']*con;assert torch.isfinite(loss);loss.backward()
   grads={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None};assert all(np.isfinite(v) for v in grads.values())
   if epoch==0 and j==0:grad_first=grads
   if epoch==0 and j==1:grad_later=grads
   torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();nnn=len(b['y']);total+=float(loss)*nnn;ce_total+=float(ce)*nnn;pen_total+=float(pen)*nnn;con_total+=float(con)*nnn;n+=nnn;positive.append(fraction);maxres=max(maxres,float(pen))
  hist[-1].update(next_train_loss=total/n,next_train_CE=ce_total/n,next_train_residual_ratio_squared=pen_total/n,next_train_SupCon=con_total/n,positive_anchor_fraction=float(np.mean(positive)))
 model.load_state_dict(beststate);model.eval();changed={k:float(v.norm()) for k,v in model.residual.named_parameters()}
 return model,dict(arm=arm,seed=seed,beta=beta,head_weight_decay=wd,best_epoch=bestepoch,valid_head_macro_f1=best,last_epoch=epoch,seconds=time.time()-start,history=hist,first_gradients=grad_first,second_step_gradients=grad_later,parameter_norms=changed,max_train_residual_ratio_squared=maxres,adapter_parameters=sum(p.numel() for p in res.parameters()),head_parameters=sum(p.numel() for p in model.head.parameters()))

def embeddings(model,b):
 out=[]
 with torch.no_grad():
  for bb in batches(b,np.arange(len(b['y']))):out.append(model.residual(bb)[0].numpy())
 return np.concatenate(out)

class Ridge:
 def __init__(self,x,y):
  self.mu=x.astype('float64').mean(0);self.sd=x.astype('float64').std(0);self.sd[self.sd<1e-12]=1.;s=(x-self.mu)/self.sd
  count=np.bincount(y,minlength=8);assert (count>0).all();w=len(y)/(8*count[y]);self.xmean=np.average(s,axis=0,weights=w);target=2*np.eye(8)[y]-1.;self.ymean=np.average(target,axis=0,weights=w);a=(s-self.xmean)*np.sqrt(w[:,None]);b=(target-self.ymean)*np.sqrt(w[:,None]);gram=a.T@a;self.e,self.v=eigh((gram+gram.T)/2,driver='evd');self.e=np.maximum(self.e,0);self.rhs=self.v.T@(a.T@b);self.a=a;self.b=b
 def fit(self,alpha):
  self.alpha=alpha;self.coef=self.v@(self.rhs/(self.e[:,None]+alpha));self.intercept=self.ymean-self.xmean@self.coef
 def scores(self,x):return ((x-self.mu)/self.sd)@self.coef+self.intercept
 def residual(self):
  g=self.a.T@(self.a@self.coef-self.b)+self.alpha*self.coef
  return float(np.abs(g).max()/max(1.,float(np.abs(self.a.T@self.b).max())))
 def arrays(self):return dict(mean=self.mu,scale=self.sd,coef=self.coef,intercept=self.intercept,alpha=self.alpha)
def fit_ridge(train,y,valid,vy):
 ridge=Ridge(train,y);trials=[];best=-1.
 for a in ALPHAS:
  ridge.fit(a);score=metrics(vy,ridge.scores(valid).argmax(1))['macro_f1'];err=ridge.residual();assert err<1e-7
  trials.append(dict(alpha=a,valid_macro_f1=score,normal_equation_relative_error=err))
  if score>best+1e-12:best=score;alpha=a
 ridge.fit(alpha)
 return ridge,dict(valid_macro_f1=best,alpha=alpha,trials=trials)

def preflight():
 O.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);files=[R/n for n in ['protocol.json','data_manifest.json','metadata_inventory.json','kang_embeddings.npz','pbmc_embeddings.npz','sae.py','run.py','core.py']];checks=[]
 old=json.loads((R/'protocol.json').read_text())
 for p,h in old['source_hashes'].items():assert sha(p)==h,p
 for fi in range(6):
  for seed in SEEDS:
   data,D,fold,src,proto=load_data(fi,seed);files.extend([src/'sae.pt',src/'sae_outputs.npz',src/'sae_result.json']);state=torch.load(src/'sae.pt',map_location='cpu',weights_only=False);module_spec=importlib.util.spec_from_file_location('historical_sae',R/'sae.py');mod=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(mod);sae=mod.TopKSAE(state['mean'],32,seed);sae.load_state_dict(state)
   for role,b in data.items():
    assert torch.isfinite(b['h']).all() and torch.isfinite(b['val']).all();assert b['idx'].min()>=0 and b['idx'].max()<2048
    take=np.unique(np.linspace(0,len(b['h'])-1,min(20,len(b['h'])),dtype=int));
    with torch.no_grad():actual,_,_=sae.encode(b['h'][take]);cache=torch.zeros_like(actual).scatter(1,b['idx'][take],b['val'][take]);error=float((actual-cache).abs().max())
    assert error<3e-6,(fi,seed,role,error)
    checks.append(dict(fold=fi,test_donor=fold['test'],seed=seed,role=role,n=len(b['y']),cache_max_error=error))
 files+=[Path(__file__).resolve()];sources={str(p):sha(p) for p in sorted(set(files))};immutable(W/'frozen_sources.json',sources);immutable(O/'PROTOCOL.json',CFG);save(O/'preflight.json',dict(status='PASS',checks=checks,source_files=len(sources),input_dim=512,dictionary_dim=2048,k=32));print('PREFLIGHT_PASS',len(checks),flush=True)

def baseline(fi,data,D):
 out=W/f'fold_{fi}'/'baseline';out.mkdir(parents=True,exist_ok=True)
 if not (out/'completion.json').exists():
  candidates=[]
  for wd in CFG['head_weight_decays']:
   m,r=train_model(data,D,'identity',1000+fi,0.,wd);cp=out/f'head_wd_{wd}.pt';torch.save(m.state_dict(),cp);save(out/f'head_wd_{wd}.json',r);candidates.append((r['valid_head_macro_f1'],wd,cp,r))
   score,wd,cp,r=max(candidates,key=lambda a:(a[0],-a[1]));state=torch.load(cp,map_location='cpu',weights_only=False);torch.save(state,out/'selected_head.pt');save(out/'completion.json',dict(selected_weight_decay=wd,valid_macro_f1=score,selected_head=sha(out/'selected_head.pt'),bindings=bindings(),trial_hashes={str(p):sha(p) for p in out.glob('head_wd_*')}))
 info=json.loads((out/'completion.json').read_text());assert info['bindings']==bindings();assert sha(out/'selected_head.pt')==info['selected_head']
 for p,h in info['trial_hashes'].items():assert sha(p)==h
 return torch.load(out/'selected_head.pt',map_location='cpu',weights_only=False),info

def develop(fi,seed):
 out=W/f'fold_{fi}'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True);data,D,fold,_,_=load_data(fi,seed);base,bi=baseline(fi,data,D);mu,sd=get_stats(data)
 identity=Model(Residual('identity',D,mu,sd,seed));identity.load_state_dict(base);_,ir=fit_ridge(data['train']['h'].numpy(),data['train']['y'].numpy(),data['valid']['h'].numpy(),data['valid']['y'].numpy());save(out/'identity_ridge.json',ir)
 for arm in ARMS:
  ao=out/arm;ao.mkdir(exist_ok=True)
  if (ao/'selection.json').exists():check_selection(ao/'selection.json');continue
  trials=[];bestscore=ir['valid_macro_f1'];chosen=None
  for beta in BETAS:
   cp=ao/f'beta_{beta}.pt';rp=ao/f'beta_{beta}.json'
   if cp.exists() and rp.exists():
    r=json.loads(rp.read_text());assert r['bindings']==bindings() and r['checkpoint_sha']==sha(cp);model=Model(Residual(arm.replace('_supcon',''),D,mu,sd,seed));model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False))
   else:
    model,r=train_model(data,D,arm,seed,beta,bi['selected_weight_decay'],base_head={k.split('head.',1)[1]:v for k,v in base.items() if k.startswith('head.')});torch.save(model.state_dict(),cp)
   tr=embeddings(model,data['train']);va=embeddings(model,data['valid']);_,probe=fit_ridge(tr,data['train']['y'].numpy(),va,data['valid']['y'].numpy());r.update(fresh_ridge=probe,bindings=bindings(),checkpoint_sha=sha(cp));save(rp,r);trials.append(dict(beta=beta,valid_macro_f1=probe['valid_macro_f1'],head_valid_macro_f1=r['valid_head_macro_f1'],checkpoint=str(cp),checkpoint_sha=sha(cp),metadata=str(rp),metadata_sha=sha(rp),best_epoch=r['best_epoch'],seconds=r['seconds']))
   if probe['valid_macro_f1']>bestscore+1e-12:bestscore=probe['valid_macro_f1'];chosen=beta
   print('TRIAL',fi,seed,arm,beta,'epoch',r['best_epoch'],'VALID',round(probe['valid_macro_f1'],5),'sec',round(r['seconds'],1),flush=True)
  selection=dict(fold=fi,test_donor=fold['test'],seed=seed,arm=arm,selected_beta=chosen,identity_selected=chosen is None,valid_macro_f1=bestscore,identity_valid_macro_f1=ir['valid_macro_f1'],trials=trials,bindings=bindings(),baseline_head_sha=sha(W/f'fold_{fi}'/'baseline/selected_head.pt'))
  save(ao/'selection.json',selection);print('SELECT',fi,seed,arm,'identity' if chosen is None else chosen,flush=True)

def knn_predictions(train,y,x):
 mu=train.astype('float64').mean(0);sd=train.astype('float64').std(0);sd[sd<1e-12]=1.;a=(train-mu)/sd;b=(x-mu)/sd;a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12);pred=[]
 for lo in range(0,len(b),256):
  dist=cdist(b[lo:lo+256],a,'sqeuclidean');ind=np.argsort(dist,axis=1,kind='stable')[:,:CFG['knn_k']];counts=np.stack([(y[ind]==c).sum(1) for c in range(8)],1);pred.extend(counts.argmax(1))
 return np.asarray(pred,np.int64)

def evaluate_all():
 selections=[W/f'fold_{fi}'/f'seed_{seed}'/arm/'selection.json' for fi in range(6) for seed in SEEDS for arm in ARMS];assert all(p.exists() for p in selections),'No TEST scoring until all development selections are finished'
 for p in selections:check_selection(p)
 lock={str(p):sha(p) for p in selections};immutable(O/'selection_lock.json',lock)
 for fi in range(6):
  for seed in SEEDS:
   data,D,fold,_,_=load_data(fi,seed);mu,sd=get_stats(data);base=torch.load(W/f'fold_{fi}'/'baseline/selected_head.pt',map_location='cpu',weights_only=False)
   for arm in ['identity']+ARMS:
    dest=O/f'fold_{fi}'/f'seed_{seed}'/arm;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():
     old=json.loads((dest/'result.json').read_text());assert old['bindings']==bindings() and old['selection_lock_sha']==sha(O/'selection_lock.json')
     for p,h in old['output_hashes'].items():assert sha(p)==h
     continue
    sel=None if arm=='identity' else json.loads((W/f'fold_{fi}'/f'seed_{seed}'/arm/'selection.json').read_text());chosen=None if sel is None else sel['selected_beta'];pure='identity' if chosen is None else arm.replace('_supcon','');model=Model(Residual(pure,D,mu,sd,seed));state=base if chosen is None else torch.load(W/f'fold_{fi}'/f'seed_{seed}'/arm/f'beta_{chosen}.pt',map_location='cpu',weights_only=False);model.load_state_dict(state);model.eval();e={role:embeddings(model,b) for role,b in data.items()};ridge,pr=fit_ridge(e['train'],data['train']['y'].numpy(),e['valid'],data['valid']['y'].numpy());np.savez_compressed(dest/'ridge.npz',**ridge.arrays());arrays={};result=dict(fold=fi,test_donor=fold['test'],valid_donor=fold['valid'],seed=seed,arm=arm,selected_beta=chosen,identity_selected=pure=='identity',probe=pr)
    for role in ['valid','test','pbmc']:
     y=data[role]['y'].numpy();scores=ridge.scores(e[role]);p=scores.argmax(1);knn=knn_predictions(e['train'],data['train']['y'].numpy(),e[role]);hp=[]
     with torch.no_grad():
      for b in batches(data[role],np.arange(len(y))):hp.append(model(b)[0].argmax(1).numpy())
     head=np.concatenate(hp);h=data[role]['h'].numpy();delta=e[role]-h;ratio=np.linalg.norm(delta,axis=1)/np.maximum(np.linalg.norm(h,axis=1),1e-12);cos=np.sum(h*e[role],1)/np.maximum(np.linalg.norm(h,axis=1)*np.linalg.norm(e[role],axis=1),1e-12)
     result[role]=dict(ridge=metrics(y,p),knn=metrics(y,knn),trained_head=metrics(y,head),residual_relative_norm_mean=float(ratio.mean()),residual_relative_norm_max=float(ratio.max()),original_cosine_mean=float(cos.mean()))
     arrays.update({role+'_label':y,role+'_row':data[role]['row'].numpy(),role+'_scores':scores,role+'_ridge_prediction':p,role+'_knn_prediction':knn,role+'_trained_head_prediction':head})
    if pure in ['gate','rotated_gate']:
     g=CFG['gate_bound']*torch.tanh(model.residual.theta).detach().numpy();np.savez_compressed(dest/'gate.npz',g=g)
    np.savez_compressed(dest/'predictions.npz',**arrays);result.update(bindings=bindings(),selection_lock_sha=sha(O/'selection_lock.json'),output_hashes={str(p):sha(p) for p in dest.glob('*.npz')});save(dest/'result.json',result);print('EVAL',fi,seed,arm,'TEST',round(result['test']['ridge']['macro_f1'],5),flush=True)
 for p,h in lock.items():assert sha(p)==h
 save(O/'completion.json',dict(status='complete',runs=6*3*8,development_fits=6*3*7*3,baseline_head_fits=6*3,TEST_selection='none; all decisions locked first',scope=CFG['scope']))

def benchmark():
 data,D,fold,_,_=load_data(0,42);records=[];start=time.time()
 for arm in ['identity']+ARMS:
  model,r=train_model(data,D,arm,42,.1,.001,max_epochs=2,force_epochs=True);records.append(r)
  mu,sd=get_stats(data);test=Residual(arm.replace('_supcon',''),D,mu,sd,42)
  with torch.no_grad():hh,delta=test({k:v[:9] for k,v in data['train'].items()})
  assert torch.equal(hh,data['train']['h'][:9]) and torch.count_nonzero(delta)==0
  rp=[v for k,v in r['second_step_gradients'].items() if k.startswith('residual.')];assert not rp or all(v>0 for v in rp),(arm,rp)
  print('BENCH',arm,round(r['seconds'],2),r['adapter_parameters'],flush=True)
 save(O/'sanity_benchmark.json',dict(status='PASS',fold=fold,viewed_roles=['train','valid'],records=records,seconds=time.time()-start,checks=['identity initialization exact','finite gradients and updates','zero-output low-rank second-step gradient flow','same sampler for all arms']))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','benchmark','develop','evaluate']);ap.add_argument('--fold',type=int);ap.add_argument('--seed',type=int);a=ap.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);O.mkdir(parents=True,exist_ok=True)
 if a.stage=='preflight':preflight()
 elif a.stage=='benchmark':check_sources();benchmark()
 elif a.stage=='develop':
  check_sources()
  for fi in ([a.fold] if a.fold is not None else range(6)):
   for seed in ([a.seed] if a.seed is not None else SEEDS):develop(fi,seed)
  check_sources()
 elif a.stage=='evaluate':check_sources();evaluate_all();check_sources()
if __name__=='__main__':main()
