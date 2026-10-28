"""Posthoc mechanism diagnosis on TRAIN/VALID only, never TEST/PBMC scoring.
Original experiment and its sealed outputs remain unchanged.
"""
from pathlib import Path
import argparse,copy,hashlib,json,math,sys,time,warnings
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_residual_diagnosis'
OLD=W.parent/'sae_residual';sys.path.insert(0,str(OLD));import experiment as m
CFG={'seed':42,'folds':list(range(6)),'scored_roles':['train','valid'],'never_scored':['test','pbmc'],
     'trajectory_arms':['gate','rotated_gate'],'head_modes':['joint','frozen'],'epochs':40,'ridge_probe_every_epochs':2,
     'beta':.1,'lr':.001,'gate_bound':.5,'clip':5.,'same_sampler_as_original':True,
     'checkpoint_probe':'epoch0, every2 epochs, and original beta0.1 best/stopping epochs',
     'head_freezing':'same selected original CE head, only parameter trainability differs',
     'branch_latents':['learned_SAE','random_TopK'],'branch_scales':[.01,.1,1.],
     'branch_alphas':m.ALPHAS,'branch_selection':'VALID MacroF1, identity wins ties; then lower scale and alpha',
     'branch_scaling':'TRAIN standardize h/z once, then scale z; no second standardization removing the scale',
     'branch_weights':'same global class-balanced cell weights as original fresh Ridge',
     'random_latent':'2048 independent unit-norm random directions, TRAIN h mean, raw TopK32, bias0; same nominal dimension/K but not activation-frequency/correlation matched',
     'scope':'posthoc exploratory diagnosis after prior TEST results; same historically explored validation donors; no new generalization estimate or proof of uselessness; only seed42 follow-up'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):m.save(p,v)
def read(p):return json.loads(Path(p).read_text())
def binding():return {'protocol_sha':sha(O/'PROTOCOL.json'),'source_sha':sha(__file__),'old_sources_sha':sha(OLD/'frozen_sources.json')}
def check():
 m.check_sources();assert read(O/'PROTOCOL.json')==CFG
 for p,h in read(W/'frozen_sources.json').items():assert sha(p)==h,p
def data_for(fi):
 proto=read(m.R/'protocol.json');fold=proto['folds'][fi]['fold'];a=np.load(m.R/'kang_embeddings.npz');don=a['donors'];_,dd=np.unique(don,return_inverse=True);src=m.R/f"test_{fold['test']}"/'seed_42/k_32';cache=np.load(src/'sae_outputs.npz');state=torch.load(src/'sae.pt',map_location='cpu',weights_only=False);d={}
 for role in ['train','valid']:
  ix=np.flatnonzero(~np.isin(don,[fold['valid'],fold['test']])) if role=='train' else np.flatnonzero(don==fold['valid']);rows=a['rows'][ix];y=a['labels'][ix];assert np.array_equal(rows,cache[role+'_rows']) and np.array_equal(y,cache[role+'_labels'])
  d[role]={'h':torch.from_numpy(a['embedding'][ix].astype('float32')),'y':torch.from_numpy(y.astype('int64')),'idx':torch.from_numpy(cache[role+'_indices'].astype('int64')),'val':torch.from_numpy(cache[role+'_values'].astype('float32')),'row':torch.from_numpy(rows.astype('int64')),'donor':torch.from_numpy(dd[ix].astype('int64'))}
 assert not set(d['train']['row'].tolist())&set(d['valid']['row'].tolist());assert torch.allclose(d['train']['h'].double().mean(0).float(),state['mean'],atol=0,rtol=0)
 base=torch.load(OLD/f'fold_{fi}/baseline/selected_head.pt',map_location='cpu',weights_only=False);bi=read(OLD/f'fold_{fi}/baseline/completion.json')
 return d,state['decoder.weight'].float(),fold,base,bi
def scale_diag(model,b):
 with torch.no_grad():
  h,delta=model.residual(b);orig=b['h'];centered=orig-model.residual.mu;white_delta=delta/model.residual.scale;white_h=centered/model.residual.scale;g=.5*torch.tanh(model.residual.theta)
  return {'relative_raw_norm':float((delta.norm(dim=1)/orig.norm(dim=1).clamp_min(1e-12)).mean()),'relative_centered_norm':float((delta.norm(dim=1)/centered.norm(dim=1).clamp_min(1e-12)).mean()),'relative_whitened_norm':float((white_delta.norm(dim=1)/white_h.norm(dim=1).clamp_min(1e-12)).mean()),'max_abs_gate':float(g.abs().max()),'gate_RMS':float(g.square().mean().sqrt()),'saturated_fraction':float((g.abs()>.45).float().mean())}
def loss_gradients(model,b):
 logits,_,delta=model(b);ce=F.cross_entropy(logits,b['y']);pen=(delta.square().sum(1)/b['h'].square().sum(1).clamp_min(1e-12)).mean();theta=model.residual.theta
 g1=torch.autograd.grad(ce,theta,retain_graph=True)[0];g2=torch.autograd.grad(CFG['beta']*pen,theta)[0];n1=float(g1.norm());n2=float(g2.norm())
 return {'CE':float(ce),'weighted_penalty':float(CFG['beta']*pen),'CE_theta_gradient_norm':n1,'penalty_theta_gradient_norm':n2,'penalty_to_CE_gradient_ratio':n2/max(n1,1e-30),'gradient_cosine':float(F.cosine_similarity(g1[None],g2[None]).item()) if n1>0 and n2>0 else None}
def valid_head_metrics(model,b):
 logits=[]
 with torch.no_grad():
  for bb in m.batches(b,np.arange(len(b['y']))):logits.append(model(bb)[0])
 score=torch.cat(logits);return m.metrics(b['y'].numpy(),score.argmax(1).numpy()),float(F.cross_entropy(score,b['y'])),score.numpy()
def trajectory(fi,arm,mode):
 out=W/'trajectories'/f'fold_{fi}'/arm/mode;out.mkdir(parents=True,exist_ok=True);done=out/'completion.json'
 if done.exists():
  r=read(done);assert r['binding']==binding()
  for p,h in r['hashes'].items():assert sha(p)==h
  return
 d,D,fold,base,bi=data_for(fi);mu,sd=m.get_stats(d);model=m.Model(m.Residual(arm,D,mu,sd,42));model.head.load_state_dict({k.split('head.',1)[1]:v for k,v in base.items() if k.startswith('head.')});fixedhead=copy.deepcopy(model.head)
 if mode=='frozen':
  for p in model.head.parameters():p.requires_grad_(False)
 groups=[{'params':model.residual.parameters(),'weight_decay':0.}]
 if mode=='joint':groups.insert(0,{'params':model.head.parameters(),'weight_decay':bi['selected_weight_decay']})
 opt=torch.optim.Adam(groups,lr=CFG['lr']);old=read(OLD/f'fold_{fi}/seed_42'/arm/'beta_0.1.json');extra={old['best_epoch'],old['last_epoch']};old_hist={h['epoch']:h for h in old['history']};baseline,info=m.fit_ridge(d['train']['h'].numpy(),d['train']['y'].numpy(),d['valid']['h'].numpy(),d['valid']['y'].numpy());records=[];start=time.time();prefix_ok=True
 grad_batch={k:v[m.order_for(d['train'],42,0)[:256]] for k,v in d['train'].items()}
 for epoch in range(CFG['epochs']+1):
  model.eval();hm,hce,heads=valid_head_metrics(model,d['valid']);rec={'fold':fi,'valid_donor':fold['valid'],'arm':arm,'head_mode':mode,'epoch':epoch,'head_valid_F1':hm['macro_f1'],'head_valid_CE':hce,'identity_Ridge_valid_F1':info['valid_macro_f1']}
  if mode=='joint' and epoch in old_hist:
   assert abs(hm['macro_f1']-old_hist[epoch]['valid_head_macro_f1'])<1e-12,(fi,arm,epoch,'original prefix mismatch')
  if mode=='joint' and epoch==old['best_epoch']:
   cp=torch.load(OLD/f'fold_{fi}/seed_42'/arm/'beta_0.1.pt',map_location='cpu',weights_only=False);assert all(torch.equal(v,cp[k]) for k,v in model.state_dict().items()),'Original selected checkpoint replay differs'
  if epoch%CFG['ridge_probe_every_epochs']==0 or epoch in extra:
   tr=m.embeddings(model,d['train']);va=m.embeddings(model,d['valid']);ridge,pr=m.fit_ridge(tr,d['train']['y'].numpy(),va,d['valid']['y'].numpy());scores=ridge.scores(va);rec.update(fresh_Ridge_valid_F1=pr['valid_macro_f1'],fresh_Ridge_alpha=pr['alpha'],fixed_original_Ridge_F1=m.metrics(d['valid']['y'].numpy(),baseline.scores(va).argmax(1))['macro_f1'])
   with torch.no_grad():fixed=fixedhead((torch.from_numpy(va)-mu)/sd).numpy()
   rec['fixed_original_CE_head_F1']=m.metrics(d['valid']['y'].numpy(),fixed.argmax(1))['macro_f1'];rec.update(scale_diag(model,d['valid']));rec['gradient_diagnostic']=loss_gradients(model,grad_batch)
   torch.save(model.state_dict(),out/f'epoch_{epoch}.pt');np.savez_compressed(out/f'epoch_{epoch}_VALID.npz',labels=d['valid']['y'].numpy(),rows=d['valid']['row'].numpy(),Ridge_scores=scores,head_scores=heads,fixed_head_scores=fixed)
   print('TRAJECTORY',fi,arm,mode,epoch,'head',round(hm['macro_f1'],5),'Ridge',round(pr['valid_macro_f1'],5),flush=True)
  records.append(rec)
  if epoch==CFG['epochs']:break
  model.train();ce_sum=0.;pen_sum=0.;n=0
  for b in m.batches(d['train'],m.order_for(d['train'],42,epoch)):
   opt.zero_grad(set_to_none=True);logits,_,delta=model(b);ce=F.cross_entropy(logits,b['y']);pen=(delta.square().sum(1)/(b['h'].square().sum(1)+1e-12)).mean();loss=ce+CFG['beta']*pen;assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['clip']);opt.step();ce_sum+=float(ce)*len(b['y']);pen_sum+=float(pen)*len(b['y']);n+=len(b['y'])
  rec.update(next_train_CE=ce_sum/n,next_train_weighted_penalty=CFG['beta']*pen_sum/n)
 save(out/'history.json',records)
 # How much of the learned correction is an ordinary linear reparameterization?
 tr=m.embeddings(model,d['train'])-d['train']['h'].numpy();va=m.embeddings(model,d['valid'])-d['valid']['h'].numpy();x=(d['train']['h'].numpy()-mu.numpy())/sd.numpy();v=(d['valid']['h'].numpy()-mu.numpy())/sd.numpy();xm=x.mean(0);ym=tr.mean(0);a=x-xm;G=dgemm(1.,a,a,trans_a=True)+np.eye(512);rhs=dgemm(1.,a,tr-ym,trans_a=True);coef=np.linalg.solve(G,rhs);pred=dgemm(1.,v-xm,coef)+ym;den=float(np.square(va-ym).sum());q2=1-float(np.square(va-pred).sum())/den if den>1e-20 else None
 save(out/'linear_reparameterization.json',{'VALID_delta_Q2_vs_TRAIN_delta_mean':q2,'TRAIN_fit_alpha':1.,'scope':'diagnostic predictability of correction, not proof of linearity/invertibility or task benefit'})
 hashes={str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json'};save(done,{'status':'complete','binding':binding(),'hashes':hashes,'seconds':time.time()-start,'original_prefix_replay':'PASS' if mode=='joint' else 'not_applicable','scored_roles':['train','valid']})
def latent_dense(b,kind,mu,random_E=None):
 if kind=='learned_SAE':return torch.zeros(len(b['h']),2048).scatter(1,b['idx'],b['val']).numpy()
 with torch.no_grad():a=(b['h']-mu)@random_E.T;v,i=torch.topk(a,32,dim=1,sorted=False);return torch.zeros_like(a).scatter(1,i,v).numpy()
def prepare_features(x,v):
 mu=x.astype('float64').mean(0);sd=x.astype('float64').std(0);active=sd>1e-12;sd[~active]=1.;return (x[:,active]-mu[active])/sd[active],(v[:,active]-mu[active])/sd[active],active
def branch(fi,kind):
 out=W/'branches'/f'fold_{fi}'/kind;out.mkdir(parents=True,exist_ok=True)
 if (out/'completion.json').exists():
  r=read(out/'completion.json');assert r['binding']==binding()
  for p,h in r['hashes'].items():assert sha(p)==h
  return
 start=time.time();d,D,fold,_,_=data_for(fi);mu,sd=m.get_stats(d);torch.manual_seed(20261019+fi);E=F.normalize(torch.randn(2048,512),dim=1);zt=latent_dense(d['train'],kind,mu,E);zv=latent_dense(d['valid'],kind,mu,E);h,hv,ha=prepare_features(d['train']['h'].numpy(),d['valid']['h'].numpy());z,zv,za=prepare_features(zt,zv);y=d['train']['y'].numpy();vy=d['valid']['y'].numpy();w=len(y)/(8*np.bincount(y,minlength=8)[y]);target=2*np.eye(8)[y]-1.;ym=np.average(target,axis=0,weights=w);B=(target-ym)*np.sqrt(w[:,None]);records=[];baseline,ir=m.fit_ridge(d['train']['h'].numpy(),y,d['valid']['h'].numpy(),vy);best=ir['valid_macro_f1'];selected=None;selected_alpha=ir['alpha'];chosen_scores=baseline.scores(d['valid']['h'].numpy());save(out/'identity.json',ir)
 for eta in CFG['branch_scales']:
  x=np.concatenate([h,eta*z],1);v=np.concatenate([hv,eta*zv],1);xm=np.average(x,axis=0,weights=w);A=(x-xm)*np.sqrt(w[:,None]);gram=dgemm(1.,A,A,trans_a=True);evals,U=eigh((gram+gram.T)/2,driver='evd');evals=np.maximum(evals,0);rhs=dgemm(1.,U,dgemm(1.,A,B,trans_a=True),trans_a=True)
  for alpha in CFG['branch_alphas']:
   coef=dgemm(1.,U,rhs/(evals[:,None]+alpha));intercept=ym-xm@coef;scores=dgemm(1.,v,coef)+intercept;assert np.isfinite(scores).all() and np.isfinite(coef).all();f=m.metrics(vy,scores.argmax(1))['macro_f1'];gradient=dgemm(1.,A,dgemm(1.,A,coef)-B,trans_a=True)+alpha*coef;den=max(1.,float(abs(dgemm(1.,A,B,trans_a=True)).max()));err=float(abs(gradient).max()/den);assert err<1e-7;records.append({'fold':fi,'valid_donor':fold['valid'],'latent':kind,'eta':eta,'alpha':alpha,'VALID_F1':f,'delta_pp':100*(f-ir['valid_macro_f1']),'normal_equation_error':err,'nonconstant_latents':int(za.sum())})
   np.savez_compressed(out/f'eta_{eta}_alpha_{alpha}_VALID.npz',scores=scores,labels=vy,rows=d['valid']['row'].numpy())
   if f>best+1e-12:best=f;selected=eta;selected_alpha=alpha;chosen_scores=scores
  print('BRANCH',fi,kind,eta,'best_VALID',round(best,5),'sec',round(time.time()-start,1),flush=True)
 np.savez_compressed(out/'selected_VALID.npz',scores=chosen_scores,labels=vy,rows=d['valid']['row'].numpy());save(out/'trials.json',records);save(out/'selection.json',{'fold':fi,'valid_donor':fold['valid'],'latent':kind,'identity_VALID_F1':ir['valid_macro_f1'],'best_VALID_F1':best,'selected_eta':selected,'selected_alpha':selected_alpha,'identity_selected':selected is None,'scope':'validation-selected exploratory signal, not held-out performance improvement'})
 hashes={str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='completion.json'};save(out/'completion.json',{'status':'complete','binding':binding(),'hashes':hashes,'seconds':time.time()-start,'scored_roles':['train','valid']})
def preflight():
 O.mkdir(parents=True,exist_ok=True);m.check_sources();m.immutable(O/'PROTOCOL.json',CFG);files=[Path(__file__),OLD/'experiment.py',OLD/'frozen_sources.json',m.O/'PROTOCOL.json',m.O/'selection_lock.json',m.R/'kang_embeddings.npz',m.R/'protocol.json']
 for fi in range(6):
  d,D,fold,base,bi=data_for(fi);files.extend([OLD/f'fold_{fi}/baseline/selected_head.pt',OLD/f'fold_{fi}/baseline/completion.json'])
  for arm in CFG['trajectory_arms']:
   files.extend([OLD/f'fold_{fi}/seed_42'/arm/'beta_0.1.json',OLD/f'fold_{fi}/seed_42'/arm/'beta_0.1.pt'])
 m.immutable(W/'frozen_sources.json',{str(p.resolve()):sha(p) for p in files});save(O/'preflight.json',{'status':'PASS','six_train_valid_splits_checked':True,'no_TEST_or_PBMC_scoring':True,'original_frozen_sources_unchanged':True});print('PREFLIGHT_PASS',flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','trajectories','branches']);a=ap.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);O.mkdir(parents=True,exist_ok=True)
 if a.stage=='preflight':preflight();return
 check()
 for fi in CFG['folds']:
  if a.stage=='trajectories':
   for arm in CFG['trajectory_arms']:
    for mode in CFG['head_modes']:trajectory(fi,arm,mode)
  else:
   for kind in CFG['branch_latents']:branch(fi,kind)
 check();save(O/(a.stage+'_completion.json'),{'status':'complete','binding':binding(),'scored_roles':['train','valid'],'never_scored':['test','pbmc']});print(a.stage.upper()+'_COMPLETE',flush=True)
if __name__=='__main__':
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');main()
  save(W/(sys.argv[1]+'_warnings.json'),{'warnings':[{'category':w.category.__name__,'message':str(w.message),'file':w.filename,'line':w.lineno} for w in caught]})
