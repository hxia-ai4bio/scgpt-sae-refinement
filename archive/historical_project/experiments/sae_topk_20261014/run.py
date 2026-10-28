"""Cell-level TopK SAE: TRAIN-only unsupervised fit, VALID MSE selection, balanced Ridge probes."""
from pathlib import Path
import argparse,json,time,copy,warnings,sys,platform
import numpy as np,torch
from sae import TopKSAE
from core import R,write,sha,metrics,reconstruction,RidgeGrid,ALPHAS
P=R.parents[1];METHODS=['original','sparse','reconstruction','concat'];SEEDS=[42,43,44]
CFG=dict(input_dim=512,dictionary_dim=2048,ks=[32,16,64],seeds=SEEDS,epochs=20,patience=3,batch_size=256,lr=.0003,optimizer='Adam',objective='raw mean squared embedding reconstruction',topk='largest raw activation VALUES; no absolute ranking/ReLU/L1; keep original signed values',decoder_bias=False,decoder_columns='unit norm after each Adam update',initialization='unit random decoder columns; encoder weight=decoder.T initially, encoder bias0; untied training',preprocess_sae='subtract fixed TRAIN mean only; no input channel scaling',probe='TRAIN StandardScaler + balanced eight-class RidgeClassifier-equivalent exact eigensolve; linear only',probe_alphas=ALPHAS,probe_selection='best VALID macroF1; first ascending-alpha tie; fit TRAIN only, no refit TRAIN+VALID',sae_selection='lowest VALID raw MSE, earliest tie; epoch0 eligible',baseline='deterministic Ridge; reused across K/seeds, not independent optimizer replicates',dtype='CPU float32 SAE; float64 probe',threads=4,train_rows='all eligible cells in four TRAIN donors; original donor roles; no class cap',macro_f1='fixed eight labels, zero_division=0; absent TEST class contributes0; supported-type macro diagnostic separately',pbmc='full eligible PBMC368k_raw4638cell local dataset; no fitting/selection',r2='1-SSE/sum||h-mean_of_evaluation_split||²; metric denominator only, not fitted preprocessing',q2='supplement 1-SSE/sum||h-TRAIN_mean||²',scope='historically explored donor and PBMC data; independent file not confirmed independent study/pretraining',no_mcr2=True,no_scgpt_training=True,no_mlp_classifier=True)

def load_kang():
 with np.load(R/'kang_embeddings.npz') as a:x=a['embedding'].copy();y=a['labels'].copy();donors=a['donors'].copy();rows=a['rows'].copy()
 meta=json.loads((R/'metadata_inventory.json').read_text());classes=sorted(meta['kang.h5ad']['cell_type']);assert len(classes)==8 and x.shape==(len(y),512) and np.isfinite(x).all()
 roles=json.loads((P/'experiments/multilayer_readout_20261012/protocol.json').read_text())['rotations'];folds=[]
 for f in roles:
  ix={s:np.flatnonzero((donors==f[s]) if s in ['valid','test'] else ~np.isin(donors,[f['valid'],f['test']])) for s in ['train','valid','test']};assert len(np.unique(np.concatenate(list(ix.values()))))==len(y);assert (np.bincount(y[ix['train']],minlength=8)>0).all();folds.append((f,ix))
 sourcepaths=[Path(__file__),R/'sae.py',R/'core.py',R/'kang_embeddings.npz',P/'experiments/multilayer_readout_20261012/protocol.json',R/'metadata_inventory.json',R/'preflight.json']
 proto=dict(config=CFG,classes=classes,source_hashes={str(p):sha(p) for p in sourcepaths},folds=[dict(fold=f,counts={s:len(i) for s,i in ix.items()},class_counts={s:np.bincount(y[i],minlength=8).tolist() for s,i in ix.items()}) for f,ix in folds],total_sae=54,core_k32_sae=18,planned_linear_representations='original6fits +54SAE×3arms=168linear families; fivealphas per family')
 if (R/'protocol.json').exists():assert json.loads((R/'protocol.json').read_text())==proto,'Immutable protocol changed; new directory required'
 else:write(R/'protocol.json',proto)
 return x,y,donors,rows,folds,proto

def sae_eval(model,x):
 model.eval();sse=0.;z=[];re=[];ii=[];vv=[]
 with torch.no_grad():
  for j in range(0,len(x),512):
   h=torch.from_numpy(x[j:j+512]);rec,a,ind,val=model(h);assert torch.isfinite(rec).all() and torch.isfinite(a).all();assert ((a!=0).sum(1)<=model.k).all();assert torch.equal(a.gather(1,ind),val)
   re.append(rec.numpy());z.append(a.numpy());ii.append(ind.numpy());vv.append(val.numpy());sse+=float((rec.double()-h.double()).square().sum())
 return sse/x.size,np.concatenate(z),np.concatenate(re),np.concatenate(ii),np.concatenate(vv)

def train_one(x,fold,ix,k,seed):
 out=R/('test_'+fold['test'])/('seed_'+str(seed))/('k_'+str(k));out.mkdir(parents=True,exist_ok=True)
 if (out/'sae_result.json').exists():return json.loads((out/'sae_result.json').read_text())
 mean=torch.from_numpy(x[ix['train']].astype(np.float64).mean(0).astype(np.float32));m=TopKSAE(mean,k,seed);initial=copy.deepcopy(m.state_dict());opt=torch.optim.Adam(m.parameters(),lr=CFG['lr']);hist=[];best=float('inf');stale=0;start=time.time();updates=0;maxnorm=0.;gradfirst=None
 for epoch in range(21):
  valid_mse,*_=sae_eval(m,x[ix['valid']]);rec=dict(epoch=epoch,updates=updates,valid_mse=valid_mse);hist.append(rec)
  if valid_mse<best:best=valid_mse;best_epoch=epoch;beststate=copy.deepcopy(m.state_dict());stale=0
  else:stale+=1
  print('SAE_EPOCH',fold['test'],seed,k,epoch,'valid_mse',round(valid_mse,7),flush=True)
  if epoch==20 or stale>=3:break
  m.train();order=np.random.default_rng(np.random.SeedSequence([seed,epoch,20261014])).permutation(ix['train']);trainerr=0.;seen=0
  for j in range(0,len(order),256):
   h=torch.from_numpy(x[order[j:j+256]]);opt.zero_grad(set_to_none=True);rh,z,_,_=m(h);loss=(rh-h).square().mean();assert torch.isfinite(loss);loss.backward();gg={n:float(p.grad.norm()) for n,p in m.named_parameters()};assert all(np.isfinite(v) and v>0 for v in gg.values())
   if updates==0:gradfirst=gg
   opt.step();m.normalize_dictionary();normerr=float((m.decoder.weight.double().norm(dim=0)-1).abs().max());maxnorm=max(maxnorm,normerr);assert normerr<2e-6;updates+=1;trainerr+=float(loss)*len(h);seen+=len(h)
  rec['next_train_mse']=trainerr/seen
 m.load_state_dict(beststate);torch.save(beststate,out/'sae.pt');result=dict(fold=fold,k=k,seed=seed,best_epoch=best_epoch,last_epoch=epoch,updates=updates,valid_best_mse=best,history=hist,initial_gradients=gradfirst,max_dictionary_norm_error=maxnorm,changed_tensors=sum(not torch.equal(v,initial[n]) for n,v in beststate.items() if n!='mean'),seconds=time.time()-start,path=str(out));write(out/'sae_result.json',result);print('SAE_DONE',fold['test'],seed,k,'sec',round(result['seconds'],1),flush=True);return result

def fit_probe(name,features,y,ix,py,out,seed):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 start=time.time();grid=RidgeGrid(features['train'],y[ix['train']]);trials=[];best=-1.;alpha=None
 for a in ALPHAS:
  grid.fit(a);vs=grid.scores(features['valid']);met=metrics(y[ix['valid']],vs.argmax(1));err=grid.verify(a);assert err<1e-7 and np.isfinite(vs).all();trials.append(dict(alpha=a,valid=met,weighted_normal_equation_relative_error=err))
  if met['macro_f1']>best:best=met['macro_f1'];alpha=a
 grid.fit(alpha);np.savez_compressed(out/'linear_probe.npz',**grid.arrays(),alpha=alpha);result=dict(method=name,seed=seed,selected_alpha=alpha,validation_best=best,trials=trials,feature_dim=features['train'].shape[1],train_nonconstant_features=len(grid.active),seconds=time.time()-start);arrays={}
 for s in ['train','valid','test','pbmc']:
  scores=grid.scores(features[s]);assert np.isfinite(scores).all();labels=py if s=='pbmc' else y[ix[s]];pred=scores.argmax(1);met=metrics(labels,pred);present=np.flatnonzero(np.bincount(labels,minlength=8)>0);met['supported_types_macro_f1']=float(np.mean(np.array(met['per_class_f1'])[present]));result[s]=met;arrays[s+'_scores']=scores;arrays[s+'_prediction']=pred;arrays[s+'_label']=labels
 np.savez_compressed(out/'predictions.npz',**arrays);write(out/'result.json',result);print('PROBE_DONE',name,seed,'test',round(result['test']['macro_f1'],4),'pbmc',round(result['pbmc']['macro_f1'],4),'sec',round(result['seconds'],1),flush=True);return result

def finish_one(x,y,rows,fold,ix,k,seed,px,py,baseline):
 out=R/('test_'+fold['test'])/('seed_'+str(seed))/('k_'+str(k))
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 state=torch.load(out/'sae.pt',map_location='cpu');m=TopKSAE(state['mean'],k,seed);m.load_state_dict(state);assert (m.decoder.weight.double().norm(dim=0)-1).abs().max()<2e-6
 z={};rh={};diag={};arrays={}
 for s,h in [('train',x[ix['train']]),('valid',x[ix['valid']]),('test',x[ix['test']]),('pbmc',px)]:
  mse,zz,re,ii,vv=sae_eval(m,h);z[s]=zz;rh[s]=re;count=(zz!=0).sum(1);freq=(zz!=0).mean(0);diag[s]={**reconstruction(h,re,state['mean'].numpy()),'mean_active_features':float(count.mean()),'min_active_features':int(count.min()),'max_active_features':int(count.max()),'retained_topk_slots':k,'dead_features':int((freq==0).sum()),'dead_feature_ratio':float((freq==0).mean()),'negative_retained_fraction':float((vv<0).mean()),'feature_activation_frequency':freq.tolist()};arrays[s+'_indices']=ii;arrays[s+'_values']=vv;arrays[s+'_reconstruction']=re;arrays[s+'_labels']=py if s=='pbmc' else y[ix[s]];arrays[s+'_rows']=np.arange(len(px)) if s=='pbmc' else rows[ix[s]]
 np.savez_compressed(out/'sae_outputs.npz',**arrays)
 orig={s:x[ix[s]] for s in ['train','valid','test']};orig['pbmc']=px;probes={'original':baseline}
 for method in ['sparse','reconstruction','concat']:
  f=z if method=='sparse' else rh if method=='reconstruction' else {s:np.concatenate([orig[s],z[s]],1) for s in orig};probes[method]=fit_probe(method,f,y,ix,py,out/method,seed)
 result=dict(fold=fold,k=k,seed=seed,sae=json.loads((out/'sae_result.json').read_text()),reconstruction=diag,probes=probes,path=str(out));write(out/'result.json',result);return result

def summarize(results):
 aggregate={}
 for k in [32,16,64]:
  rs=[r for r in results if r['k']==k]
  if len(rs)==18:
   aa={}
   for m in METHODS:
    aa[m]={}
    for s in ['test','pbmc']:
     vals=[r['probes'][m][s]['macro_f1'] for r in rs];byseed=[float(np.mean([r['probes'][m][s]['macro_f1'] for r in rs if r['seed']==seed])) for seed in SEEDS];aa[m][s]=dict(mean=float(np.mean(vals)),sd_18_fold_seed=float(np.std(vals,ddof=1)),seed_fold_mean=byseed,sd_three_seed_means=float(np.std(byseed,ddof=1)),per_class_f1=np.mean([r['probes'][m][s]['per_class_f1'] for r in rs],0).tolist(),accuracy_mean=float(np.mean([r['probes'][m][s]['accuracy'] for r in rs])))
   aa['sae_metrics']={s:{key:float(np.mean([r['reconstruction'][s][key] for r in rs])) for key in ['mse','r2_test_centered','q2_train_mean','dead_feature_ratio','mean_active_features']} for s in ['train','valid','test','pbmc']};aggregate[str(k)]=aa
 write(R/'summary.json',dict(status='complete' if len(results)==54 else 'running',completed_sae_candidates=len(results),planned_sae_candidates=54,aggregate=aggregate,results=results))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-only',action='store_true');ap.add_argument('--ks',type=int,nargs='+',default=[32,16,64]);args=ap.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);x,y,donors,rows,folds,proto=load_kang();results=[]
 if not args.train_only:
  manifest=json.loads((R/'data_manifest.json').read_text());assert manifest['classes']==proto['classes'];assert sha(R/'kang_embeddings.npz')==manifest['sources'][str(R/'kang_embeddings.npz')]
  with np.load(R/'pbmc_embeddings.npz') as a:px=a['embedding'];py=a['labels']
  assert sha(R/'pbmc_embeddings.npz')==manifest['sources'][str(R/'pbmc_embeddings.npz')]
 for k in args.ks:
  for fold,ix in folds:
   np.savez_compressed(R/('split_'+fold['test']+'.npz'),**{s:rows[v] for s,v in ix.items()})
   baseline=None
   if not args.train_only:
    features={s:x[ix[s]] for s in ['train','valid','test']};features['pbmc']=px;baseline=fit_probe('original',features,y,ix,py,R/('test_'+fold['test'])/'original',None)
   for seed in SEEDS:
    train_one(x,fold,ix,k,seed)
    if not args.train_only:results.append(finish_one(x,y,rows,fold,ix,k,seed,px,py,baseline));summarize(results)
  print('K_STAGE_COMPLETE',k,flush=True)
 if not args.train_only:print('ALL54_SAE_AND_PROBES_COMPLETE',flush=True)
if __name__=='__main__':
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always')
  try:main()
  finally:write(R/('train_only_warnings.json' if '--train-only' in sys.argv else 'run_warnings.json'),dict(count=len(ws),messages=sorted(set(str(w.message) for w in ws))))
