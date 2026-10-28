"""Independent output, selection, metric and linear-normal-equation checks.
Does not import the training implementation. No training or hyperparameter search.
"""
from pathlib import Path
import json,hashlib,warnings
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_residual'
R=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def reconstruct(h,idx,val,state,kind):
 hh=torch.from_numpy(h);ii=torch.from_numpy(idx.astype('int64'));vv=torch.from_numpy(val);out=[]
 with torch.no_grad():
  for lo in range(0,len(h),256):
   b=hh[lo:lo+256];i=ii[lo:lo+256];v=vv[lo:lo+256]
   if kind=='identity':delta=torch.zeros_like(b)
   elif kind in ['gate','rotated_gate']:
    weight=.5*torch.tanh(state['residual.theta'][i])*v
    delta=torch.einsum('bk,bkd->bd',weight,state['residual.DT'][i])
   elif kind=='lowrank':delta=(state['residual.V'][i]*v[...,None]).sum(1)@state['residual.U']
   else:delta=torch.clamp((b-state['residual.mu'])@state['residual.V'],min=0)@state['residual.U']
   out.append((b+delta).numpy())
 return np.concatenate(out)
def main():
 torch.set_num_threads(4);cfg=json.loads((O/'PROTOCOL.json').read_text());binding={'sources_sha':sha(W/'frozen_sources.json'),'protocol_sha':sha(O/'PROTOCOL.json')};lock=json.loads((O/'selection_lock.json').read_text())
 for p,h in json.loads((W/'frozen_sources.json').read_text()).items():assert sha(p)==h,p
 for p,h in lock.items():assert sha(p)==h,p
 assert len(lock)==126
 a=np.load(R/'kang_embeddings.npz');pb=np.load(R/'pbmc_embeddings.npz');protocol=json.loads((R/'protocol.json').read_text());errs=[];normal=[];checked=0;predictions=0;near_ties=[];baseline={}
 selections=[]
 for dest in sorted(O.glob('fold_*/seed_*/*/result.json')):
  r=json.loads(dest.read_text());fi=r['fold'];seed=r['seed'];arm=r['arm'];fold=protocol['folds'][fi]['fold'];assert r['bindings']==binding and r['selection_lock_sha']==sha(O/'selection_lock.json')
  for p,h in r['output_hashes'].items():assert sha(p)==h
  ix={s:np.flatnonzero((a['donors']==fold[s]) if s!='train' else ~np.isin(a['donors'],[fold['valid'],fold['test']])) for s in ['train','valid','test']};assert len(np.unique(np.concatenate(list(ix.values()))))==len(a['labels'])
  src=R/f"test_{fold['test']}"/f'seed_{seed}'/'k_32';cache=np.load(src/'sae_outputs.npz')
  if arm=='identity':state=torch.load(W/f'fold_{fi}/baseline/selected_head.pt',map_location='cpu',weights_only=False);kind='identity'
  else:
   sp=W/f'fold_{fi}'/f'seed_{seed}'/arm/'selection.json';sel=json.loads(sp.read_text());assert sel['bindings']==binding
   for t in sel['trials']:assert sha(t['checkpoint'])==t['checkpoint_sha'] and sha(t['metadata'])==t['metadata_sha']
   best=sel['identity_valid_macro_f1'];chosen=None
   for t in sel['trials']:
    if t['valid_macro_f1']>best+1e-12:best=t['valid_macro_f1'];chosen=t['beta']
   assert chosen==sel['selected_beta']==r['selected_beta'] and abs(best-sel['valid_macro_f1'])<1e-12
   selections.append({'fold':fi,'seed':seed,'arm':arm,'identity_selected':chosen is None})
   kind='identity' if chosen is None else arm.replace('_supcon','');cp=W/f'fold_{fi}/baseline/selected_head.pt' if chosen is None else W/f'fold_{fi}'/f'seed_{seed}'/arm/f'beta_{chosen}.pt';state=torch.load(cp,map_location='cpu',weights_only=False)
  D=torch.load(src/'sae.pt',map_location='cpu',weights_only=False)['decoder.weight']
  if kind=='gate':assert torch.equal(state['residual.DT'],D.T.contiguous())
  if kind=='rotated_gate':
   cols=np.arange(0,2048,16);orig=D[:,cols].double().T@D[:,cols].double();other=state['residual.DT'][cols].double()@state['residual.DT'][cols].double().T;assert float((orig-other).abs().max())<3e-6
  def e(role):
   h=a['embedding'][ix[role]] if role!='pbmc' else pb['embedding']
   return reconstruct(h.astype('float32'),cache[role+'_indices'],cache[role+'_values'].astype('float32'),state,kind)
  train=e('train');y=a['labels'][ix['train']];ridge=np.load(dest.parent/'ridge.npz');assert all(np.isfinite(ridge[k]).all() for k in ridge.files);mean=train.astype('float64').mean(0);scale=train.astype('float64').std(0);scale[scale<1e-12]=1.;assert np.max(abs(mean-ridge['mean']))<3e-6 and np.max(abs(scale-ridge['scale']))<3e-6
  x=(train-ridge['mean'])/ridge['scale'];coef=ridge['coef'];intercept=ridge['intercept'];weights=len(y)/(8*np.bincount(y,minlength=8)[y]);target=2*np.eye(8)[y]-1.;score=np.einsum('nd,dc->nc',x,coef)+intercept;res=score-target;grad=np.einsum('nd,nc,n->dc',x,res,weights)+float(ridge['alpha'])*coef;den=max(1.,float(np.abs(np.einsum('nd,nc,n->dc',x,target,weights)).max()));error=float(abs(grad).max()/den);normal.append(error);assert error<5e-5,(dest,error)
  arrays=np.load(dest.parent/'predictions.npz')
  for role in ['valid','test','pbmc']:
   yy=a['labels'][ix[role]] if role!='pbmc' else pb['labels'];rr=a['rows'][ix[role]] if role!='pbmc' else np.arange(len(yy));assert np.array_equal(yy,arrays[role+'_label']) and np.array_equal(rr,arrays[role+'_row']);emb=e(role);scores=np.einsum('nd,dc->nc',(emb-ridge['mean'])/ridge['scale'],coef)+intercept;old=arrays[role+'_scores'];assert np.isfinite(old).all();err=float(np.max(abs(scores-old)));errs.append(err);assert err<5e-4,(dest,role,err)
   assert np.array_equal(old.argmax(1),arrays[role+'_ridge_prediction'])
   mismatch=scores.argmax(1)!=old.argmax(1)
   if mismatch.any():
    sorted_scores=np.sort(old,axis=1);margin=sorted_scores[:,-1]-sorted_scores[:,-2];assert (margin[mismatch]<=2*err+1e-10).all();near_ties.append({'file':str(dest),'role':role,'n':int(mismatch.sum()),'max_error':err})
   for name,key in [('ridge','ridge_prediction'),('knn','knn_prediction'),('trained_head','trained_head_prediction')]:
    pred=arrays[role+'_'+key];values=f1_score(yy,pred,labels=np.arange(8),average=None,zero_division=0);support=np.bincount(yy,minlength=8);m=r[role][name];assert abs(values.mean()-m['macro_f1'])<1e-12;assert abs(values[support>0].mean()-m['supported_macro_f1'])<1e-12;assert np.allclose(values,m['per_class_f1'],atol=1e-12);assert np.array_equal(confusion_matrix(yy,pred,labels=np.arange(8)),m['confusion_matrix']);assert abs(accuracy_score(yy,pred)-m['accuracy'])<1e-12;predictions+=len(yy)
   if arm=='identity':
    key=(fi,role)
    if key in baseline:assert np.array_equal(baseline[key],old)
    else:baseline[key]=old
  checked+=1
 assert checked==144 and len(selections)==126
 save(O/'verification.json',dict(status='PASS',runs_checked=checked,selected_adapters_checked=len(selections),metric_predictions_checked=predictions,max_independent_score_error=max(errs),max_independent_normal_equation_relative_error=max(normal),near_tie_prediction_differences=near_ties,all_sources_and_selection_hashes_unchanged=True,scope='input/provenance/numerical verification; not evidence of performance gain or biological mechanism'))
 print('VERIFY_PASS',checked,flush=True)
if __name__=='__main__':
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');main();save(O/'verification_warnings.json',[{'category':str(w.category.__name__),'message':str(w.message),'filename':w.filename,'line':w.lineno} for w in caught])
