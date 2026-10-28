from pathlib import Path
import time,json,copy,warnings
import torch,numpy as np
from sae import TopKSAE
from core import write,ridge_smoke
R=Path(__file__).resolve().parent;torch.set_num_threads(4)
with np.load(R/'kang_embeddings.npz') as a:x=torch.from_numpy(a['embedding']);y=a['labels'];donors=a['donors']
ix=np.flatnonzero(~np.isin(donors,['patient_1016','patient_1039']));h=x[ix[:256]];mean=x[ix].mean(0);checks={}
for k in [16,32,64]:
 m=TopKSAE(mean,k,42);opt=torch.optim.Adam(m.parameters(),lr=3e-4);ls=[];start=time.time()
 for step in range(30):
  opt.zero_grad(set_to_none=True);re,z,inds,vals=m(h);a=m.encoder(h-m.mean);sortedvals=torch.sort(a,dim=1,descending=True).values[:,:k];assert torch.allclose(torch.sort(vals,dim=1,descending=True).values,sortedvals)
  assert inds.shape==(len(h),k) and ((z!=0).sum(1)<=k).all();assert torch.equal(z.gather(1,inds),vals);assert torch.isfinite(re).all();loss=(re-h).square().mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in m.parameters());opt.step();m.normalize_dictionary();ls.append(float(loss));assert (m.decoder.weight.double().norm(dim=0)-1).abs().max()<2e-6
 assert ls[-1]<ls[0]
 checks[str(k)]=dict(first_mse=ls[0],last_mse=ls[-1],mean_actual_nonzero=float((z!=0).sum(1).float().mean()),dictionary_max_unit_error=float((m.decoder.weight.norm(dim=0)-1).abs().max()),seconds_30_training_steps=time.time()-start)
neg=TopKSAE(mean,32,19)
with torch.no_grad():neg.encoder.weight.zero_();neg.encoder.bias.copy_(-torch.arange(1,2049,dtype=torch.float32))
z,inds,vals=neg.encode(h[:2]);assert set(inds[0].tolist())==set(range(32));assert (vals<0).all() and (z!=0).sum(1).tolist()==[32,32]
# Small TRAIN-only class-complete probe; same target weights as established sklearn protocol.
ids=np.concatenate([ix[y[ix]==c][:20] for c in range(8)]);assert len(np.unique(y[ids]))==8
with warnings.catch_warnings(record=True) as w:
 warnings.simplefilter('always');probe=ridge_smoke(x[ids].numpy().astype(float),y[ids],x[ix[:30]].numpy().astype(float))
write(R/'preflight.json',dict(status='PASS',checks=checks,negative_topk_preserves_values=True,no_abs_topk_no_relu=True,probe=probe,smoke_train_rows=ids.tolist(),weights_discarded=True,only_first_fold_TRAIN=True,probe_warning_count=len(w),probe_warning_messages=sorted(set(str(t.message) for t in w)),scgpt_loaded_or_trained=False))
print(json.dumps(checks,indent=2));print('PREFLIGHT_PASS',flush=True)
