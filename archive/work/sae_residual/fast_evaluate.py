"""Identical k=15 standardized-cosine neighbors via blocked BLAS distances.
For near ties, replay the original cdist/stable-sort rule exactly. Training,
selection and all other evaluation code remain in the frozen experiment.py.
"""
from pathlib import Path
import json,time,shutil
import numpy as np
from scipy.linalg.blas import dgemm
from scipy.spatial.distance import cdist
import experiment as m
W=Path(__file__).resolve().parent;O=m.O
def fast_knn(train,y,x):
 mu=train.astype('float64').mean(0);sd=train.astype('float64').std(0);sd[sd<1e-12]=1.;a=(train-mu)/sd;b=(x-mu)/sd;a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12);an=np.einsum('ij,ij->i',a,a);pred=[];k=m.CFG['knn_k']
 for lo in range(0,len(b),256):
  bb=b[lo:lo+256];dist=dgemm(-2.,bb,a,trans_b=True);dist+=np.einsum('ij,ij->i',bb,bb)[:,None]+an[None,:];assert np.isfinite(dist).all()
  ids=np.argpartition(dist,k,axis=1)[:,:k+1];vals=np.take_along_axis(dist,ids,axis=1);ii=np.argsort(vals,axis=1);ids=np.take_along_axis(ids,ii,axis=1);vals=np.take_along_axis(vals,ii,axis=1);near=(vals[:,k]-vals[:,k-1])<1e-10
  chosen=ids[:,:k].copy()
  if near.any():chosen[near]=np.argsort(cdist(bb[near],a,'sqeuclidean'),axis=1,kind='stable')[:,:k]
  counts=np.stack([(y[chosen]==c).sum(1) for c in range(8)],1);pred.extend(counts.argmax(1))
 return np.asarray(pred,np.int64)
def parity():
 checks=[];start=time.time()
 # All already produced initial evaluations are immutable references.
 for rp in sorted(O.glob('fold_*/seed_*/*/result.json')):
  r=json.loads(rp.read_text());data,D,fold,_,_=m.load_data(r['fold'],r['seed']);mu,sd=m.get_stats(data);chosen=r['selected_beta'];kind='identity' if chosen is None else r['arm'].replace('_supcon','');model=m.Model(m.Residual(kind,D,mu,sd,r['seed']));cp=W/f"fold_{r['fold']}"/'baseline/selected_head.pt' if chosen is None else W/f"fold_{r['fold']}"/f"seed_{r['seed']}"/r['arm']/f'beta_{chosen}.pt';model.load_state_dict(m.torch.load(cp,map_location='cpu',weights_only=False));tr=m.embeddings(model,data['train']);old=np.load(rp.parent/'predictions.npz')
  for role in ['valid','test','pbmc']:
   pred=fast_knn(tr,data['train']['y'].numpy(),m.embeddings(model,data[role]));assert np.array_equal(pred,old[role+'_knn_prediction']),(rp,role);checks.append({'result':str(rp),'role':role,'queries':len(pred),'prediction_differences':0})
 # Duplicated/equidistant rows must follow original stable index tie behavior.
 rng=np.random.default_rng(30);a=rng.normal(size=(50,12));a[:20]=a[0];y=np.arange(50)%8;b=np.vstack([a[0],rng.normal(size=(5,12))]);assert np.array_equal(fast_knn(a,y,b),m.knn_predictions(a,y,b))
 assert checks,'Need already saved reference predictions before changing backend'
 record={'status':'PASS','backend':'blocked scipy BLAS squared distances + partial selection; cdist stable-sort replay at near ties','same_k':15,'source_sha':m.sha(__file__),'reference_checks':checks,'tie_test':'PASS','seconds':time.time()-start,'scope':'performance-only implementation; mathematical distance and neighbor rule unchanged; no test-based parameter selection'}
 m.immutable(O/'EVALUATION_BACKEND.json',record)
 print('FAST_KNN_PARITY_PASS',sum(c['queries'] for c in checks),'queries',round(record['seconds'],2),'seconds',flush=True)
def main():
 m.torch.set_num_threads(4);m.torch.use_deterministic_algorithms(True);m.check_sources()
 if not (O/'EVALUATION_BACKEND.json').exists():parity()
 else:assert json.loads((O/'EVALUATION_BACKEND.json').read_text())['source_sha']==m.sha(__file__)
 m.knn_predictions=fast_knn;m.evaluate_all();m.check_sources()
if __name__=='__main__':main()
