"""Metrics and TRAIN-only standardized balanced Ridge linear probes.
Exact weighted Ridge eigensolve reused for five regularizers; no neural classifier.
Ridge cost O(n p²+p³) for n>=p; O(p n²+n³) for n<p.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
R=Path(__file__).resolve().parent;ALPHAS=[.01,.1,1.,10.,100.]
def write(p,j):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(j,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def metrics(y,p):
 cm=confusion_matrix(y,p,labels=np.arange(8));return dict(macro_f1=float(f1_score(y,p,labels=np.arange(8),average='macro',zero_division=0)),accuracy=float(accuracy_score(y,p)),per_class_f1=f1_score(y,p,labels=np.arange(8),average=None,zero_division=0).tolist(),confusion_matrix=cm.tolist(),support=cm.sum(1).tolist())
def reconstruction(x,recon,mean):
 err=float(np.square(x.astype(np.float64)-recon).sum());x64=x.astype(np.float64);den=float(np.square(x64-x64.mean(0)).sum());tden=float(np.square(x64-mean.astype(np.float64)).sum())
 return dict(mse=err/x.size,r2_test_centered=1-err/den if den>0 else None,q2_train_mean=1-err/tden if tden>0 else None,sse=err,sst_split_mean=den,sst_train_mean=tden)

class RidgeGrid:
 def __init__(self,x,y):
  self.scaler=StandardScaler().fit(x);s=self.scaler.transform(x.astype(np.float64));self.active=np.flatnonzero(self.scaler.var_>0);s=s[:,self.active];n,p=s.shape
  count=np.bincount(y,minlength=8);assert (count>0).all(),'all eight labels required in TRAIN'
  self.weights=n/(8*count[y]);self.xmean=np.average(s,axis=0,weights=self.weights);target=2*np.eye(8)[y]-1.;self.ymean=np.average(target,axis=0,weights=self.weights)
  a=(s-self.xmean)*np.sqrt(self.weights)[:,None];b=(target-self.ymean)*np.sqrt(self.weights)[:,None];self.a=a;self.b=b
  self.dual=n<p;gram=a@a.T if self.dual else a.T@a;gram=(gram+gram.T)/2;self.eigen,self.v=eigh(gram,check_finite=True,driver='evd');self.eigen=np.maximum(self.eigen,0)
  self.rhs=self.v.T@b if self.dual else self.v.T@(a.T@b)
 def fit(self,alpha):
  q=self.v@(self.rhs/(self.eigen[:,None]+alpha));self.coef=self.a.T@q if self.dual else q;self.intercept=self.ymean-self.xmean@self.coef;return self
 def scores(self,x):return self.scaler.transform(x.astype(np.float64))[:,self.active]@self.coef+self.intercept
 def arrays(self):return dict(scaler_mean=self.scaler.mean_,scaler_scale=self.scaler.scale_,active_columns=self.active,coef=self.coef,intercept=self.intercept)
 def verify(self,alpha):
  gradient=self.a.T@(self.a@self.coef-self.b)+alpha*self.coef;den=max(float(np.abs(self.a.T@self.b).max()),1.)
  return float(np.abs(gradient).max()/den)

def ridge_smoke(x,y,valid):
 grid=RidgeGrid(x,y);grid.fit(10.);ref=RidgeClassifier(alpha=10.,class_weight='balanced',solver='cholesky').fit(grid.scaler.transform(x.astype(np.float64)),y);a=grid.scores(valid);b=ref.decision_function(grid.scaler.transform(valid.astype(np.float64)));err=float(np.max(np.abs(a-b)));assert err<1e-6,err
 return dict(status='PASS',sklearn_scores_max_error=err,weighted_normal_equation_relative_error=grid.verify(10.))
