"""Definitions extracted verbatim from the original experiments; see provenance.json.
SAE O(B*d*m), relation KL O(B^2*d), Ridge O(N*d^2+d^3).
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm
from sklearn.metrics import f1_score, confusion_matrix

class SAE(nn.Module):
    def __init__(self, mean, width, k, seed, mode='raw'):
        super().__init__()
        torch.manual_seed(seed)
        self.width, self.k, self.mode = width, k, mode
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.encoder = nn.Linear(512, width)
        self.decoder = nn.Linear(width, 512, bias=False)
        nn.init.zeros_(self.encoder.bias)
        self.normalize()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)

    @torch.no_grad()
    def normalize(self):
        norms = self.decoder.weight.norm(dim=0, keepdim=True)
        assert torch.isfinite(norms).all() and (norms > 1e-12).all()
        self.decoder.weight.div_(norms)

    def forward(self, x):
        a = self.encoder(x-self.mean)
        score = a.abs() if self.mode == 'abs' else a
        ix = torch.topk(score, self.k, dim=1, sorted=False).indices
        values = a.gather(1, ix)
        z = torch.zeros_like(a).scatter(1, ix, values)
        return self.decoder(z)+self.mean, z

def infer(model, h, latent=False):
    model.eval()
    results = []
    with torch.no_grad():
        for j in range(0, len(h), 512):
            rec, z = model(torch.from_numpy(h[j:j+512]))
            assert torch.isfinite(rec).all() and torch.isfinite(z).all()
            assert ((z != 0).sum(1) <= model.k).all()
            results.append((z if latent else rec).numpy())
    return np.concatenate(results)

def relation(h,rec,mean,mask=None,anchor_h=None,anchor_rec=None):
    anchor_h=h if anchor_h is None else anchor_h
    anchor_rec=rec if anchor_rec is None else anchor_rec
    x=F.normalize(h.double()-mean.double(),dim=1,eps=1e-12)
    a=F.normalize(anchor_h.double()-mean.double(),dim=1,eps=1e-12)
    y=F.normalize(rec.double()-mean.double(),dim=1,eps=1e-12)
    ar=F.normalize(anchor_rec.double()-mean.double(),dim=1,eps=1e-12)
    teacher=(x@a.T).detach();student=y@ar.T
    if mask is not None:
        assert mask.sum(1).unique().numel()==1
        teacher=teacher[mask].reshape(len(h),-1);student=student[mask].reshape(len(h),-1)
    logp=F.log_softmax(teacher/.1,dim=1)
    logq=F.log_softmax(student/.1,dim=1)
    p=logp.exp()
    assert not logp.requires_grad
    assert torch.isfinite(logp).all() and torch.isfinite(logq).all()
    assert float((p.sum(1)-1).abs().max())<1e-10
    assert float((logq.exp().sum(1)-1).abs().max())<1e-10
    kl=(p*(logp-logq)).sum(1).mean()
    assert torch.isfinite(kl) and float(kl.detach())>=-1e-12
    return kl,float((-(p*logp).sum(1)).exp().mean())

def mm(a,b,ta=False,tb=False):
    return dgemm(1.,np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64),trans_a=ta,trans_b=tb)

def metrics(y,scores):
    pred=np.asarray(scores).argmax(1);sup=np.bincount(y,minlength=8)
    f=f1_score(y,pred,labels=np.arange(8),average=None,zero_division=0)
    return {'macro_f1':float(f.mean()),'supported_macro_f1':float(f[sup>0].mean()),
            'accuracy':float(np.mean(pred==y)),'per_class_f1':f.tolist(),'support':sup.tolist(),
            'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(8)).tolist()}

def weights(y):
    c=np.bincount(y,minlength=8);return len(y)/(np.count_nonzero(c)*c[y])

class Ridge:
    def __init__(self,x,y,target=None,scale=True):
        x=np.asarray(x,dtype=np.float64)
        self.mu=x.mean(0) if scale else np.zeros(x.shape[1]);self.sd=x.std(0) if scale else np.ones(x.shape[1]);self.sd[self.sd<1e-12]=1.
        s=(x-self.mu)/self.sd;w=weights(y)
        target=2*np.eye(8)[y]-1. if target is None else np.asarray(target,dtype=np.float64)
        self.xm=np.average(s,axis=0,weights=w);self.ym=np.average(target,axis=0,weights=w)
        a=(s-self.xm)*np.sqrt(w[:,None]);b=(target-self.ym)*np.sqrt(w[:,None])
        self.gram=mm(a,a,ta=True);self.gram=(self.gram+self.gram.T)/2
        self.rawrhs=mm(a,b,ta=True)
        e,self.u=eigh(self.gram,driver='evd',check_finite=True);self.e=np.maximum(e,0);self.rhs=mm(self.u,self.rawrhs,ta=True)
    def fit(self,alpha):
        self.alpha=float(alpha);self.coef=mm(self.u,self.rhs/(self.e[:,None]+alpha))
        self.intercept=self.ym-np.einsum('i,ij->j',self.xm,self.coef)
        grad=mm(self.gram,self.coef)+alpha*self.coef-self.rawrhs
        self.err=float(np.abs(grad).max()/max(1.,float(np.abs(self.rawrhs).max())))
        assert self.err<1e-7 and np.isfinite(self.coef).all(),self.err
        return self
    def scores(self,x):
        s=mm((np.asarray(x,dtype=np.float64)-self.mu)/self.sd,self.coef)+self.intercept
        assert np.isfinite(s).all();return s
    def arrays(self): return {'mean':self.mu,'scale':self.sd,'coef':self.coef,'intercept':self.intercept,'alpha':np.asarray(self.alpha)}
