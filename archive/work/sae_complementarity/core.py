"""Frozen-representation analysis; all numeric products use explicit SciPy BLAS.

Ridge fit: O(n p^2 + p^3) time, O(n p + p^2) memory. No neural training.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm
from sklearn.metrics import f1_score, confusion_matrix

W = Path(__file__).resolve().parent
ROOT = W.parents[1]
O = ROOT / 'outputs/sae_complementarity'
R = Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
ALPHAS = [10. ** v for v in range(-4, 7)]
SEEDS = [42, 43, 44]
KINDS = ['SAE', 'random', 'linear']
ETA = [.01, .1, 1.]
GAMMA = [.25, .5, 1.]
MIX_ALPHA = [0., -1., -.5, -.25, .25, .5, 1.]
MIX_LAMBDA = [1., 0., .5, 2/3, .8, 4/3, 2.]
CFG = {
    'folds': list(range(6)), 'seeds': SEEDS, 'k': 32,
    'ridge_alphas': ALPHAS, 'branch_eta': ETA, 'correction_gamma': GAMMA,
    'fixed_alpha': MIX_ALPHA, 'residual_lambda': MIX_LAMBDA,
    'classes': ['B cells','CD14+ Monocytes','CD4 T cells','CD8 T cells',
                'Dendritic cells','FCGR3A+ Monocytes','Megakaryocytes','NK cells'],
    'selection': 'VALID fixed-eight-class MacroF1; identity wins ties; then ascending grid order. All 18 selections locked before any new TEST/PBMC scoring.',
    'fit': 'four TRAIN donors only; no TRAIN+VALID refit; TRAIN-only unweighted feature mean/std; class-balanced weighted Ridge with intercept, targets 2*onehot-1; weights sum to TRAIN n',
    'scope': 'posthoc exploratory re-use of historically explored Kang and PBMC, not blind confirmatory validation; frozen upstream checkpoints fit four TRAIN donors and selected by VALID reconstruction MSE',
    'crossfit': 'TRAIN four-donor LODO base predictions. For each held-out TRAIN donor, choose base alpha by three-donor LODO inside remaining three donors, refit remaining three, predict excluded donor. Targets 2*onehot-1 minus these honest label-crossfit scores. Frozen upstream SAE is NOT refit inside these inner splits; its unsupervised training included inner-held-out h. Cross-fitting claim concerns label/readout fitting only.',
    'correction': 'fit weighted Ridge of TRAIN z to label-crossfit residuals; VALID-select correction alpha and gamma against separately VALID-selected original h Ridge. Identity candidate included. Report TEST residual MSE against final original-h errors separately from class metrics.',
    'concat': 'TRAIN standardize h and z separately once, then concatenate [h_std, eta*z_std]; no second scaler; select eta and alpha jointly; identity included',
    'controls': 'random: 2048 unit Gaussian directions, TRAIN mean, raw-value TopK32, three paired seeds, nominal dimension/K/search budget matched, not activation frequency/correlation/semantics. linear: SAME SAE encoder pre-TopK affine activations; deterministic affine function of h; can only change linear-readout regularization, not function class.',
    'fixed_equivalence': 'q=Dz, h=mean+q+r. h+alpha*q=(1+alpha)*(mean+q+r/(1+alpha))-alpha*mean for alpha!=-1. With TRAIN channel scaling + intercept the paired mixtures yield identical standardized coordinates up to roundoff. alpha=-1 isolates r; lambda=0 isolates reconstruction. The two grids overlap and are not independent corroboration.',
    'statistics': 'primary original-vs-concat_SAE and concat_SAE-vs-concat_random/linear; TEST MacroF1 averaged over 3 seeds within each of 6 donors, then donor mean/paired bootstrap interval. Only 6 donors; overlapping training sets; descriptive intervals, no independent-cell inference. PBMC has one dataset, repeated fold/seed fits are not independent external datasets.',
    'stop_rule': 'complete predefined grids on all 18 frozen checkpoints; no test-guided retuning or additional complex networks',
}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)

def read(p): return json.loads(Path(p).read_text())

def immutable(p,v):
    if Path(p).exists(): assert read(p)==v, str(p)
    else: save(p,v)

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

def standardize(train,other):
    mu=np.asarray(train,dtype=np.float64).mean(0);sd=np.asarray(train,dtype=np.float64).std(0)
    sd[sd<1e-12]=1.
    return (train-mu)/sd,(other-mu)/sd,mu,sd

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

def score_model(p,x):
    with np.load(p) as a: return mm((x-a['mean'])/a['scale'],a['coef'])+a['intercept']

def src_for(fi,seed):
    fold=read(R/'protocol.json')['folds'][fi]['fold'];return R/f"test_{fold['test']}"/f'seed_{seed}'/'k_32'

def load(fi,seed,roles=('train','valid')):
    proto=read(R/'protocol.json');fold=proto['folds'][fi]['fold'];src=src_for(fi,seed)
    a=np.load(R/'kang_embeddings.npz');cache=np.load(src/'sae_outputs.npz')
    state=torch.load(src/'sae.pt',map_location='cpu',weights_only=False)
    state={k:v.detach().cpu().numpy().astype(np.float64) for k,v in state.items()};out={}
    for role in roles:
        if role=='pbmc':
            b=np.load(R/'pbmc_embeddings.npz');h=b['embedding'].astype(np.float64);y=b['labels'].astype(np.int64);rows=np.arange(len(h));don=np.full(len(h),'PBMC')
        else:
            ix=np.flatnonzero(~np.isin(a['donors'],[fold['valid'],fold['test']])) if role=='train' else np.flatnonzero(a['donors']==fold[role])
            h=a['embedding'][ix].astype(np.float64);y=a['labels'][ix].astype(np.int64);rows=a['rows'][ix];don=a['donors'][ix]
        assert np.array_equal(rows,cache[role+'_rows']) and np.array_equal(y,cache[role+'_labels'])
        z=np.zeros((len(h),2048),dtype=np.float64);np.put_along_axis(z,cache[role+'_indices'],cache[role+'_values'],axis=1)
        q=mm(z,state['decoder.weight'],tb=True);rec=q+state['mean'];r=h-rec
        err=float(np.max(np.abs(rec-cache[role+'_reconstruction'])))
        assert err<1e-5,(fi,seed,role,err)
        out[role]={'h':h,'y':y,'rows':rows,'donor':don,'z':z,'q':q,'recon':rec,'r':r,'cache_error':err}
    if 'train' in out:
        assert np.array_equal(out['train']['h'].mean(0).astype(np.float32),state['mean'].astype(np.float32))
    return out,state,fold

def latent(b,state,kind,fi,seed):
    if kind=='SAE':return b['z']
    if kind=='linear':return mm(b['h']-state['mean'],state['encoder.weight'],tb=True)+state['encoder.bias']
    rng=np.random.default_rng(np.random.SeedSequence([20261019,fi,seed,32]));e=rng.normal(size=(2048,512));e/=np.linalg.norm(e,axis=1,keepdims=True)
    a=mm(b['h']-state['mean'],e,tb=True);ix=np.argpartition(a,-32,axis=1)[:,-32:]
    z=np.zeros_like(a);np.put_along_axis(z,ix,np.take_along_axis(a,ix,axis=1),axis=1);return z

def fixed_features(b,kind,param=None):
    if kind=='original_h':return b['h']
    if kind=='sparse_z':return b['z']
    if kind=='reconstruction':return b['recon']
    if kind=='reconstruction_error':return b['r']
    if kind=='fixed_Dz':return b['h']+param*b['q']
    if kind=='residual_mix':return b['recon']+param*b['r']
    raise ValueError(kind)

def binding():return {'protocol_sha':sha(O/'PROTOCOL.json'),'sources_sha':sha(W/'frozen_sources.json')}

def check():
    assert read(O/'PROTOCOL.json')==CFG
    for p,h in read(W/'frozen_sources.json').items(): assert sha(p)==h,p

def completed(p):
    if not Path(p).exists():return False
    v=read(p);assert v['binding']==binding()
    for f,h in v['hashes'].items():assert sha(f)==h,f
    return True

def complete(out,start):
    save(out/'completion.json',{'status':'complete','binding':binding(),'seconds':time.time()-start,
          'hashes':{str(p.resolve()):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='completion.json'}})
