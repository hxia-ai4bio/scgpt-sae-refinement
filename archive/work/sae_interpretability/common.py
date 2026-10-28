"""Read-only access to historical experiments; new artifacts stay in this run.

Dense activation storage O(n * 2048); model forward O(B * L * (s*d*d+s*s*d)).
No fitting or modification of source models is performed.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys, os
import numpy as np
import torch
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE/'numba_cache'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'matplotlib_cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')
OUT = HERE.parents[1] / 'outputs' / 'sae_interpretability'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7')
OLD = ROOT / 'experiments/sae_topk_20261014'
sys.path.insert(0, str(OLD))
from sae import TopKSAE
torch.set_num_threads(4)
torch.backends.mha.set_fastpath_enabled(False)

def js(path, obj):
    def default(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.generic): return o.item()
        if isinstance(o, Path): return str(o)
        raise TypeError(type(o).__name__)
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=default, allow_nan=False))
    tmp.replace(path)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def load_npz(path):
    with np.load(path, allow_pickle=False) as a: return {k:a[k] for k in a.files}

def load_sae(fold='patient_1016', seed=42):
    path=OLD / ('test_'+fold) / ('seed_'+str(seed)) / 'k_32/sae.pt'
    state=torch.load(path, map_location='cpu', weights_only=True)
    model=TopKSAE(state['mean'],32,seed); model.load_state_dict(state)
    return model.eval().requires_grad_(False), path

@torch.inference_mode()
def encode(model, h):
    return np.concatenate([model.encode(torch.from_numpy(h[i:i+256]))[0].numpy()
                           for i in range(0,len(h),256)])

def auc_columns(z, y):
    """Tie-aware Mann-Whitney AUC, no scipy asymptotic cell-level p values."""
    n1=int(y.sum()); n0=len(y)-n1
    if not n1 or not n0: return np.full(z.shape[1],np.nan)
    r=rankdata(z,axis=0,method='average')
    return (r[y].sum(0)-n1*(n1+1)/2)/(n1*n0)

def corr_columns(x, v):
    x=x.astype(np.float64); v=np.asarray(v,dtype=np.float64)
    x-=x.mean(0); v=v-v.mean()
    den=np.sqrt(np.einsum('ij,ij->j',x,x)*np.dot(v,v))
    return np.divide(np.einsum('ij,i->j',x,v),den,out=np.zeros(x.shape[1]),where=den>1e-12)

def residualize(x, design):
    """SVD basis handles absent/constant QC fields without projecting arbitrary axes."""
    u,s,v=torch.linalg.svd(torch.as_tensor(design,dtype=torch.float64),full_matrices=False)
    q=u[:,s>s.max()*1e-10]
    t=torch.as_tensor(x,dtype=torch.float64)
    return (t-q@(q.T@t)).numpy()

def prep_module():
    spec=importlib.util.spec_from_file_location('historical_prepare',OLD/'prepare.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def dataset():
    manifest=json.loads((OLD/'data_manifest.json').read_text())
    return load_npz(OLD/'kang_embeddings.npz'),load_npz(OLD/'pbmc_embeddings.npz'),manifest

def raw_data():
    prep=prep_module()
    manifest=json.loads((OLD/'data_manifest.json').read_text())
    cfg=manifest['preprocess']
    x,g,meta=prep.read(cfg['dataset'], True)
    px,pg,pm=prep.read('/Users/apple/Desktop/Llava-Cell/dataset/PBMC368k_raw.h5ad',False)
    return prep,cfg,x,g,meta,px,pg,pm

def qc(x, genes, rows):
    xx=x[rows]; total=np.asarray(xx.sum(1)).ravel()
    mt=np.char.startswith(genes.astype(str),'MT-')
    return np.column_stack([np.log1p(total), np.log1p(xx.getnnz(1)),
                            np.asarray(xx[:,mt].sum(1)).ravel()/np.maximum(total,1)])

def bh(p):
    p=np.asarray(p,float); order=np.argsort(p); q=np.empty_like(p)
    q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    return np.minimum(q,1)
