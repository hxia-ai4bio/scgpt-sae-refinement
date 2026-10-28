"""TRAIN-only geometry and gradient diagnostics; no optimizer, labels or TEST score.

Time O(folds*(N*d+B*d*m+B**2*d)), batch B=256, d=512, m=2048.
Memory O(N*d+B*m+B**2+d*m); only one fold/model resident.
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/sae_refine'))
import run as refine


def grad_norm(loss,params):
    grads=torch.autograd.grad(loss,params,retain_graph=True)
    return float(torch.sqrt(sum(g.double().square().sum() for g in grads)))


def main():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out=ROOT/'outputs/sae_relation_preflight'
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for fi in range(6):
        fold=refine.read(refine.old.R/'protocol.json')['folds'][fi]['fold']
        with np.load(refine.old.R/'kang_embeddings.npz') as a:
            indices=np.flatnonzero(~np.isin(a['donors'],[fold['valid'],fold['test']]))
            h=a['embedding'][indices].astype(np.float32)
            donors=a['donors'][indices]
        groups=np.unique(donors)
        rng=np.random.default_rng(20261019+fi)
        chosen=np.concatenate([rng.choice(np.flatnonzero(donors==g),64,replace=False) for g in groups])
        batch=torch.from_numpy(h[chosen])
        batchdon=donors[chosen]
        mask=torch.from_numpy(batchdon[:,None]!=batchdon[None,:])
        assert mask.sum(1).unique().tolist()==[192]
        state=torch.load(refine.old.src_for(fi,42)/'sae.pt',map_location='cpu',weights_only=False)
        mean=torch.from_numpy(h.astype(np.float64).mean(0).astype(np.float32))
        np.testing.assert_array_equal(mean.numpy(),state['mean'].numpy())
        model=refine.SAE(mean,2048,32,42)
        model.load_state_dict(state)
        rec,z=model(batch)
        total_variance=float(np.mean((h.astype(np.float64)-mean.numpy())**2))
        mse=(rec-batch).square().mean()/total_variance
        params=tuple(model.parameters())
        msegrad=grad_norm(mse,params)
        scores=[]
        for center in [False,True]:
            x=batch.double()-(mean.double() if center else 0.)
            y=rec.double()-(mean.double() if center else 0.)
            xn=F.normalize(x,dim=1);yn=F.normalize(y,dim=1)
            teacher=(xn@xn.T)[mask].reshape(256,192).detach()
            student=(yn@yn.T)[mask].reshape(256,192)
            for tau in [.05,.1,.2]:
                logp=F.log_softmax(teacher/tau,dim=1)
                logq=F.log_softmax(student/tau,dim=1)
                kl=(logp.exp()*(logp-logq)).sum(1).mean()
                entropy=-(logp.exp()*logp).sum(1)
                g=grad_norm(kl,params)
                scores.append(dict(centered=center,tau=tau,teacher_cosine_quantiles=torch.quantile(teacher.flatten(),torch.tensor([.05,.5,.95],dtype=torch.float64)).tolist(),mean_effective_neighbors=float(entropy.exp().mean()),relative_entropy=float((entropy/np.log(192)).mean()),kl=float(kl),kl_gradient_norm=g,normalized_mse_gradient_norm=msegrad,gradient_ratio_kl_over_mse=g/msegrad))
        # Histories only describe previously run models, not selection of a new method.
        histories=[]
        for seed in [42,43,44]:
            result=refine.read(refine.old.src_for(fi,seed)/'sae_result.json')
            first=next(e['valid_mse'] for e in result['history'] if e['epoch']==15)
            last=result['history'][-1]['valid_mse']
            histories.append(dict(seed=seed,best_epoch=result['best_epoch'],last_epoch=result['last_epoch'],relative_valid_mse_drop_last5=(first-last)/first))
        records.append(dict(fold=fi,train_donors=groups.tolist(),n_train=len(h),sampled_per_donor=64,batch_size=256,anchors_per_cell=192,normalized_reconstruction_mse=float(mse),centered_train_mean_square=total_variance,geometry=scores,existing_training=histories))
        print('TRAIN_GEOMETRY_CHECKED',fi,flush=True)
    result=dict(scope='read-only TRAIN geometry + gradients at old seed42 checkpoint; no optimizer steps, labels, VALID/TEST embeddings or new prediction scores',sources={'runner_sha':refine.old.sha(Path(__file__)),'embedding_sha':refine.old.sha(refine.old.R/'kang_embeddings.npz')},records=records)
    (out/'CHECKS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for center in [False,True]:
        r=[s for f in records for s in f['geometry'] if s['centered']==center and s['tau']==.1]
        print(json.dumps(dict(centered=center,tau=.1,mean_effective_neighbors=float(np.mean([s['mean_effective_neighbors'] for s in r])),mean_relative_entropy=float(np.mean([s['relative_entropy'] for s in r])),median_kl_to_mse_gradient_ratio=float(np.median([s['gradient_ratio_kl_over_mse'] for s in r])))))


if __name__=='__main__':main()
