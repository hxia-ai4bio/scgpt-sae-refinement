"""Readback of exported embeddings, real-label metric checks and Cholesky probes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor,cho_solve
import torch
from threadpoolctl import threadpool_limits

W=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('relation_experiment',W/'run.py')
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)


def manual_metrics(y,scores):
    pred=scores.argmax(1);cm=np.bincount(8*y+pred,minlength=64).reshape(8,8)
    den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(8,dtype=float),where=den>0)
    return f,cm


def features(fi,seed,method,parts):
    if method=='original_h':return {r:b['h'].astype(np.float64) for r,b in parts.items()}
    if method=='existing_raw20':return exp.refine.cached(fi,int(seed),tuple(parts))
    m=exp.restore(fi,int(seed),method)
    return {r:exp.refine.infer(m,b['h']) for r,b in parts.items()}


def main():
    torch.set_num_threads(4)
    exp.validate_sources()
    sealed=exp.read(exp.O/'SELECTION_LOCK.json')
    assert sealed['binding']==exp.binding()
    for p,h in sealed['hashes'].items():assert exp.old.sha(p)==h,p
    records=exp.read(exp.O/'all_metrics.json');assert len(records)==150
    checks=[];embedding_files=[];trainchecks=[]
    for fi in range(6):
        parts,fold=exp.data(fi,('train','valid','test'))
        tc={};b=parts['test'];train=parts['train'];valid=parts['valid']
        for rec in [r for r in records if r['fold']==fi]:
            seed,method,variant=(rec[k] for k in ['seed','method','variant'])
            predpath=exp.O/'predictions'/f'fold_{fi}'/f'seed_{seed}'/method/variant
            selpath=exp.W/'probes'/f'fold_{fi}'/f'seed_{seed}'/method/variant
            pred=np.load(predpath/'test.npz');model=np.load(selpath/'model.npz');sel=exp.read(selpath/'selection.json')
            np.testing.assert_array_equal(pred['rows'],b['rows']);np.testing.assert_array_equal(pred['labels'],b['y'])
            assert np.isfinite(pred['scores']).all()
            f,cm=manual_metrics(pred['labels'],pred['scores'])
            np.testing.assert_allclose(f,rec['per_class_f1'],atol=1e-14,rtol=0)
            np.testing.assert_array_equal(cm,rec['confusion_matrix'])
            assert abs(float(f.mean())-rec['macro_f1'])<1e-14
            key=(method,seed)
            if key not in tc:tc[key]=features(fi,seed,method,parts)
            feat=tc[key]
            if method=='original_h':x={r:parts[r]['h'].astype(np.float64) for r in parts}
            else:x={r:parts[r]['h'].astype(np.float64)+sel['mix_alpha']*(feat[r].astype(np.float64)-parts[r]['h']) for r in parts}
            p=exp.O/'embeddings'/f'fold_{fi}'/f'seed_{seed}'/method/f'{variant}.npz'
            with np.load(p) as emb:
                np.testing.assert_array_equal(emb['embedding'],x['test'])
                np.testing.assert_array_equal(emb['rows'],b['rows'])
                np.testing.assert_array_equal(emb['labels'],b['y'])
                assert emb['embedding'].shape==(len(b['h']),512) and np.isfinite(emb['embedding']).all()
            embedding_files.append(str(p.relative_to(exp.O)))
            coef_scores=np.einsum('ni,ij->nj',(x['test']-model['mean'])/model['scale'],model['coef'],optimize=False)+model['intercept']
            scoreerr=float(np.max(np.abs(coef_scores-pred['scores'])));assert scoreerr<1e-8
            # Independent Cholesky, including weighted centering / intercept, of the selected alpha.
            mu=x['train'].mean(0);sd=x['train'].std(0);sd[sd<1e-12]=1.
            np.testing.assert_allclose(mu,model['mean'],atol=1e-14,rtol=0)
            np.testing.assert_allclose(sd,model['scale'],atol=1e-14,rtol=0)
            counts=np.bincount(train['y'],minlength=8);assert (counts>0).all()
            weights=len(train['y'])/(8*counts[train['y']])
            s=(x['train']-mu)/sd;target=2*np.eye(8)[train['y']]-1
            xm=np.average(s,axis=0,weights=weights);ym=np.average(target,axis=0,weights=weights)
            a=(s-xm)*np.sqrt(weights[:,None]);bb=(target-ym)*np.sqrt(weights[:,None])
            gram=exp.old.mm(a,a,ta=True);gram=(gram+gram.T)/2
            gram.flat[::513]+=float(model['alpha'])
            rhs=exp.old.mm(a,bb,ta=True)
            coef=cho_solve(cho_factor(gram,lower=True),rhs)
            intercept=ym-np.einsum('i,ij->j',xm,coef)
            ref=np.einsum('ni,ij->nj',(x['valid']-mu)/sd,coef,optimize=False)+intercept
            saved=np.load(selpath/'valid.npz')
            np.testing.assert_array_equal(saved['labels'],valid['y']);np.testing.assert_array_equal(saved['rows'],valid['rows'])
            err=float(np.max(np.abs(ref-saved['scores'])));assert err<1e-7,err
            vf,_=manual_metrics(valid['y'],saved['scores']);assert abs(vf.mean()-sel['VALID_F1'])<1e-14
            trials=exp.read(selpath/'trials.json');best=max(t['VALID_F1'] for t in trials)
            winner=next(t for t in trials if t['VALID_F1']>=best-1e-12)
            assert winner['ridge_alpha']==sel['ridge_alpha'] and abs(best-sel['VALID_F1'])<1e-12
            checks.append(dict(fold=fi,seed=seed,method=method,variant=variant,TEST_einsum_score_max_error=scoreerr,VALID_independent_cholesky_score_max_error=err))
        for seed in exp.SEEDS:
            batch=exp.W/'batches'/f'fold_{fi}_seed_{seed}.npz'
            with np.load(batch) as a:
                ix=a['indices'];assert ix.shape==(1000,256)
                np.testing.assert_array_equal(a['rows'],train['rows'][ix])
                for start,donor in zip([0,64,128,192],np.unique(train['donor'])):
                    assert (train['donor'][ix[:,start:start+64]]==donor).all()
                    assert all(len(np.unique(row))==64 for row in ix[:,start:start+64])
            for arm,lam in exp.ARMS.items():
                path=exp.network_dir(fi,seed,arm);r=exp.read(path/'training.json')
                assert r['additional_steps']==1000 and r['selected_step']==1000 and r['lambda_KL']==lam and r['optimizer_reset']
                assert r['batch_sha']==exp.old.sha(batch)
                assert r['checkpoint_sha']==exp.old.sha(path/'sae.pt')
                assert r['decoder_norm_error']<2e-6 and r['weight_delta_norms']['mean']==0
                assert all(v>0 for k,v in r['weight_delta_norms'].items() if k!='mean')
                assert [h['step'] for h in r['monitor_history']]==[0,250,500,750,1000]
                assert [h['step'] for h in r['gradient_diagnostics']]==[1,250,500,750,1000]
                assert all(g['mse_gradient_norm']>0 and g['kl_gradient_norm']>0 for g in r['gradient_diagnostics'])
                assert all(abs(g['weighted_gradient_ratio']-lam*g['kl_gradient_norm']/g['mse_gradient_norm'])<1e-12 for g in r['gradient_diagnostics'])
                trainchecks.append({k:r[k] for k in ['fold','seed','arm','lambda_KL','additional_steps','seconds','weight_delta_norms','decoder_norm_error','monitor_history','gradient_diagnostics']})
        print('VERIFIED_FOLD',fi,flush=True)
    assert len(trainchecks)==54
    summary=exp.read(exp.O/'summary.json')
    original=next(s for s in summary if s['method']=='original_h')
    oldrec=next(s for s in summary if s['method']=='existing_raw20' and s['variant']=='reconstruction')
    assert abs(original['macro_f1_percent']-82.7539600326646)<1e-10
    assert abs(oldrec['macro_f1_percent']-82.26544439130096)<1e-10
    exp.save(exp.O/'training_summary.json',trainchecks)
    exp.save(exp.O/'embedding_manifest.json',dict(status='readback_PASS',files=embedding_files,count=len(embedding_files),role='held-out TEST donor only per fold; no full-data refit',dtype='float64',dimension=512))
    exp.save(exp.O/'verification.json',dict(status='PASS',prediction_models=150,training_runs=54,embedding_files_read_back=150,checks=checks,batches_paired=True,baseline_reproduction=True,selection_lock_sha=exp.old.sha(exp.O/'SELECTION_LOCK.json')))
    tracker=exp.read(exp.O/'EXPERIMENT_TRACKER.json');tracker.update(status='VERIFIED_PENDING_REVIEW',verification='PASS')
    exp.save(exp.O/'EXPERIMENT_TRACKER.json',tracker)
    print('VERIFICATION_PASS',len(checks),'selected models;',len(trainchecks),'training runs',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
