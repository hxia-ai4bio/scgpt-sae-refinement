"""Read back real embeddings and labels; selected Ridge independently via Cholesky."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import torch
from threadpoolctl import threadpool_limits

W=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lambda_experiment',W/'run.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def metric(y,s):
    cm=np.bincount(8*y+s.argmax(1),minlength=64).reshape(8,8)
    den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(8,dtype=float),where=den>0)
    return f,cm

def main():
    torch.set_num_threads(4);e.validate()
    seal=e.read(e.O/'SELECTION_LOCK.json');assert seal['binding']==e.runtime.binding()
    for p,h in seal['hashes'].items():assert e.sha(p)==h,p
    records=e.read(e.O/'all_metrics.json');assert len(records)==132
    checks=[];embeddings=[];training=[];groups={}
    for fi in range(6):
        d,fold=e.base.data(fi,('train','valid','test'));tr=d['train'];va=d['valid'];te=d['test']
        for method,seed,p in e.selections(fi):
            r=next(r for r in records if r['fold']==fi and r['method']==method and str(r['seed'])==str(seed))
            sel=e.read(p/'selection.json');m=np.load(p/'model.npz')
            if method=='original_h':x={role:b['h'].astype('float64') for role,b in d.items()}
            else:
                model=e.restore(fi,seed,method)
                x={role:e.base.refine.infer(model,b['h']).astype('float64') for role,b in d.items()}
                state=model.state_dict();initial=e.base.initial(fi,seed).state_dict()
                t=e.read(e.checkpoint(fi,seed,method).parent/'training.json')
                assert t['lambda_KL']==e.METHODS[method] and t['additional_steps']==t['selected_step']==1000
                assert t['checkpoint_sha']==e.sha(e.checkpoint(fi,seed,method))
                assert t['start_checkpoint_sha']==e.sha(e.old.src_for(fi,seed)/'sae.pt')
                delta={k:float((v-initial[k]).double().norm()) for k,v in state.items()}
                assert delta==t['weight_delta_norms'] and delta['mean']==0
                assert all(v>0 for k,v in delta.items() if k!='mean')
                np.testing.assert_array_equal(state['mean'].numpy(),tr['h'].astype('float64').mean(0).astype('float32'))
                assert float((state['decoder.weight'].double().norm(dim=0)-1).abs().max())<2e-6
                home=e.W if method in e.NEW else e.base.W
                batch=home/'batches'/f'fold_{fi}_seed_{seed}.npz';assert t['batch_sha']==e.sha(batch)
                with np.load(batch) as a,np.load(e.base.W/'batches'/f'fold_{fi}_seed_{seed}.npz') as b:
                    np.testing.assert_array_equal(a['indices'],b['indices']);np.testing.assert_array_equal(a['rows'],tr['rows'][a['indices']])
                assert [g['step'] for g in t['gradient_diagnostics']]==[1,250,500,750,1000]
                for g in t['gradient_diagnostics']:
                    assert g['mse_gradient_norm']>0 and g['kl_gradient_norm']>0
                    assert abs(g['weighted_gradient_ratio']-t['lambda_KL']*g['kl_gradient_norm']/g['mse_gradient_norm'])<1e-12
                    assert all(np.isfinite(z) and z>0 for z in g['total_parameter_gradient_norms'].values())
                for role in ['train','valid']:
                    err=float(np.mean((x[role]-d[role]['h'].astype('float64'))**2))
                    assert abs(err-t['monitor_history'][-1][role+'_raw_mse'])<1e-12
                training.append(dict(fold=fi,seed=seed,method=method,lambda_KL=t['lambda_KL'],reused=method not in e.NEW,actual_checkpoint_sha=t['checkpoint_sha'],weight_delta_norms=delta))
            ep=e.O/'embeddings'/f'fold_{fi}'/f'seed_{seed}'/f'{method}.npz'
            with np.load(ep) as b:
                np.testing.assert_array_equal(b['embedding'],x['test']);np.testing.assert_array_equal(b['rows'],te['rows']);np.testing.assert_array_equal(b['labels'],te['y'])
                assert b['embedding'].shape==(len(te['h']),512) and np.isfinite(b['embedding']).all()
            embeddings.append(dict(fold=fi,seed=seed,method=method,path=str(ep.relative_to(e.O)),sha256=e.sha(ep),shape=list(x['test'].shape)))
            pp=e.O/'predictions'/f'fold_{fi}'/f'seed_{seed}'/method/'test.npz'
            with np.load(pp) as pred:
                np.testing.assert_array_equal(pred['rows'],te['rows']);np.testing.assert_array_equal(pred['labels'],te['y'])
                f,cm=metric(te['y'],pred['scores']);np.testing.assert_allclose(f,r['per_class_f1'],atol=1e-14,rtol=0);np.testing.assert_array_equal(cm,r['confusion_matrix'])
                assert abs(f.mean()-r['macro_f1'])<1e-14
                explicit=np.einsum('ni,ij->nj',(x['test']-m['mean'])/m['scale'],m['coef'],optimize=False)+m['intercept']
                scoreerr=float(np.max(np.abs(explicit-pred['scores'])));assert scoreerr<1e-8
                if method not in e.NEW:
                    variant='original' if method=='original_h' else 'reconstruction'
                    with np.load(e.base.O/'predictions'/f'fold_{fi}'/f'seed_{seed}'/method/variant/'test.npz') as oldpred:
                        np.testing.assert_array_equal(oldpred['rows'],pred['rows']);np.testing.assert_array_equal(oldpred['labels'],pred['labels'])
                        np.testing.assert_allclose(oldpred['scores'],pred['scores'],atol=1e-10,rtol=0)
            # Fit the chosen alpha independently with Cholesky instead of executor eigensystem.
            mu=x['train'].mean(0);sd=x['train'].std(0);sd[sd<1e-12]=1
            np.testing.assert_allclose(mu,m['mean'],atol=1e-14,rtol=0);np.testing.assert_allclose(sd,m['scale'],atol=1e-14,rtol=0)
            counts=np.bincount(tr['y'],minlength=8);assert (counts>0).all()
            w=len(tr['y'])/(8*counts[tr['y']]);a=(x['train']-mu)/sd;b=2*np.eye(8)[tr['y']]-1
            xm=np.average(a,axis=0,weights=w);ym=np.average(b,axis=0,weights=w)
            aa=(a-xm)*np.sqrt(w[:,None]);bb=(b-ym)*np.sqrt(w[:,None])
            gram=e.old.mm(aa,aa,ta=True);gram=(gram+gram.T)/2;gram.flat[::513]+=float(m['alpha'])
            coef=cho_solve(cho_factor(gram,lower=True),e.old.mm(aa,bb,ta=True));intercept=ym-np.einsum('i,ij->j',xm,coef)
            ref=np.einsum('ni,ij->nj',(x['valid']-mu)/sd,coef,optimize=False)+intercept
            with np.load(p/'valid.npz') as v:
                np.testing.assert_array_equal(v['rows'],va['rows']);np.testing.assert_array_equal(v['labels'],va['y'])
                verror=float(np.max(np.abs(ref-v['scores'])));assert verror<1e-7
                vf,_=metric(va['y'],v['scores']);assert abs(vf.mean()-sel['VALID_F1'])<1e-14
            trials=e.read(p/'trials.json');best=max(t['VALID_F1'] for t in trials)
            assert [t['ridge_alpha'] for t in trials]==e.old.ALPHAS
            assert next(t for t in trials if t['VALID_F1']>=best-1e-12)['ridge_alpha']==sel['ridge_alpha']
            checks.append(dict(fold=fi,seed=seed,method=method,TEST_einsum_score_max_error=scoreerr,VALID_independent_cholesky_score_max_error=verror))
            groups.setdefault(method,{}).setdefault(fi,[]).append(float(f.mean()))
        print('VERIFIED_FOLD',fi,flush=True)
    assert len(training)==126 and sum(not t['reused'] for t in training)==72
    for row in e.read(e.O/'summary.json'):
        vals=np.array([np.mean(groups[row['method']][f]) for f in range(6)])
        assert abs(row['macro_f1_percent']-100*vals.mean())<1e-12
    original=next(r for r in e.read(e.O/'summary.json') if r['method']=='original_h')
    assert abs(original['macro_f1_percent']-82.7539600326646)<1e-10
    e.save(e.O/'embedding_manifest.json',embeddings);e.save(e.O/'training_verification.json',training)
    e.save(e.O/'verification.json',dict(status='PASS',prediction_models=132,training_runs=126,new_training_runs=72,reused_training_runs=54,embedding_files_read_back=len(embeddings),selected_probe_checks=checks,batches_paired=True,reused_TEST_predictions_reproduced=True,selection_lock_sha=e.sha(e.O/'SELECTION_LOCK.json')))
    e.tracker('VERIFIED_PENDING_REVIEW',verification='PASS',review='pending')
    print('VERIFICATION_PASS 132selectedmodels;126training;72new',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
