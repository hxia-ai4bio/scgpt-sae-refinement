"""Verify exported real embeddings, paired updates, and independently fitted Ridge."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
import torch
from threadpoolctl import threadpool_limits

W=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('extension_experiment',W/'run.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def metrics(y,scores):
    cm=np.bincount(8*y+scores.argmax(1),minlength=64).reshape(8,8);den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(8,dtype=float),where=den>0)
    return f,cm

def main():
    torch.set_num_threads(4);e.validate();seal=e.read(e.O/'SELECTION_LOCK.json')
    for p,h in seal['hashes'].items():assert e.sha(p)==h,p
    for ds,h in seal['per_dataset_locks'].items():assert e.sha(e.dsout(ds)/'SELECTION_LOCK.json')==h
    records=e.read(e.O/'all_metrics.json');assert len(records)==176
    checks=[];embedding_manifest=[];trainchecks=[];warmups=[];groups={}
    for ds in e.DATASETS:
        rt=e.configure(ds);n=e.nfold(ds)
        for fi in range(n):
            d,fold=e.data(ds,fi,('train','valid','test'));tr=d['train'];va=d['valid'];te=d['test']
            # Actual persisted arrays must match the dataset's real group semantics.
            for seed in e.SEEDS:
                ix,mask,batchpath=rt.batches(fi,seed,tr)
                assert ix.shape==(1000,256) and (mask.sum(1)==e.cfg(ds)['relation_anchor_count']).all()
                with np.load(batchpath) as a:np.testing.assert_array_equal(a['rows'],tr['rows'][ix])
                if ds=='stim':
                    for j,g in enumerate(np.unique(tr['donor'])):
                        block=ix[:,j*64:(j+1)*64];assert (tr['donor'][block]==g).all()
                        assert all(len(set(row))==64 for row in block)
                    assert set(tr['donor']).isdisjoint(va['donor']) and set(tr['donor']).isdisjoint(te['donor'])
                else:
                    assert all(len(set(row))==256 for row in ix)
                    assert not mask.diag().any() and len(set(tr['donor']))==1
                    assert set(tr['donor'])==set(va['donor']) and set(tr['donor']).isdisjoint(te['donor'])
                init=rt.initial(fi,seed);state=init.state_dict();p=e.initialdir(ds,fi,seed)
                t=e.read(p/'training.json');assert rt.checked_completion(p/'completion.json')
                assert t['dataset']==ds and t['checkpoint_sha']==e.sha(p/'sae.pt')
                assert t['train_input_sha']==e.base.array_sha(tr['h']) and t['valid_input_sha']==e.base.array_sha(va['h'])
                assert 0<t['selected_epoch']<=20 and t['selected_updates']==t['history'][t['selected_epoch']]['updates']
                best=min(r['valid_mse'] for r in t['history']);assert t['selected_epoch']==next(r['epoch'] for r in t['history'] if r['valid_mse']==best)
                mu=tr['h'].astype('float64').mean(0).astype('float32');np.testing.assert_array_equal(state['mean'].numpy(),mu)
                rec=e.base.refine.infer(init,va['h']);assert abs(float(np.mean((rec.astype('float64')-va['h'].astype('float64'))**2))-best)<1e-12
                assert t['changed_weight_norms']['mean']==0 and all(v>0 for k,v in t['changed_weight_norms'].items() if k!='mean')
                warmups.append(dict(dataset=ds,fold=fi,seed=seed,selected_epoch=t['selected_epoch'],selected_updates=t['selected_updates'],checkpoint_sha=t['checkpoint_sha']))
            for method,seed,p in e.selections(ds,fi):
                r=next(z for z in records if z['dataset']==ds and z['fold']==fi and z['method']==method and str(z['seed'])==str(seed))
                sel=e.read(p/'selection.json');m=np.load(p/'model.npz')
                if method=='original_h':x={role:b['h'].astype('float64') for role,b in d.items()}
                else:
                    model=rt.restore(fi,seed,method);x={role:e.base.refine.infer(model,b['h']).astype('float64') for role,b in d.items()}
                    state=model.state_dict();initial=rt.initial(fi,seed).state_dict();tp=rt.network_dir(fi,seed,method);t=e.read(tp/'training.json')
                    assert t['lambda_KL']==e.ARMS[method] and t['additional_steps']==t['selected_step']==1000 and t['optimizer_reset']
                    assert t['start_checkpoint_sha']==e.sha(e.initialdir(ds,fi,seed)/'sae.pt') and t['checkpoint_sha']==e.sha(tp/'sae.pt')
                    assert t['batch_sha']==e.sha(rt.W/'batches'/f'fold_{fi}_seed_{seed}.npz')
                    delta={k:float((v-initial[k]).double().norm()) for k,v in state.items()};assert delta==t['weight_delta_norms']
                    assert delta['mean']==0 and all(v>0 for k,v in delta.items() if k!='mean')
                    np.testing.assert_array_equal(state['mean'].numpy(),tr['h'].astype('float64').mean(0).astype('float32'))
                    assert float((state['decoder.weight'].double().norm(dim=0)-1).abs().max())<2e-6
                    assert [z['step'] for z in t['gradient_diagnostics']]==[1,250,500,750,1000]
                    for g in t['gradient_diagnostics']:
                        assert g['mse_gradient_norm']>0 and g['kl_gradient_norm']>0
                        assert abs(g['weighted_gradient_ratio']-t['lambda_KL']*g['kl_gradient_norm']/g['mse_gradient_norm'])<1e-12
                        assert all(np.isfinite(v) and v>0 for v in g['total_parameter_gradient_norms'].values())
                    for role in ['train','valid']:
                        err=float(np.mean((x[role]-d[role]['h'].astype('float64'))**2));assert abs(err-t['monitor_history'][-1][role+'_raw_mse'])<1e-12
                    trainchecks.append(dict(dataset=ds,fold=fi,seed=seed,method=method,weight_delta_norms=delta,checkpoint_sha=t['checkpoint_sha']))
                ep=e.dsout(ds)/'embeddings'/f'fold_{fi}'/f'seed_{seed}'/f'{method}.npz'
                with np.load(ep) as a:
                    np.testing.assert_array_equal(a['embedding'],x['test']);np.testing.assert_array_equal(a['rows'],te['rows']);np.testing.assert_array_equal(a['labels'],te['y']);np.testing.assert_array_equal(a['groups'],te['donor'])
                    assert a['embedding'].shape==(len(te['h']),512) and np.isfinite(a['embedding']).all()
                embedding_manifest.append(dict(dataset=ds,fold=fi,seed=seed,method=method,path=str(ep.relative_to(e.O)),sha256=e.sha(ep),shape=list(x['test'].shape)))
                pp=e.dsout(ds)/'predictions'/f'fold_{fi}'/f'seed_{seed}'/method/'test.npz'
                with np.load(pp) as pred:
                    np.testing.assert_array_equal(pred['rows'],te['rows']);np.testing.assert_array_equal(pred['labels'],te['y'])
                    f,cm=metrics(te['y'],pred['scores']);np.testing.assert_allclose(f,r['per_class_f1'],atol=1e-14,rtol=0);np.testing.assert_array_equal(cm,r['confusion_matrix']);assert abs(f.mean()-r['macro_f1'])<1e-14
                    scores=np.einsum('ni,ij->nj',(x['test']-m['mean'])/m['scale'],m['coef'],optimize=False)+m['intercept'];serr=float(abs(scores-pred['scores']).max());assert serr<1e-8
                    relative=float(np.linalg.norm(x['test']-te['h'])/np.linalg.norm(te['h'].astype('float64')));assert abs(relative-r['relative_embedding_change'])<1e-12
                # Independent chosen-alpha Cholesky: classbalance uses actual TRAIN classes.
                mu=x['train'].mean(0);sd=x['train'].std(0);sd[sd<1e-12]=1
                np.testing.assert_allclose(mu,m['mean'],atol=1e-14,rtol=0);np.testing.assert_allclose(sd,m['scale'],atol=1e-14,rtol=0)
                counts=np.bincount(tr['y'],minlength=8);assert (counts>0).all();w=len(tr['y'])/(8*counts[tr['y']])
                a=(x['train']-mu)/sd;b=2*np.eye(8)[tr['y']]-1;xm=np.average(a,axis=0,weights=w);ym=np.average(b,axis=0,weights=w)
                aa=(a-xm)*np.sqrt(w[:,None]);bb=(b-ym)*np.sqrt(w[:,None]);gram=e.old.mm(aa,aa,ta=True);gram=(gram+gram.T)/2;gram.flat[::513]+=float(m['alpha'])
                coef=cho_solve(cho_factor(gram,lower=True),e.old.mm(aa,bb,ta=True));intercept=ym-np.einsum('i,ij->j',xm,coef)
                ref=np.einsum('ni,ij->nj',(x['valid']-mu)/sd,coef,optimize=False)+intercept
                with np.load(p/'valid.npz') as v:
                    np.testing.assert_array_equal(v['rows'],va['rows']);np.testing.assert_array_equal(v['labels'],va['y']);verr=float(abs(ref-v['scores']).max());assert verr<1e-7
                    vf,_=metrics(va['y'],v['scores']);assert abs(vf.mean()-sel['VALID_F1'])<1e-14
                trials=e.read(p/'trials.json');assert [z['ridge_alpha'] for z in trials]==e.old.ALPHAS
                best=max(z['VALID_F1'] for z in trials);assert next(z for z in trials if z['VALID_F1']>=best-1e-12)['ridge_alpha']==sel['ridge_alpha']
                checks.append(dict(dataset=ds,fold=fi,seed=seed,method=method,TEST_einsum_max_error=serr,VALID_cholesky_max_error=verr))
                groups.setdefault((ds,method),{}).setdefault(fi,[]).append(float(f.mean()))
            print('VERIFIED_FOLD',ds,fi,flush=True)
    assert len(trainchecks)==168 and len(warmups)==24
    for row in e.read(e.O/'summary.json'):
        values=[np.mean(groups[(row['dataset'],row['method'])][fi]) for fi in range(e.nfold(row['dataset']))]
        assert abs(row['macro_f1_percent']-100*np.mean(values))<1e-12
    e.save(e.O/'embedding_manifest.json',embedding_manifest);e.save(e.O/'warmup_verification.json',warmups);e.save(e.O/'training_verification.json',trainchecks)
    e.save(e.O/'verification.json',dict(status='PASS',prediction_models=176,continuation_training_runs=168,warmup_training_runs=24,embedding_files_read_back=len(embedding_manifest),selected_probe_checks=checks,group_semantics_verified=True,selection_lock_sha=e.sha(e.O/'SELECTION_LOCK.json')))
    e.update('VERIFIED_PENDING_REVIEW',verification='PASS',review='pending');print('VERIFICATION_PASS 176models,168continuation,24warmups',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
