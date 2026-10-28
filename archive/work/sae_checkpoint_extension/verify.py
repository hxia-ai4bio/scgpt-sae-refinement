"""Checkpoint trajectory, VALID selection, frozen readout and kNN replay."""
import importlib.util,os,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
W=Path(__file__).resolve().parent
os.environ['SAE_CHECKPOINT_DATASET']=sys.argv[1]
s=importlib.util.spec_from_file_location('selected_checkpoint_runtime',W/'run.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)

def main():
    r.check_inputs();lock=r.read(r.O/'SELECTION_LOCK.json');assert lock['source_sha']==r.sha(W/'run.py')
    global_lock=r.read(r.O.parent/'GLOBAL_SELECTION_LOCK.json')
    assert r.sha(r.O/'SELECTION_LOCK.json')==global_lock['dataset_lock_hashes'][r.DS]
    for p,h in lock['hashes'].items():assert r.sha(p)==h,p
    df=pd.read_csv(r.O/'all_metrics.csv');kn=pd.read_csv(r.O/'knn_metrics.csv');assert len(df)==r.N*3 and len(kn)==r.N*6
    old=pd.read_csv(r.q.O/'all_metrics.csv');oldkn=pd.read_csv(r.m.O/'all_metrics.csv');a=r.m.canonical(r.DS);checks=0;checkpoints=0;neighbors=0;headchecks=0
    provenance=r.read(r.m.W/'data'/r.DS/'binding.json')
    with np.load(provenance['canonical_path']) as z:
        for key in ['rows','embedding','labels','cell_ids','condition','groups']:np.testing.assert_array_equal(a[key],z[key])
    assert r.sha(provenance['raw_path'])==provenance['raw_sha']
    assert r.sha(provenance['canonical_path'])==provenance['canonical_sha']
    for p,h in provenance['sources'].items():assert r.sha(p)==h,p
    for fi in range(r.NF):
        d,f,ix=r.m.fold_data(a,r.DS,fi);pn=r.m.panel(r.DS,fi,a,ix);t=r.m.targets(a,pn)
        for seed in r.SEEDS:
            home=r.bodyhome(fi,seed);selection=r.read(home/'selection.json');assert [x['step'] for x in selection['history']]==r.STEPS
            assert selection['selected_step']==min(selection['history'],key=lambda x:x['valid_objective'])['step']
            assert selection['batch_array_sha']==r.m.arraysha(r.m.batches(r.DS,fi,seed,d))
            for item in selection['history']:
                model=r.restore_body(fi,seed,item['step']);valid=r.body_valid(model,a,t,ix['valid'],pn)
                for key in ['valid_objective','valid_MAE','entries']:assert abs(valid[key]-item[key])<1e-12
                state=model.state_dict();assert all(torch.isfinite(v).all() for v in state.values());np.testing.assert_allclose(state['network.decoder.weight'].norm(dim=0).numpy(),1,rtol=0,atol=2e-6)
                if item['step']==1000:
                    prior=torch.load(r.m.W/'networks'/r.DS/f'fold_{fi}'/f'seed_{seed}'/'C_sae/model.pt',map_location='cpu',weights_only=False)
                    assert all(torch.equal(state[k],prior[k]) for k in state)
                if item['step']==0:
                    warm=torch.load(r.m.source_warm(r.DS,fi,seed),map_location='cpu',weights_only=False)
                    assert all(torch.equal(state['network.'+k],v) for k,v in warm.items())
                checkpoints+=1
            model=r.restore_body(fi,seed);state=model.state_dict();headhome=r.headhome(fi,seed)
            with np.load(headhome/'features.npz') as z:feat={k:z[k].copy() for k in z.files}
            with np.load(headhome/'scaler.npz') as z:mu=z['mean'];sd=z['std']
            np.testing.assert_array_equal(feat['train_rows'],d['train']['rows']);np.testing.assert_array_equal(feat['valid_rows'],d['valid']['rows'])
            np.testing.assert_allclose(mu,feat['train'].astype('float64').mean(0),rtol=0,atol=1e-12);np.testing.assert_allclose(sd,feat['train'].astype('float64').std(0).clip(1e-6),rtol=0,atol=1e-12)
            out=r.O/'predictions'/f'fold_{fi}'/f'seed_{seed}'
            with np.load(out/'expression.npz') as z:test={k:z[k].copy() for k in z.files}
            # Bounded independent checkpoint forward for each role.
            for role,value in [('train',feat['train']),('valid',feat['valid']),('test',test['embedding'])]:
                raw=torch.from_numpy(a['masked_embedding'][ix[role]][:64].astype('float32'));center=state['network.mean'];score=torch.nn.functional.linear(raw-center,state['network.encoder.weight'],state['network.encoder.bias']);values,indices=score.topk(32,dim=1,sorted=False);latent=torch.zeros_like(score).scatter(1,indices,values);expected=torch.nn.functional.linear(latent,state['network.decoder.weight'])+center
                np.testing.assert_allclose(value[:len(raw)],expected.numpy(),rtol=0,atol=2e-5)
            hs=torch.load(headhome/'head.pt',map_location='cpu',weights_only=False);head=r.q.head(seed);expected_initial={k:v.detach().numpy() for k,v in head.state_dict().items()}
            with np.load(headhome/'initial_head.npz') as z:
                for key in ['weight','bias']:np.testing.assert_array_equal(z[key],expected_initial[key])
            # Check same probe initialization and budget as BOTH sealed comparisons.
            for arm in ['original_h','C_sae']:
                ph=r.q.dest(r.DS,fi,seed,arm)
                with np.load(ph/'initial_head.npz') as z:
                    for key in ['weight','bias']:np.testing.assert_array_equal(z[key],expected_initial[key])
                assert r.read(ph/'training.json')['batch_array_sha']==selection['batch_array_sha']
            log=r.read(headhome/'training.json');assert log['steps_trained']==1000 and log['refiner_frozen'];assert log['selected_step']==min(log['history'],key=lambda x:x['valid_MAE'])['step'];assert log['body_selected_step']==selection['selected_step']
            hw=hs['weight'].numpy().astype('float64');hb=hs['bias'].numpy();x=((test['embedding']-mu)/sd).astype('float32');gi=test['gene_index'];rowmap={int(row):i for i,row in enumerate(d['test']['rows'])};ri=np.array([rowmap[int(row)] for row in test['rows']]);pred=(np.einsum('ni,ji->nj',x.astype('float64'),hw,optimize=True)+hb)[ri,gi]*pn['std'][gi]+pn['mean'][gi]
            np.testing.assert_allclose(pred,test['prediction'],rtol=0,atol=2e-5)
            with np.load(headhome/'valid_predictions.npz') as z:
                assert abs(np.abs(z['prediction']-z['target']).mean()-log['valid_MAE'])<1e-12
                vm={int(row):i for i,row in enumerate(ix['valid'])};vr=np.array([vm[int(row)] for row in z['canonical_index']]);vg=z['gene_index'];vx=((feat['valid']-mu)/sd).astype('float32');vp=(np.einsum('ni,ji->nj',vx.astype('float64'),hw,optimize=True)+hb)[vr,vg]*pn['std'][vg]+pn['mean'][vg];np.testing.assert_allclose(vp,z['prediction'],rtol=0,atol=2e-5)
            prior=r.q.O/'predictions'/r.DS/f'fold_{fi}'/f'seed_{seed}'/'C_sae/test.npz'
            with np.load(prior) as z:
                for key in ['rows','gene_names','target']:np.testing.assert_array_equal(test[key],z[key])
                if selection['selected_step']==1000:
                    np.testing.assert_array_equal(test['embedding'],z['embedding']);np.testing.assert_array_equal(test['prediction'],z['prediction'])
                    oldhs=torch.load(r.q.dest(r.DS,fi,seed,'C_sae')/'head.pt',map_location='cpu',weights_only=False)
                    assert all(torch.equal(hs[k],oldhs[k]) for k in hs)
            row=df[(df.fold==fi)&(df.seed==seed)&(df.method=='C_selected')].iloc[0]
            for key,value in r.q.metrics(test['prediction'],test['target'],gi,pn).items():assert abs(row[key]-value)<1e-12
            headchecks+=1
            for arm,method in [('original_h','original_h'),('C_sae','C_1000')]:
                row=df[(df.fold==fi)&(df.seed==seed)&(df.method==method)].iloc[0];prev=old[(old.dataset==r.DS)&(old.fold==fi)&(old.seed==seed)&(old.method==arm)].iloc[0]
                for key in ['MAE','normalized_MSE','gene_macro_MAE','gene_mean_baseline_MAE','entries']:assert row[key]==prev[key]
                for view in ['clean','masked']:
                    row=kn[(kn.fold==fi)&(kn.seed==seed)&(kn.method==method)&(kn.view==view)].iloc[0];prev=oldkn[(oldkn.dataset==r.DS)&(oldkn.fold==fi)&(oldkn.seed.astype(str)==('shared' if method=='original_h' else str(seed)))&(oldkn.method==arm)&(oldkn.view==view)].iloc[0];assert row.knn_macro_f1==prev.knn_macro_f1
            train=r.m.inference(model,a['embedding'][ix['train']]);center=d['train']['h'].astype('float64').mean(0)
            for view in ['clean','masked']:
                with np.load(out/f'knn_{view}.npz') as z:
                    xt=z['embedding'];nr=z['neighbor_train_rows'];labels={int(row):int(y) for row,y in zip(d['train']['rows'],d['train']['y'])};labs=np.array([[labels[int(row)] for row in rr] for rr in nr]);vote=np.stack([(labs==c).sum(1) for c in range(8)],1).argmax(1);np.testing.assert_array_equal(vote,z['prediction']);np.testing.assert_array_equal(z['labels'],d['test']['y']);np.testing.assert_array_equal(z['rows'],d['test']['rows'])
                    score=r.m.f1_score(z['labels'],vote,labels=np.arange(8),average='macro',zero_division=0);row=kn[(kn.fold==fi)&(kn.seed==seed)&(kn.method=='C_selected')&(kn.view==view)].iloc[0];assert abs(score-row.knn_macro_f1)<1e-12
                    indices,_=r.m.ev.top(r.m.ev.norm(xt[:16]-center),r.m.ev.norm(train-center),15);np.testing.assert_array_equal(d['train']['rows'][indices],nr[:16]);neighbors+=1
                    expected=r.m.inference(model,a['embedding'][ix['test']][:64] if view=='clean' else a['masked_embedding'][ix['test']][:64]);np.testing.assert_allclose(xt[:len(expected)],expected,rtol=0,atol=2e-5)
            checks+=1
    assert checks==r.N and checkpoints==r.N*7
    r.save(r.O/'verification.json',dict(status='PASS',body_trajectories=checks,checkpoint_VALID_objective_replays=checkpoints,all_step1000_exactly_equal_old=True,new_probe_TEST_and_selected_VALID_prediction_replays=headchecks,paired_initialization_and_batch_checks=r.N,bounded_checkpoint_forward_roles=r.N*3,knn_metrics_and_16query_replays=neighbors,baseline_metric_identity_checks=True,input_and_selection_hashes_rechecked=True,
        limits='No refitting of unselected probe checkpoints; bounded manual body forward and bounded exact kNN queries. Historical exploratory TEST and reused VALID remain scope limits.'))
    print('VERIFICATION_PASS',checks,checkpoints,flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
