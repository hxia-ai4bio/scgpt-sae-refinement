"""Replay paired frozen expression readouts and their TRAIN/VALID provenance."""
import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
W=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('probe_runtime',W/'run.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q)

def main():
    q.check_inputs();lock=q.read(q.O/'SELECTION_LOCK.json')
    for p,h in lock['hashes'].items():assert q.sha(p)==h,p
    assert lock['source_sha']==q.sha(W/'run.py')
    df=pd.read_csv(q.O/'all_metrics.csv');assert len(df)==84
    n=0;manual=0;coverage=[];checked={}
    for ds in q.m.DATASETS:
        a=q.m.canonical(ds)
        for fi in range(2 if ds=='pbmc' else 6):
            d,f,ix=q.m.fold_data(a,ds,fi);pn=q.m.panel(ds,fi,a,ix);t=q.m.targets(a,pn)
            ng=np.bincount(t['cols'][ix['train']][t['valid'][ix['train']]],minlength=2048)
            coverage.append(dict(dataset=ds,fold=fi,genes_without_TRAIN_masked_targets=int((ng==0).sum()),TEST_entries_for_unseen_genes=int((ng[np.maximum(t['cols'][ix['test']],0)]==0)[t['valid'][ix['test']]].sum())))
            for seed in q.SEEDS:
                initial=[];batches=[];paired=[]
                for arm in q.ARMS:
                    home=q.dest(ds,fi,seed,arm);log=q.read(home/'training.json');batches.append(log['batch_array_sha'])
                    assert log['steps_trained']==1000 and log['refiner_frozen']
                    assert log['selected_step']==min(log['history'],key=lambda x:x['valid_MAE'])['step']
                    assert log['valid_MAE']==min(v['valid_MAE'] for v in log['history'])
                    with np.load(home/'initial_head.npz') as z:initial.append({k:z[k].copy() for k in z.files})
                    with np.load(home/'features.npz') as z:ft={k:z[k].copy() for k in z.files}
                    for role in ['train','valid']:np.testing.assert_array_equal(ft[role+'_rows'],d[role]['rows'])
                    with np.load(home/'scaler.npz') as z:mu=z['mean'];sd=z['std']
                    np.testing.assert_allclose(mu,ft['train'].astype('float64').mean(0),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(sd,ft['train'].astype('float64').std(0).clip(1e-6),rtol=0,atol=1e-12)
                    assert q.m.arraysha(ft['train'])==log['train_feature_sha'] and q.m.arraysha(ft['valid'])==log['valid_feature_sha']
                    assert batches[-1]==q.m.arraysha(q.m.batches(ds,fi,seed,d))
                    state=torch.load(home/'head.pt',map_location='cpu',weights_only=False);hw=state['weight'].numpy();hb=state['bias'].numpy()
                    predfile=q.O/'predictions'/ds/f'fold_{fi}'/f'seed_{seed}'/arm/'test.npz'
                    with np.load(predfile) as z:test={k:z[k].copy() for k in z.files}
                    np.testing.assert_array_equal(test['embedding_rows'],d['test']['rows'])
                    # Replay TRAIN/VALID/TEST links to frozen original or C checkpoint.
                    cp=q.m.W/'networks'/ds/f'fold_{fi}'/f'seed_{seed}'/'C_sae/model.pt'
                    cstate=None if arm=='original_h' else torch.load(cp,map_location='cpu',weights_only=False)
                    for role,h in [('train',ft['train']),('valid',ft['valid']),('test',test['embedding'])]:
                        raw=a['masked_embedding'][ix[role]]
                        if arm=='original_h':np.testing.assert_array_equal(h,raw)
                        else:
                            raw=torch.from_numpy(raw[:64].astype('float32'));center=cstate['network.mean']
                            score=torch.nn.functional.linear(raw-center,cstate['network.encoder.weight'],cstate['network.encoder.bias'])
                            values,indices=score.topk(32,dim=1,sorted=False);latent=torch.zeros_like(score).scatter(1,indices,values)
                            expected=torch.nn.functional.linear(latent,cstate['network.decoder.weight'])+center
                            np.testing.assert_allclose(h[:len(raw)],expected.numpy(),rtol=0,atol=2e-5);manual+=1
                    # Target identity and ordering match the independently audited parent C test output.
                    parent=q.PARENT/'predictions'/ds/f'fold_{fi}'/f'seed_{seed}'/'C_sae/masked_expression_predictions.npz'
                    with np.load(parent) as z:
                        for key in ['rows','gene_names','target']:np.testing.assert_array_equal(test[key],z[key])
                    gi=test['gene_index'];rowpos={int(r):i for i,r in enumerate(d['test']['rows'])};ri=np.array([rowpos[int(r)] for r in test['rows']])
                    # Independent matrix prediction replay in CPU float64 after original float32 scaling.
                    x=((test['embedding']-mu)/sd).astype('float32');pp=(np.einsum('ni,ji->nj',x.astype('float64'),hw.astype('float64'),optimize=True)+hb)[ri,gi]*pn['std'][gi]+pn['mean'][gi]
                    np.testing.assert_allclose(pp,test['prediction'],rtol=0,atol=2e-5)
                    r=df[(df.dataset==ds)&(df.fold==fi)&(df.seed==seed)&(df.method==arm)].iloc[0]
                    stats=q.metrics(test['prediction'],test['target'],gi,pn)
                    for key,value in stats.items():assert abs(value-r[key])<1e-12
                    with np.load(home/'valid_predictions.npz') as z:
                        assert abs(np.abs(z['prediction']-z['target']).mean()-log['valid_MAE'])<1e-12
                        vgi=z['gene_index'];vrows=z['canonical_index'];vpos={int(row):i for i,row in enumerate(ix['valid'])};vri=np.array([vpos[int(row)] for row in vrows]);vx=((ft['valid']-mu)/sd).astype('float32');vp=(np.einsum('ni,ji->nj',vx.astype('float64'),hw.astype('float64'),optimize=True)+hb)[vri,vgi]*pn['std'][vgi]+pn['mean'][vgi]
                        np.testing.assert_allclose(vp,z['prediction'],rtol=0,atol=2e-5)
                    paired.append((test['rows'],test['gene_index'],test['target']));checked[str(predfile)]=q.sha(predfile);n+=1
                expected_initial=q.head(seed).state_dict()
                for k in ['weight','bias']:
                    np.testing.assert_array_equal(initial[0][k],initial[1][k])
                    np.testing.assert_array_equal(initial[0][k],expected_initial[k].numpy())
                torch.manual_seed(seed+2003);upstream=torch.nn.Linear(512,2048)
                torch.nn.init.normal_(upstream.weight,std=.002)
                assert not np.array_equal(initial[0]['weight'],upstream.weight.detach().numpy())
                assert batches[0]==batches[1]
                for v1,v2 in zip(paired[0],paired[1]):np.testing.assert_array_equal(v1,v2)
    assert n==84
    q.save(q.O/'verification.json',dict(status='PASS',head_prediction_checks=n,VALID_prediction_checks=n,TRAIN_scaler_checks=n,paired_initialization_batch_target_checks=42,
        bounded_manual_C_forward_checks=manual,input_bindings_rechecked=True,selection_lock_rechecked=True,coverage=coverage,
        scope='All TEST and selected VALID predictions independently replayed from fresh heads/scalers; C manual frozen checkpoint forward bounded64cells perrole. Earlier unselected VALID histories not retrained independently.',prediction_hashes=checked))
    print('VERIFICATION_PASS',n,flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):main()
