# -*- coding: utf-8 -*-
"""Stim/PBMC checkpoint-selection replication; fixed SAE training trajectory.
SAE training O(S*B*512*2048), frozen linear readout O(S*B*512*2048).
Seven checkpoints saved; only VALID chooses the evaluated SAE, never TEST.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='4'
import copy,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
DS=sys.argv[2] if len(sys.argv)>2 else os.environ.get('SAE_CHECKPOINT_DATASET','stim')
assert DS in ['stim','pbmc'],DS
NF=6 if DS=='stim' else 2;N=NF*3
W=BASE/DS;O=ROOT/'outputs/sae_checkpoint_extension'/DS
s=importlib.util.spec_from_file_location('checkpoint_probe_parent',ROOT/'work/sae_expression_probe/run.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
m=q.m;sha=q.sha;read=q.read;save=q.save
STEPS=[0,25,50,100,200,500,1000];SEEDS=[42,43,44]

def bodyhome(fi,seed):return W/'bodies'/f'fold_{fi}'/f'seed_{seed}'
def headhome(fi,seed):return W/'heads'/f'fold_{fi}'/f'seed_{seed}'
def protocol():
    return dict(dataset=DS,folds=NF,seeds=SEEDS,checkpoints=STEPS,topk=32,width=2048,
        question='Does VALID-only selection of an earlier C checkpoint improve frozen masked-gene expression readout versus original_h and the fixed step1000 C checkpoint?',
        body_training='Exactly replay sealed C: same warmup, original joint head seed+2003, masked input, TRAIN panel/target,1000x256 inherited minibatches, Adam3e-4->1e-5 cosine, decoder normalization after every update. Extra evaluation/save calls do not update weights.',
        body_selection='Full VALID standardized masked-positive expression MSE through jointly trained C head at seven prespecified checkpoints; earliest minimum. Discard joint head for downstream expression probes.',
        head_training=q.protocol()['training'],head_scaling=q.protocol()['scaling'],
        comparisons=['original_h','C_1000','C_selected'],
        baseline_reuse='Reuse same-dataset sealed original_h/C_1000 fresh-linear-probe models,predictions,metrics. New step1000 body must exactly match old state before using these references.',
        knn='Same fixed15NN centered cosine, clean TRAIN reference, clean/masked TEST queries, original clean TRAIN mean; eight-class MacroF1. Existing original/C_1000 metrics reused, selected body newly evaluated.',
        primary='Raw-logCP10k masked-positive token MAE, paired C_selected vs original_h and C_1000',
        secondary=['gene-macro MAE','normalized MSE','clean/masked15NN MacroF1'],
        aggregation='Average3seeds within heldout group then equal groups: stim six historical Kang donors, PBMC two batch directions. No TEST choice among checkpoint,method,seed.',
        scope='Same fixed rule transferred from ctrl, not new tuning: historical stim shares Kang donors with ctrl; PBMC is two batch directions, not verified donor generalization. One fixed25pct mask on originally selected positive tokens; no zero-expression/all-gene imputation. Same architecture/loss/budget; finite readout optimization and upstream VALID reused for head selection. No TEST-based adjustment.',
        new_body_trajectories=N,new_fresh_heads=N,no_new_hyperparameter_sweep=True)

def preflight():
    O.mkdir(parents=True,exist_ok=True);m.core.immutable(O/'PROTOCOL.json',protocol())
    q.check_inputs();bindings=read(q.O/'INPUT_BINDINGS.json').copy()
    for p in [Path(__file__),O/'PROTOCOL.json',q.O/'all_metrics.csv',m.O/'all_metrics.csv',q.O/'FINAL_MANIFEST.json',m.O/'FINAL_MANIFEST.json']:
        bindings[str(p)]=sha(p)
    provenance=read(m.W/'data'/DS/'binding.json')
    assert sha(provenance['raw_path'])==provenance['raw_sha']
    assert sha(provenance['canonical_path'])==provenance['canonical_sha']
    for path,h in provenance['sources'].items():assert sha(path)==h,path
    extras=[Path(provenance['raw_path']),Path(provenance['canonical_path']),m.W/'data'/DS/'binding.json',m.O/f'{DS}_DATA_MANIFEST.json',m.p.ext.dsout(DS)/'SPLITS.json',ROOT/'work/sae_checkpoint_selection/run.py',ROOT/'outputs/sae_checkpoint_selection/FINAL_MANIFEST.json']
    for path in extras+list(map(Path,provenance['sources'])):bindings[str(path)]=sha(path)
    a=m.canonical(DS)
    with np.load(provenance['canonical_path']) as cache:
        for key in ['rows','embedding','labels','cell_ids','condition','groups']:np.testing.assert_array_equal(a[key],cache[key])
    checks=[]
    for fi in range(NF):
        d,f,ix=m.fold_data(a,DS,fi);pn=m.panel(DS,fi,a,ix)
        assert not set(d['train']['rows'])&set(d['valid']['rows'])
        assert not set(d['test']['donor'])&(set(d['train']['donor'])|set(d['valid']['donor']))
        if DS=='stim':assert not set(d['train']['donor'])&set(d['valid']['donor'])
        for seed in SEEDS:
            for arm in ['original_h','C_sae']:
                home=q.dest(DS,fi,seed,arm)
                with np.load(home/'features.npz') as oldfeat:
                    for role in ['train','valid']:np.testing.assert_array_equal(oldfeat[role+'_rows'],d[role]['rows'])
                for p in home.iterdir():
                    if p.is_file():bindings[str(p)]=sha(p)
                p=q.O/'predictions'/DS/f'fold_{fi}'/f'seed_{seed}'/arm/'test.npz';bindings[str(p)]=sha(p)
            p=m.W/'networks'/DS/f'fold_{fi}'/f'seed_{seed}'/'C_sae/model.pt';bindings[str(p)]=sha(p)
        assert not(set(d['train']['rows'])&set(d['test']['rows']) or set(d['valid']['rows'])&set(d['test']['rows']))
        checks.append(dict(fold=fi,roles={r:dict(rows=d[r]['rows'].tolist(),groups=np.unique(d[r]['donor']).tolist()) for r in d}))
    m.core.immutable(O/'INPUT_BINDINGS.json',bindings);save(O/'PREFLIGHT.json',dict(status='PASS',folds=checks))
    print('PREFLIGHT_PASS',DS,flush=True)

def check_inputs():
    for p,h in read(O/'INPUT_BINDINGS.json').items():assert sha(p)==h,p

@torch.no_grad()
def body_valid(model,a,t,ii,pn):
    total=0.;absolute=0.;count=0;model.eval()
    for j in range(0,len(ii),256):
        rows=ii[j:j+256];co=np.maximum(t['cols'][rows],0);mask=t['valid'][rows]
        rec=model(torch.from_numpy(a['masked_embedding'][rows].astype('float32')))
        pred=np.take_along_axis(model.predict_genes(rec).numpy(),co,axis=1)
        error=(pred-t['target'][rows]).astype('float64')[mask]
        total+=float(np.square(error).sum());count+=len(error)
        raw=pred*pn['std'][co]+pn['mean'][co];absolute+=float(np.abs(raw-a['target_log_expression'][rows])[mask].sum())
    return dict(valid_objective=total/count,valid_MAE=absolute/count,entries=count)

def train_body(fi,seed,a,d,ix,pn,t):
    home=bodyhome(fi,seed);home.mkdir(parents=True,exist_ok=True)
    if (home/'completion.json').exists():
        for p,h in read(home/'completion.json')['hashes'].items():assert sha(p)==h
        return
    started=time.time();model=m.Refiner(DS,fi,seed,'C_sae',d['train']['h']);opt=torch.optim.Adam(model.parameters(),lr=.0003)
    bi=m.batches(DS,fi,seed,d);ii=ix['train'];h=torch.from_numpy(a['masked_embedding'][ii].astype('float32'));clean=torch.from_numpy(a['embedding'][ii].astype('float32'));co=torch.from_numpy(t['cols'][ii]);va=torch.from_numpy(t['valid'][ii]);ta=torch.from_numpy(t['target'][ii]);history=[];grads=[]
    for step in range(1001):
        if step in STEPS:
            value=body_valid(model,a,t,ix['valid'],pn);assert all(np.isfinite(v) for v in value.values())
            history.append(dict(step=step,**value));torch.save(model.state_dict(),home/f'step_{step}.pt')
        if step==1000:break
        model.train();ids=bi[step];lr=.00001+.5*(.0003-.00001)*(1+math.cos(math.pi*step/999))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss=m.mask_loss(model,h[ids],clean[ids],co[ids],va[ids],ta[ids],1.);assert torch.isfinite(loss);loss.backward()
        if step in [0,1,999]:
            norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None};assert all(np.isfinite(v) for v in norms.values());grads.append(dict(step=step+1,loss=float(loss),norms=norms))
        opt.step();model.normalize()
    previous=m.W/'networks'/DS/f'fold_{fi}'/f'seed_{seed}'/'C_sae/model.pt';old=torch.load(previous,map_location='cpu',weights_only=False);state=model.state_dict()
    assert set(state)==set(old)
    differences={key:float((state[key]-old[key]).abs().max()) for key in state}
    assert max(differences.values())==0.,differences
    selected=min(history,key=lambda x:x['valid_objective'])
    save(home/'selection.json',dict(selected_step=selected['step'],selected_valid_objective=selected['valid_objective'],selected_valid_MAE=selected['valid_MAE'],history=history,selection_role='VALID',TRAIN_only_fit=True,
        steps_trained=1000,seconds=time.time()-started,gradients=grads,batch_array_sha=m.arraysha(bi),warmup_sha=sha(m.source_warm(DS,fi,seed)),old1000_sha=sha(previous),exact_old1000_replay=True,parameter_max_differences=differences))
    save(home/'completion.json',dict(hashes={str(p):sha(p) for p in home.iterdir() if p.is_file() and p.name!='completion.json'}))
    print('BODY_COMPLETE',DS,fi,seed,'selected',selected['step'],'exact1000',True,'seconds',round(time.time()-started,2),flush=True)

def restore_body(fi,seed,step=None):
    home=bodyhome(fi,seed);step=read(home/'selection.json')['selected_step'] if step is None else step
    model=m.Refiner(DS,fi,seed,'C_sae');model.load_state_dict(torch.load(home/f'step_{step}.pt',map_location='cpu',weights_only=False));return model.eval().requires_grad_(False)

def train_probe(fi,seed,a,d,ix,pn,t):
    home=headhome(fi,seed);home.mkdir(parents=True,exist_ok=True)
    if (home/'completion.json').exists():
        for p,h in read(home/'completion.json')['hashes'].items():assert sha(p)==h
        return
    started=time.time();body=restore_body(fi,seed);feat={r:m.inference(body,a['masked_embedding'][ix[r]]).astype('float32') for r in ['train','valid']}
    assert all(not p.requires_grad and p.grad is None for p in body.parameters())
    mu=feat['train'].astype('float64').mean(0);sd=feat['train'].astype('float64').std(0).clip(1e-6);x={r:((feat[r]-mu)/sd).astype('float32') for r in feat}
    np.savez_compressed(home/'features.npz',**feat,train_rows=d['train']['rows'],valid_rows=d['valid']['rows'])
    model=q.head(seed);initial=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.001);batch=m.batches(DS,fi,seed,d)
    tr=ix['train'];h=torch.from_numpy(x['train']);co=torch.from_numpy(t['cols'][tr]);mask=torch.from_numpy(t['valid'][tr]);yt=torch.from_numpy(t['target'][tr]);best=float('inf');history=[];grad=[]
    for step in range(1001):
        if step in [0,100,250,500,750,1000]:
            model.eval();pred,target,gene,rows=q.predict(model,x['valid'],t,pn,ix['valid']);mae=float(np.abs(pred-target).mean());assert np.isfinite(mae);history.append(dict(step=step,valid_MAE=mae))
            if mae<best:best=mae;selected=step;state=copy.deepcopy(model.state_dict())
        if step==1000:break
        model.train();ids=batch[step];lr=.00001+.5*(.001-.00001)*(1+math.cos(math.pi*step/999))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);pred=model(h[ids]).gather(1,co[ids].clamp(min=0));loss=(pred[mask[ids]]-yt[ids][mask[ids]]).square().mean();assert torch.isfinite(loss);loss.backward()
        if step in [0,999]:
            norm={name:float(p.grad.norm()) for name,p in model.named_parameters()};assert all(np.isfinite(v) and v>0 for v in norm.values());grad.append(dict(step=step+1,loss=float(loss),norms=norm))
        opt.step()
    model.load_state_dict(state);torch.save(state,home/'head.pt');np.savez_compressed(home/'scaler.npz',mean=mu,std=sd);np.savez_compressed(home/'initial_head.npz',**{k:v.numpy() for k,v in initial.items()})
    pred,target,gene,rows=q.predict(model,x['valid'],t,pn,ix['valid']);np.savez_compressed(home/'valid_predictions.npz',prediction=pred,target=target,gene_index=gene,canonical_index=rows)
    save(home/'training.json',dict(selected_step=selected,valid_MAE=best,history=history,steps_trained=1000,seconds=time.time()-started,gradients=grad,batch_array_sha=m.arraysha(batch),body_selected_step=read(bodyhome(fi,seed)/'selection.json')['selected_step'],body_checkpoint_sha=sha(bodyhome(fi,seed)/f"step_{read(bodyhome(fi,seed)/'selection.json')['selected_step']}.pt"),refiner_frozen=True))
    save(home/'completion.json',dict(hashes={str(p):sha(p) for p in home.iterdir() if p.is_file() and p.name!='completion.json'}))
    print('PROBE_COMPLETE',DS,fi,seed,'selected',selected,'seconds',round(time.time()-started,2),flush=True)

def develop(pilot=False):
    check_inputs();a=m.canonical(DS)
    for fi in range(1 if pilot else NF):
        d,f,ix=m.fold_data(a,DS,fi);pn=m.panel(DS,fi,a,ix);t=m.targets(a,pn);t['raw']=a['target_log_expression']
        for seed in ([42] if pilot else SEEDS):
            train_body(fi,seed,a,d,ix,pn,t);train_probe(fi,seed,a,d,ix,pn,t)
            count=len(list((W/'heads').glob('*/*/completion.json')));save(O/'EXPERIMENT_TRACKER.json',dict(status='TRAIN_VALID_ONLY',bodies_and_heads_done=count,total=N))
    if not pilot:
        paths=[p for home in ['bodies','heads'] for p in (W/home).rglob('*') if p.is_file()]
        assert len(list((W/'heads').glob('*/*/completion.json')))==N
        m.core.immutable(O/'SELECTION_LOCK.json',dict(source_sha=sha(__file__),hashes={str(p):sha(p) for p in paths},roles_used=['TRAIN','VALID'],TEST_evaluated=False));print('SELECTION_LOCKED',flush=True)

def evaluate():
    check_inputs();global_lock=read(O.parent/'GLOBAL_SELECTION_LOCK.json')
    assert global_lock['source_sha']==sha(__file__)
    assert global_lock['all_new_datasets_selected_before_TEST']
    assert set(global_lock['dataset_lock_hashes'])=={'stim','pbmc'}
    for ds,h in global_lock['dataset_lock_hashes'].items():assert sha(O.parent/ds/'SELECTION_LOCK.json')==h,ds
    lock=read(O/'SELECTION_LOCK.json')
    for p,h in lock['hashes'].items():assert sha(p)==h,p
    a=m.canonical(DS);old=pd.read_csv(q.O/'all_metrics.csv');old=old[old.dataset==DS];oldkn=pd.read_csv(m.O/'all_metrics.csv');oldkn=oldkn[oldkn.dataset==DS];records=[];knn=[]
    for fi in range(NF):
        d,f,ix=m.fold_data(a,DS,fi);pn=m.panel(DS,fi,a,ix);t=m.targets(a,pn);t['raw']=a['target_log_expression'];mu=d['train']['h'].astype('float64').mean(0)
        for seed in SEEDS:
            for method,parentmethod in [('original_h','original_h'),('C_1000','C_sae')]:
                r=old[(old.fold==fi)&(old.seed==seed)&(old.method==parentmethod)].iloc[0].to_dict();r['method']=method;r['source']='sealed_expression_probe';records.append(r)
                for view in ['clean','masked']:
                    s=oldkn[(oldkn.fold==fi)&(oldkn.seed.astype(str)==('shared' if method=='original_h' else str(seed)))&(oldkn.method==parentmethod)&(oldkn.view==view)].iloc[0]
                    knn.append(dict(dataset=DS,fold=fi,group=s['group'],seed=seed,method=method,view=view,knn_macro_f1=float(s.knn_macro_f1)))
            body=restore_body(fi,seed);home=headhome(fi,seed);out=O/'predictions'/f'fold_{fi}'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
            feat=m.inference(body,a['masked_embedding'][ix['test']]).astype('float32')
            with np.load(home/'scaler.npz') as s:x=((feat-s['mean'])/s['std']).astype('float32')
            head=q.head(seed);head.load_state_dict(torch.load(home/'head.pt',map_location='cpu',weights_only=False));head.eval();pp,yy,gg,rr=q.predict(head,x,t,pn,ix['test'])
            np.savez_compressed(out/'expression.npz',prediction=pp,target=yy,gene_index=gg,gene_names=pn['gene_names'][gg],rows=a['rows'][rr],embedding=feat,embedding_rows=d['test']['rows'])
            selected=read(bodyhome(fi,seed)/'selection.json')['selected_step'];records.append(dict(dataset=DS,fold=fi,group=str(d['test']['donor'][0]),seed=seed,method='C_selected',selected_body_step=selected,selected_step=read(home/'training.json')['selected_step'],source='new_selected_checkpoint',**q.metrics(pp,yy,gg,pn)))
            trainclean=m.inference(body,a['embedding'][ix['train']]);ref=m.ev.norm(trainclean-mu)
            for view in ['clean','masked']:
                xt=m.inference(body,a['embedding'][ix['test']]) if view=='clean' else feat.astype('float64');indices,_=m.ev.top(m.ev.norm(xt-mu),ref,15);labels=d['train']['y'][indices];prediction=np.stack([(labels==c).sum(1) for c in range(8)],1).argmax(1)
                score=float(m.f1_score(d['test']['y'],prediction,labels=np.arange(8),average='macro',zero_division=0));knn.append(dict(dataset=DS,fold=fi,group=str(d['test']['donor'][0]),seed=seed,method='C_selected',view=view,knn_macro_f1=score))
                np.savez_compressed(out/f'knn_{view}.npz',rows=d['test']['rows'],embedding=xt,labels=d['test']['y'],prediction=prediction,neighbor_train_rows=d['train']['rows'][indices])
            print('TEST_COMPLETE',DS,fi,seed,'body',selected,flush=True)
    assert len(records)==N*3 and len(knn)==N*6
    pd.DataFrame(records).to_csv(O/'all_metrics.csv',index=False);save(O/'all_metrics.json',records);pd.DataFrame(knn).to_csv(O/'knn_metrics.csv',index=False)
    save(O/'EXPERIMENT_TRACKER.json',dict(status='EVALUATED_PENDING_REVIEW',new_bodies=N,new_heads=N,expression_results=N*3,knn_results=N*6));print('ALL_EVALUATIONS_COMPLETE',flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        stage=sys.argv[1]
        if stage=='preflight':preflight()
        elif stage=='pilot':develop(True)
        elif stage=='develop':develop()
        elif stage=='evaluate':evaluate()
