# -*- coding: utf-8 -*-
"""Matched frozen linear probes of masked gene expression; no refiner updates.
Training O(steps*batch*512*2048), working memory O(N*512+N*tokens).
All new outputs are separate from sealed parent experiments.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='4'
import copy, importlib.util, json, math, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

W=Path(__file__).resolve().parent
ROOT=W.parents[1]
O=ROOT/'outputs/sae_expression_probe'
PARENT=ROOT/'outputs/sae_masked_ac'
spec=importlib.util.spec_from_file_location('expression_parent',ROOT/'work/sae_masked_ac/pipeline.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ARMS=['original_h','C_sae']; SEEDS=[42,43,44]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(path)

def protocol():
    return dict(datasets=m.DATASETS,seeds=SEEDS,arms=ARMS,head='Linear512->2048; no refiner/backbone gradients',
        question='Does frozen C-refined masked embedding support lower masked-positive expression MAE than original masked scGPT with an identically trained fresh linear head?',
        scope='Exploratory historical cohorts; C is already supervised on TRAIN expression. Compares task-specific frozen representations, not equal total upstream training or SAE necessity.',
        mask='Reuse fixed25pct originally selected expressed tokens, same panel/target/hidden positions as sealed parent. Positive tokens only; not zeros or full-transcriptome imputation.',
        input='Masked input for both TRAIN/VALID/TEST; C checkpoint fixed at parent step1000; discard original jointly trained C head.',
        scaling='Separate per-representation TRAIN-masked coordinate mean/std, floor1e-6; targets use identical inherited TRAIN-positive gene mean/std.',
        training=dict(steps=1000,batch_size=256,optimizer='Adam',initial_lr=.001,final_lr=.00001,weight_decay=0,
            objective='Mean squared error of standardized masked-positive expression entries',
            initialization='paired torch.manual_seed(seed+400003), Normal(0,.002) weights and zero bias; identical initial heads per arm',
            batches='Reuse identical parent1000x256 TRAIN-balanced minibatch indices for paired arms',
            selection='Full VALID raw-logCP10k token MAE at steps0,100,250,500,750,1000; earliest minimum; no TRAIN+VALID refit'),
        primary='Paired TEST raw-logCP10k masked-positive token MAE, C_sae minus original_h',
        secondary=['gene-macro MAE','normalized MSE','TRAIN-positive gene-mean baseline'],
        aggregation='Mean3 paired seeds within fold then equal heldout groups; Kang6donors per condition, PBMC2batch directions; ctrl/stim share donors',
        count_heads=84,historical_TEST_explored=True,no_new_TEST_based_selection=True,
        limits='Single fixed mask, single1000-step probe budget, fixed prior C checkpoint; C has upstream expression supervision, original has none.')

def dest(ds,fi,seed,arm):return W/'heads'/ds/f'fold_{fi}'/f'seed_{seed}'/arm

def get_features(ds,fi,seed,arm,a,ix,roles):
    model=None if arm=='original_h' else m.restore(ds,fi,seed,'C_sae')
    if model is not None:model.eval().requires_grad_(False)
    out={r:a['masked_embedding'][ix[r]].astype('float32') if model is None else m.inference(model,a['masked_embedding'][ix[r]]).astype('float32') for r in roles}
    if model is not None:assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return out

def preflight():
    O.mkdir(parents=True,exist_ok=True)
    m.core.immutable(O/'PROTOCOL.json',protocol())
    # Parent code, consumed data, checkpoints, panels, and batches are immutable inputs.
    bindings={str(Path(__file__)):sha(__file__),str(O/'PROTOCOL.json'):sha(O/'PROTOCOL.json')}
    for p,h in read(m.W/'sources.json').items():
        assert sha(p)==h,p
        bindings[p]=h
    records=[]
    for ds in m.DATASETS:
        a=m.canonical(ds)
        for fi in range(2 if ds=='pbmc' else 6):
            d,f,ix=m.fold_data(a,ds,fi);pn=m.panel(ds,fi,a,ix)
            assert not (set(d['train']['rows'])&set(d['valid']['rows']) or set(d['train']['rows'])&set(d['test']['rows']) or set(d['valid']['rows'])&set(d['test']['rows']))
            t=m.targets(a,pn);assert all(t['valid'][ix[r]].any() for r in ['train','valid','test'])
            for seed in SEEDS:
                cp=m.W/'networks'/ds/f'fold_{fi}'/f'seed_{seed}'/'C_sae/model.pt'
                assert sha(cp)==read(cp.parent/'training.json')['checkpoint_sha'];bindings[str(cp)]=sha(cp)
                bi=m.batches(ds,fi,seed,d)
                home=m.base.W if ds=='ctrl' else m.p.ext.dswork(ds)
                q=home/'batches'/f'fold_{fi}_seed_{seed}.npz';bindings[str(q)]=sha(q)
                assert bi.min()>=0 and bi.max()<len(ix['train'])
            records.append(dict(dataset=ds,fold=fi,roles={r:dict(rows=d[r]['rows'].tolist(),entries=int(t['valid'][ix[r]].sum())) for r in ix}))
    m.core.immutable(O/'INPUT_BINDINGS.json',bindings)
    save(O/'PREFLIGHT.json',dict(status='PASS',folds=records));print('PREFLIGHT_PASS',flush=True)

def check_inputs():
    for p,h in read(O/'INPUT_BINDINGS.json').items():assert sha(p)==h,p

def head(seed):
    torch.manual_seed(seed+400003);model=nn.Linear(512,2048)
    nn.init.normal_(model.weight,std=.002);nn.init.zeros_(model.bias)
    return model

@torch.no_grad()
def predict(model,x,t,pn,indices):
    pp=[];yy=[];gg=[];rr=[]
    for start in range(0,len(indices),256):
        rows=indices[start:start+256];co=np.maximum(t['cols'][rows],0);mask=t['valid'][rows]
        out=model(torch.from_numpy(x[start:start+len(rows)])).numpy()
        raw=np.take_along_axis(out,co,axis=1)*pn['std'][co]+pn['mean'][co]
        pp.append(raw[mask]);yy.append(t['raw'][rows][mask]);gg.append(co[mask]);rr.append(np.broadcast_to(rows[:,None],co.shape)[mask])
    return tuple(np.concatenate(v) for v in [pp,yy,gg,rr])

def train_one(ds,fi,seed,arm,a,d,ix,pn,t,feat):
    home=dest(ds,fi,seed,arm);home.mkdir(parents=True,exist_ok=True)
    if (home/'completion.json').exists():
        for p,h in read(home/'completion.json')['hashes'].items():assert sha(p)==h
        return
    started=time.time();mu=feat['train'].astype('float64').mean(0);sd=feat['train'].astype('float64').std(0).clip(1e-6)
    x={r:((feat[r]-mu)/sd).astype('float32') for r in ['train','valid']}
    np.savez_compressed(home/'features.npz',train=feat['train'],valid=feat['valid'],train_rows=d['train']['rows'],valid_rows=d['valid']['rows'])
    model=head(seed);initial=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.001)
    batch=m.batches(ds,fi,seed,d);tr=ix['train'];h=torch.from_numpy(x['train']);co=torch.from_numpy(t['cols'][tr]);mask=torch.from_numpy(t['valid'][tr]);yt=torch.from_numpy(t['target'][tr])
    history=[];best=float('inf');selected=None;state=None;grads=[]
    for step in range(1001):
        if step in [0,100,250,500,750,1000]:
            model.eval();pred,target,gene,rows=predict(model,x['valid'],t,pn,ix['valid']);mae=float(np.abs(pred-target).mean())
            assert np.isfinite(mae);history.append(dict(step=step,valid_MAE=mae))
            if mae<best:
                best=mae;selected=step;state=copy.deepcopy(model.state_dict())
        if step==1000:break
        model.train();ii=batch[step]
        lr=.00001+.5*(.001-.00001)*(1+math.cos(math.pi*step/999))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);pred=model(h[ii]).gather(1,co[ii].clamp(min=0));loss=(pred[mask[ii]]-yt[ii][mask[ii]]).square().mean()
        assert torch.isfinite(loss);loss.backward()
        if step in [0,999]:
            grad={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert all(np.isfinite(v) and v>0 for v in grad.values());grads.append(dict(step=step+1,loss=float(loss),norms=grad))
        opt.step()
    model.load_state_dict(state);assert selected is not None
    torch.save(state,home/'head.pt');np.savez_compressed(home/'scaler.npz',mean=mu,std=sd)
    pred,target,gene,rows=predict(model,x['valid'],t,pn,ix['valid']);np.savez_compressed(home/'valid_predictions.npz',prediction=pred,target=target,gene_index=gene,canonical_index=rows)
    np.savez_compressed(home/'initial_head.npz',weight=initial['weight'].numpy(),bias=initial['bias'].numpy())
    record=dict(dataset=ds,fold=fi,seed=seed,arm=arm,selected_step=selected,valid_MAE=best,history=history,gradients=grads,steps_trained=1000,seconds=time.time()-started,
        source_sha=sha(__file__),batch_array_sha=m.arraysha(batch),train_feature_sha=m.arraysha(feat['train']),valid_feature_sha=m.arraysha(feat['valid']),
        parameter_count=sum(p.numel() for p in model.parameters()),selected_head_change=float((state['weight']-initial['weight']).norm()),refiner_frozen=True)
    save(home/'training.json',record)
    save(home/'completion.json',dict(hashes={str(p):sha(p) for p in home.iterdir() if p.is_file() and p.name!='completion.json'}))
    print('HEAD_COMPLETE',ds,fi,seed,arm,'selected',selected,'seconds',round(record['seconds'],2),flush=True)

def train(pilot=False):
    check_inputs()
    for ds in (['ctrl'] if pilot else m.DATASETS):
        a=m.canonical(ds)
        for fi in range(1 if pilot else (2 if ds=='pbmc' else 6)):
            d,f,ix=m.fold_data(a,ds,fi);pn=m.panel(ds,fi,a,ix);t=m.targets(a,pn);t['raw']=a['target_log_expression']
            original=get_features(ds,fi,42,'original_h',a,ix,['train','valid'])
            for seed in ([42] if pilot else SEEDS):
                for arm in ARMS:
                    feat=original if arm=='original_h' else get_features(ds,fi,seed,arm,a,ix,['train','valid'])
                    train_one(ds,fi,seed,arm,a,d,ix,pn,t,feat)
                done=len(list((W/'heads').glob('*/*/*/*/completion.json')))
                save(O/'EXPERIMENT_TRACKER.json',dict(status='FITTING_TRAIN_VALID_ONLY',heads_done=done,heads_total=84))
    if not pilot:
        paths=[p for p in (W/'heads').rglob('*') if p.is_file()];assert len(list((W/'heads').glob('*/*/*/*/completion.json')))==84
        m.core.immutable(O/'SELECTION_LOCK.json',dict(hashes={str(p):sha(p) for p in paths},source_sha=sha(__file__),roles_used=['TRAIN','VALID'],TEST_scored=False))
        print('SELECTION_LOCKED',flush=True)

def metrics(pred,target,gene,pn):
    err=np.abs(pred-target);count=np.bincount(gene,minlength=2048);used=count>0
    return dict(MAE=float(err.mean()),normalized_MSE=float(np.mean(((pred-target)/pn['std'][gene])**2)),
        gene_macro_MAE=float((np.bincount(gene,weights=err,minlength=2048)[used]/count[used]).mean()),
        gene_mean_baseline_MAE=float(np.abs(pn['mean'][gene]-target).mean()),entries=len(err))

def evaluate():
    check_inputs();lock=read(O/'SELECTION_LOCK.json')
    for p,h in lock['hashes'].items():assert sha(p)==h,p
    records=[]
    for ds in m.DATASETS:
        a=m.canonical(ds)
        for fi in range(2 if ds=='pbmc' else 6):
            d,f,ix=m.fold_data(a,ds,fi);pn=m.panel(ds,fi,a,ix);t=m.targets(a,pn);t['raw']=a['target_log_expression']
            for seed in SEEDS:
                for arm in ARMS:
                    home=dest(ds,fi,seed,arm);feat=get_features(ds,fi,seed,arm,a,ix,['test'])['test']
                    with np.load(home/'scaler.npz') as s:x=((feat-s['mean'])/s['std']).astype('float32')
                    model=head(seed);model.load_state_dict(torch.load(home/'head.pt',map_location='cpu',weights_only=False));model.eval()
                    pp,yy,gg,rr=predict(model,x,t,pn,ix['test']);out=O/'predictions'/ds/f'fold_{fi}'/f'seed_{seed}'/arm;out.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(out/'test.npz',prediction=pp,target=yy,gene_index=gg,rows=a['rows'][rr],gene_names=pn['gene_names'][gg],embedding=feat,embedding_rows=d['test']['rows'])
                    selected=read(home/'training.json');records.append(dict(dataset=ds,fold=fi,group=str(d['test']['donor'][0]),seed=seed,method=arm,selected_step=selected['selected_step'],valid_MAE=selected['valid_MAE'],**metrics(pp,yy,gg,pn)))
            print('TEST_COMPLETE',ds,fi,flush=True)
    assert len(records)==84
    pd.DataFrame(records).to_csv(O/'all_metrics.csv',index=False);save(O/'all_metrics.json',records)
    save(O/'EXPERIMENT_TRACKER.json',dict(status='EVALUATED_PENDING_VERIFICATION_AND_REVIEW',heads_done=84,test_results=84))
    print('ALL_EVALUATIONS_COMPLETE',flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        stage=sys.argv[1]
        if stage=='preflight':preflight()
        elif stage=='pilot':train(True)
        elif stage=='develop':train()
        elif stage=='evaluate':evaluate()
