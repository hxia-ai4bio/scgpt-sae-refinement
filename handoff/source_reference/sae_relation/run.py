"""Paired SAE continuation: MSE vs centered cross-donor relational reconstruction.

Neural time O(steps*(B*d*m+B**2*d)), memory O(N*d+B*m+B**2+d*m).
Ridge time O(N*d**2+d**3), memory O(N*d+d**2), d=512,m=2048,B=256.
No backbone training; all supervised selection confined to VALID Ridge alpha.
"""
import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','4')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/sae_refine'))
import run as refine
old=refine.old
W=Path(__file__).resolve().parent
O=ROOT/'outputs/sae_relation'
PROPOSAL=ROOT/'outputs/sae_relation_preflight/PROPOSED_PROTOCOL.json'
ARMS={'continue_mse':0.,'relation_weak':.03,'relation_strong':.3}
SEEDS=[42,43,44]
STEPS=1000
VARIANTS={'reconstruction':1.,'fixed_half':.5}
CFG={
    'folds':list(range(6)),'seeds':SEEDS,'arms':ARMS,'additional_steps':STEPS,
    'batch_size':256,'cells_per_train_donor':64,'width':2048,'topk':32,
    'topk_operator':'largest raw values, selected signs retained; unchanged architecture',
    'temperature':.1,'mean':'fixed TRAIN mean buffer, same for original/reconstructed cosine',
    'objective':'raw MSE / TRAIN centered mean-square + lambda * mean_query KL(P_teacher||P_student)',
    'relations':'centered L2-normalized cosine, cross-TRAIN-donor batch anchors only,192 perquery; detached original teacher; float64 log_softmax normalized rows',
    'optimizer':'Adam reset identically for all arms, no weight decay',
    'lr_start':.0003,'lr_end':.00001,
    'schedule':'cosine by zero-based update step, first3e-4,last1e-5',
    'batch_seed':'SeedSequence([20261019,fold,seed,1000]); same persisted batch indices for all arms',
    'checkpoint':'fixed final step1000, no VALID checkpoint/model/strength selection',
    'initialization':'same existing fold/seed raw-value TopK32 SAE checkpoint, trained fourTRAIN donors and historically VALID-MSE selected at epoch20',
    'backbone':'frozen cached final layer signed mean512D scGPT; no new scGPT inference or optimization',
    'reference_arms':['original_h','existing_raw20'],
    'variants':VARIANTS,'main_variant':'reconstruction','secondary_variant':'fixed_half; fixed in advance, no mixture or identity search',
    'ridge_alphas':old.ALPHAS,
    'probe_selection':'TRAIN-only unweighted channel mean/std; balanced eight-class weighted Ridge with intercept and 2*onehot-1 targets; best VALID fixed8 MacroF1; ascending-alpha ties; no refit TRAIN+VALID',
    'pilot':{'folds':[0,2],'seed':42,'arms':list(ARMS),'use':'training health, no pilot readout or TEST score; completed pilot models reused in full run'},
    'primary_contrasts':['relation_weak vs original_h','relation_weak vs continue_mse'],
    'secondary_contrasts':['relation_strong vs controls','continue_mse vs existing_raw20','fixed_half variants'],
    'advancement':'weak main: >=0.5pp vsoriginal_h, >=4/6 positive donors and>=2/3 positive seed means; additionally positive average vs matchedMSE, report donor/seed consistency for both. Descriptive engineering gate, not a significance test.',
    'statistics':'equal donor means after within-donor seed averaging;6 exploratory historic donors, overlapping training sets; descriptive paired donor bootstrap only; seeds/cells not independent donors',
    'scope':'historically explored Kang ctrl9577cells,6donors,8types; not a new blind test; no PBMC, no supervised feature ranking; relational geometry is a self-supervised proxy, not ground-truth biology',
    'stop_rule':'complete all54 continuation fits and fixed readout grids; do not retune from TEST or promote secondary arm posthoc',
}


def save(p,v):old.save(p,v)
def read(p):return old.read(p)
def immutable(p,v):old.immutable(p,v)


def atomic_torch_save(v,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');torch.save(v,temp);temp.replace(p)


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def data(fi,roles=('train','valid')):return refine.data(fi,roles)


def initial(fi,seed):
    state=torch.load(old.src_for(fi,seed)/'sae.pt',map_location='cpu',weights_only=False)
    model=refine.SAE(state['mean'],2048,32,seed)
    model.load_state_dict(state)
    return model


def network_dir(fi,seed,arm):return W/'networks'/f'fold_{fi}'/f'seed_{seed}'/arm


def restore(fi,seed,arm):
    model=initial(fi,seed)
    model.load_state_dict(torch.load(network_dir(fi,seed,arm)/'sae.pt',map_location='cpu',weights_only=False))
    return model


def batches(fi,seed,b):
    groups=np.unique(b['donor']);assert len(groups)==4
    rng=np.random.default_rng(np.random.SeedSequence([20261019,fi,seed,STEPS]))
    choices=[np.flatnonzero(b['donor']==g) for g in groups]
    assert all(len(c)>=64 for c in choices)
    ix=np.asarray([np.concatenate([rng.choice(c,64,replace=False) for c in choices]) for _ in range(STEPS)],dtype=np.int64)
    p=W/'batches'/f'fold_{fi}_seed_{seed}.npz';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        with np.load(p) as a:
            np.testing.assert_array_equal(a['indices'],ix)
            np.testing.assert_array_equal(a['rows'],b['rows'][ix])
    else:np.savez_compressed(p,indices=ix,rows=b['rows'][ix],donors=groups)
    donor_numbers=torch.arange(4).repeat_interleave(64)
    mask=donor_numbers[:,None]!=donor_numbers[None,:]
    assert torch.equal(mask.sum(1),torch.full((256,),192))
    return ix,mask,p


def relation(h,rec,mean,mask=None,anchor_h=None,anchor_rec=None):
    anchor_h=h if anchor_h is None else anchor_h
    anchor_rec=rec if anchor_rec is None else anchor_rec
    x=F.normalize(h.double()-mean.double(),dim=1,eps=1e-12)
    a=F.normalize(anchor_h.double()-mean.double(),dim=1,eps=1e-12)
    y=F.normalize(rec.double()-mean.double(),dim=1,eps=1e-12)
    ar=F.normalize(anchor_rec.double()-mean.double(),dim=1,eps=1e-12)
    teacher=(x@a.T).detach();student=y@ar.T
    if mask is not None:
        assert mask.sum(1).unique().numel()==1
        teacher=teacher[mask].reshape(len(h),-1);student=student[mask].reshape(len(h),-1)
    logp=F.log_softmax(teacher/.1,dim=1)
    logq=F.log_softmax(student/.1,dim=1)
    p=logp.exp()
    assert not logp.requires_grad
    assert torch.isfinite(logp).all() and torch.isfinite(logq).all()
    assert float((p.sum(1)-1).abs().max())<1e-10
    assert float((logq.exp().sum(1)-1).abs().max())<1e-10
    kl=(p*(logp-logq)).sum(1).mean()
    assert torch.isfinite(kl) and float(kl.detach())>=-1e-12
    return kl,float((-(p*logp).sum(1)).exp().mean())


def grad_norm(loss,params):
    grads=torch.autograd.grad(loss,params,retain_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)
    return float(torch.sqrt(sum(g.double().square().sum() for g in grads)))


def monitor(model,d,den,monitor_ix,valid_ix,mask,step):
    rec={r:refine.infer(model,b['h']) for r,b in d.items()}
    out={'step':step}
    for r,b in d.items():
        err=float(np.mean((rec[r].astype(np.float64)-b['h'].astype(np.float64))**2))
        out[r+'_raw_mse']=err;out[r+'_normalized_mse']=err/den
    with torch.no_grad():
        h=torch.from_numpy(d['train']['h'][monitor_ix]);r=torch.from_numpy(rec['train'][monitor_ix])
        kl,eff=relation(h,r,model.mean,mask)
        out['train_monitor_kl']=float(kl);out['teacher_effective_neighbors']=eff
        vh=torch.from_numpy(d['valid']['h'][valid_ix]);vr=torch.from_numpy(rec['valid'][valid_ix])
        kl,eff=relation(vh,vr,model.mean,anchor_h=h,anchor_rec=r)
        out['valid_to_train_kl']=float(kl);out['valid_teacher_effective_neighbors']=eff
    model.train()
    return out


def binding():return {'protocol_sha':old.sha(O/'PROTOCOL.json'),'source_manifest_sha':old.sha(W/'sources.json')}


def checked_completion(p):
    if not Path(p).exists():return False
    c=read(p);assert c['binding']==binding()
    for q,h in c['hashes'].items():assert old.sha(q)==h,q
    return True


def complete(path,start):
    save(path/'completion.json',dict(binding=binding(),seconds=time.time()-start,hashes={str(p):old.sha(p) for p in path.rglob('*') if p.is_file() and p.name!='completion.json'}))


def train_one(fi,seed,arm,d):
    out=network_dir(fi,seed,arm)
    if checked_completion(out/'completion.json'):return
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    model=initial(fi,seed);initial_state=copy.deepcopy(model.state_dict())
    tr=d['train'];ix,mask,batchpath=batches(fi,seed,tr)
    den=float(np.mean((tr['h'].astype(np.float64)-model.mean.numpy().astype(np.float64))**2))
    assert den>0
    rng=np.random.default_rng(np.random.SeedSequence([20261019,fi,seed,128]))
    vi=rng.choice(len(d['valid']['h']),128,replace=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=CFG['lr_start'])
    params=tuple(model.parameters());progress=out/'progress.pt'
    history=[];gradients=[];step0=0;elapsed=0.
    if progress.exists():
        checkpoint=torch.load(progress,map_location='cpu',weights_only=False)
        assert checkpoint['binding']==binding() and checkpoint['batch_sha']==old.sha(batchpath)
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        step0=checkpoint['step'];history=checkpoint['history'];gradients=checkpoint['gradients'];elapsed=checkpoint['elapsed']
    else:history.append(monitor(model,d,den,ix[0],vi,mask,0))
    lam=ARMS[arm]
    for step in range(step0,STEPS):
        lr=CFG['lr_end']+.5*(CFG['lr_start']-CFG['lr_end'])*(1+math.cos(math.pi*step/(STEPS-1)))
        optimizer.param_groups[0]['lr']=lr
        h=torch.from_numpy(tr['h'][ix[step]])
        optimizer.zero_grad(set_to_none=True)
        rec,z=model(h)
        mse=(rec-h).square().mean()/den
        checkgrad=step in [0,249,499,749,999]
        if lam or checkgrad:
            kl,eff=relation(h,rec,model.mean,mask)
        else:kl=torch.zeros((),dtype=torch.float64)
        loss=mse+lam*kl
        assert torch.isfinite(loss)
        if checkgrad:
            gm=grad_norm(mse,params);gk=grad_norm(kl,params)
            assert gm>0 and gk>0
            gradients.append(dict(step=step+1,lr=lr,normalized_mse=float(mse.detach()),kl=float(kl.detach()),mse_gradient_norm=gm,kl_gradient_norm=gk,weighted_gradient_ratio=lam*gk/gm,teacher_effective_neighbors=eff))
        loss.backward()
        if checkgrad:
            gn={n:float(p.grad.norm()) for n,p in model.named_parameters()}
            assert all(np.isfinite(g) and g>0 for g in gn.values())
            gradients[-1]['total_parameter_gradient_norms']=gn
        optimizer.step();model.normalize()
        if (step+1)%250==0:
            history.append(monitor(model,d,den,ix[0],vi,mask,step+1))
            atomic_torch_save(dict(binding=binding(),model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,batch_sha=old.sha(batchpath),history=history,gradients=gradients,elapsed=elapsed+time.time()-start),progress)
            print('TRAIN_PROGRESS',fi,seed,arm,step+1,'train_mse',round(history[-1]['train_raw_mse'],7),'valid_mse',round(history[-1]['valid_raw_mse'],7),flush=True)
    state=model.state_dict();assert all(torch.isfinite(v).all() for v in state.values())
    delta={n:float((v-initial_state[n]).double().norm()) for n,v in state.items()}
    assert delta['mean']==0. and all(delta[n]>0 for n in delta if n!='mean')
    normerror=float((model.decoder.weight.double().norm(dim=0)-1).abs().max());assert normerror<2e-6
    atomic_torch_save(state,out/'sae.pt')
    save(out/'training.json',dict(fold=fi,seed=seed,arm=arm,lambda_KL=lam,additional_steps=STEPS,train_cells=len(tr['h']),train_donors=np.unique(tr['donor']).tolist(),batch_path=str(batchpath),batch_sha=old.sha(batchpath),start_checkpoint=str(old.src_for(fi,seed)/'sae.pt'),start_checkpoint_sha=old.sha(old.src_for(fi,seed)/'sae.pt'),checkpoint_sha=old.sha(out/'sae.pt'),fixed_train_mean_square=den,selected_step=STEPS,optimizer_reset=True,monitor_history=history,gradient_diagnostics=gradients,weight_delta_norms=delta,decoder_norm_error=normerror,seconds=elapsed+time.time()-start,train_input_sha=array_sha(tr['h']),valid_input_sha=array_sha(d['valid']['h'])))
    progress.unlink(missing_ok=True)
    complete(out,start)
    print('TRAIN_COMPLETE',fi,seed,arm,'seconds',round(elapsed+time.time()-start,2),flush=True)


def preflight():
    O.mkdir(parents=True,exist_ok=True)
    immutable(O/'PROTOCOL.json',CFG)
    srcs=[Path(__file__),ROOT/'work/sae_refine/run.py',ROOT/'work/sae_complementarity/core.py',PROPOSAL,old.R/'protocol.json',old.R/'kang_embeddings.npz']
    source_protocol=read(old.R/'protocol.json')
    assert old.sha(old.R/'kang_embeddings.npz')==source_protocol['source_hashes'][str(old.R/'kang_embeddings.npz')]
    splits={};checks=[]
    for fi in range(6):
        d,fold=data(fi,('train','valid','test'))
        splits[str(fi)]={'donors':fold,'rows':{r:b['rows'].tolist() for r,b in d.items()}}
        assert len(set(np.concatenate([b['rows'] for b in d.values()]).tolist()))==9577
        for seed in SEEDS:
            model=initial(fi,seed)
            np.testing.assert_array_equal(model.mean.numpy(),d['train']['h'].astype(np.float64).mean(0).astype(np.float32))
            p=old.src_for(fi,seed)
            srcs.extend([p/'sae.pt',p/'sae_outputs.npz',p/'sae_result.json'])
            with np.load(p/'sae_outputs.npz') as a:
                for role in ['train','valid']:
                    np.testing.assert_array_equal(d[role]['rows'],a[role+'_rows'])
                    ix=np.unique(np.linspace(0,len(d[role]['h'])-1,12,dtype=int))
                    err=float(np.max(np.abs(refine.infer(model,d[role]['h'][ix])-a[role+'_reconstruction'][ix])))
                    assert err<5e-6
                    checks.append(dict(fold=fi,seed=seed,role=role,cache_max_error=err))
    immutable(W/'sources.json',{str(p):old.sha(p) for p in srcs})
    immutable(O/'SPLITS.json',splits)
    # Independent loss sanity: exact identity gives zero; manual row probabilities agree.
    h=torch.from_numpy(d['train']['h'][:8]);rec=h.clone().requires_grad_()
    mask=torch.arange(4).repeat_interleave(2);mask=mask[:,None]!=mask[None,:]
    kl,_=relation(h,rec,model.mean,mask);assert abs(float(kl))<1e-14
    altered=(h+.02*torch.sin(torch.arange(h.numel()).reshape_as(h))).requires_grad_()
    kl,_=relation(h,altered,model.mean,mask)
    x=F.normalize(h.double()-model.mean.double(),dim=1)
    y=F.normalize(altered.double()-model.mean.double(),dim=1)
    a=(x@x.T)[mask].reshape(8,6).detach().numpy()/.1
    b=(y@y.T)[mask].reshape(8,6).detach().numpy()/.1
    a-=a.max(1,keepdims=True);b-=b.max(1,keepdims=True)
    p=np.exp(a);p/=p.sum(1,keepdims=True);q=np.exp(b);q/=q.sum(1,keepdims=True)
    manual=float(np.mean(np.sum(p*(np.log(p)-np.log(q)),axis=1)))
    assert abs(manual-float(kl))<1e-12
    gg=torch.autograd.grad(kl,altered)[0];assert torch.isfinite(gg).all() and float(gg.norm())>0
    save(O/'preflight.json',dict(status='PASS',checks=checks,identity_kl=0.,manual_KL_error=abs(manual-float(kl)),nonzero_student_gradient=float(gg.norm()),no_new_TEST_score=True))
    save(O/'INTENDED_CLAIMS.json',dict(primary='weak relational reconstruction improves fixed512D donor-heldout MacroF1 over original_h and equal-budget MSE continuation',secondary='strong fixed relation arm and fixed_half are secondary, no TEST-based promotion',scope='exploratory historical Kang, no general SAE benefit or causal biology claim',new_training_fits=54,selected_test_models=150))
    save(O/'EXPERIMENT_TRACKER.json',dict(status='PREFLIGHT_PASS',pilot='NOT_STARTED',development='NOT_STARTED',evaluation='NOT_STARTED',review='NOT_STARTED'))
    save(ROOT/'outputs/sae_relation_preflight/EXECUTION_LINK.json',dict(status='STARTED',tracker=str(O/'EXPERIMENT_TRACKER.json'),protocol=str(O/'PROTOCOL.json')))
    print('PREFLIGHT_PASS',len(checks),'cache checks',flush=True)


def validate_sources():
    assert read(O/'PROTOCOL.json')==CFG
    for p,h in read(W/'sources.json').items():assert old.sha(p)==h,p


def pilot():
    for fi in [0,2]:
        d,_=data(fi)
        for arm in ARMS:train_one(fi,42,arm,d)
    runs=[read(network_dir(fi,42,a)/'training.json') for fi in [0,2] for a in ARMS]
    save(O/'PILOT.json',dict(status='PASS',runs=[{k:r[k] for k in ['fold','arm','additional_steps','seconds','decoder_norm_error']} for r in runs],no_pilot_classification_or_TEST=True,budget_decision='keep proposed1000 extra steps for all arms; this is a fixed-budget comparison, not proof of convergence'))
    save(O/'EXPERIMENT_TRACKER.json',dict(status='PILOT_PASS',pilot='complete',development='NOT_STARTED',evaluation='NOT_STARTED',review='NOT_STARTED'))
    print('PILOT_ALL6_PASS',flush=True)


def probe(fi,seed,method,variant,features,d):
    out=W/'probes'/f'fold_{fi}'/f'seed_{seed}'/method/variant
    if checked_completion(out/'completion.json'):return
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    grid=old.Ridge(features['train'],d['train']['y']);best=-1.;trials=[]
    for alpha in old.ALPHAS:
        grid.fit(alpha);scores=grid.scores(features['valid']);f=old.metrics(d['valid']['y'],scores)['macro_f1']
        r=dict(ridge_alpha=alpha,VALID_F1=f,normal_equation_error=grid.err);trials.append(r)
        if f>best+1e-12:
            best=f
            np.savez_compressed(out/'model.npz',**grid.arrays())
            np.savez_compressed(out/'valid.npz',scores=scores,labels=d['valid']['y'],rows=d['valid']['rows'])
            save(out/'selection.json',dict(r,fold=fi,seed=seed,method=method,variant=variant,mix_alpha=VARIANTS.get(variant,0.)))
    save(out/'trials.json',trials);complete(out,start)
    print('PROBE_COMPLETE',fi,seed,method,variant,flush=True)


def develop():
    assert read(O/'PILOT.json')['status']=='PASS'
    for fi in range(6):
        d,_=data(fi)
        h={r:b['h'] for r,b in d.items()}
        probe(fi,'shared','original_h','original',h,d)
        for seed in SEEDS:
            recs={'existing_raw20':refine.cached(fi,seed,tuple(d))}
            for arm in ARMS:
                train_one(fi,seed,arm,d)
                m=restore(fi,seed,arm)
                recs[arm]={r:refine.infer(m,b['h']) for r,b in d.items()}
            for method,rec in recs.items():
                for variant,alpha in VARIANTS.items():
                    f={r:h[r].astype(np.float64)+alpha*(rec[r].astype(np.float64)-h[r]) for r in d}
                    probe(fi,seed,method,variant,f,d)
            print('DEVELOP_COMPLETE',fi,seed,flush=True)


def lock():
    for fi in range(6):
        assert checked_completion(W/'probes'/f'fold_{fi}'/'seed_shared/original_h/original/completion.json')
        for seed in SEEDS:
            for arm in ARMS:assert checked_completion(network_dir(fi,seed,arm)/'completion.json')
            for method in ['existing_raw20']+list(ARMS):
                for variant in VARIANTS:
                    assert checked_completion(W/'probes'/f'fold_{fi}'/f'seed_{seed}'/method/variant/'completion.json')
    files={str(p):old.sha(p) for name in ['networks','probes','batches'] for p in (W/name).rglob('*') if p.is_file()}
    immutable(O/'SELECTION_LOCK.json',dict(binding=binding(),hashes=files,locked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),roles_seen_for_selection=['TRAIN','VALID']))
    print('ALL_SELECTIONS_LOCKED',len(files),flush=True)


def evaluate():
    seal=read(O/'SELECTION_LOCK.json');assert seal['binding']==binding()
    for p,h in seal['hashes'].items():assert old.sha(p)==h,p
    allresults=[]
    for fi in range(6):
        d,fold=data(fi,('test',));b=d['test'];cache={}
        for selected in sorted((W/'probes'/f'fold_{fi}').glob('seed_*/*/*/selection.json')):
            sel=read(selected);method=sel['method'];variant=sel['variant'];seed=sel['seed']
            if method=='original_h':x=b['h'].astype(np.float64)
            else:
                key=(method,seed)
                if key not in cache:
                    cache[key]=refine.cached(fi,int(seed),('test',))['test'] if method=='existing_raw20' else refine.infer(restore(fi,int(seed),method),b['h'])
                x=b['h'].astype(np.float64)+sel['mix_alpha']*(cache[key].astype(np.float64)-b['h'])
            scores=old.score_model(selected.parent/'model.npz',x);assert np.isfinite(scores).all()
            met=old.metrics(b['y'],scores)
            met.update(dict(fold=fi,test_donor=fold['test'],seed=seed,method=method,variant=variant,selection=sel,relative_embedding_change=float(np.sqrt(np.sum((x-b['h'])**2)/np.sum(b['h'].astype(np.float64)**2)))))
            dest=O/'predictions'/f'fold_{fi}'/f'seed_{seed}'/method/variant;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'test.npz',scores=scores,labels=b['y'],rows=b['rows'])
            save(dest/'metrics.json',met)
            emb=O/'embeddings'/f'fold_{fi}'/f'seed_{seed}'/method;emb.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(emb/f'{variant}.npz',embedding=x,labels=b['y'],rows=b['rows'],donor=b['donor'])
            allresults.append(met)
        print('TEST_FOLD_COMPLETE',fi,flush=True)
    assert len(allresults)==150
    save(O/'all_metrics.json',allresults)
    summarize(allresults)


def summarize(records):
    flat=[{k:r[k] for k in ['fold','test_donor','seed','method','variant','macro_f1','supported_macro_f1','accuracy','relative_embedding_change']}|{'ridge_alpha':r['selection']['ridge_alpha']} for r in records]
    df=pd.DataFrame(flat);df.to_csv(O/'all_metrics.csv',index=False)
    donor=df.groupby(['method','variant','fold','test_donor'],as_index=False)['macro_f1'].mean()
    baseline=donor[donor.method=='original_h'].set_index('fold')['macro_f1']
    rng=np.random.default_rng(20261019);summaries=[]
    for (method,variant),group in donor.groupby(['method','variant']):
        group=group.sort_values('fold');delta=group.macro_f1.to_numpy()-baseline.loc[group.fold].to_numpy()
        intervals=np.quantile(delta[rng.integers(0,6,size=(20000,6))].mean(1)*100,[.025,.975]).tolist()
        sub=df[(df.method==method)&(df.variant==variant)];sd=[]
        for seed,g in sub.groupby('seed'):
            if str(seed)!='shared':sd.append(dict(seed=str(seed),delta_pp=float(100*np.mean(g.macro_f1.to_numpy()-baseline.loc[g.fold].to_numpy()))))
        summaries.append(dict(method=method,variant=variant,macro_f1_percent=float(100*group.macro_f1.mean()),delta_pp=float(100*delta.mean()),positive_donors=int(np.sum(delta>1e-12)),negative_donors=int(np.sum(delta< -1e-12)),descriptive_ci95_pp=intervals,seed_deltas=sd,fits=len(sub)))
    save(O/'summary.json',summaries)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['seed_deltas','descriptive_ci95_pp']} for r in summaries]).to_csv(O/'summary.csv',index=False)
    donor['delta_pp']=100*(donor.macro_f1-donor.fold.map(baseline));donor.to_csv(O/'paired_donor_metrics.csv',index=False)
    contrasts=[]
    for variant in VARIANTS:
        for left,right in [('relation_weak','original_h'),('relation_weak','continue_mse'),('relation_strong','original_h'),('relation_strong','continue_mse'),('continue_mse','existing_raw20')]:
            a=donor[(donor.method==left)&(donor.variant==variant)].set_index('fold').macro_f1
            b=baseline if right=='original_h' else donor[(donor.method==right)&(donor.variant==variant)].set_index('fold').macro_f1
            diff=a-b;ci=np.quantile(diff.to_numpy()[rng.integers(0,6,size=(20000,6))].mean(1)*100,[.025,.975]).tolist()
            seed_delta=[]
            for s in SEEDS:
                aa=df[(df.method==left)&(df.variant==variant)&(df.seed.astype(str)==str(s))].set_index('fold').macro_f1
                bb=baseline if right=='original_h' else df[(df.method==right)&(df.variant==variant)&(df.seed.astype(str)==str(s))].set_index('fold').macro_f1
                seed_delta.append(dict(seed=s,delta_pp=float(100*(aa-bb).mean())))
            contrasts.append(dict(left=left,right=right,variant=variant,delta_pp=float(100*diff.mean()),positive_donors=int((diff>1e-12).sum()),descriptive_ci95_pp=ci,seed_deltas=seed_delta,donor_deltas_pp=(100*diff).to_dict()))
    save(O/'planned_contrasts.json',contrasts)
    weak=next(x for x in contrasts if x['left']=='relation_weak' and x['right']=='original_h' and x['variant']=='reconstruction')
    matched=next(x for x in contrasts if x['left']=='relation_weak' and x['right']=='continue_mse' and x['variant']=='reconstruction')
    save(O/'advance_gate.json',dict(pass_gate=bool(weak['delta_pp']>=.5 and weak['positive_donors']>=4 and sum(x['delta_pp']>0 for x in weak['seed_deltas'])>=2 and matched['delta_pp']>0),vs_original=weak,vs_matched_MSE=matched,note='descriptive development gate, not significance; only predeclared weak reconstruction arm is primary'))
    pc=[]
    for r in records:
        for j,name in enumerate(old.CFG['classes']):pc.append(dict(method=r['method'],variant=r['variant'],fold=r['fold'],seed=r['seed'],cell_type=name,f1=r['per_class_f1'][j],support=r['support'][j]))
    pd.DataFrame(pc).to_csv(O/'per_class_metrics.csv',index=False)
    cm=pd.DataFrame(pc).groupby(['method','variant','fold','cell_type'],as_index=False).f1.mean().groupby(['method','variant','cell_type'],as_index=False).f1.mean()
    base=cm[cm.method=='original_h'].set_index('cell_type').f1
    cm['delta_pp']=100*(cm.f1-cm.cell_type.map(base));cm.to_csv(O/'per_class_summary.csv',index=False)
    save(O/'EXPERIMENT_TRACKER.json',dict(status='EVALUATED_PENDING_VERIFICATION',pilot='complete',development='complete',evaluation='complete',review='NOT_STARTED',selection_lock_sha=old.sha(O/'SELECTION_LOCK.json')))
    print(pd.DataFrame([{k:r[k] for k in ['method','variant','macro_f1_percent','delta_pp','positive_donors']} for r in summaries]).to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','pilot','develop','lock','evaluate']);args=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        if args.stage=='preflight':preflight();return
        validate_sources()
        globals()[args.stage]()


if __name__=='__main__':main()
