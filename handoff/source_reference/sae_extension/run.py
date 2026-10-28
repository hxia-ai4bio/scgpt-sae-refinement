"""Dataset-specific SAE refinement on stim and PBMC, fixed seven-point lambda grid.

Warmup O(20*N*d*m); continuation O(1000*(B*d*m+B^2*d)); memory O(N*d+B*m+B^2+d*m).
Ridge O(N*d^2+d^3), d512,m2048,B256. No cross-dataset SAE reuse or lambda selection.
"""
import os
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_extension'
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(key,'4')
import argparse,copy,importlib.util,time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

def import_file(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=import_file('extension_frozen_relation',ROOT/'work/sae_relation/run.py')
old=base.old;save=old.save;read=old.read;sha=old.sha
LAMBDAS=[0.,.01,.03,.1,.3,1.,3.]
ARMS=dict(zip(['lambda_0','lambda_0p01','lambda_0p03','lambda_0p1','lambda_0p3','lambda_1','lambda_3'],LAMBDAS))
SEEDS=[42,43,44]
DATASETS=['stim','pbmc']
R=old.R

def dsout(ds):return O/ds
def dswork(ds):return W/ds
def nfold(ds):return 6 if ds=='stim' else 2
def initialdir(ds,fi,seed):return dswork(ds)/'initial'/f'fold_{fi}'/f'seed_{seed}'

def update(status,**values):
    p=O/'EXPERIMENT_TRACKER.json';d=read(p) if p.exists() else {}
    d.update(status=status,updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values);save(p,d)

def canonical(ds):
    with np.load(W/'data'/f'{ds}.npz') as a:return {k:a[k].copy() for k in a.files}

def make_splits(ds,a):
    out=[]
    if ds=='stim':
        p=read(R/'protocol.json')
        for fi in range(6):
            f=p['folds'][fi]['fold'];ix={r:np.flatnonzero(~np.isin(a['groups'],[f['valid'],f['test']])) if r=='train' else np.flatnonzero(a['groups']==f[r]) for r in ['train','valid','test']}
            out.append(dict(fold=fi,test_group=f['test'],valid_group=f['valid'],roles={r:v.tolist() for r,v in ix.items()}))
    else:
        groups=sorted(np.unique(a['groups']).tolist())
        assert len(groups)==2
        # VALID is a fixed stratified subset of the SOURCE batch; outer TEST batch untouched.
        for fi,target in enumerate(reversed(groups)):
            source=next(g for g in groups if g!=target);src=np.flatnonzero(a['groups']==source)
            rng=np.random.default_rng(np.random.SeedSequence([20261020,fi,200]))
            train=[];valid=[]
            for label in range(8):
                ix=src[a['labels'][src]==label];assert len(ix)>=2
                ix=rng.permutation(ix);nv=min(len(ix)-1,max(1,int(round(.2*len(ix)))))
                valid.extend(ix[:nv].tolist());train.extend(ix[nv:].tolist())
            out.append(dict(fold=fi,test_group=target,valid_group=source+' / fixed20pct',source_group=source,roles=dict(train=sorted(train),valid=sorted(valid),test=np.flatnonzero(a['groups']==target).tolist())))
    return out

def data(ds,fi,roles=('train','valid')):
    a=canonical(ds);f=read(dsout(ds)/'SPLITS.json')[fi];d={}
    for role in roles:
        ix=np.array(f['roles'][role],dtype=np.int64)
        d[role]=dict(h=a['embedding'][ix].astype('float32'),y=a['labels'][ix].astype('int64'),rows=a['rows'][ix],donor=a['groups'][ix])
    return d,f

def cfg(ds):
    c=copy.deepcopy(base.CFG)
    c.update(dataset=ds,folds=list(range(nfold(ds))),arms=ARMS,seeds=SEEDS,reference_arms=['original_h'],variants={'reconstruction':1.},secondary_variant=None,
        warmup='Dataset-specific MSE SAE: maximum20epochs, VALID rawMSE earliest best, epoch0 eligible, patience3; same seed/order/Adam3e-4 as ctrl initialization recipe. Four TRAIN donors for stim; source-batch TRAIN cells for PBMC.',
        initialization='Same dataset/fold/seed warmup checkpoint for all7lambdas; no weights or mean from Kang ctrl or the other dataset.',
        batch_seed='SeedSequence([20261020,dataset_index,fold,seed,1000]); fixed persisted indices paired across all7lambdas',
        checkpoint='Fixed continuation finalstep1000; warmup selection independently fixed by VALID MSE. No continuation checkpoint selection.',
        primary_contrasts=['Predeclared reference lambda0.3 vs original_h and same-dataset equal-budget lambda0'],
        secondary_contrasts=['lambda0.1 and complete sevenpoint sensitivity; no per-dataset winner selection'],
        advancement='No bestlambda or automatic followup decision; report full fixedgrid and all heldout groups.',
        pilot={'use':'Extraction replay and representative training health only; no TEST readout or adaptive budget decision'},
        selection='For each fixedlambda same11Ridge alpha choices using source TRAIN scaling/classbalanced weightedRidge, VALID fixed8class MacroF1, earliest ascendingalpha ties; no refit TRAIN+VALID or selectinglambda.',
        stop_rule='Complete all7fixedlambdas,3seeds,all declared splits; no TEST-driven new params, no temperature or mixture search.',
        statistics='Average seeds within each heldout group then equal group mean; seed variation is not biological replication. No pooled donor/batch significance or cell bootstrap.',
        scope='Dataset-specific refinement on historically explored data; scGPT frozen. stim shares donors with prior ctrl and is not an independent cohort. PBMC has batch labels only, no verified donors/study independence; previous crossdataset-transfer scores not directly comparable.')
    if ds=='stim':
        c.update(split='Six original Kang donor rotations:4TRAIN,1VALID,1TEST; stim only; original two reserved donors excluded.',group_unit='donor',batch_size=256,cells_per_train_donor=64,relations='Centered cosine teacher||student KL tau0.1;192 other-TRAIN-donor anchors/query; detachedteacher;float64 logsoftmax.',relation_anchor_count=192,relation_mask='cross_train_donor',validation='One entire donor',descriptive_ci='paired six-donor bootstrap,20000draws,seed20261020; overlapping TRAIN folds,descriptive only')
    else:
        c.update(split='Two directions:Batch1→Batch2 and Batch2→Batch1. Fixed class-stratified20pct SOURCE batch VALID; remainder TRAIN; entire otherbatch TEST. Same split for all seeds/lambdas.',group_unit='batch',cells_per_train_donor=None,relations='Same centered cosine teacher||student KL tau0.1, but255 other TRAIN cells/query (nonself) because SOURCE TRAIN has onlyonebatch; not crossdonor/crossbatch relation training.',relation_anchor_count=255,relation_mask='all_nonself_train',validation='Stratified cells within SOURCE batch only; labels inform split but not SAE losses',descriptive_ci=None,pbmc_limitation='Cannot instantiate crossdonor relation constraint with missing donor IDs. This explicit withinbatch relation variant has different anchor/mask semantics from Kang; no pseudodonors are invented.')
    return c

class ReadoutProxy:
    def __init__(self,ds):self.ds=ds
    def __getattr__(self,key):return getattr(old,key)
    def src_for(self,fi,seed):return initialdir(self.ds,fi,seed)

def configure(ds):
    rt=import_file('relation_runtime_'+ds,ROOT/'work/sae_relation/run.py')
    rt.W=dswork(ds);rt.O=dsout(ds);rt.ARMS=ARMS;rt.VARIANTS={'reconstruction':1.};rt.CFG=cfg(ds);rt.old=ReadoutProxy(ds)
    def init(fi,seed):
        state=torch.load(initialdir(ds,fi,seed)/'sae.pt',map_location='cpu',weights_only=False)
        model=base.refine.SAE(state['mean'],2048,32,seed);model.load_state_dict(state);return model
    def batch(fi,seed,b):
        rng=np.random.default_rng(np.random.SeedSequence([20261020,DATASETS.index(ds),fi,seed,1000]))
        if ds=='stim':
            groups=np.unique(b['donor']);assert len(groups)==4
            choices=[np.flatnonzero(b['donor']==g) for g in groups];assert all(len(c)>=64 for c in choices)
            ix=np.asarray([np.concatenate([rng.choice(c,64,replace=False) for c in choices]) for _ in range(1000)],dtype=np.int64)
            g=torch.arange(4).repeat_interleave(64);mask=g[:,None]!=g[None,:]
        else:
            groups=np.unique(b['donor']);assert len(groups)==1 and len(b['h'])>=256
            ix=np.asarray([rng.choice(len(b['h']),256,replace=False) for _ in range(1000)],dtype=np.int64);mask=~torch.eye(256,dtype=torch.bool)
        assert (mask.sum(1)==cfg(ds)['relation_anchor_count']).all()
        p=dswork(ds)/'batches'/f'fold_{fi}_seed_{seed}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            with np.load(p) as a:np.testing.assert_array_equal(a['indices'],ix);np.testing.assert_array_equal(a['rows'],b['rows'][ix])
        else:np.savez_compressed(p,indices=ix,rows=b['rows'][ix],groups=groups)
        return ix,mask,p
    rt.initial=init;rt.batches=batch;rt.data=lambda fi,roles=('train','valid'):data(ds,fi,roles)
    return rt

def train_initial(ds,fi,seed,d,rt):
    p=initialdir(ds,fi,seed)
    if rt.checked_completion(p/'completion.json'):return
    p.mkdir(parents=True,exist_ok=True);start=time.time();tr=d['train']['h'];va=d['valid']['h']
    mean=tr.astype('float64').mean(0).astype('float32');model=base.refine.SAE(mean,2048,32,seed)
    initial=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.0003)
    best=float('inf');history=[];stale=0;updates=0;first_grad=None
    for epoch in range(21):
        rec=base.refine.infer(model,va);mse=float(np.mean((rec.astype('float64')-va.astype('float64'))**2))
        history.append(dict(epoch=epoch,valid_mse=mse,updates=updates))
        if mse<best:best=mse;best_epoch=epoch;beststate=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if epoch==20 or stale>=3:break
        order=np.random.default_rng(np.random.SeedSequence([seed,epoch,20261014])).permutation(len(tr));model.train()
        for j in range(0,len(order),256):
            h=torch.from_numpy(tr[order[j:j+256]]);opt.zero_grad(set_to_none=True);r,z=model(h);loss=(r-h).square().mean();assert torch.isfinite(loss)
            loss.backward()
            if updates==0:
                first_grad={n:float(v.grad.norm()) for n,v in model.named_parameters()};assert all(np.isfinite(v) and v>0 for v in first_grad.values())
            opt.step();model.normalize();updates+=1
    model.load_state_dict(beststate);assert best_epoch>0
    rt.atomic_torch_save(beststate,p/'sae.pt')
    save(p/'training.json',dict(dataset=ds,fold=fi,seed=seed,history=history,selected_epoch=best_epoch,updates_executed=updates,selected_updates=history[best_epoch]['updates'],valid_mse=best,first_gradient_norms=first_grad,train_input_sha=base.array_sha(tr),valid_input_sha=base.array_sha(va),train_groups=np.unique(d['train']['donor']).tolist(),checkpoint_sha=sha(p/'sae.pt'),changed_weight_norms={n:float((v-initial[n]).double().norm()) for n,v in beststate.items()},seconds=time.time()-start))
    rt.complete(p,start);print('INITIAL_COMPLETE',ds,fi,seed,'epoch',best_epoch,'seconds',round(time.time()-start,2),flush=True)

def preflight(selected=None):
    selected=DATASETS if selected is None else selected
    upstream=read(R/'data_manifest.json')
    srcpaths=[Path(__file__),W/'prepare.py',ROOT/'work/sae_relation/run.py',ROOT/'work/sae_refine/run.py',ROOT/'work/sae_complementarity/core.py',R/'data_manifest.json',R/'prepare.py',Path(upstream['preprocess']['dataset']),Path('/Users/apple/Desktop/Llava-Cell/dataset/PBMC368k_raw.h5ad'),R/'pbmc_embeddings.npz',Path(upstream['preprocess']['checkpoint_dir'])/'best_model.pt',Path(upstream['preprocess']['checkpoint_dir'])/'args.json',Path(upstream['preprocess']['checkpoint_dir'])/'vocab.json']
    sources={str(p):sha(p) for p in srcpaths}
    for p,h in sources.items():
        if p in upstream['sources']:assert h==upstream['sources'][p],p
    old.immutable(W/'sources.json',sources)
    details=read(O/'preflight.json')['splits'] if (O/'preflight.json').exists() else []
    for ds in selected:
        if ds=='stim':
            m=read(O/'DATA_MANIFEST.json');assert m['status']=='PASS'
            for p,h in {**m['sources'],**m['canonical_files']}.items():assert sha(p)==h,p
        else:
            bench=read(O/'EXTRACTION_BENCHMARK.json');assert bench['pbmc_replay8_max_error']<3e-5
        a=canonical(ds);splits=make_splits(ds,a);path=dsout(ds);path.mkdir(exist_ok=True)
        ds_sources={**sources,str(W/'data'/f'{ds}.npz'):sha(W/'data'/f'{ds}.npz')}
        if ds=='stim':ds_sources[str(O/'DATA_MANIFEST.json')]=sha(O/'DATA_MANIFEST.json')
        old.immutable(path/'PROTOCOL.json',cfg(ds));old.immutable(path/'SPLITS.json',splits);old.immutable(dswork(ds)/'sources.json',ds_sources)
        if ds=='pbmc':
            with np.load(R/'pbmc_embeddings.npz') as pb:
                for key,src in [('embedding','embedding'),('labels','labels'),('rows','rows'),('groups','batches'),('cell_ids','cell_ids')]:np.testing.assert_array_equal(a[key],pb[src])
        assert np.isfinite(a['embedding']).all() and a['embedding'].shape==(len(a['rows']),512) and len(set(a['cell_ids']))==len(a['rows'])
        for f in splits:
            ix={r:np.array(v) for r,v in f['roles'].items()}
            assert not set(ix['train'])&set(ix['valid']) and not set(ix['train'])&set(ix['test']) and not set(ix['valid'])&set(ix['test'])
            assert set(np.concatenate(list(ix.values())))==set(range(len(a['rows'])))
            assert not set(a['groups'][ix['train']])&set(a['groups'][ix['test']])
            if ds=='stim':assert not set(a['groups'][ix['train']])&set(a['groups'][ix['valid']])
            counts={r:np.bincount(a['labels'][v],minlength=8).tolist() for r,v in ix.items()}
            assert all(v>0 for v in counts['train']) and min(len(ix[r]) for r in ix)>128
            row=dict(dataset=ds,fold=f['fold'],test_group=f['test_group'],counts={r:len(v) for r,v in ix.items()},class_counts=counts,groups={r:np.unique(a['groups'][v]).tolist() for r,v in ix.items()})
            prior=[v for v in details if v['dataset']==ds and v['fold']==f['fold']]
            if prior:assert prior[0]==row
            else:details.append(row)
    save(O/'preflight.json',dict(status='PASS',datasets_ready=sorted(set(r['dataset'] for r in details)),splits=details,continuation_fits=168,warmup_fits=24,selected_readouts=176,VALID_candidates=1936,no_TEST_score=True))
    save(O/'INTENDED_CLAIMS.json',dict(question='Does fixed relation reconstruction show same-dataset refinement gains on Kang stim and PBMC?',reference='lambda0.3 frozen from prior ctrl; also reportlambda0.1 and entiregrid without choosing per-dataset winner',scope='stim same historical donors/newcondition, not independentcohort; PBMC dataset-specific bidirectional batchholdout and withinTRAIN nonself relation variant, not donor-generalization or SAE transfer',not_supported_by_design=['Universal SAE-specific benefit versus nonSAE nonlinear controls','Causal biological mechanism','Best lambda selection from TEST','Crossdonor PBMC validation'],new_warmup_fits=24,new_continuation_fits=168,new_TEST_readouts=176))
    warm=sum(len(list((dswork(ds)/'initial').glob('fold_*/seed_*/completion.json'))) for ds in DATASETS)
    cont=sum(len(list((dswork(ds)/'networks').glob('fold_*/seed_*/*/completion.json'))) for ds in DATASETS)
    update('PREFLIGHT_PASS',warmup_completed=warm,warmup_total=24,continuation_completed=cont,continuation_total=168,evaluation='NOT_STARTED',review='NOT_STARTED')
    print('PREFLIGHT_PASS',details,flush=True)

def validate(selected=None):
    selected=DATASETS if selected is None else selected
    for p,h in read(W/'sources.json').items():assert sha(p)==h,p
    for ds in selected:
        assert read(dsout(ds)/'PROTOCOL.json')==cfg(ds)
        for p,h in read(dswork(ds)/'sources.json').items():assert sha(p)==h,p

def develop(selected=None):
    warm=sum(len(list((dswork(ds)/'initial').glob('fold_*/seed_*/completion.json'))) for ds in DATASETS)
    cont=sum(len(list((dswork(ds)/'networks').glob('fold_*/seed_*/*/completion.json'))) for ds in DATASETS)
    # PBMC first: its validated embedding cache is smaller; same predeclared protocol.
    for ds in (['pbmc','stim'] if selected is None else selected):
        rt=configure(ds)
        for fi in range(nfold(ds)):
            d,_=data(ds,fi);h={r:b['h'] for r,b in d.items()};rt.probe(fi,'shared','original_h','original',h,d)
            for seed in SEEDS:
                train_initial(ds,fi,seed,d,rt)
                warm=sum(len(list((dswork(s)/'initial').glob('fold_*/seed_*/completion.json'))) for s in DATASETS)
                for arm in ARMS:
                    rt.train_one(fi,seed,arm,d);model=rt.restore(fi,seed,arm)
                    features={r:base.refine.infer(model,b['h']).astype('float64') for r,b in d.items()};rt.probe(fi,seed,arm,'reconstruction',features,d)
                    cont=sum(len(list((dswork(s)/'networks').glob('fold_*/seed_*/*/completion.json'))) for s in DATASETS)
                    update('DEVELOPING',dataset=ds,fold=fi,seed=seed,warmup_completed=warm,continuation_completed=cont)
                print('BLOCK_COMPLETE',ds,fi,seed,cont,'/168',flush=True)

def lock():
    allfiles={}
    for ds in DATASETS:
        rt=configure(ds);files={}
        for fi in range(nfold(ds)):
            assert rt.checked_completion(rt.W/'probes'/f'fold_{fi}'/'seed_shared/original_h/original/completion.json')
            for seed in SEEDS:
                assert rt.checked_completion(initialdir(ds,fi,seed)/'completion.json')
                for arm in ARMS:
                    assert rt.checked_completion(rt.network_dir(fi,seed,arm)/'completion.json')
                    assert rt.checked_completion(rt.W/'probes'/f'fold_{fi}'/f'seed_{seed}'/arm/'reconstruction/completion.json')
        for name in ['initial','networks','probes','batches']:
            for p in (rt.W/name).rglob('*'):
                if p.is_file():files[str(p)]=sha(p)
        old.immutable(rt.O/'SELECTION_LOCK.json',dict(binding=rt.binding(),hashes=files,roles_seen_for_selection=['TRAIN','VALID'],locked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        allfiles.update(files)
    old.immutable(O/'SELECTION_LOCK.json',dict(hashes=allfiles,per_dataset_locks={ds:sha(dsout(ds)/'SELECTION_LOCK.json') for ds in DATASETS},no_new_TEST_readout_before_lock=True))
    update('LOCKED',warmup_completed=24,continuation_completed=168);print('ALL_SELECTIONS_LOCKED',len(allfiles),flush=True)

def selections(ds,fi):
    home=dswork(ds)/'probes'/f'fold_{fi}'
    yield 'original_h','shared',home/'seed_shared/original_h/original'
    for seed in SEEDS:
        for arm in ARMS:yield arm,seed,home/f'seed_{seed}'/arm/'reconstruction'

def evaluate():
    seal=read(O/'SELECTION_LOCK.json')
    for p,h in seal['hashes'].items():assert sha(p)==h,p
    allrecords=[]
    for ds in DATASETS:
        rt=configure(ds);records=[]
        for fi in range(nfold(ds)):
            d,f=data(ds,fi,('test',));b=d['test']
            for method,seed,p in selections(ds,fi):
                sel=read(p/'selection.json');x=b['h'].astype('float64') if method=='original_h' else base.refine.infer(rt.restore(fi,seed,method),b['h']).astype('float64')
                scores=old.score_model(p/'model.npz',x);met=old.metrics(b['y'],scores)
                met.update(dataset=ds,fold=fi,test_group=f['test_group'],group_unit=cfg(ds)['group_unit'],seed=seed,method=method,lambda_KL=ARMS.get(method),probe_path=str(p),selection=sel,relative_embedding_change=float(np.linalg.norm(x-b['h'])/np.linalg.norm(b['h'].astype('float64'))))
                dest=dsout(ds)/'predictions'/f'fold_{fi}'/f'seed_{seed}'/method;dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/'test.npz',scores=scores,labels=b['y'],rows=b['rows']);save(dest/'metrics.json',met)
                ep=dsout(ds)/'embeddings'/f'fold_{fi}'/f'seed_{seed}';ep.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(ep/f'{method}.npz',embedding=x,labels=b['y'],rows=b['rows'],groups=b['donor'])
                records.append(met)
            print('TEST_FOLD_COMPLETE',ds,fi,flush=True)
        assert len(records)==22*nfold(ds);save(dsout(ds)/'all_metrics.json',records);allrecords.extend(records)
    assert len(allrecords)==176;save(O/'all_metrics.json',allrecords);summarize(allrecords);update('EVALUATED_PENDING_VERIFICATION',evaluation='complete')

def summarize(records):
    flat=[{k:r[k] for k in ['dataset','fold','test_group','group_unit','seed','method','lambda_KL','macro_f1','supported_macro_f1','accuracy','relative_embedding_change']} for r in records]
    df=pd.DataFrame(flat);df.to_csv(O/'all_metrics.csv',index=False)
    group=df.groupby(['dataset','method','fold','test_group','group_unit'],as_index=False).macro_f1.mean();summaries=[];contrasts=[];sens=[];seedrows=[]
    for ds in DATASETS:
        g=group[group.dataset==ds];h=g[g.method=='original_h'].set_index('fold').macro_f1;zero=g[g.method=='lambda_0'].set_index('fold').macro_f1
        draw=np.random.default_rng(20261020).integers(0,nfold(ds),size=(20000,nfold(ds))) if ds=='stim' else None
        for method in ['original_h']+list(ARMS):
            v=g[g.method==method].sort_values('fold').set_index('fold').macro_f1
            row=dict(dataset=ds,group_unit=cfg(ds)['group_unit'],method=method,lambda_KL=ARMS.get(method),macro_f1_percent=float(v.mean()*100),groups=nfold(ds))
            for control,ref in [('original_h',h),('lambda_0',zero)]:
                delta=(v-ref)*100;sd=[]
                for seed in ([] if method=='original_h' else SEEDS):
                    a=df[(df.dataset==ds)&(df.method==method)&(df.seed.astype(str)==str(seed))].set_index('fold').macro_f1
                    b=h if control=='original_h' else df[(df.dataset==ds)&(df.method=='lambda_0')&(df.seed.astype(str)==str(seed))].set_index('fold').macro_f1
                    sd.append(dict(seed=seed,delta_pp=float(100*(a-b).mean())));seedrows.append(dict(dataset=ds,method=method,lambda_KL=ARMS.get(method),seed=seed,control=control,macro_f1_percent=float(a.mean()*100),delta_pp=sd[-1]['delta_pp']))
                ci=None if draw is None else np.quantile(delta.to_numpy()[draw].mean(1),[.025,.975]).tolist()
                c=dict(dataset=ds,group_unit=cfg(ds)['group_unit'],method=method,lambda_KL=ARMS.get(method),control=control,delta_pp=float(delta.mean()),positive_groups=int((delta>1e-10).sum()),positive_seed_means=sum(s['delta_pp']>1e-10 for s in sd),seed_deltas=sd,group_deltas_pp=delta.to_dict(),descriptive_ci95_pp=ci)
                contrasts.append(c)
                if control=='original_h':row.update(delta_pp=c['delta_pp'],positive_groups=c['positive_groups'],positive_seed_means=c['positive_seed_means'],descriptive_ci95_pp=ci)
                else:row.update(delta_vs_MSE_pp=c['delta_pp'],positive_groups_vs_MSE=c['positive_groups'],positive_seed_means_vs_MSE=c['positive_seed_means'])
                for dropped in range(nfold(ds)):sens.append(dict(dataset=ds,method=method,lambda_KL=ARMS.get(method),control=control,excluded_fold=dropped,delta_pp=float(delta.drop(dropped).mean())))
            summaries.append(row)
        mask=group.dataset==ds;group.loc[mask,'delta_vs_h_pp']=100*(group.loc[mask,'macro_f1']-group.loc[mask,'fold'].map(h));group.loc[mask,'delta_vs_MSE_pp']=100*(group.loc[mask,'macro_f1']-group.loc[mask,'fold'].map(zero))
    group['lambda_KL']=group.method.map(ARMS);group.to_csv(O/'paired_group_metrics.csv',index=False)
    save(O/'summary.json',summaries);pd.DataFrame([{k:v for k,v in r.items() if k!='descriptive_ci95_pp'} for r in summaries]).to_csv(O/'summary.csv',index=False)
    save(O/'planned_contrasts.json',contrasts);pd.DataFrame(seedrows).to_csv(O/'seed_metrics.csv',index=False);pd.DataFrame(sens).to_csv(O/'leave_one_group_out_sensitivity.csv',index=False)
    curve=[];grad=[];training=[];initial=[]
    for ds in DATASETS:
        rt=configure(ds)
        for fi in range(nfold(ds)):
            for seed in SEEDS:
                initial.append(read(initialdir(ds,fi,seed)/'training.json'))
                for method,lam in ARMS.items():
                    t=read(rt.network_dir(fi,seed,method)/'training.json');training.append(dict(dataset=ds,**t))
                    for m in t['monitor_history']:curve.append(dict(dataset=ds,fold=fi,seed=seed,method=method,lambda_KL=lam,**m))
                    for m in t['gradient_diagnostics']:grad.append(dict(dataset=ds,fold=fi,seed=seed,method=method,lambda_KL=lam,**m))
    save(O/'training_summary.json',training);save(O/'initial_training_summary.json',initial);pd.DataFrame(curve).to_csv(O/'learning_curves.csv',index=False);pd.DataFrame(grad).to_csv(O/'gradient_diagnostics.csv',index=False)
    pc=[]
    for r in records:
        for j,name in enumerate(old.CFG['classes']):pc.append(dict(dataset=r['dataset'],method=r['method'],lambda_KL=r['lambda_KL'],fold=r['fold'],seed=r['seed'],cell_type=name,f1=r['per_class_f1'][j],support=r['support'][j]))
    pd.DataFrame(pc).to_csv(O/'per_class_metrics.csv',index=False)
    print(pd.DataFrame([{k:r[k] for k in ['dataset','method','macro_f1_percent','delta_pp','positive_groups']} for r in summaries]).to_string(index=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','develop','lock','evaluate']);ap.add_argument('--dataset',choices=DATASETS);a=ap.parse_args();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    selected=[a.dataset] if a.dataset else None
    with threadpool_limits(limits=4):
        if a.stage=='preflight':preflight(selected)
        elif a.stage=='develop':validate(selected);develop(selected)
        else:
            assert selected is None,'Lock/evaluate both datasets together'
            validate();globals()[a.stage]()

if __name__=='__main__':main()
