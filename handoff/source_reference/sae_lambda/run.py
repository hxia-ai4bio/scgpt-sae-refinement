"""Fixed-grid descriptive lambda sensitivity, using the frozen paired runtime.

Training complexity per fit O(S*(B*d*m+B^2*d)); memory O(N*d+B*m+B^2+d*m).
No TEST-driven winner, budget adaptation, or temperature/architecture change.
"""
import copy
import importlib.util
import os
from pathlib import Path
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
O=ROOT/'outputs/sae_lambda'

def load_runtime(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'work/sae_relation/run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

base=load_runtime('frozen_relation_runtime')
runtime=load_runtime('extended_relation_runtime')
old=base.old
LAMBDAS=[0.,.01,.03,.1,.3,1.,3.]
NEW={'lambda_0p01':.01,'lambda_0p1':.1,'lambda_1':1.,'lambda_3':3.}
REUSED={'continue_mse':0.,'relation_weak':.03,'relation_strong':.3}
METHODS={**REUSED,**NEW}
SEEDS=[42,43,44]
CFG=copy.deepcopy(base.CFG)
CFG.update(arms=METHODS,lambda_grid=LAMBDAS,new_arms=NEW,reused_arms=REUSED,
    variants={'reconstruction':1.},secondary_variant=None,
    pilot={'use':'No new classification pilot. Existing audited runtime and measured speed reused; all new fits retain the fixed1000 budget.'},
    primary_contrasts=['Each fixed lambda vs original_h and equal-budget lambda0 MSE; full grid, no winning lambda selected'],
    secondary_contrasts=['Predeclared leave-one-donor-out sensitivity, all six exclusions for every lambda'],
    advancement='No advance gate or best-lambda decision. Descriptive sensitivity study only.',
    scope='Historically explored Kang ctrl9577cells,6donors,8types. Grid motivated after prior TEST results; not a new blind test, not confirmatory evidence. VALID labels select Ridge alpha only. No PBMC or temperature search.',
    stop_rule='Complete seven fixedlambda curves across sixfolds and threeseeds.72new+54reused trainings;132readouts including6original_h. No additional search based on new TEST scores.',
    reuse='Original54 final networks and54 reconstruction Ridge selections plus6original_h selections reused by verified content hashes, not retrained or treated as new replicates.',
    selection='Each lambda has its own same11alpha VALID Ridge search; no lambda selection by VALID or TEST in this descriptive experiment.',
    sensitivity='Equal-donor mean after within-donor three-seed averaging; show all donor and seed curves; use same bootstrap donor index draws for all contrasts. No cell-level or seed-level independent replicate claims.')
runtime.W=W;runtime.O=O;runtime.ARMS=NEW;runtime.VARIANTS={'reconstruction':1.};runtime.CFG=CFG
save=base.save;read=base.read;sha=old.sha

def checkpoint(fi,seed,method):
    return (runtime if method in NEW else base).network_dir(fi,seed,method)/'sae.pt'

def probe_dir(fi,seed,method):
    home=W if method in NEW else base.W
    variant='original' if method=='original_h' else 'reconstruction'
    return home/'probes'/f'fold_{fi}'/f'seed_{seed}'/method/variant

def restore(fi,seed,method):
    return (runtime if method in NEW else base).restore(fi,seed,method)

def tracker(status,**kwargs):
    p=O/'EXPERIMENT_TRACKER.json';v=read(p) if p.exists() else {}
    v.update(status=status,updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs);save(p,v)

def validate():
    assert read(O/'PROTOCOL.json')==CFG
    for p,h in read(W/'sources.json').items():assert sha(p)==h,p

def preflight():
    O.mkdir(parents=True,exist_ok=True);base.validate_sources()
    oldseal=read(base.O/'SELECTION_LOCK.json')
    for p,h in oldseal['hashes'].items():assert sha(p)==h,p
    base.immutable(O/'PROTOCOL.json',CFG)
    sources={str(p):sha(p) for p in [Path(__file__),base.W/'run.py',base.W/'sources.json',base.O/'PROTOCOL.json',base.O/'SELECTION_LOCK.json',base.O/'AUDIT_SEAL.json',ROOT/'work/sae_refine/run.py',ROOT/'work/sae_complementarity/core.py']}
    sources.update(read(base.W/'sources.json'));base.immutable(W/'sources.json',sources)
    base.immutable(O/'SPLITS.json',read(base.O/'SPLITS.json'))
    manifest=[];checks=[]
    for fi in range(6):
        d,_=base.data(fi)
        assert base.checked_completion(probe_dir(fi,'shared','original_h')/'completion.json')
        for seed in SEEDS:
            ix,mask,p=runtime.batches(fi,seed,d['train'])
            with np.load(base.W/'batches'/f'fold_{fi}_seed_{seed}.npz') as a:
                np.testing.assert_array_equal(ix,a['indices']);np.testing.assert_array_equal(d['train']['rows'][ix],a['rows'])
            for method,lam in REUSED.items():
                network=base.network_dir(fi,seed,method)
                assert base.checked_completion(network/'completion.json')
                assert base.checked_completion(probe_dir(fi,seed,method)/'completion.json')
                r=read(network/'training.json');assert r['lambda_KL']==lam and r['additional_steps']==1000 and r['selected_step']==1000
                manifest.append(dict(fold=fi,seed=seed,method=method,lambda_KL=lam,checkpoint=str(checkpoint(fi,seed,method)),checkpoint_sha=sha(checkpoint(fi,seed,method)),probe=str(probe_dir(fi,seed,method))))
            checks.append(dict(fold=fi,seed=seed,paired_batch_array_sha=base.array_sha(ix),cross_donor_anchors=int(mask.sum(1)[0])))
    save(O/'REUSE_MANIFEST.json',manifest)
    save(O/'preflight.json',dict(status='PASS',reused_networks=54,reused_selected_probes=60,paired_batch_checks=checks,original_lock_files_verified=len(oldseal['hashes']),no_new_TEST_evaluation=True))
    save(O/'INTENDED_CLAIMS.json',dict(question='Does the positive relation-reconstruction signal persist across adjacent lambda values and multiple donors?',scope='Descriptive full fixed grid, no winner selection and no confirmatory/general improvement claim',new_training_fits=72,reused_training_fits=54,test_readouts=132,VALID_candidates=1452))
    tracker('PREFLIGHT_PASS',new_completed=0,new_total=72,reused=54,evaluation='NOT_STARTED',review='NOT_STARTED')
    print('PREFLIGHT_PASS 54network/60probe reuse;72newfits;132totalreadouts',flush=True)

def develop():
    done=0
    for fi in range(6):
        d,_=base.data(fi)
        for seed in SEEDS:
            for arm in NEW:
                runtime.train_one(fi,seed,arm,d)
                model=runtime.restore(fi,seed,arm)
                feat={r:base.refine.infer(model,b['h']).astype('float64') for r,b in d.items()}
                runtime.probe(fi,seed,arm,'reconstruction',feat,d)
                done+=1;tracker('DEVELOPING',new_completed=done)
            print('GRID_BLOCK_COMPLETE',fi,seed,done,'/72',flush=True)

def lock():
    files={}
    for fi in range(6):
        p=probe_dir(fi,'shared','original_h');assert base.checked_completion(p/'completion.json')
        for q in p.rglob('*'):
            if q.is_file():files[str(q)]=sha(q)
        for seed in SEEDS:
            for method in METHODS:
                module=runtime if method in NEW else base
                n=module.network_dir(fi,seed,method);p=probe_dir(fi,seed,method)
                assert module.checked_completion(n/'completion.json') and module.checked_completion(p/'completion.json')
                for folder in [n,p]:
                    for q in folder.rglob('*'):
                        if q.is_file():files[str(q)]=sha(q)
            for home in [W,base.W]:
                p=home/'batches'/f'fold_{fi}_seed_{seed}.npz';files[str(p)]=sha(p)
    base.immutable(O/'SELECTION_LOCK.json',dict(binding=runtime.binding(),hashes=files,locked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),roles_seen_for_new_selection=['TRAIN','VALID'],historical_TEST_previously_seen=True))
    tracker('LOCKED',new_completed=72,evaluation='NOT_STARTED')
    print('SELECTIONS_LOCKED',len(files),flush=True)

def selections(fi):
    yield 'original_h','shared',probe_dir(fi,'shared','original_h')
    for seed in SEEDS:
        for method in sorted(METHODS,key=METHODS.get):yield method,seed,probe_dir(fi,seed,method)

def evaluate():
    seal=read(O/'SELECTION_LOCK.json');assert seal['binding']==runtime.binding()
    for p,h in seal['hashes'].items():assert sha(p)==h,p
    records=[]
    for fi in range(6):
        d,fold=base.data(fi,('test',));b=d['test']
        for method,seed,p in selections(fi):
            sel=read(p/'selection.json')
            x=b['h'].astype('float64') if method=='original_h' else base.refine.infer(restore(fi,seed,method),b['h']).astype('float64')
            scores=old.score_model(p/'model.npz',x);assert np.isfinite(scores).all()
            r=old.metrics(b['y'],scores)
            r.update(fold=fi,test_donor=fold['test'],seed=seed,method=method,lambda_KL=METHODS.get(method),selection=sel,probe_path=str(p),checkpoint=None if method=='original_h' else str(checkpoint(fi,seed,method)),reused=method not in NEW,relative_embedding_change=float(np.linalg.norm(x-b['h'])/np.linalg.norm(b['h'].astype('float64'))))
            dest=O/'predictions'/f'fold_{fi}'/f'seed_{seed}'/method;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'test.npz',scores=scores,labels=b['y'],rows=b['rows'])
            save(dest/'metrics.json',r)
            dest=O/'embeddings'/f'fold_{fi}'/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'{method}.npz',embedding=x,labels=b['y'],rows=b['rows'],donor=b['donor'])
            records.append(r)
        print('TEST_FOLD_COMPLETE',fi,flush=True)
    assert len(records)==132;save(O/'all_metrics.json',records)
    tracker('EVALUATED_PENDING_VERIFICATION',evaluation='complete')
    summarize(records)

def summarize(records):
    df=pd.DataFrame([{k:r[k] for k in ['fold','test_donor','seed','method','lambda_KL','macro_f1','supported_macro_f1','accuracy','relative_embedding_change','reused']} for r in records]);df.to_csv(O/'all_metrics.csv',index=False)
    donors=df.groupby(['method','fold','test_donor'],as_index=False).macro_f1.mean()
    h=donors[donors.method=='original_h'].set_index('fold').macro_f1
    mse=donors[donors.method=='continue_mse'].set_index('fold').macro_f1
    draw=np.random.default_rng(20261019).integers(0,6,size=(20000,6))
    rows=[];contrasts=[];sens=[];seedrows=[]
    for method in ['original_h']+sorted(METHODS,key=METHODS.get):
        a=donors[donors.method==method].sort_values('fold').set_index('fold').macro_f1
        r=dict(method=method,lambda_KL=METHODS.get(method),macro_f1_percent=float(a.mean()*100),reused=method not in NEW)
        for control,ref in [('original_h',h),('continue_mse',mse)]:
            delta=100*(a-ref);ci=np.quantile(delta.to_numpy()[draw].mean(1),[.025,.975]).tolist()
            seed_delta=[]
            if method!='original_h':
                for seed in SEEDS:
                    q=df[(df.method==method)&(df.seed.astype(str)==str(seed))].set_index('fold').macro_f1
                    z=h if control=='original_h' else df[(df.method=='continue_mse')&(df.seed.astype(str)==str(seed))].set_index('fold').macro_f1
                    sd=float(100*(q-z).mean());seed_delta.append(dict(seed=seed,delta_pp=sd))
                    seedrows.append(dict(method=method,lambda_KL=METHODS[method],seed=seed,control=control,delta_pp=sd,macro_f1_percent=float(q.mean()*100)))
            c=dict(method=method,lambda_KL=METHODS.get(method),control=control,delta_pp=float(delta.mean()),positive_donors=int((delta>1e-10).sum()),descriptive_ci95_pp=ci,positive_seed_means=sum(s['delta_pp']>1e-10 for s in seed_delta),seed_deltas=seed_delta,donor_deltas_pp=delta.to_dict())
            contrasts.append(c)
            if control=='original_h':r.update(delta_pp=c['delta_pp'],positive_donors=c['positive_donors'],positive_seed_means=c['positive_seed_means'],descriptive_ci95_pp=ci)
            else:r.update(delta_vs_MSE_pp=c['delta_pp'],positive_donors_vs_MSE=c['positive_donors'],positive_seed_means_vs_MSE=c['positive_seed_means'])
            for dropped in range(6):sens.append(dict(method=method,lambda_KL=METHODS.get(method),control=control,excluded_fold=dropped,delta_pp=float(delta.drop(dropped).mean())))
        rows.append(r)
    donors['lambda_KL']=donors.method.map(METHODS);donors['delta_vs_h_pp']=100*(donors.macro_f1-donors.fold.map(h));donors['delta_vs_MSE_pp']=100*(donors.macro_f1-donors.fold.map(mse))
    donors.to_csv(O/'paired_donor_metrics.csv',index=False);pd.DataFrame(seedrows).to_csv(O/'seed_metrics.csv',index=False)
    pd.DataFrame(sens).to_csv(O/'leave_one_donor_out_sensitivity.csv',index=False)
    save(O/'summary.json',rows);pd.DataFrame([{k:v for k,v in r.items() if k!='descriptive_ci95_pp'} for r in rows]).to_csv(O/'summary.csv',index=False)
    save(O/'planned_contrasts.json',contrasts)
    training=[];grad=[]
    for fi in range(6):
        for seed in SEEDS:
            for method,lam in METHODS.items():
                p=checkpoint(fi,seed,method).parent;t=read(p/'training.json');training.append(t)
                for g in t['gradient_diagnostics']:grad.append(dict(fold=fi,seed=seed,method=method,lambda_KL=lam,**g))
    save(O/'training_summary.json',training)
    pd.DataFrame(grad).to_csv(O/'gradient_diagnostics.csv',index=False)
    curve=[]
    for t in training:
        for m in t['monitor_history']:curve.append(dict(fold=t['fold'],seed=t['seed'],method=t['arm'],lambda_KL=t['lambda_KL'],**m))
    pd.DataFrame(curve).to_csv(O/'learning_curves.csv',index=False)
    pc=[]
    for r in records:
        for j,name in enumerate(old.CFG['classes']):pc.append(dict(method=r['method'],lambda_KL=r['lambda_KL'],fold=r['fold'],seed=r['seed'],cell_type=name,f1=r['per_class_f1'][j],support=r['support'][j]))
    pd.DataFrame(pc).to_csv(O/'per_class_metrics.csv',index=False)
    print(pd.DataFrame([{k:r[k] for k in ['method','lambda_KL','macro_f1_percent','delta_pp','positive_donors']} for r in rows]).to_string(index=False),flush=True)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','develop','lock','evaluate','all']);a=ap.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        if a.stage in ['all','preflight']:preflight()
        if a.stage!='preflight':
            validate()
            for stage in (['develop','lock','evaluate'] if a.stage=='all' else [a.stage]):
                print('PIPELINE_STAGE',stage,flush=True);globals()[stage]()
    print('PIPELINE_COMPLETE',flush=True)

if __name__=='__main__':main()
