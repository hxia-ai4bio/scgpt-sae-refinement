"""Development and locked evaluation are separate commands, with no test sweep."""
import argparse, time, shutil, sys
import numpy as np
import torch
from core import *

def original(fi):
    out=W/'shared'/f'fold_{fi}'/'original_h'
    if completed(out/'completion.json'):return
    start=time.time();d,_,_=load(fi,42);out.mkdir(parents=True,exist_ok=True)
    fit_family(out,d,'original_h',[None]);complete(out,start)

def save_selected(out,model,scores,sel,b):
    np.savez_compressed(out/'model.npz',**model.arrays())
    np.savez_compressed(out/'selected_VALID.npz',scores=scores,labels=b['y'],rows=b['rows'])
    save(out/'selection.json',sel)

def fit_family(out,d,kind,params):
    out.mkdir(parents=True,exist_ok=True);tr=d['train'];va=d['valid'];records=[];allscore=[];best=-1.;bestsel=None
    for param in params:
        x=fixed_features(tr,kind,param);v=fixed_features(va,kind,param);model=Ridge(x,tr['y'])
        for alpha in ALPHAS:
            model.fit(alpha);s=model.scores(v);f=metrics(va['y'],s)['macro_f1']
            row={'param':param,'alpha':alpha,'VALID_F1':f,'normal_equation_error':model.err};records.append(row);allscore.append(s)
            if f>best+1e-12:
                best=f;bestsel=dict(row,method=kind,identity_selected=(kind=='original_h' or (kind=='fixed_Dz' and param==0) or (kind=='residual_mix' and param==1)))
                save_selected(out,model,s,bestsel,va)
        del model
    save(out/'trials.json',records);np.savez_compressed(out/'grid_VALID.npz',scores=np.stack(allscore),labels=va['y'],rows=va['rows'])

def crossfit(fi):
    out=W/'shared'/f'fold_{fi}'/'crossfit'
    if completed(out/'completion.json'):return
    start=time.time();out.mkdir(parents=True,exist_ok=True);d,_,_=load(fi,42);tr=d['train'];h=tr['h'];y=tr['y'];don=tr['donor'];pred=np.zeros((len(h),8));records=[]
    for held in np.unique(don):
        fit=don!=held;groups=np.unique(don[fit]);cv=np.zeros((len(ALPHAS),len(groups)));trials=[]
        for j,valid in enumerate(groups):
            ti=fit&(don!=valid);vi=don==valid;model=Ridge(h[ti],y[ti])
            for k,a in enumerate(ALPHAS):
                model.fit(a);cv[k,j]=metrics(y[vi],model.scores(h[vi]))['macro_f1']
            trials.append({'inner_valid':str(valid),'fit_donors':np.unique(don[ti]).tolist(),'fit_class_counts':np.bincount(y[ti],minlength=8).tolist()})
        # Ascending alpha ties, mean of donor F1 (not pooled cells).
        k=int(np.argmax(cv.mean(1)));alpha=ALPHAS[k];model=Ridge(h[fit],y[fit]).fit(alpha);pred[~fit]=model.scores(h[~fit])
        records.append({'heldout_donor':str(held),'fit_donors':groups.tolist(),'alpha':alpha,'inner_mean_F1':cv.mean(1).tolist(),'inner_trials':trials,'heldout_rows':tr['rows'][~fit].tolist(),'fit_class_counts':np.bincount(y[fit],minlength=8).tolist()})
    residual=2*np.eye(8)[y]-1.-pred
    np.savez_compressed(out/'targets.npz',scores=pred,residual=residual,labels=y,rows=tr['rows'],donors=don)
    save(out/'crossfit.json',{'fold':fi,'records':records,'label_crossfit':True,'SAE_crossfit':False,'OOF_metrics':metrics(y,pred)})
    complete(out,start);print('CROSSFIT',fi,'seconds',round(time.time()-start,1),flush=True)

def branch(out,d,state,kind,fi,seed):
    out.mkdir(parents=True,exist_ok=True);tr=d['train'];va=d['valid']
    zt=latent(tr,state,kind,fi,seed);zv=latent(va,state,kind,fi,seed)
    h,hv,hm,hs=standardize(tr['h'],va['h']);z,zv,zm,zs=standardize(zt,zv)
    np.savez_compressed(out/'block_scalers.npz',hmean=hm,hscale=hs,zmean=zm,zscale=zs)
    orig=W/'shared'/f'fold_{fi}'/'original_h';base=read(orig/'selection.json');bs=np.load(orig/'selected_VALID.npz')['scores']
    shutil.copy2(orig/'model.npz',out/'model.npz');np.savez_compressed(out/'selected_VALID.npz',scores=bs,labels=va['y'],rows=va['rows'])
    best=base['VALID_F1'];sel={'method':'concat_'+kind,'identity_selected':True,'eta':0.,'alpha':base['alpha'],'VALID_F1':best};save(out/'selection.json',sel)
    records=[];scores=[]
    for eta in ETA:
        x=np.concatenate([h,eta*z],1);v=np.concatenate([hv,eta*zv],1);model=Ridge(x,tr['y'],scale=False)
        for alpha in ALPHAS:
            model.fit(alpha);s=model.scores(v);f=metrics(va['y'],s)['macro_f1'];scores.append(s)
            records.append({'eta':eta,'alpha':alpha,'VALID_F1':f,'normal_equation_error':model.err})
            if f>best+1e-12:
                best=f;sel={'method':'concat_'+kind,'identity_selected':False,'eta':eta,'alpha':alpha,'VALID_F1':f};save_selected(out,model,s,sel,va)
        del model
    save(out/'trials.json',records);np.savez_compressed(out/'grid_VALID.npz',scores=np.stack(scores),labels=va['y'],rows=va['rows'])
    save(out/'feature_diagnostics.json',{'kind':kind,'train_nonconstant_columns':int(np.sum(np.std(zt,axis=0)>1e-12)),'mean_nonzero_slots':float(np.count_nonzero(zt,axis=1).mean()),'not_activation_frequency_matched':kind=='random'})

def correction(out,d,state,kind,fi,seed):
    out.mkdir(parents=True,exist_ok=True);tr=d['train'];va=d['valid'];target=np.load(W/'shared'/f'fold_{fi}'/'crossfit/targets.npz')
    assert np.array_equal(target['rows'],tr['rows']);e=target['residual'];zt=latent(tr,state,kind,fi,seed);zv=latent(va,state,kind,fi,seed)
    orig=W/'shared'/f'fold_{fi}'/'original_h';base=read(orig/'selection.json');bs=np.load(orig/'selected_VALID.npz')['scores']
    best=base['VALID_F1'];sel={'method':'crossfit_residual_'+kind,'identity_selected':True,'gamma':0.,'alpha':None,'VALID_F1':best}
    np.savez_compressed(out/'selected_VALID.npz',scores=bs,correction=np.zeros_like(bs),labels=va['y'],rows=va['rows']);save(out/'selection.json',sel)
    model=Ridge(zt,tr['y'],target=e);records=[];scores=[]
    for alpha in ALPHAS:
        model.fit(alpha);es=model.scores(zv)
        for gamma in GAMMA:
            s=bs+gamma*es;f=metrics(va['y'],s)['macro_f1'];scores.append(s)
            records.append({'alpha':alpha,'gamma':gamma,'VALID_F1':f,'normal_equation_error':model.err})
            if f>best+1e-12:
                best=f;sel={'method':'crossfit_residual_'+kind,'identity_selected':False,'gamma':gamma,'alpha':alpha,'VALID_F1':f};save_selected(out,model,s,sel,va)
                np.savez_compressed(out/'selected_VALID.npz',scores=s,correction=gamma*es,labels=va['y'],rows=va['rows'])
    save(out/'trials.json',records);np.savez_compressed(out/'grid_VALID.npz',scores=np.stack(scores),labels=va['y'],rows=va['rows'])

def develop(fi,seed):
    out=W/'models'/f'fold_{fi}'/f'seed_{seed}'
    if completed(out/'completion.json'):return
    start=time.time();original(fi);crossfit(fi);d,state,fold=load(fi,seed);out.mkdir(parents=True,exist_ok=True)
    eq=[]
    for a in MIX_ALPHA:
        if a==-1:continue
        lam=1/(1+a);x=fixed_features(d['train'],'fixed_Dz',a);v=fixed_features(d['valid'],'fixed_Dz',a)
        xx=fixed_features(d['train'],'residual_mix',lam);vv=fixed_features(d['valid'],'residual_mix',lam)
        sx,sv,_,_=standardize(x,v);tx,tv,_,_=standardize(xx,vv)
        error=max(float(np.max(np.abs(sx-tx))),float(np.max(np.abs(sv-tv))));assert error<1e-9,error
        eq.append({'alpha':a,'equivalent_lambda':lam,'standardized_max_error':error})
    save(out/'equivalence.json',eq)
    fixed=[('sparse_z',[None]),('reconstruction',[None]),('reconstruction_error',[None]),('fixed_Dz',MIX_ALPHA),('residual_mix',MIX_LAMBDA)]
    for kind,params in fixed:
        sub=out/kind
        if not completed(sub/'completion.json'):
            t=time.time();fit_family(sub,d,kind,params);complete(sub,t)
        print('FIXED',fi,seed,kind,read(sub/'selection.json')['VALID_F1'],flush=True)
    for kind in KINDS:
        sub=out/('concat_'+kind)
        if not completed(sub/'completion.json'):
            t=time.time();branch(sub,d,state,kind,fi,seed);complete(sub,t)
        print('CONCAT',fi,seed,kind,read(sub/'selection.json'),flush=True)
        sub=out/('crossfit_residual_'+kind)
        if not completed(sub/'completion.json'):
            t=time.time();correction(sub,d,state,kind,fi,seed);complete(sub,t)
        print('CORRECTION',fi,seed,kind,read(sub/'selection.json'),flush=True)
    save(out/'fold.json',{'fold':fi,'seed':seed,**fold});complete(out,start)
    print('DEVELOP_DONE',fi,seed,'seconds',round(time.time()-start,1),flush=True)

def preflight():
    O.mkdir(parents=True,exist_ok=True);immutable(O/'PROTOCOL.json',CFG)
    paths=[W/'core.py',W/'run.py',R/'protocol.json',R/'kang_embeddings.npz',R/'pbmc_embeddings.npz',R/'sae.py']
    checks=[]
    for fi in CFG['folds']:
        for seed in SEEDS:
            d,state,fold=load(fi,seed);src=src_for(fi,seed);paths.extend([src/'sae.pt',src/'sae_outputs.npz',src/'sae_result.json'])
            assert len(np.unique(d['train']['donor']))==4
            assert not set(d['train']['rows'])&set(d['valid']['rows'])
            # Recompute cached activations from source encoder on a bounded TRAIN/VALID sample.
            for role,b in d.items():
                ix=np.unique(np.linspace(0,len(b['h'])-1,12,dtype=int));a=mm(b['h'][ix]-state['mean'],state['encoder.weight'],tb=True)+state['encoder.bias']
                ii=np.argpartition(a,-32,axis=1)[:,-32:];z=np.zeros_like(a);np.put_along_axis(z,ii,np.take_along_axis(a,ii,axis=1),axis=1)
                err=float(np.max(np.abs(z-b['z'][ix])));assert err<5e-6,err
                checks.append({'fold':fi,'seed':seed,'role':role,'n':len(b['h']),'cache_activation_error':err,'cache_reconstruction_error':b['cache_error'],'classes':np.bincount(b['y'],minlength=8).tolist()})
    immutable(W/'frozen_sources.json',{str(p.resolve()):sha(p) for p in paths})
    save(O/'preflight.json',{'status':'PASS','checks':checks,'no_TEST_PBMC_scoring':True})
    # Keep original outputs and checkpoint workspace immutable without copying them.
    orig=ROOT/'outputs/sae_residual/FINAL_MANIFEST.json';immutable(W/'original_manifest_binding.json',{'path':str(orig),'sha':sha(orig)})
    save(O/'EXPERIMENT_TRACKER.json',{'status':'RUNNING','development':'RUNNING','evaluation':'NOT_STARTED','review':'NOT_STARTED'})
    print('PREFLIGHT_PASS',len(checks),flush=True)

def lock():
    if (O/'SELECTION_LOCK.json').exists():
        sealed=read(O/'SELECTION_LOCK.json');assert sealed['binding']==binding()
        for p,h in sealed['hashes'].items():assert sha(p)==h,p
        print('EXISTING_SELECTION_LOCK_VERIFIED',len(sealed['hashes']),flush=True);return
    files={}
    for fi in CFG['folds']:
        assert completed(W/'shared'/f'fold_{fi}'/'original_h/completion.json')
        assert completed(W/'shared'/f'fold_{fi}'/'crossfit/completion.json')
        for seed in SEEDS:
            out=W/'models'/f'fold_{fi}'/f'seed_{seed}';assert completed(out/'completion.json')
    for p in list((W/'models').rglob('*'))+list((W/'shared').rglob('*')):
        if p.is_file():files[str(p.resolve())]=sha(p)
    immutable(O/'SELECTION_LOCK.json',{'binding':binding(),'locked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hashes':files,'roles_seen_for_selection':['train','valid']})
    print('ALL_SELECTIONS_LOCKED',len(files),flush=True)

def evaluate():
    lockp=O/'SELECTION_LOCK.json';assert lockp.exists();sealed=read(lockp);assert sealed['binding']==binding()
    for p,h in sealed['hashes'].items():assert sha(p)==h,p
    rows=[]
    for fi in CFG['folds']:
        orig=W/'shared'/f'fold_{fi}'/'original_h'
        for seed in SEEDS:
            out=O/'evaluations'/f'fold_{fi}'/f'seed_{seed}'
            if completed(out/'completion.json'):continue
            start=time.time();out.mkdir(parents=True,exist_ok=True);d,state,fold=load(fi,seed,roles=('test','pbmc'))
            modelroot=W/'models'/f'fold_{fi}'/f'seed_{seed}';methods=['original_h']+[p.name for p in modelroot.iterdir() if p.is_dir()]
            for role,b in d.items():
                base=score_model(orig/'model.npz',b['h']);target=2*np.eye(8)[b['y']]-1.;e=target-base
                for method in sorted(methods):
                    sub=orig if method=='original_h' else modelroot/method;sel=read(sub/'selection.json');cor=None
                    if method=='original_h' or sel.get('identity_selected',False):s=base.copy()
                    elif method.startswith('concat_'):
                        kind=method[len('concat_'):];z=latent(b,state,kind,fi,seed);sc=np.load(sub/'block_scalers.npz')
                        x=np.concatenate([(b['h']-sc['hmean'])/sc['hscale'],sel['eta']*(z-sc['zmean'])/sc['zscale']],1);s=score_model(sub/'model.npz',x)
                    elif method.startswith('crossfit_residual_'):
                        kind=method[len('crossfit_residual_'):];cor=sel['gamma']*score_model(sub/'model.npz',latent(b,state,kind,fi,seed));s=base+cor
                    else:s=score_model(sub/'model.npz',fixed_features(b,method,sel.get('param')))
                    met=metrics(b['y'],s);bp=base.argmax(1);p=s.argmax(1)
                    met.update({'fold':fi,'seed':seed,'test_donor':fold['test'],'role':role,'method':method,'identity_selected':sel.get('identity_selected',False),'VALID_F1':sel['VALID_F1'],
                                'corrected_errors':int(np.sum((bp!=b['y'])&(p==b['y']))),'harmed_correct':int(np.sum((bp==b['y'])&(p!=b['y']))),
                                'score_target_MSE':float(np.mean((target-s)**2)),'baseline_score_target_MSE':float(np.mean(e**2)),'selection_lock_sha':sha(lockp)})
                    if method.startswith('crossfit_residual_'):
                        if cor is None:cor=np.zeros_like(base)
                        met['residual_prediction_Q2']=float(1-np.sum((e-cor)**2)/np.sum(e**2))
                    save(out/f'{method}_{role}.json',met);np.savez_compressed(out/f'{method}_{role}.npz',scores=s,labels=b['y'],rows=b['rows'])
            complete(out,start);print('EVALUATED',fi,seed,flush=True)
    save(O/'evaluation_completion.json',{'status':'complete','binding':binding(),'selection_lock_sha':sha(lockp),'no_test_selection':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','develop','lock','evaluate']);ap.add_argument('--fold',type=int);ap.add_argument('--seed',type=int);args=ap.parse_args()
    np.seterr(over='raise',divide='raise',invalid='raise');torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if args.stage=='preflight':preflight();return
    check()
    if args.stage=='develop':
        for fi in ([args.fold] if args.fold is not None else CFG['folds']):
            for seed in ([args.seed] if args.seed is not None else SEEDS):develop(fi,seed)
    elif args.stage=='lock':lock()
    else:evaluate()
    check()
if __name__=='__main__':main()
