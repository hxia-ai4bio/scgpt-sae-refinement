"""Recompute saved metrics/selection, row provenance and affine-control parity."""
import time
import numpy as np
from core import *

def main():
    start=time.time();check();lock=read(O/'SELECTION_LOCK.json')
    for p,h in lock['hashes'].items():assert sha(p)==h,p
    checked=0;maxmetric=0.;selectionchecks=0;maxnorm=0.;affine=[];valparity=[];equiv=[]
    for fi in CFG['folds']:
        cfdir=W/'shared'/f'fold_{fi}'/'crossfit';cf=read(cfdir/'crossfit.json');target=np.load(cfdir/'targets.npz')
        assert np.allclose(target['residual'],2*np.eye(8)[target['labels']]-1-target['scores'],atol=0,rtol=0)
        seen=[]
        for rec in cf['records']:
            held=rec['heldout_donor'];assert held not in rec['fit_donors'];assert len(rec['fit_donors'])==3
            for inner in rec['inner_trials']:
                assert held not in inner['fit_donors'] and held!=inner['inner_valid']
                assert inner['inner_valid'] not in inner['fit_donors'] and len(inner['fit_donors'])==2
            seen.extend(rec['heldout_rows'])
        assert sorted(seen)==sorted(target['rows'].tolist())
        for seed in SEEDS:
            modelroot=W/'models'/f'fold_{fi}'/f'seed_{seed}';d,state,fold=load(fi,seed);va=d['valid'];orig=W/'shared'/f'fold_{fi}'/'original_h';base=np.load(orig/'selected_VALID.npz')['scores'];basef=metrics(va['y'],base)['macro_f1']
            for sub in [orig]+sorted([p for p in modelroot.iterdir() if p.is_dir()]):
                sel=read(sub/'selection.json');grid=np.load(sub/'grid_VALID.npz');trials=read(sub/'trials.json');assert len(trials)==len(grid['scores']);assert np.array_equal(grid['rows'],va['rows']);assert np.array_equal(grid['labels'],va['y'])
                fs=[]
                for row,s in zip(trials,grid['scores']):
                    f=metrics(va['y'],s)['macro_f1'];assert abs(f-row['VALID_F1'])<1e-13;fs.append(f);maxnorm=max(maxnorm,row['normal_equation_error'])
                expected=max(fs+[basef]) if sub.name.startswith(('concat_','crossfit_')) else max(fs)
                assert abs(sel['VALID_F1']-expected)<1e-12,(sub,sel,expected)
                if sel.get('identity_selected') and sub.name.startswith(('concat_','crossfit_')):assert max(fs)<=basef+1e-12
                selected=np.load(sub/'selected_VALID.npz');assert np.array_equal(selected['rows'],va['rows']);assert abs(metrics(va['y'],selected['scores'])['macro_f1']-sel['VALID_F1'])<1e-13
                if sub.name=='original_h' or sel.get('identity_selected',False):s=base
                elif sub.name.startswith('concat_'):
                    kind=sub.name[len('concat_'):];z=latent(va,state,kind,fi,seed);sc=np.load(sub/'block_scalers.npz');x=np.concatenate([(va['h']-sc['hmean'])/sc['hscale'],sel['eta']*(z-sc['zmean'])/sc['zscale']],1);s=score_model(sub/'model.npz',x)
                    if kind=='linear':
                        # Explicit affine classifier on raw h, independent of expanded-input scoring.
                        m=np.load(sub/'model.npz');cc=m['coef']/m['scale'][:,None];cc0=m['intercept']-np.einsum('i,ij->j',m['mean'],cc);ch=cc[:512]/sc['hscale'][:,None];cz=sel['eta']*cc[512:]/sc['zscale'][:,None]
                        coef=ch+mm(state['encoder.weight'],cz,ta=True)
                        offset=state['encoder.bias']-np.einsum('ij,j->i',state['encoder.weight'],state['mean'])-sc['zmean']
                        intercept=cc0-np.einsum('i,ij->j',sc['hmean'],ch)+np.einsum('i,ij->j',offset,cz)
                        er=float(np.max(np.abs(s-(mm(va['h'],coef)+intercept))));assert er<1e-8;affine.append({'fold':fi,'seed':seed,'score_max_error':er})
                elif sub.name.startswith('crossfit_'):
                    kind=sub.name[len('crossfit_residual_'):];s=base+sel['gamma']*score_model(sub/'model.npz',latent(va,state,kind,fi,seed))
                else:s=score_model(sub/'model.npz',fixed_features(va,sub.name,sel.get('param')))
                er=float(np.max(np.abs(s-selected['scores'])));assert er<1e-9,(sub,er);valparity.append(er);selectionchecks+=1
            # Same-alpha paired mixtures should have identical predictions after TRAIN scaling.
            ga=np.load(modelroot/'fixed_Dz/grid_VALID.npz')['scores'];gl=np.load(modelroot/'residual_mix/grid_VALID.npz')['scores']
            for ia,a in enumerate(MIX_ALPHA):
                if a==-1:continue
                lam=1/(1+a);il=next(j for j,v in enumerate(MIX_LAMBDA) if abs(v-lam)<1e-12)
                aa=ga[ia*len(ALPHAS):(ia+1)*len(ALPHAS)];bb=gl[il*len(ALPHAS):(il+1)*len(ALPHAS)];er=float(np.max(np.abs(aa-bb)))
                assert er<1e-7 and np.array_equal(aa.argmax(2),bb.argmax(2)),(fi,seed,a,er)
                equiv.append({'fold':fi,'seed':seed,'alpha':a,'lambda':lam,'max_score_error':er})
            for role in ['test','pbmc']:
                bbase=np.load(O/'evaluations'/f'fold_{fi}'/f'seed_{seed}'/f'original_h_{role}.npz');y=bbase['labels'];base=bbase['scores'];target_y=2*np.eye(8)[y]-1
                for p in (O/'evaluations'/f'fold_{fi}'/f'seed_{seed}').glob(f'*_{role}.json'):
                    r=read(p);a=np.load(p.with_suffix('.npz'));assert np.array_equal(a['labels'],y) and np.array_equal(a['rows'],bbase['rows']);met=metrics(y,a['scores'])
                    for key in ['macro_f1','supported_macro_f1','accuracy']:
                        err=abs(met[key]-r[key]);maxmetric=max(maxmetric,err);assert err<1e-13
                    for key in ['per_class_f1','support','confusion_matrix']:assert np.allclose(met[key],r[key],atol=1e-13,rtol=0)
                    if r['identity_selected']:assert np.array_equal(a['scores'],base)
                    assert abs(r['score_target_MSE']-float(np.mean((target_y-a['scores'])**2)))<1e-12
                    checked+=1
    # Historical original experiment remains bitwise untouched.
    old=read(W/'original_manifest_binding.json');assert sha(old['path'])==old['sha'];manifest=read(old['path'])
    for p,h in manifest.items():
        pp=Path(p) if Path(p).is_absolute() else ROOT/p;assert sha(pp)==h,pp
    result={'status':'PASS','evaluation_predictions_checked':checked,'validation_selected_models_checked':selectionchecks,
            'unique_validation_selected_model_directories':204,'baseline_replayed_for_each_SAE_seed':True,
            'max_metric_error':maxmetric,'max_normal_equation_error':maxnorm,'max_selected_VALID_score_replay_error':max(valparity),
            'fixed_mixture_affine_checks':equiv,'selected_linear_branch_affine_checks':affine,'label_crossfit_donor_exclusion':'PASS',
            'original_manifest_unchanged_entries':len(manifest),'selection_lock_unchanged':'PASS','seconds':time.time()-start}
    assert checked==432 and selectionchecks==216
    save(O/'verification.json',result);print('VERIFICATION_PASS',checked,selectionchecks,round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
