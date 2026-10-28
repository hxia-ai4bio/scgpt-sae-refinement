"""Read-only bounded reviewer checks; writes only review_trace artifacts."""
from pathlib import Path
import argparse, hashlib, json, sys
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dgemm

ROOT=Path(__file__).resolve().parents[2]
W=ROOT/'work/sae_residual_diagnosis'; O=ROOT/'outputs/sae_residual_diagnosis'
OLD=ROOT/'work/sae_residual'; TRACE=W/'review_trace'; TRACE.mkdir(exist_ok=True)
SRC=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def f1(y,p):
    cm=np.bincount(8*y+p,minlength=64).reshape(8,8)
    den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm),den,out=np.zeros(8),where=den>0).mean())
def check_manifest(p):
    j=read(p)
    for name,h in j.items(): assert sha(ROOT/name)==h,(p,name)
    return len(j)
def data(fi):
    a=np.load(SRC/'kang_embeddings.npz'); f=read(SRC/'protocol.json')['folds'][fi]['fold']
    ix={r:np.flatnonzero(~np.isin(a['donors'],[f['test'],f['valid']])) if r=='train' else np.flatnonzero(a['donors']==f['valid']) for r in ['train','valid']}
    cache=np.load(SRC/f"test_{f['test']}"/'seed_42/k_32/sae_outputs.npz')
    out={r:dict(h=a['embedding'][i].astype(np.float64), y=a['labels'][i], rows=a['rows'][i], donors=a['donors'][i]) for r,i in ix.items()}
    for r in out:
        assert np.array_equal(out[r]['y'],cache[r+'_labels'])
        assert np.array_equal(out[r]['rows'],cache[r+'_rows'])
        z=np.zeros((len(ix[r]),2048),dtype=np.float32)
        z[np.arange(len(z))[:,None],cache[r+'_indices']]=cache[r+'_values']
        out[r]['z']=z
    assert not set(out['train']['rows']) & set(out['valid']['rows'])
    return out
def static():
    result={'old_manifest_files_checked':check_manifest(ROOT/'outputs/sae_residual/FINAL_MANIFEST.json'),
            'old_source_files_checked':check_manifest(OLD/'frozen_sources.json'),
            'new_source_files_checked':check_manifest(W/'frozen_sources.json')}
    a=np.load(SRC/'kang_embeddings.npz'); classes=read(SRC/'protocol.json')['classes']
    with h5py.File('/Users/apple/Desktop/Llava-Cell/dataset/kang.h5ad','r') as f:
        obs=f['obs'][:]; cats=[x.decode() if isinstance(x,bytes) else str(x) for x in f['uns/cell_type_categories'][:]]
        labels=np.array([classes.index(cats[i]) for i in obs['cell_type'][a['rows']]])
        assert np.array_equal(labels,a['labels'])
    result['dataset_annotation_rows_verified']=len(labels)
    cf=pd.read_csv(O/'historical_candidate_diagnostics.csv'); funnel=[]
    for fi in range(6):
        for seed in [42,43,44]:
            sel=read(OLD/f'fold_{fi}/seed_{seed}/gate/selection.json')
            all0=all(t['best_epoch']==0 for t in sel['trials'])
            funnel.append({'fold':fi,'seed':seed,'all_beta_checkpoints_epoch0':all0,'identity_selected':sel['identity_selected'],
                           'rejection_stage':'checkpoint_epoch0' if all0 else 'fresh_Ridge' if sel['identity_selected'] else 'retained'})
    assert funnel==read(O/'gate_rejection_funnel.json')
    for r in cf.to_dict('records'):
        p=OLD/f"fold_{r['fold']}/seed_{r['seed']}/{r['arm']}/beta_{r['beta']}.json"; j=read(p)
        ir=read(p.parents[1]/'identity_ridge.json'); hist=j['history']
        assert r['best_epoch']==j['best_epoch'] and r['last_epoch']==j['last_epoch']
        assert abs(r['ridge_delta_pp']-100*(j['fresh_ridge']['valid_macro_f1']-ir['valid_macro_f1']))<1e-10
        assert abs(r['head_delta_pp']-100*(j['valid_head_macro_f1']-hist[0]['valid_head_macro_f1']))<1e-10
        ce=[h['next_train_CE'] for h in hist if 'next_train_CE' in h]
        assert abs(r['train_CE_drop']-(ce[0]-ce[-1]))<1e-10
    result['historical_rows_verified']=len(cf)
    result['funnel_counts']=dict(pd.Series([r['rejection_stage'] for r in funnel]).value_counts())
    result['funnel_counts']={k:int(v) for k,v in result['funnel_counts'].items()}
    ds=[]
    for fi in range(6):
        d=data(fi); tr=d['train']; h=tr['h']; centered=h-h.mean(0)
        ds.append({'fold':fi,'train_cells':len(h),'valid_cells':len(d['valid']['h']),'train_donors':len(set(tr['donors'])),
                   'train_valid_disjoint':True,'valid_support':np.bincount(d['valid']['y'],minlength=8).tolist(),
                   'raw_to_centered_median':float(np.median(np.linalg.norm(h,axis=1)/np.linalg.norm(centered,axis=1)))})
    result['splits']=ds
    write(TRACE/'static_checks.json',result); print(json.dumps(result,indent=2))
def snapshots(require_complete=False):
    counts={'trajectories':0,'branches':0,'trajectory_predictions':0,'branch_predictions':0,'bitwise_replays':0}; maxerr=0.
    for fi in range(6):
        d=data(fi); y=d['valid']['y']; rows=d['valid']['rows']
        for arm in ['gate','rotated_gate']:
            for mode in ['joint','frozen']:
                p=W/f'trajectories/fold_{fi}/{arm}/{mode}'; comp=p/'completion.json'
                if not comp.exists(): continue
                j=read(comp)
                for name,h in j['hashes'].items(): assert sha(name)==h,name
                hist=read(p/'history.json'); assert len(hist)==41
                for rec in hist:
                    if 'fresh_Ridge_valid_F1' not in rec: continue
                    z=np.load(p/f"epoch_{rec['epoch']}_VALID.npz")
                    assert np.array_equal(z['rows'],rows) and np.array_equal(z['labels'],y)
                    for key,col in [('Ridge_scores','fresh_Ridge_valid_F1'),('head_scores','head_valid_F1'),('fixed_head_scores','fixed_original_CE_head_F1')]:
                        err=abs(f1(y,z[key].argmax(1))-rec[col]); assert err<1e-12; maxerr=max(maxerr,err)
                    counts['trajectory_predictions']+=1
                if mode=='joint':
                    old=read(OLD/f'fold_{fi}/seed_42/{arm}/beta_0.1.json')
                    aa=torch.load(p/f"epoch_{old['best_epoch']}.pt",map_location='cpu',weights_only=False)
                    bb=torch.load(OLD/f'fold_{fi}/seed_42/{arm}/beta_0.1.pt',map_location='cpu',weights_only=False)
                    assert aa.keys()==bb.keys() and all(torch.equal(aa[k],bb[k]) for k in aa)
                    for r in old['history']: assert hist[r['epoch']]['head_valid_F1']==r['valid_head_macro_f1']
                    counts['bitwise_replays']+=1
                counts['trajectories']+=1
        for kind in ['learned_SAE','random_TopK']:
            p=W/f'branches/fold_{fi}/{kind}'; comp=p/'completion.json'
            if not comp.exists():continue
            j=read(comp)
            for name,h in j['hashes'].items(): assert sha(name)==h,name
            trials=read(p/'trials.json'); assert len(trials)==33
            sel=read(p/'selection.json'); best=sel['identity_VALID_F1']; pick=None; alpha=read(p/'identity.json')['alpha']
            for t in trials:
                z=np.load(p/f"eta_{t['eta']}_alpha_{t['alpha']}_VALID.npz")
                assert np.array_equal(z['rows'],rows) and np.array_equal(z['labels'],y)
                err=abs(f1(y,z['scores'].argmax(1))-t['VALID_F1']); assert err<1e-12; maxerr=max(maxerr,err)
                if t['VALID_F1']>best+1e-12: best=t['VALID_F1'];pick=t['eta'];alpha=t['alpha']
                counts['branch_predictions']+=1
            assert best==sel['best_VALID_F1'] and pick==sel['selected_eta'] and alpha==sel['selected_alpha']
            assert abs(f1(y,np.load(p/'selected_VALID.npz')['scores'].argmax(1))-best)<1e-12
            counts['branches']+=1
    if require_complete: assert counts['trajectories']==24 and counts['branches']==12,counts
    result={'status':'PASS','counts':counts,'max_metric_absolute_error':maxerr,'TEST_PBMC_scoring_performed':False}
    write(TRACE/'prediction_checks.json',result);print(json.dumps(result,indent=2))
def numerical():
    fi=0;d=data(fi);h=d['train']['h'];hv=d['valid']['h'];y=d['train']['y'];checks=[]
    for kind in ['learned_SAE','random_TopK']:
        p=W/f'branches/fold_{fi}/{kind}';sel=read(p/'selection.json');eta=sel['selected_eta'];alpha=sel['selected_alpha']
        assert eta is not None
        if kind=='learned_SAE':z=d['train']['z'];zv=d['valid']['z']
        else:
            torch.manual_seed(20261019+fi); E=torch.nn.functional.normalize(torch.randn(2048,512),dim=1)
            mu=torch.from_numpy(h.mean(0).astype(np.float32)); zs=[]
            for hh in [h,hv]:
                act=(torch.from_numpy(hh.astype(np.float32))-mu)@E.T
                v,i=torch.topk(act,32,dim=1,sorted=False);zs.append(torch.zeros_like(act).scatter(1,i,v).numpy())
            z,zv=zs
        parts=[];vparts=[]
        for x,v,mult in [(h,hv,1.),(z,zv,eta)]:
            mu=x.astype(np.float64).mean(0);sd=x.astype(np.float64).std(0);keep=sd>1e-12
            parts.append(mult*(x[:,keep]-mu[keep])/sd[keep]);vparts.append(mult*(v[:,keep]-mu[keep])/sd[keep])
        x=np.concatenate(parts,1);v=np.concatenate(vparts,1);w=len(y)/(8*np.bincount(y)[y]);t=2*np.eye(8)[y]-1
        xm=np.average(x,axis=0,weights=w);ym=np.average(t,axis=0,weights=w);A=(x-xm)*np.sqrt(w[:,None]);B=(t-ym)*np.sqrt(w[:,None])
        G=dgemm(1.,A,A,trans_a=True);rhs=dgemm(1.,A,B,trans_a=True);G.flat[::len(G)+1]+=alpha
        coef=cho_solve(cho_factor(G),rhs);scores=dgemm(1.,v-xm,coef)+ym
        saved=np.load(p/'selected_VALID.npz')['scores'];err=float(abs(scores-saved).max());pred_equal=bool(np.array_equal(scores.argmax(1),saved.argmax(1)))
        assert err<1e-5 and pred_equal,(kind,err)
        checks.append({'fold':fi,'kind':kind,'eta':eta,'alpha':alpha,'independent_solver':'Cholesky normal equations','max_score_error':err,'predictions_equal':pred_equal})
    write(TRACE/'independent_numerical_checks.json',{'status':'PASS','checks':checks});print(json.dumps(checks,indent=2))
def gradients():
    d=data(0); y=d['train']['y']; donors=d['train']['donors']; rng=np.random.default_rng(np.random.SeedSequence([42,0,20261019])); order=[]
    for c in range(8):
        pools={dd:np.flatnonzero((y==c)&(donors==dd)) for dd in np.unique(donors[y==c])}
        ds=np.resize(rng.permutation(list(pools)),32);rng.shuffle(ds);order.extend(int(rng.choice(pools[dd])) for dd in ds)
    rng.shuffle(order); ix=np.asarray(order);h=torch.tensor(d['train']['h'][ix],dtype=torch.float32); yy=torch.tensor(y[ix],dtype=torch.long)
    fold=read(SRC/'protocol.json')['folds'][0]['fold']; cache=np.load(SRC/f"test_{fold['test']}"/'seed_42/k_32/sae_outputs.npz')
    idx=torch.tensor(cache['train_indices'][ix].astype(np.int64)); vals=torch.tensor(cache['train_values'][ix]);checks=[]
    for arm in ['gate','rotated_gate']:
        for mode in ['joint','frozen']:
            p=W/f'trajectories/fold_0/{arm}/{mode}';s=torch.load(p/'epoch_40.pt',map_location='cpu',weights_only=False);theta=s['residual.theta'].clone().requires_grad_()
            delta=torch.einsum('bk,bkd->bd',.5*theta[idx].tanh()*vals,s['residual.DT'][idx]);out=torch.nn.functional.linear((h+delta-s['residual.mu'])/s['residual.scale'],s['head.weight'],s['head.bias'])
            ce=torch.nn.functional.cross_entropy(out,yy);pen=.1*(delta.square().sum(1)/h.square().sum(1).clamp_min(1e-12)).mean()
            ga=torch.autograd.grad(ce,theta,retain_graph=True)[0];gb=torch.autograd.grad(pen,theta)[0];a=float(ga.norm());b=float(gb.norm());r=read(p/'history.json')[-1]['gradient_diagnostic']
            for value,key in [(a,'CE_theta_gradient_norm'),(b,'penalty_theta_gradient_norm')]:assert abs(value-r[key])<1e-10,(arm,mode,key,value,r[key])
            checks.append({'fold':0,'epoch':40,'arm':arm,'mode':mode,'CE_gradient':a,'penalty_gradient':b,'ratio':b/a})
    write(TRACE/'gradient_checks.json',{'status':'PASS','scope':'one fixed class/donor-balanced TRAIN minibatch, epoch40, fold0, four arms/modes','checks':checks});print(json.dumps(checks,indent=2))
def selected_readouts():
    """All 24 trajectory maxima and six epoch0 identities, without NumPy matmul."""
    checks=[]
    for fi in range(6):
        d=data(fi);fold=read(SRC/'protocol.json')['folds'][fi]['fold'];cache=np.load(SRC/f"test_{fold['test']}"/'seed_42/k_32/sae_outputs.npz')
        targets=[('gate','joint',0)]
        for arm in ['gate','rotated_gate']:
            for mode in ['joint','frozen']:
                p=W/f'trajectories/fold_{fi}/{arm}/{mode}';hh=read(p/'history.json');rr=[x for x in hh if 'fresh_Ridge_valid_F1' in x];best=max(rr,key=lambda x:x['fresh_Ridge_valid_F1']);targets.append((arm,mode,best['epoch']))
        for arm,mode,epoch in targets:
            p=W/f'trajectories/fold_{fi}/{arm}/{mode}';s=torch.load(p/f'epoch_{epoch}.pt',map_location='cpu',weights_only=False);features=[]
            for role in ['train','valid']:
                h=torch.tensor(d[role]['h'],dtype=torch.float32);idx=torch.tensor(cache[role+'_indices'].astype(np.int64));v=torch.tensor(cache[role+'_values']);parts=[]
                for lo in range(0,len(h),256):
                    ii=idx[lo:lo+256];delta=torch.einsum('bk,bkd->bd',.5*s['residual.theta'][ii].tanh()*v[lo:lo+256],s['residual.DT'][ii]);parts.append((h[lo:lo+256]+delta).numpy())
                features.append(np.concatenate(parts).astype(np.float64))
            tr,va=features;mu=tr.mean(0);sd=tr.std(0);sd[sd<1e-12]=1.;x=(tr-mu)/sd;v=(va-mu)/sd;y=d['train']['y'];vy=d['valid']['y'];w=len(y)/(8*np.bincount(y)[y]);t=2*np.eye(8)[y]-1
            xm=np.average(x,axis=0,weights=w);ym=np.average(t,axis=0,weights=w);A=(x-xm)*np.sqrt(w[:,None]);B=(t-ym)*np.sqrt(w[:,None]);G=dgemm(1.,A,A,trans_a=True);rhs=dgemm(1.,A,B,trans_a=True);best=-1.;chosen=None;best_scores=None
            for alpha in [10.**k for k in range(-4,7)]:
                gg=G.copy();gg.flat[::len(G)+1]+=alpha;coef=cho_solve(cho_factor(gg),rhs);scores=dgemm(1.,v-xm,coef)+ym;score=f1(vy,scores.argmax(1))
                if score>best+1e-12: best=score;chosen=alpha;best_scores=scores
            rec=next(z for z in read(p/'history.json') if z['epoch']==epoch);saved=np.load(p/f'epoch_{epoch}_VALID.npz')['Ridge_scores'];err=float(abs(best_scores-saved).max());pred_equal=bool(np.array_equal(best_scores.argmax(1),saved.argmax(1)))
            assert chosen==rec['fresh_Ridge_alpha'] and best==rec['fresh_Ridge_valid_F1'] and err<1e-5 and pred_equal,(fi,arm,mode,epoch,chosen,err)
            checks.append({'fold':fi,'arm':arm,'mode':mode,'epoch':epoch,'alpha':chosen,'VALID_F1':best,'max_score_error':err,'predictions_equal':pred_equal})
        print('READOUT_CHECK_FOLD',fi,flush=True)
    write(TRACE/'selected_readout_checks.json',{'status':'PASS','independent_solver':'SciPy dgemm and Cholesky, no NumPy matmul','families':len(checks),'alpha_fits':11*len(checks),'checks':checks});print('ALL_SELECTED_READOUTS_PASS',len(checks),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['static','snapshots','numerical','gradients','selected_readouts']);ap.add_argument('--complete',action='store_true');a=ap.parse_args();torch.set_num_threads(2)
    if a.stage=='static':static()
    elif a.stage=='snapshots':snapshots(a.complete)
    elif a.stage=='numerical':numerical()
    elif a.stage=='gradients':gradients()
    else:selected_readouts()
