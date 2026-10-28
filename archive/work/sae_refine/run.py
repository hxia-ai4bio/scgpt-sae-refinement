"""Frozen scGPT refinement; TRAIN fit / VALID selection / locked TEST evaluation.

SAE cost O(epochs*n*512*width), memory O(n*512+batch*width+512*width).
512D Ridge cost O(n*512**2+512**3); sparse diagnostic uses width=2048.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work/sae_complementarity'))
import core as old

W = Path(__file__).resolve().parent
O = ROOT / 'outputs/sae_refine'
SEEDS = [42, 43, 44]
MIX = [0., .25, .5, .75, 1.]
CONFIG = {
    'folds': list(range(6)), 'seeds': SEEDS, 'mix_alpha': MIX,
    'ridge_alpha': old.ALPHAS, 'epochs_max': 20, 'patience': 3,
    'batch_size': 256, 'lr': .0003, 'optimizer': 'Adam', 'threads': 4,
    'device': 'cpu', 'sae_dtype': 'float32', 'probe_dtype': 'float64',
    'backbone': 'frozen cached scGPT; no backbone or supervised adapter training',
    'data': 'Kang ctrl historical 9577 cells / 6 donors / 8 labels',
    'fit': 'four TRAIN donors only; fixed TRAIN mean; dictionary columns unit-norm after every update; no decoder bias',
    'sae_selection': 'lowest VALID MSE, earliest tie, epoch0 eligible; same as historical SAE',
    'probe_selection': 'VALID fixed-eight-class MacroF1; alpha=0 identity wins ties, then ascending mixture and Ridge alpha; no TRAIN+VALID refit',
    'arms': ['original_h', 'single_raw', 'ensemble_existing3', 'wide6144_k96', 'abs2048_k32', 'abs_sparse_z', 'bag2_1024_k16', 'boost2_1024_k16'],
    'variants': 'each reconstruction arm reports independently selected forced reconstruction (mix=1) and mixed512 (mix includes0); z is diagnostic only',
    'capacity_controls': 'existing3 ensemble: 6144 dictionary/96 slots vs single6144/96; bag2 and boost2: 2048 dictionary/32 slots vs single2048/32. Maximum total dictionary-weighted epochs matched; actual early stopping and compute recorded, not exactly equal walltime.',
    'boost': 'first SAE fit h, second fit h-first_reconstruction; residual TRAIN mean only. First1024 shared with bag2; second same initialization seed as bag2 second, different target.',
    'bag': 'average independently reconstructed h; member seeds s and s+10000',
    'existing_ensemble': 'one ensemble of existing seeds42,43,44 per fold, not three independent repeats',
    'scope': 'exploratory reuse of historically inspected Kang donors; no new blind confirmation; no PBMC transfer',
    'statistics': 'average seeds within donor then equal mean across6 donors; donor paired bootstrap descriptive only, overlapping training sets; seeds not independent donors',
    'advance_rule': 'descriptive >=0.5pp gain, >=4/6 positive donors; >=2/3 positive seed means when applicable; report every class including rare classes',
    'DIAL': 'not scheduled: no validated nuisance-only directions yet; technical suppression is conditional, not silently omitted after results',
}


def save(p, value):
    old.save(p, value)


def read(p):
    return old.read(p)


def immutable(p, value):
    old.immutable(p, value)


def data(fi, roles=('train', 'valid')):
    proto = read(old.R / 'protocol.json')
    fold = proto['folds'][fi]['fold']
    with np.load(old.R / 'kang_embeddings.npz') as a:
        h, y, donor, rows = (a[k] for k in ['embedding', 'labels', 'donors', 'rows'])
        out = {}
        for role in roles:
            ix = np.flatnonzero(~np.isin(donor, [fold['valid'], fold['test']])) if role == 'train' else np.flatnonzero(donor == fold[role])
            out[role] = dict(h=h[ix].astype(np.float32), y=y[ix].astype(np.int64), rows=rows[ix], donor=donor[ix])
    if 'train' in out:
        assert len(np.unique(out['train']['donor'])) == 4
    if 'train' in out and 'valid' in out:
        assert not set(out['train']['rows']) & set(out['valid']['rows'])
    return out, fold


class SAE(nn.Module):
    def __init__(self, mean, width, k, seed, mode='raw'):
        super().__init__()
        torch.manual_seed(seed)
        self.width, self.k, self.mode = width, k, mode
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.encoder = nn.Linear(512, width)
        self.decoder = nn.Linear(width, 512, bias=False)
        nn.init.zeros_(self.encoder.bias)
        self.normalize()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)

    @torch.no_grad()
    def normalize(self):
        norms = self.decoder.weight.norm(dim=0, keepdim=True)
        assert torch.isfinite(norms).all() and (norms > 1e-12).all()
        self.decoder.weight.div_(norms)

    def forward(self, x):
        a = self.encoder(x-self.mean)
        score = a.abs() if self.mode == 'abs' else a
        ix = torch.topk(score, self.k, dim=1, sorted=False).indices
        values = a.gather(1, ix)
        z = torch.zeros_like(a).scatter(1, ix, values)
        return self.decoder(z)+self.mean, z


def infer(model, h, latent=False):
    model.eval()
    results = []
    with torch.no_grad():
        for j in range(0, len(h), 512):
            rec, z = model(torch.from_numpy(h[j:j+512]))
            assert torch.isfinite(rec).all() and torch.isfinite(z).all()
            assert ((z != 0).sum(1) <= model.k).all()
            results.append((z if latent else rec).numpy())
    return np.concatenate(results)


def newpath(fi, seed, name):
    return W / 'networks' / f'fold_{fi}' / f'seed_{seed}' / name


def restore(path):
    cfg = read(path / 'training.json')
    state = torch.load(path / 'sae.pt', map_location='cpu', weights_only=False)
    model = SAE(state['mean'], cfg['width'], cfg['k'], cfg['seed'], cfg['mode'])
    model.load_state_dict(state)
    return model


def train(fi, seed, name, tr, va, width, k, mode='raw', init_seed=None):
    path = newpath(fi, seed, name)
    if (path / 'training.json').exists():
        cfg = read(path / 'training.json')
        assert cfg['train_input_sha'] == array_sha(tr) and cfg['valid_input_sha'] == array_sha(va)
        assert cfg['checkpoint_sha'] == old.sha(path / 'sae.pt')
        return restore(path)
    start = time.time()
    path.mkdir(parents=True, exist_ok=True)
    init_seed = seed if init_seed is None else init_seed
    mean = tr.astype(np.float64).mean(0).astype(np.float32)
    model = SAE(mean, width, k, init_seed, mode)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    history, best, stale, updates = [], float('inf'), 0, 0
    gradient = None
    for epoch in range(CONFIG['epochs_max']+1):
        rec = infer(model, va)
        vmse = float(np.mean((rec.astype(np.float64)-va.astype(np.float64))**2))
        row = {'epoch': epoch, 'valid_mse': vmse, 'updates': updates}
        history.append(row)
        if vmse < best:
            best, best_epoch, beststate, stale = vmse, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch == CONFIG['epochs_max'] or stale >= CONFIG['patience']:
            break
        model.train()
        order = np.random.default_rng(np.random.SeedSequence([init_seed, epoch, 20261014])).permutation(len(tr))
        loss_sum = 0.
        for j in range(0, len(order), CONFIG['batch_size']):
            x = torch.from_numpy(tr[order[j:j+CONFIG['batch_size']]])
            opt.zero_grad(set_to_none=True)
            recon, z = model(x)
            loss = (recon-x).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            if updates == 0:
                gradient = {n: float(p.grad.norm()) for n, p in model.named_parameters()}
                assert all(np.isfinite(v) and v > 0 for v in gradient.values())
            opt.step()
            model.normalize()
            updates += 1
            loss_sum += float(loss.detach())*len(x)
        row['next_train_mse'] = loss_sum / len(tr)
    model.load_state_dict(beststate)
    torch.save(beststate, path / 'sae.pt')
    norm_error = float((model.decoder.weight.double().norm(dim=0)-1).abs().max())
    assert norm_error < 2e-6
    cfg = dict(width=width, k=k, mode=mode, seed=init_seed, fold=fi,
               train_input_sha=array_sha(tr), valid_input_sha=array_sha(va),
               history=history, best_epoch=best_epoch, last_epoch=epoch, updates=updates,
               first_gradients=gradient, dictionary_norm_error=norm_error,
               changed_tensors=sum(not torch.equal(v, initial[n]) for n, v in beststate.items() if n != 'mean'),
               selected_valid_mse=best, seconds=time.time()-start,
               checkpoint_sha=old.sha(path / 'sae.pt'))
    save(path / 'training.json', cfg)
    print('TRAIN_DONE', fi, seed, name, 'best_epoch', best_epoch, 'seconds', round(cfg['seconds'], 2), flush=True)
    return model


def array_sha(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def cached(fi, seed, roles):
    src = old.src_for(fi, seed)
    with np.load(src / 'sae_outputs.npz') as a:
        return {role: a[role+'_reconstruction'].copy() for role in roles}


def probe(fi, seed, method, features, d, mixtures=True):
    out = W / 'probes' / f'fold_{fi}' / f'seed_{seed}' / method
    if (out / 'completion.json').exists():
        for p, h in read(out / 'completion.json')['hashes'].items():
            assert old.sha(p) == h
        return
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    h = {r: d[r]['h'].astype(np.float64) for r in d}
    trials = []
    best_mixed, best_forced = -1., -1.
    for mix in (MIX if mixtures else [1.]):
        xt = h['train']+mix*(features['train'].astype(np.float64)-h['train']) if mixtures else features['train']
        xv = h['valid']+mix*(features['valid'].astype(np.float64)-h['valid']) if mixtures else features['valid']
        ridge = old.Ridge(xt, d['train']['y'])
        for alpha in old.ALPHAS:
            ridge.fit(alpha)
            vs = ridge.scores(xv)
            vf = old.metrics(d['valid']['y'], vs)['macro_f1']
            row = dict(mix_alpha=mix, ridge_alpha=alpha, valid_f1=vf, normal_equation_error=ridge.err)
            trials.append(row)
            choices = []
            if vf > best_mixed+1e-12:
                best_mixed = vf
                choices.append('mixed512' if mixtures else 'z_only')
            if mixtures and mix == 1. and vf > best_forced+1e-12:
                best_forced = vf
                choices.append('forced_reconstruction')
            for variant in choices:
                path = out / variant
                path.mkdir(exist_ok=True)
                np.savez_compressed(path / 'model.npz', **ridge.arrays())
                np.savez_compressed(path / 'valid_predictions.npz', scores=vs, labels=d['valid']['y'], rows=d['valid']['rows'])
                save(path / 'selection.json', dict(row, method=method, variant=variant, identity=(mixtures and mix == 0), seed=seed, fold=fi))
    save(out / 'trials.json', trials)
    diagnostics = {}
    for role, f in features.items():
        if mixtures:
            h64 = h[role]
            err = np.sum((f.astype(np.float64)-h64)**2)
            den = np.sum((h64-h['train'].mean(0))**2)
            diagnostics[role] = dict(mse=float(err/h64.size), q2_train_mean=float(1-err/den), relative_displacement=float(np.sqrt(err/np.sum(h64*h64))))
        else:
            diagnostics[role] = dict(mean_active=float(np.count_nonzero(f, axis=1).mean()), negative_active_fraction=float(np.sum(f < 0)/max(1, np.count_nonzero(f))), dead_features=int(np.sum(np.count_nonzero(f, axis=0)==0)))
    save(out / 'feature_diagnostics.json', diagnostics)
    save(out / 'completion.json', {'seconds': time.time()-start, 'hashes': {str(p): old.sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'completion.json'}})
    print('PROBE_DONE', fi, seed, method, 'seconds', round(time.time()-start, 2), flush=True)


def develop(fi, seed):
    d, fold = data(fi)
    roles = tuple(d)
    tr, va = (d[r]['h'] for r in ['train', 'valid'])
    if seed == 42:
        probe(fi, 'shared', 'original_h', {r: d[r]['h'] for r in roles}, d)
        ens = {r: np.mean(np.stack([cached(fi, s, roles)[r].astype(np.float64) for s in SEEDS]), axis=0) for r in roles}
        probe(fi, 'ensemble', 'ensemble_existing3', ens, d)
    probe(fi, seed, 'single_raw', cached(fi, seed, roles), d)
    absmodel = train(fi, seed, 'abs2048_k32', tr, va, 2048, 32, 'abs')
    probe(fi, seed, 'abs2048_k32', {r: infer(absmodel, d[r]['h']) for r in roles}, d)
    probe(fi, seed, 'abs_sparse_z', {r: infer(absmodel, d[r]['h'], True) for r in roles}, d, False)
    del absmodel
    wide = train(fi, seed, 'wide6144_k96', tr, va, 6144, 96)
    probe(fi, seed, 'wide6144_k96', {r: infer(wide, d[r]['h']) for r in roles}, d)
    del wide
    m1 = train(fi, seed, 'first1024_k16', tr, va, 1024, 16)
    rec1 = {r: infer(m1, d[r]['h']) for r in roles}
    m2 = train(fi, seed, 'bag_second1024_k16', tr, va, 1024, 16, init_seed=seed+10000)
    bag = {r: (rec1[r].astype(np.float64)+infer(m2, d[r]['h']).astype(np.float64))/2 for r in roles}
    probe(fi, seed, 'bag2_1024_k16', bag, d)
    del m1, m2, bag
    residual = {r: np.asarray(d[r]['h']-rec1[r], dtype=np.float32) for r in roles}
    m2 = train(fi, seed, 'boost_second1024_k16', residual['train'], residual['valid'], 1024, 16, init_seed=seed+10000)
    boost = {r: rec1[r].astype(np.float64)+infer(m2, residual[r]).astype(np.float64) for r in roles}
    probe(fi, seed, 'boost2_1024_k16', boost, d)
    print('FOLD_SEED_DEVELOPED', fi, seed, flush=True)


def preflight():
    O.mkdir(parents=True, exist_ok=True)
    immutable(O / 'PROTOCOL.json', CONFIG)
    sources = [Path(__file__), ROOT / 'work/sae_complementarity/core.py', old.R / 'protocol.json', old.R / 'kang_embeddings.npz']
    manifest = read(old.R / 'protocol.json')
    assert old.sha(old.R / 'kang_embeddings.npz') == manifest['source_hashes'][str(old.R / 'kang_embeddings.npz')]
    checks = []
    splits = {}
    for fi in CONFIG['folds']:
        d, fold = data(fi, ('train', 'valid', 'test'))
        assert sum(len(b['h']) for b in d.values()) == 9577
        splits[str(fi)] = {'donors': fold, 'rows': {r: b['rows'].tolist() for r, b in d.items()}}
        for seed in SEEDS:
            src = old.src_for(fi, seed)
            sources += [src / 'sae.pt', src / 'sae_outputs.npz', src / 'sae_result.json']
            state = torch.load(src / 'sae.pt', map_location='cpu', weights_only=False)
            m = SAE(state['mean'], 2048, 32, seed)
            m.load_state_dict(state)
            np.testing.assert_array_equal(state['mean'].numpy(), d['train']['h'].astype(np.float64).mean(0).astype(np.float32))
            with np.load(src / 'sae_outputs.npz') as c:
                for role in ('train', 'valid'):
                    np.testing.assert_array_equal(c[role+'_rows'], d[role]['rows'])
                    sample = np.unique(np.linspace(0, len(d[role]['h'])-1, 12, dtype=int))
                    err = float(np.max(np.abs(infer(m, d[role]['h'][sample])-c[role+'_reconstruction'][sample])))
                    assert err < 5e-6
                    checks.append(dict(fold=fi, seed=seed, role=role, reconstruction_cache_error=err))
    immutable(W / 'sources.json', {str(p): old.sha(p) for p in sources})
    immutable(O / 'SPLITS.json', splits)
    save(O / 'preflight.json', {'status': 'PASS', 'checks': checks, 'no_test_scoring': True})
    import shutil
    shutil.copy2(ROOT / 'work/sae_refine_resources.json', O / 'resources.json')
    # Bounded reproduction: same TRAIN order and initialization as existing raw SAE.
    d, _ = data(0)
    m = train(0, 42, 'smoke_raw_reproduction', d['train']['h'], d['valid']['h'], 2048, 32)
    ref = torch.load(old.src_for(0,42) / 'sae.pt', map_location='cpu', weights_only=False)
    errors = {n: float((v-ref[n]).abs().max()) for n,v in m.state_dict().items()}
    assert max(errors.values()) < 2e-6, errors
    save(O / 'smoke.json', {'raw_checkpoint_reproduction_max_errors': errors, 'status': 'PASS'})
    print('PREFLIGHT_AND_REPRODUCTION_PASS', flush=True)


def validate_sources():
    assert read(O / 'PROTOCOL.json') == CONFIG
    for p, h in read(W / 'sources.json').items():
        assert old.sha(p) == h, p


def lock():
    expected = []
    for fi in CONFIG['folds']:
        expected += [(fi, 'shared', 'original_h'), (fi, 'ensemble', 'ensemble_existing3')]
        expected += [(fi,s,m) for s in SEEDS for m in CONFIG['arms'] if m not in ['original_h','ensemble_existing3']]
    for fi,s,m in expected:
        p = W / 'probes' / f'fold_{fi}' / f'seed_{s}' / m / 'completion.json'
        assert p.exists(), p
        for q,h in read(p)['hashes'].items():
            assert old.sha(q) == h
    hashes = {str(p): old.sha(p) for root in ['probes','networks'] for p in (W/root).rglob('*') if p.is_file()}
    immutable(O / 'SELECTION_LOCK.json', {'source_manifest_sha': old.sha(W/'sources.json'), 'protocol_sha': old.sha(O/'PROTOCOL.json'), 'hashes': hashes})
    print('SELECTION_LOCKED', len(expected), 'families', flush=True)


def test_features(fi, seed, method, h):
    if method == 'original_h':
        return h
    if method == 'ensemble_existing3':
        return np.mean(np.stack([cached(fi,s,('test',))['test'].astype(np.float64) for s in SEEDS]), axis=0)
    if method == 'single_raw':
        return cached(fi,int(seed),('test',))['test']
    if method in ['abs2048_k32','abs_sparse_z','wide6144_k96']:
        name = 'abs2048_k32' if method == 'abs_sparse_z' else method
        return infer(restore(newpath(fi,int(seed),name)), h, method == 'abs_sparse_z')
    first = restore(newpath(fi,int(seed),'first1024_k16'))
    r1 = infer(first, h)
    if method == 'bag2_1024_k16':
        second = restore(newpath(fi,int(seed),'bag_second1024_k16'))
        return (r1.astype(np.float64)+infer(second,h).astype(np.float64))/2
    second = restore(newpath(fi,int(seed),'boost_second1024_k16'))
    return r1.astype(np.float64)+infer(second,np.asarray(h-r1,dtype=np.float32)).astype(np.float64)


def evaluate():
    sealed = read(O / 'SELECTION_LOCK.json')
    assert sealed['source_manifest_sha'] == old.sha(W / 'sources.json')
    assert sealed['protocol_sha'] == old.sha(O / 'PROTOCOL.json')
    for p,h in sealed['hashes'].items():
        assert old.sha(p) == h
    records = []
    for fi in CONFIG['folds']:
        d, fold = data(fi, ('test',))
        b = d['test']
        base_dir = W / 'probes' / f'fold_{fi}'
        for selection in sorted(base_dir.glob('seed_*/*/*/selection.json')):
            sel = read(selection)
            method, variant, seed = (sel[k] for k in ['method','variant','seed'])
            f = test_features(fi,seed,method,b['h'])
            x = b['h'].astype(np.float64)+sel['mix_alpha']*(f.astype(np.float64)-b['h'].astype(np.float64)) if variant != 'z_only' else f
            scores = old.score_model(selection.parent / 'model.npz', x)
            out = O / 'predictions' / f'fold_{fi}' / f'seed_{seed}' / method / variant
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out/'test.npz',scores=scores,labels=b['y'],rows=b['rows'])
            met = old.metrics(b['y'], scores)
            relative = float(np.linalg.norm(x-b['h'])/np.linalg.norm(b['h'])) if x.shape==b['h'].shape else None
            record = dict(fold=fi,test_donor=fold['test'],seed=seed,method=method,variant=variant,**met,selection=sel,relative_embedding_change=relative)
            save(out / 'metrics.json', record)
            records.append(record)
        print('TEST_EVALUATED_FOLD',fi,flush=True)
    save(O / 'all_metrics.json',records)
    summarize(records)


def summarize(records):
    flat=[]
    for r in records:
        flat.append({k:r[k] for k in ['fold','test_donor','seed','method','variant','macro_f1','supported_macro_f1','accuracy','relative_embedding_change']} | {'mix_alpha':r['selection']['mix_alpha'],'ridge_alpha':r['selection']['ridge_alpha'],'identity':r['selection']['identity']})
    df=pd.DataFrame(flat)
    df.to_csv(O/'all_metrics.csv',index=False)
    donor=df.groupby(['method','variant','fold','test_donor'],as_index=False)['macro_f1'].mean()
    baseline=donor[(donor.method=='original_h')&(donor.variant=='mixed512')].set_index('fold')['macro_f1']
    summaries=[]
    rng=np.random.default_rng(20261019)
    for (method,variant), g in donor.groupby(['method','variant']):
        g=g.sort_values('fold')
        delta=g['macro_f1'].to_numpy()-baseline.loc[g['fold']].to_numpy()
        boot=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1)*100
        subset=df[(df.method==method)&(df.variant==variant)]
        seedmeans=[]
        for seed,sg in subset.groupby('seed'):
            if str(seed) not in ['shared','ensemble']:
                sd=sg['macro_f1'].to_numpy()-baseline.loc[sg['fold']].to_numpy()
                seedmeans.append({'seed':str(seed),'delta_pp':float(sd.mean()*100)})
        sm=dict(method=method,variant=variant,macro_f1_percent=float(g['macro_f1'].mean()*100),delta_pp=float(delta.mean()*100),positive_donors=int(np.sum(delta>1e-12)),negative_donors=int(np.sum(delta< -1e-12)),donors=6,descriptive_ci95_pp=np.quantile(boot,[.025,.975]).tolist(),identity_selected=int(subset['identity'].sum()),fits=len(subset),seed_deltas=seedmeans)
        sm['advance_gate']=bool(sm['delta_pp']>=.5 and sm['positive_donors']>=4 and (not seedmeans or sum(s['delta_pp']>0 for s in seedmeans)>=2))
        summaries.append(sm)
    save(O/'summary.json',summaries)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['seed_deltas','descriptive_ci95_pp']} for r in summaries]).to_csv(O/'summary.csv',index=False)
    donor['delta_pp']=100*(donor['macro_f1']-donor['fold'].map(baseline))
    donor.to_csv(O/'paired_donor_metrics.csv',index=False)
    pcs=[]
    for r in records:
        for k,name in enumerate(old.CFG['classes']):
            pcs.append(dict(method=r['method'],variant=r['variant'],fold=r['fold'],seed=r['seed'],cell_type=name,f1=r['per_class_f1'][k],support=r['support'][k]))
    pd.DataFrame(pcs).to_csv(O/'per_class_metrics.csv',index=False)
    classmean=pd.DataFrame(pcs).groupby(['method','variant','fold','cell_type'],as_index=False)['f1'].mean().groupby(['method','variant','cell_type'],as_index=False)['f1'].mean()
    baseclass=classmean[(classmean.method=='original_h')&(classmean.variant=='mixed512')].set_index('cell_type')['f1']
    classmean['delta_pp']=100*(classmean.f1-classmean.cell_type.map(baseclass))
    classmean.to_csv(O/'per_class_summary.csv',index=False)
    save(O/'EXPERIMENT_TRACKER.json',dict(status='EVALUATED_PENDING_VERIFICATION',development='complete',test='complete',DIAL='conditional_not_run',selection_lock_sha=old.sha(O/'SELECTION_LOCK.json')))
    print(pd.DataFrame([{k:r[k] for k in ['method','variant','macro_f1_percent','delta_pp','positive_donors','advance_gate']} for r in summaries]).to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['preflight','develop','lock','evaluate'])
    ap.add_argument('--fold',type=int)
    ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        if args.stage=='preflight':
            preflight()
        else:
            validate_sources()
            if args.stage=='develop':
                for fi in ([args.fold] if args.fold is not None else CONFIG['folds']):
                    for seed in ([args.seed] if args.seed is not None else SEEDS):
                        develop(fi,seed)
            elif args.stage=='lock':
                lock()
            else:
                evaluate()


if __name__=='__main__':
    main()
