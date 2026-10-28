"""Recompute saved metrics, labels, selected models and export usable test embeddings."""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as exp


def main():
    torch.set_num_threads(4)
    exp.validate_sources()
    sealed = exp.read(exp.O/'SELECTION_LOCK.json')
    for path, checksum in sealed['hashes'].items():
        assert exp.old.sha(path) == checksum, path
    allrows = exp.read(exp.O/'all_metrics.json')
    assert len(allrows) == 222, len(allrows)
    with np.load(exp.old.R/'kang_embeddings.npz') as source:
        source_labels = dict(zip(source['rows'].tolist(),source['labels'].tolist()))
    checked, exported, sklearn_checks, geometry = [], [], [], []
    sourcesizes = {}
    for fi in exp.CONFIG['folds']:
        test, fold = exp.data(fi, ('test',))
        dev, _ = exp.data(fi)
        b = test['test']
        train, valid = dev['train'], dev['valid']
        cache = {}
        for rec in [r for r in allrows if r['fold'] == fi]:
            method, variant, seed = (rec[k] for k in ['method','variant','seed'])
            out = exp.O/'predictions'/f'fold_{fi}'/f'seed_{seed}'/method/variant
            arr = np.load(out/'test.npz')
            np.testing.assert_array_equal(arr['rows'],b['rows'])
            np.testing.assert_array_equal(arr['labels'],[source_labels[int(r)] for r in arr['rows']])
            pred = arr['scores'].argmax(1)
            f = f1_score(arr['labels'],pred,labels=np.arange(8),average=None,zero_division=0)
            np.testing.assert_allclose(f, rec['per_class_f1'], atol=1e-14, rtol=0)
            assert abs(float(f.mean())-rec['macro_f1']) < 1e-14
            np.testing.assert_array_equal(confusion_matrix(arr['labels'],pred,labels=np.arange(8)),rec['confusion_matrix'])
            key = (method,seed)
            if key not in cache:
                cache[key] = exp.test_features(fi,seed,method,b['h'])
            features = cache[key]
            sel = rec['selection']
            x = b['h'].astype(np.float64)+sel['mix_alpha']*(features.astype(np.float64)-b['h'].astype(np.float64)) if variant != 'z_only' else features
            folder = exp.W/'probes'/f'fold_{fi}'/f'seed_{seed}'/method/variant
            model = np.load(folder/'model.npz')
            score = np.einsum('ni,ij->nj',(x-model['mean'])/model['scale'],model['coef'],optimize=False)+model['intercept']
            error = float(np.max(np.abs(score-arr['scores'])))
            assert error < 1e-8, error
            trials = exp.read(folder.parent/'trials.json')
            eligible = trials if variant != 'forced_reconstruction' else [t for t in trials if t['mix_alpha']==1.]
            best = max(t['valid_f1'] for t in eligible)
            assert abs(sel['valid_f1']-best) < 1e-12
            first = next(t for t in eligible if t['valid_f1'] >= best-1e-12)
            assert sel['mix_alpha']==first['mix_alpha'] and sel['ridge_alpha']==first['ridge_alpha']
            checked.append(dict(fold=fi,seed=seed,method=method,variant=variant,score_recomputation_max_error=error))
            if variant == 'mixed512':
                dest = exp.O/'embeddings'/f'fold_{fi}'/f'seed_{seed}'/f'{method}.npz'
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest,embedding=x,rows=b['rows'],labels=b['y'],donor=b['donor'],mix_alpha=sel['mix_alpha'])
                exported.append(str(dest.relative_to(exp.O)))
                geometry.append(dict(fold=fi,seed=seed,method=method,relative_change=rec['relative_embedding_change'],identity=sel['identity']))
                # Forced and selected feature diagnostics must not be mistaken for classifier evidence.
                den = float(np.sum((b['h'].astype(np.float64)-train['h'].astype(np.float64).mean(0))**2))
                mse = float(np.mean((features.astype(np.float64)-b['h'].astype(np.float64))**2))
                sourcesizes[str((fi,seed,method))] = dict(test_reconstruction_mse=mse,test_reconstruction_q2=1-mse*b['h'].size/den)
        # Verify a full weighted Ridge against sklearn using original 512D h, all folds.
        folder = exp.W/'probes'/f'fold_{fi}'/'seed_shared'/'original_h'/'mixed512'
        model = np.load(folder/'model.npz')
        tr = (train['h'].astype(np.float64)-model['mean'])/model['scale']
        va = (valid['h'].astype(np.float64)-model['mean'])/model['scale']
        ref = RidgeClassifier(alpha=float(model['alpha']),class_weight='balanced',solver='cholesky').fit(tr,train['y'])
        target = np.load(folder/'valid_predictions.npz')['scores']
        err = float(np.max(np.abs(ref.decision_function(va)-target)))
        assert err < 1e-6
        sklearn_checks.append(dict(fold=fi,score_max_error=err))
        print('VERIFIED_AND_EXPORTED_FOLD',fi,flush=True)
    assert len(exported)==102
    # No claim of 18 independently sampled donors or three independent 3-member ensembles.
    train_records=[]
    for path in (exp.W/'networks').glob('fold_*/seed_*/*/training.json'):
        r=exp.read(path)
        assert r['checkpoint_sha']==exp.old.sha(path.parent/'sae.pt')
        assert r['updates']>0 and r['changed_tensors']==3 and r['dictionary_norm_error']<2e-6
        assert all(v>0 and np.isfinite(v) for v in r['first_gradients'].values())
        assert r['best_epoch'] == min(r['history'],key=lambda a:a['valid_mse'])['epoch']
        train_records.append(dict(path=str(path),**{k:r[k] for k in ['width','k','seed','mode','best_epoch','last_epoch','updates','seconds','selected_valid_mse']}))
    assert len(train_records)==91, len(train_records)
    exp.save(exp.O/'training_summary.json',train_records)
    exp.save(exp.O/'embedding_manifest.json',dict(role='held-out TEST cells per fold, not full-data refit',dtype='float64',columns=512,files=exported))
    exp.save(exp.O/'test_geometry.json',dict(reconstruction=sourcesizes,selected_changes=geometry))
    exp.save(exp.O/'verification.json',dict(status='PASS',prediction_files_verified=len(checked),training_runs_verified=len(train_records),test_embedding_files=len(exported),sklearn_checks=sklearn_checks,checks=checked,selection_lock_sha=exp.old.sha(exp.O/'SELECTION_LOCK.json')))
    print('ALL_VERIFICATIONS_PASS',len(checked),'predictions',len(train_records),'training runs',len(exported),'embedding files',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
