"""Portable continuation from cached embeddings and an existing SAE checkpoint.

Uses the archived SAE, KL and balanced Ridge definitions verbatim. This entry
point is a convenience interface, not a rerun of all historical experiment arms.
CPU is intentional: the relation loss uses float64, unsupported by Apple MPS.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .core import SAE, Ridge, infer, metrics, relation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_data(cache, split_file, cross_group):
    with np.load(cache, allow_pickle=False) as a:
        h = a['embedding'].astype(np.float32)
        raw_y = a['labels']
        if not np.issubdtype(raw_y.dtype, np.integer):
            raise ValueError('labels must be integer IDs 0..7')
        y = raw_y.astype(np.int64)
        key = next((k for k in ('groups', 'donors', 'batches') if k in a), None)
        if key is None:
            raise ValueError('Cache needs groups, donors, or batches')
        groups = a[key].astype(str)
    if h.ndim != 2 or h.shape[1] != 512 or not np.isfinite(h).all():
        raise ValueError('Expected finite embedding array with shape [N, 512]')
    if y.shape != (len(h),) or groups.shape != y.shape or not np.isin(y, np.arange(8)).all():
        raise ValueError('Inconsistent metadata or labels outside 0..7')
    splits = json.loads(Path(split_file).read_text())
    out = {}
    for role in ('train', 'valid', 'test'):
        ix = np.asarray(splits[role])
        if ix.ndim != 1 or not len(ix) or not np.issubdtype(ix.dtype, np.integer):
            raise ValueError('Splits require nonempty integer cache-position arrays')
        if (ix < 0).any() or (ix >= len(h)).any() or len(np.unique(ix)) != len(ix):
            raise ValueError('Out-of-range or duplicate split indices')
        out[role] = {'h': h[ix], 'y': y[ix], 'groups': groups[ix], 'indices': ix}
    all_ix = np.concatenate([b['indices'] for b in out.values()])
    if len(all_ix) != len(np.unique(all_ix)):
        raise ValueError('TRAIN / VALID / TEST overlap')
    tr, va, te = (set(out[r]['groups']) for r in ('train', 'valid', 'test'))
    if te & (tr | va) or (cross_group and tr & va):
        raise ValueError('Group leakage: TEST must be held out; cross-group VALID also held out')
    if set(out['train']['y']) != set(range(8)):
        raise ValueError('All eight labels must occur in TRAIN')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='TRAIN-fitted, VALID-selected raw SAE state_dict; never TEST-selected')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lambda-kl', type=float, default=.3)
    p.add_argument('--anchor-mode', choices=('cross-group', 'nonself'), default='cross-group')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.steps < 2 or args.batch_size < 2 or args.threads < 1 or not math.isfinite(args.lambda_kl) or args.lambda_kl < 0:
        p.error('Require steps >= 2, batch-size >= 2, threads >= 1 and finite lambda >= 0')
    if args.out.exists():
        p.error('Output directory already exists; use a new run directory')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    d = load_data(args.cache, args.splits, args.anchor_mode == 'cross-group')
    tr = d['train']
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = SAE(state['mean'], 2048, 32, args.seed)
    model.load_state_dict(state, strict=True)
    for value in state.values():
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite checkpoint tensor')
    np.testing.assert_allclose(model.mean.numpy(), tr['h'].mean(0), atol=2e-5, rtol=2e-5,
                               err_msg='Checkpoint TRAIN mean differs from this split')
    den = float(np.mean((tr['h'].astype(np.float64)-model.mean.numpy().astype(np.float64))**2))
    if den <= 0:
        raise ValueError('Zero TRAIN variance')
    rng = np.random.default_rng(np.random.SeedSequence([20261019, args.fold, args.seed, args.steps]))
    if args.anchor_mode == 'cross-group':
        groups = np.unique(tr['groups'])
        if len(groups) < 2 or args.batch_size % len(groups):
            raise ValueError('Batch size must be divisible by number of TRAIN groups (>=2)')
        n = args.batch_size // len(groups)
        choices = [np.flatnonzero(tr['groups'] == g) for g in groups]
        if min(map(len, choices)) < n:
            raise ValueError('Too few cells per TRAIN group for sampling without replacement')
        batches = np.array([np.concatenate([rng.choice(c, n, replace=False) for c in choices]) for _ in range(args.steps)])
        g = torch.arange(len(groups)).repeat_interleave(n)
        mask = g[:, None] != g[None, :]
    else:
        if len(tr['h']) < args.batch_size:
            raise ValueError('Too few TRAIN cells for batch-size')
        batches = np.array([rng.choice(len(tr['h']), args.batch_size, replace=False) for _ in range(args.steps)])
        mask = ~torch.eye(args.batch_size, dtype=torch.bool)
    args.out.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['input_sha256'] = {k: digest(getattr(args, k)) for k in ('cache', 'splits', 'checkpoint')}
    config['selection'] = 'Fixed final SAE step; Ridge alpha by VALID only; no TRAIN+VALID refit'
    config['scope'] = 'Portable continuation; PBMC sampling stream differs from historical extension. Not a regenerated paper result.'
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    history = []
    for step, indices in enumerate(batches):
        opt.param_groups[0]['lr'] = 1e-5 + .5*(.0003-1e-5)*(1+math.cos(math.pi*step/(args.steps-1)))
        h = torch.from_numpy(tr['h'][indices])
        opt.zero_grad(set_to_none=True)
        rec, _ = model(h)
        mse = (rec-h).square().mean()/den
        kl, _ = relation(h, rec, model.mean, mask)
        loss = mse + args.lambda_kl*kl
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Missing or non-finite gradients')
        opt.step()
        model.normalize()
        if step == 0 or (step+1) % 250 == 0 or step+1 == args.steps:
            history.append(dict(step=step+1, normalized_mse=float(mse.detach()), kl=float(kl.detach())))
            print(history[-1], flush=True)
    torch.save(model.state_dict(), args.out/'sae.pt')
    (args.out/'training.json').write_text(json.dumps(history, indent=2)+'\n')
    representations = {'original': {r: b['h'] for r, b in d.items()},
                       'reconstruction': {r: infer(model, b['h']) for r, b in d.items()}}
    selected = {}
    # Select both readouts before computing either TEST metric.
    with threadpool_limits(limits=args.threads):
        for name, feat in representations.items():
            grid = Ridge(feat['train'], tr['y'])
            best, alpha, trials = -1., None, []
            for a in [10.**v for v in range(-4, 7)]:
                grid.fit(a)
                f = metrics(d['valid']['y'], grid.scores(feat['valid']))['macro_f1']
                trials.append({'alpha': a, 'valid_macro_f1': f})
                if f > best:
                    best, alpha = f, a
            grid.fit(alpha)
            np.savez_compressed(args.out/f'{name}_ridge.npz', **grid.arrays())
            selected[name] = (grid, {'alpha': alpha, 'valid_macro_f1': best, 'trials': trials})
        (args.out/'selection.json').write_text(json.dumps({k: v[1] for k, v in selected.items()}, indent=2)+'\n')
        result = {k: metrics(d['test']['y'], grid.scores(representations[k]['test'])) for k, (grid, _) in selected.items()}
    np.savez_compressed(args.out/'test_reconstruction.npz', embedding=representations['reconstruction']['test'], indices=d['test']['indices'])
    (args.out/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v['macro_f1'] for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
