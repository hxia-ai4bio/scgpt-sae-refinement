"""Frozen scGPT extraction for six-donor Kang stim; reuse audited PBMC embeddings.

Raw RNA stays sparse. Transformer memory O(B*L^2*heads + B*L*d);
time O(N*layers*(L*d^2+L^2*d)), L512,d512,B8. Chunked resume is hash-bound.
"""
import os
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_extension'
os.environ.setdefault('MPLCONFIGDIR',str(W/'mplcache'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(W/'numbacache'))
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(k,'4')
import importlib.util,json,hashlib,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

R=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
spec=importlib.util.spec_from_file_location('frozen_preparation',R/'prepare.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
def save(p,d):prep.write(p,d)
def sha(p):return prep.sha(p)

def main():
    start=time.time();torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    m=json.loads((R/'data_manifest.json').read_text());cfg=m['preprocess'].copy();classes=m['classes']
    for path in [cfg['dataset'],str(Path(cfg['checkpoint_dir'])/'best_model.pt'),str(R/'pbmc_embeddings.npz'),str(R/'prepare.py')]:assert sha(path)==m['sources'][path],path
    cfg.update(condition='stim',excluded_cell_types=[])
    model,vocab,args,audit=prep.atlas.load_dense(cfg);model.eval().requires_grad_(False);model=model.cpu()
    x,genes,meta=prep.read(cfg['dataset'],True);c=prep.cohort(x,genes,meta,cfg,vocab,classes)
    rawrows=np.flatnonzero(np.isin(meta['replicate'],cfg['development_donors'])&(meta['label']=='stim'))
    eligible=np.asarray([np.sum(c.vocab_ids[x[int(r)].indices]>=0)>=16 for r in rawrows]);rows=rawrows[eligible]
    assert len(rows)>9000 and set(meta['cell_type'][rows])==set(classes)
    with np.load(R/'kang_embeddings.npz') as a:cr=a['rows'][:8].copy();ch=a['embedding'][:8].copy()
    # Deterministic source-row tokenizer is unchanged between ctrl and stim.
    replay=prep.extract(model,c,cr,torch.device('cpu'));err=float(abs(replay-ch).max());assert err<3e-5,err
    pbraw=Path('/Users/apple/Desktop/Llava-Cell/dataset/PBMC368k_raw.h5ad')
    assert sha(pbraw)==m['sources'][str(pbraw)]
    px,pg,pm=prep.read(pbraw,False);pc=prep.cohort(px,pg,pm,cfg,vocab,classes)
    with np.load(R/'pbmc_embeddings.npz') as a:
        ph={k:a[k].copy() for k in a.files}
    pbreplay=prep.extract(model,pc,ph['rows'][:8],torch.device('cpu'));pberr=float(abs(pbreplay-ph['embedding'][:8]).max());assert pberr<3e-5,pberr
    np.testing.assert_array_equal(ph['cell_ids'],pm['cell_ids'][ph['rows']]);np.testing.assert_array_equal(ph['batches'],pm['replicate'][ph['rows']])
    np.testing.assert_array_equal(np.array(classes)[ph['labels']],pm['cell_type'][ph['rows']])
    dat=W/'data';dat.mkdir(exist_ok=True)
    np.savez_compressed(dat/'pbmc.npz',embedding=ph['embedding'],labels=ph['labels'],rows=ph['rows'],groups=ph['batches'],cell_ids=ph['cell_ids'],condition=np.repeat('unannotated',len(ph['rows'])))
    del px,pc
    binding=dict(script=sha(__file__),checkpoint=m['checkpoint_sha'],dataset=sha(cfg['dataset']),tokenizer=sha(prep.AT/'run_error_atlas.py'),config=cfg,rows_sha=hashlib.sha256(rows.tobytes()).hexdigest())
    chunkdir=dat/'stim_chunks';chunkdir.mkdir(exist_ok=True);bp=chunkdir/'binding.json'
    if bp.exists():assert json.loads(bp.read_text())==binding
    else:save(bp,binding)
    chunks=[];timings=[]
    for i in range(0,len(rows),128):
        rr=rows[i:i+128];p=chunkdir/f'{i:06d}.npz';tick=time.time()
        if p.exists():
            with np.load(p) as a:np.testing.assert_array_equal(a['rows'],rr);h=a['embedding'].copy()
        else:
            h=prep.extract(model,c,rr,torch.device('cpu'));temp=p.with_suffix('.tmp.npz');np.savez_compressed(temp,rows=rr,embedding=h);temp.replace(p)
        assert h.shape==(len(rr),512) and np.isfinite(h).all();chunks.append(h);timings.append(time.time()-tick)
        if i==0:save(O/'EXTRACTION_BENCHMARK.json',dict(cells=len(rr),seconds=timings[-1],estimated_remaining_seconds=timings[-1]*((len(rows)-len(rr))/len(rr)),estimate_only=True,ctrl_replay8_max_error=err,pbmc_replay8_max_error=pberr))
        save(O/'PREPARATION_TRACKER.json',dict(status='EXTRACTING_STIM',done=min(i+128,len(rows)),total=len(rows),elapsed_seconds=time.time()-start))
        print('STIM_CHUNK',min(i+128,len(rows)),'/',len(rows),'seconds',round(timings[-1],2),flush=True)
    h=np.concatenate(chunks)
    np.savez_compressed(dat/'stim.npz',embedding=h,rows=rows,labels=np.array([classes.index(t) for t in meta['cell_type'][rows]]),groups=meta['replicate'][rows],cell_ids=meta['cell_ids'][rows],condition=meta['label'][rows])
    assert all(p.grad is None for p in model.parameters()) and sha(Path(cfg['checkpoint_dir'])/'best_model.pt')==m['checkpoint_sha']
    sources={str(p):sha(p) for p in [Path(__file__),R/'prepare.py',prep.AT/'run_error_atlas.py',prep.AT/'config.json',Path(cfg['dataset']),pbraw,Path(cfg['checkpoint_dir'])/'best_model.pt',Path(cfg['checkpoint_dir'])/'args.json',Path(cfg['checkpoint_dir'])/'vocab.json',R/'pbmc_embeddings.npz',R/'kang_embeddings.npz']}
    save(O/'DATA_MANIFEST.json',dict(status='PASS',classes=classes,stim_cells=len(rows),stim_QC_excluded=int((~eligible).sum()),pbmc_cells=len(ph['rows']),pooling=m['pooling'],preprocessing=cfg,checkpoint_audit=audit,sources=sources,canonical_files={str(p):sha(p) for p in [dat/'stim.npz',dat/'pbmc.npz']},ctrl_replay8_max_error=err,pbmc_replay8_max_error=pberr,seconds=time.time()-start,scope='stim six historical development donors only; two original untouched Kang donors excluded; PBMC two batches not verified donors; no backbone training or label-based feature fitting'))
    save(O/'PREPARATION_TRACKER.json',dict(status='COMPLETE',done=len(rows),total=len(rows),elapsed_seconds=time.time()-start))
    print('DATA_READY',len(rows),len(ph['rows']),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
