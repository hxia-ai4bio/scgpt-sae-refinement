"""Mask gene observations BEFORE scGPT, recompute visible-only expression bins.
No dense transcriptome. Transformer cost O(N*L^2*d), memory O(B*L^2*heads).
Mask selected expressed gene tokens, pad hidden slots; gene IDs remain target metadata.
"""
import os
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_masked_ac'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
os.environ['MPLCONFIGDIR']=str(W/'mpl');os.environ['NUMBA_CACHE_DIR']=str(W/'numba')
import importlib.util,json,time,hashlib,sys
import numpy as np,torch
from scipy import sparse
from threadpoolctl import threadpool_limits

def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ext=module('masked_ext',ROOT/'work/sae_extension/run.py');base=ext.base;core=ext.old
prep=module('masked_prep',core.R/'prepare.py');save=core.save;sha=core.sha

def canonical(ds):
 if ds!='ctrl':return ext.canonical(ds)
 with np.load(core.R/'kang_embeddings.npz') as a:return {**{k:a[k].copy() for k in a.files},'groups':a['donors'].copy()}

def tokens(c,rows,mask=True):
 clean=c.tokenize(rows);genes=clean['genes'].numpy().copy();values=clean['values'].numpy().copy();padding=clean['padding'].numpy().copy();cols=np.full(genes.shape,-1,dtype=np.int64);targets=np.zeros(genes.shape,np.float32);hidden=np.zeros(genes.shape,bool)
 gene_to_col={int(v):i for i,v in enumerate(c.vocab_ids) if v>=0}
 for bi,r in enumerate(map(int,rows)):
  raw=c.x[r];count=dict(zip(raw.indices.tolist(),raw.data.tolist()));selected=np.flatnonzero(~padding[bi]);cc=np.array([gene_to_col[int(genes[bi,i])] for i in selected]);cols[bi,selected]=cc
  total=float(raw.data.astype('float64').sum());targets[bi,selected]=np.log1p(np.array([count[int(k)] for k in cc])/total*1e4).astype('float32')
  if not mask:continue
  rng=np.random.default_rng(np.random.SeedSequence([20261021,r,250]));pick=rng.choice(selected,max(1,int(np.floor(.25*len(selected)))),replace=False);hidden[bi,pick]=True;maskedcols=set(cols[bi,pick].tolist());visible=~np.isin(raw.indices,list(maskedcols));idx=raw.indices[visible];counts=raw.data[visible].astype('float64');tot=counts.sum();keep=c.vocab_ids[idx]>=0;idx=idx[keep];counts=counts[keep]
  normalized=np.log1p(counts/tot*1e4);edges=np.quantile(normalized,np.linspace(0,1,c.config['n_bins']-1));left=np.digitize(normalized,edges);right=np.digitize(normalized,edges,right=True);ties=np.random.default_rng(np.random.SeedSequence([c.config['seed'],r,11])).random(len(c.gene_names));bins=np.ceil(ties[idx]*(right-left)+left).astype('float32');bycol=dict(zip(idx.tolist(),bins.tolist()))
  for j in selected:
   if hidden[bi,j]:genes[bi,j]=c.pad_id;values[bi,j]=-2;padding[bi,j]=True
   else:values[bi,j]=bycol[int(cols[bi,j])]
  assert np.all(padding[bi,hidden[bi]]) and not set(cols[bi,~padding[bi]])&maskedcols
 return dict(genes=torch.from_numpy(genes),values=torch.from_numpy(values),padding=torch.from_numpy(padding),columns=cols,target_log_expression=targets,hidden=hidden)

@torch.no_grad()
def infer(model,t,device):
 outs=[]
 for j in range(0,len(t['genes']),8):
  g=t['genes'][j:j+8].to(device);v=t['values'][j:j+8].to(device);p=t['padding'][j:j+8].to(device);h=model.encoder(g)+model.value_encoder(v)
  for layer in model.transformer_encoder.layers:h=layer(h,src_key_padding_mask=p)
  out=(h*(~p).unsqueeze(-1)).sum(1)/(~p).sum(1,keepdim=True);assert torch.isfinite(out).all();outs.append(out.cpu().numpy())
 if device.type=='mps':torch.mps.synchronize()
 return np.concatenate(outs)

def load(ds):
 cfg=core.read(core.R/'data_manifest.json')['preprocess'].copy();cfg.update(condition='ctrl' if ds=='ctrl' else 'stim',excluded_cell_types=[])
 p=Path('/Users/apple/Desktop/Llava-Cell/dataset')/('PBMC368k_raw.h5ad' if ds=='pbmc' else 'kang.h5ad');x,genes,meta=prep.read(p,ds!='pbmc');return cfg,x,genes,meta,p

def run(stage):
 O.mkdir(exist_ok=True);W.mkdir(exist_ok=True);torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(20261021)
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
 cfg,x,genes,meta,p=load('ctrl');model,vocab,args,audit=prep.atlas.load_dense(cfg);model.eval().requires_grad_(False);model=model.to(device);c=prep.cohort(x,genes,meta,cfg,vocab,core.CFG['classes']);a=canonical('ctrl');rows=a['rows'][:16]
 t=tokens(c,rows,False);tick=time.time();h=infer(model,t,device);err=float(np.max(np.abs(h-a['embedding'][:16])));assert err<1e-3,(device,err)
 # Masked-input invariance: changing hidden gene magnitudes must not alter visible tokens.
 t=tokens(c,rows);r=int(rows[0]);before=c.x[r].copy();pos=np.flatnonzero(t['hidden'][0]);cc=t['columns'][0,pos];indices=c.x[r].indices;lo=c.x.indptr[r];hit=np.isin(indices,cc);old=c.x.data[lo:lo+len(indices)].copy();c.x.data[lo:lo+len(indices)][hit]*=7
 again=tokens(c,rows[:1]);c.x.data[lo:lo+len(indices)]=old
 for key in ['genes','values','padding']:np.testing.assert_array_equal(t[key][:1].numpy(),again[key].numpy())
 tick=time.time();mh=infer(model,t,device);seconds=time.time()-tick;assert not np.allclose(mh,h)
 bench=dict(device=str(device),cells=len(rows),masked_seconds=seconds,seconds_per_cell=seconds/len(rows),clean_replay16_max_error=err,hidden_expression_invariance='PASS',mask_fraction=.25,estimate_ctrl_seconds=seconds/len(rows)*len(a['rows']),estimate_all3_seconds=seconds/len(rows)*(9577+9728+4638),estimated_only=True);save(O/'EXTRACTION_BENCHMARK.json',bench);print('BENCHMARK',bench,flush=True)
 if stage=='benchmark':return
 protocol=core.read(O/'PROTOCOL.json');assert protocol['mask_fraction']==.25
 sourcepaths=[Path(__file__),core.R/'prepare.py',prep.AT/'run_error_atlas.py',prep.AT/'config.json',Path(cfg['checkpoint_dir'])/'best_model.pt',Path(cfg['checkpoint_dir'])/'args.json',Path(cfg['checkpoint_dir'])/'vocab.json']
 src={str(q):sha(q) for q in sourcepaths};core.immutable(W/'extraction_sources.json',src)
 for ds in protocol['datasets']:
  cfg,x,genes,meta,p=load(ds);a=canonical(ds);c=prep.cohort(x,genes,meta,cfg,vocab,core.CFG['classes']);np.testing.assert_array_equal(a['cell_ids'],meta['cell_ids'][a['rows']]);np.testing.assert_array_equal(a['groups'],meta['replicate'][a['rows']]);np.testing.assert_array_equal(np.array(core.CFG['classes'])[a['labels']],meta['cell_type'][a['rows']]);cache=core.R/'kang_embeddings.npz' if ds=='ctrl' else ext.W/'data'/f'{ds}.npz';binding=dict(sources=src,raw_path=str(p),raw_sha=sha(p),canonical_path=str(cache),canonical_sha=sha(cache),protocol_sha=sha(O/'PROTOCOL.json'),device=str(device));dest=W/'data'/ds;dest.mkdir(parents=True,exist_ok=True);core.immutable(dest/'binding.json',binding);chunks=[];columns=[];targets=[];hidden=[];start=time.time()
  for j in range(0,len(a['rows']),128):
   rr=a['rows'][j:j+128];path=dest/f'{j:06d}.npz'
   if path.exists():
    with np.load(path) as q:np.testing.assert_array_equal(q['rows'],rr);v={k:q[k].copy() for k in q.files}
   else:
    t=tokens(c,rr);hh=infer(model,t,device);v=dict(rows=rr,masked_embedding=hh,columns=t['columns'],target_log_expression=t['target_log_expression'],hidden=t['hidden']);temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**v);temp.replace(path)
   assert np.isfinite(v['masked_embedding']).all();chunks.append(v['masked_embedding']);columns.append(v['columns']);targets.append(v['target_log_expression']);hidden.append(v['hidden']);save(O/'EXPERIMENT_TRACKER.json',dict(status='EXTRACTING_MASKED_INPUTS',dataset=ds,done=min(j+128,len(a['rows'])),total=len(a['rows']),elapsed_seconds=time.time()-start));print('MASKED_CHUNK',ds,min(j+128,len(a['rows'])),'/',len(a['rows']),flush=True)
  expr=c.normalized_expression(a['rows']);sparse.save_npz(dest/'normalized_expression.npz',expr);
  target=W/'data'/f'{ds}.npz';np.savez_compressed(target,**a,masked_embedding=np.concatenate(chunks),columns=np.concatenate(columns),target_log_expression=np.concatenate(targets),hidden=np.concatenate(hidden),gene_names=genes);save(O/f'{ds}_DATA_MANIFEST.json',dict(binding=binding,canonical_data_sha=sha(target),raw_genes=len(genes),cells=len(a['rows']),model_frozen=all(not v.requires_grad and v.grad is None for v in model.parameters()),scope='Mask25pct original expressed selected tokens; pad hidden positions and recompute bins/library totals without hidden counts. C targets logCP10k expression of masked positive tokens in TRAIN-selected gene panel, not all-gene/zero-expression imputation. One fixed mask per cell shared across methods/seeds.'))
 print('EXTRACTION_COMPLETE',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4):run(sys.argv[1])
