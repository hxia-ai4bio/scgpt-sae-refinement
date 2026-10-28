"""Reuse immutable whole-human mean-pooled Kang cache; extract missing8th-class and fullPBMC only."""
from pathlib import Path
import json,hashlib,time,importlib.util,sys
import h5py,numpy as np,torch
from scipy import sparse
R=Path(__file__).resolve().parent;P=R.parents[1];AT=P/'experiments/kang_error_atlas_20261012'
spec=importlib.util.spec_from_file_location('atlas_source',AT/'run_error_atlas.py');atlas=importlib.util.module_from_spec(spec);spec.loader.exec_module(atlas)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,j):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(j,indent=2,allow_nan=False));tmp.replace(p)
def decode(a):return np.asarray([x.decode() if isinstance(x,bytes) else str(x) for x in a])
def read(path,kang):
 with h5py.File(path,'r') as f:
  if kang:
   obs=f['obs'][:];meta={k:decode(f['uns'][k+'_categories'][:])[obs[k]] for k in ['replicate','label','cell_type']};meta['cell_ids']=decode(obs['index']);genes=decode(f['var'][:]['name']);x=sparse.csc_matrix((f['X/data'][:],f['X/indices'][:],f['X/indptr'][:]),shape=tuple(f['X'].attrs['h5sparse_shape'])).tocsr()
  else:
   cat=lambda k:decode(f['obs'][k]['categories'][:])[f['obs'][k]['codes'][:]]
   meta=dict(replicate=cat('batch'),cell_type=cat('celltype'),label=np.repeat('unannotated',len(f['obs/CellID'])),cell_ids=decode(f['obs/CellID'][:]));genes=decode(f['var/Gene'][:]);x=sparse.csr_matrix((f['X/data'][:],f['X/indices'][:],f['X/indptr'][:]),shape=tuple(f['X'].attrs['shape']))
 x.eliminate_zeros();x.sort_indices();assert np.isfinite(x.data).all() and (x.data>=0).all() and np.allclose(x.data,np.rint(x.data),atol=1e-6,rtol=0);assert len(set(genes))==len(genes);assert len(set(meta['cell_ids']))==len(meta['cell_ids']);return x,genes,meta

def cohort(x,genes,meta,cfg,vocab,classes):
 c=object.__new__(atlas.KangDevelopmentCohort);c.x=x;c.gene_names=genes;c.meta=meta;c.config=cfg;c.vocab_ids=np.array([vocab.get(g,-1) for g in genes]);c.pad_id=vocab['<pad>'];c.classes=classes;c.class_to_id={k:i for i,k in enumerate(classes)};return c
@torch.no_grad()
def extract(model,c,rows,device):
 outs=[];start=time.time()
 for i in range(0,len(rows),8):
  b={k:v.to(device) for k,v in c.tokenize(rows[i:i+8]).items()};h=model.encoder(b['genes'])+model.value_encoder(b['values'])
  for layer in model.transformer_encoder.layers:h=layer(h,src_key_padding_mask=b['padding'])
  v=~b['padding'];z=(h*v.unsqueeze(-1)).sum(1)/v.sum(1,keepdim=True);assert torch.isfinite(z).all();outs.append(z.cpu().numpy())
  if i%512==0:print('EXTRACT',i,len(rows),'seconds',round(time.time()-start,1),flush=True)
 return np.concatenate(outs) if outs else np.empty((0,512),np.float32)
def main():
 torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False);cfg=json.loads((AT/'config.json').read_text());oldmanifest=json.loads((AT/'cache/feature_manifest.json').read_text());ckpt=Path(cfg['checkpoint_dir'])/'best_model.pt';assert sha(ckpt)==oldmanifest['checkpoint_sha256'];assert sha(cfg['dataset'])==oldmanifest['dataset_sha256']
 model,vocab,args,audit=atlas.load_dense(cfg);model.eval().requires_grad_(False);device=torch.device('cpu');model=model.to(device)
 assert len(model.transformer_encoder.layers)==12 and model.encoder.embedding.embedding_dim==512
 x,genes,meta=read(cfg['dataset'],True);classes=sorted(set(meta['cell_type']));assert len(classes)==8 and 'Megakaryocytes' in classes;c=cohort(x,genes,meta,cfg,vocab,classes)
 rawrows=np.flatnonzero(np.isin(meta['replicate'],cfg['development_donors'])&(meta['label']=='ctrl'));eligible=np.asarray([np.sum(c.vocab_ids[x[int(r)].indices]>=0)>=16 for r in rawrows]);rows=rawrows[eligible]
 with np.load(AT/'cache/features.npz') as old:oldrows=old['rows'];oldh=old['layer12'];oldy=old['labels']
 oldclasses=sorted(set(meta['cell_type'][oldrows]));assert np.array_equal(np.array(oldclasses)[oldy],meta['cell_type'][oldrows]);assert set(oldrows)==set(rows[meta['cell_type'][rows]!='Megakaryocytes'])
 replay=extract(model,c,oldrows[:8],device);err=float(np.max(np.abs(replay-oldh[:8])));assert err<3e-5,err
 missing=np.array(sorted(set(rows)-set(oldrows)),dtype=np.int64);newh=extract(model,c,missing,device);byrow={int(r):h for r,h in zip(oldrows,oldh)};byrow.update({int(r):h for r,h in zip(missing,newh)});kh=np.stack([byrow[int(r)] for r in rows]);ky=np.array([classes.index(t) for t in meta['cell_type'][rows]])
 np.savez_compressed(R/'kang_embeddings.npz',embedding=kh,rows=rows,labels=ky,donors=meta['replicate'][rows],cell_ids=meta['cell_ids'][rows],condition=meta['label'][rows])
 pbpath=Path('/Users/apple/Desktop/Llava-Cell/dataset/PBMC368k_raw.h5ad');px,pg,pm=read(pbpath,False);assert set(pm['cell_type'])==set(classes);pc=cohort(px,pg,pm,cfg,vocab,classes);pr=np.arange(px.shape[0]);pe=np.asarray([np.sum(pc.vocab_ids[px[int(r)].indices]>=0)>=16 for r in pr]);pr=pr[pe];ph=extract(model,pc,pr,device);py=np.array([classes.index(t) for t in pm['cell_type'][pr]])
 np.savez_compressed(R/'pbmc_embeddings.npz',embedding=ph,rows=pr,labels=py,batches=pm['replicate'][pr],cell_ids=pm['cell_ids'][pr])
 assert all(p.grad is None for p in model.parameters());assert sha(ckpt)==oldmanifest['checkpoint_sha256'];assert np.isfinite(kh).all() and np.isfinite(ph).all()
 folds=json.loads((P/'experiments/multilayer_readout_20261012/protocol.json').read_text())['rotations'];splitstats=[]
 for f in folds:
  ff={**f,'train':[d for d in cfg['development_donors'] if d not in [f['test'],f['valid']]]};s={k:np.flatnonzero(np.isin(meta['replicate'][rows],ff[k] if k=='train' else [ff[k]])) for k in ['train','valid','test']};assert len(np.unique(np.concatenate(list(s.values()))))==len(rows);splitstats.append({**ff,'indices':{k:v.tolist() for k,v in s.items()},'counts':{k:len(v) for k,v in s.items()},'class_counts':{k:np.bincount(ky[v],minlength=8).tolist() for k,v in s.items()}})
 write(R/'data_manifest.json',dict(status='PASS',classes=classes,checkpoint_audit=audit,checkpoint_sha=sha(ckpt),pooling='final12 transformer output; signed masked mean over nonpadding512 gene tokens; no CLS; no model tuning',preprocess=cfg,reused_kang_cells=len(oldrows),new_kang_cells=len(missing),kang_excluded_input_qc=int((~eligible).sum()),pbmc_cells=len(pr),pbmc_excluded_input_qc=int((~pe).sum()),old_cache_replay8_max_error=err,feature_extraction_device=str(device),sources={str(p):sha(p) for p in [Path(__file__),AT/'run_error_atlas.py',AT/'config.json',AT/'cache/features.npz',AT/'cache/feature_manifest.json',ckpt,Path(cfg['checkpoint_dir'])/'args.json',Path(cfg['checkpoint_dir'])/'vocab.json',Path(cfg['dataset']),pbpath,R/'kang_embeddings.npz',R/'pbmc_embeddings.npz']},splits=splitstats,pbmc_class_counts=np.bincount(py,minlength=8).tolist(),pbmc_batches={b:int((pm['replicate'][pr]==b).sum()) for b in np.unique(pm['replicate'][pr])},scope='PBMC historical exploratory external dataset; donors/study and pretraining overlap unknown; noPBMCmodelselection; Kang full eligible TRAIN instead of historical64donor/type cap'))
 print('DATA_READY',len(rows),len(pr),classes,flush=True)
if __name__=='__main__':main()
