"""Raw H5 label/id verification and exact shared-gene count-profile overlap; no fitting."""
from pathlib import Path
import hashlib,numpy as np
from prepare import read,sha,write,R
kp=Path('/Users/apple/Desktop/Llava-Cell/dataset/kang.h5ad');pp=Path('/Users/apple/Desktop/Llava-Cell/dataset/PBMC368k_raw.h5ad');kx,kg,km=read(kp,True);px,pg,pm=read(pp,False)
with np.load(R/'kang_embeddings.npz') as a:kr=a['rows'];ky=a['labels'];kd=a['donors'];ki=a['cell_ids']
with np.load(R/'pbmc_embeddings.npz') as a:pr=a['rows'];py=a['labels'];pi=a['cell_ids']
classes=sorted(set(km['cell_type']));assert np.array_equal(ky,[classes.index(c) for c in km['cell_type'][kr]]);assert np.array_equal(kd,km['replicate'][kr]);assert np.array_equal(ki,km['cell_ids'][kr]);assert (km['label'][kr]=='ctrl').all();assert np.array_equal(py,[classes.index(c) for c in pm['cell_type'][pr]]);assert np.array_equal(pi,pm['cell_ids'][pr])
common,ka,pa=np.intersect1d(kg,pg,return_indices=True)
def fp(x):
 x=x.tocsr();x.eliminate_zeros();x.sort_indices();v=[]
 for row in range(x.shape[0]):
  s,e=x.indptr[row:row+2];v.append(hashlib.sha256(x.indices[s:e].astype('<i8').tobytes()+x.data[s:e].astype('<f8').tobytes()).hexdigest())
 return v
kf=fp(kx[kr][:,ka]);pf=fp(px[pr][:,pa]);over=set(kf)&set(pf)
write(R/'raw_data_verification.json',dict(status='PASS',kang_cells=len(kr),pbmc_cells=len(pr),classes=classes,raw_labels_donors_ids_match=True,shared_genes=len(common),identical_shared_gene_count_profiles=len(over),overlap_kang_cells=sum(f in over for f in kf),overlap_pbmc_cells=sum(f in over for f in pf),scope='exact observed shared-gene profile overlap only; not proof of independent studies/donors or pretraining exclusion',sources={str(kp):sha(kp),str(pp):sha(pp)}));print('RAW_DATA_PASS',len(over),flush=True)
