"""Frozen B-feature multi-reference attribution. O(pairs * tokens * forward cost).
Reference -> prototype -> donor equal averaging. Same-cell coordinate diagnostics,
not a global comparison of interpretability or biological causal intervention.
"""
from pathlib import Path
import sys,json,time,importlib.util,os,copy
import numpy as np,pandas as pd,torch
WORK=Path(__file__).resolve().parent;FEATURE=int(os.environ['SAE_FEATURE']);BASE=WORK/f'F{FEATURE}'
OLD=WORK.parent/'sae_interpretability';OUT=WORK.parents[1]/'outputs/sae_B_remaining'/f'F{FEATURE}'
sys.path.insert(0,str(OLD))
from common import js,sha,load_npz
spec=importlib.util.spec_from_file_location('forward_old',OLD/'02_forward.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
NAMES=[f'sae_F{FEATURE}',f'encoder_F{FEATURE}','embedding_neg201','pca3']
@torch.inference_mode()
def scores(f,b):
 h=f.embedding(b);a=f.sae.encoder(h-f.sae.mean);z=f.sae.encode(h)[0]
 mu=torch.as_tensor(f.pca['mean'],dtype=h.dtype);d=torch.as_tensor(f.pca['directions'][:,3],dtype=h.dtype)
 return torch.stack([z[:,FEATURE],a[:,FEATURE],-h[:,201],(h-mu)@d],1).numpy()
class DeviceForward(mod.Forward):
 def __init__(self):
  super().__init__();self.cpu_model=self.model;self.gpu_model=copy.deepcopy(self.model).to('mps') if torch.backends.mps.is_available() else None;self.on_cpu=False
 @torch.inference_mode()
 def embedding(self,b):
  gpu=self.gpu_model is not None and not self.on_cpu;self.model=self.gpu_model if gpu else self.cpu_model
  bb={k:v.to('mps') for k,v in b.items()} if gpu else b
  return super().embedding(bb).cpu()
def cpu_scores(f,b):
 f.on_cpu=True
 try:return scores(f,b)
 finally:f.on_cpu=False
def fixed(path,obj):
 if path.exists():assert json.loads(path.read_text())==obj, str(path)+' changed'
 else:js(path,obj)
def setup():
 f=DeviceForward();torch.set_num_threads(8);torch.backends.mha.set_fastpath_enabled(False)
 z=load_npz(OLD/'primary_activations.npz')['kang']; k=f.k;split=f.manifest['splits'][0];ci=f.manifest['classes'].index('B cells');pairs=[]
 for role,n in [('train',2),('test',4)]:
  ix=np.array(split['indices'][role])
  for donor in sorted(np.unique(k['donors'][ix])):
   cells=ix[(k['donors'][ix]==donor)&(k['labels'][ix]==ci)]
   positive=cells[z[cells,FEATURE]>0];inactive=cells[z[cells,FEATURE]==0]
   assert len(positive)>=n and len(inactive)>=1
   for rank,pi in enumerate(sorted(positive,key=lambda i:(-float(z[i,FEATURE]),int(i)))[:n]):
    delta=k['embedding'][inactive].astype(float)-k['embedding'][pi];dist=np.einsum('ij,ij->i',delta,delta)
    refs=sorted(range(len(inactive)),key=lambda j:(dist[j],int(inactive[j])))[:3]
    for rr,j in enumerate(refs):
     ri=inactive[j];pairs.append({'pair_id':len(pairs),'role':role,'donor':str(donor),'prototype_index':int(pi),'reference_index':int(ri),'prototype_raw_row':int(k['rows'][pi]),'reference_raw_row':int(k['rows'][ri]),'prototype_cell_id':str(k['cell_ids'][pi]),'reference_cell_id':str(k['cell_ids'][ri]),'prototype_rank':rank,'reference_rank':rr,'reference_distance':float(np.sqrt(dist[j])),'n_references':len(refs)})
 protocol={'feature':FEATURE,'target':'B cells','models':'frozen historical primary SAE K32/scGPT','prototype_rule':'top2 active B cells per 4 TRAIN donors; top4 per single TEST donor','reference_rule':'up to3 nearest same-donor same-type zero-SAE cells; no fallback, no replacement','averaging':'reference equal within prototype, prototype equal within donor, donor equal within split','intervention':'fixed original gene tokens/padding; replace a single binned value from reference full-cell bins; 0 when absent; no raw-count causal interpretation','score_names':NAMES,'backend':'production MPS float32 batch16; each pair largest-effect CPU slow replay; CPU verification allowed','comparison':'same prototypes/references selected by SAE; raw coordinates are diagnostic controls, not unbiased global interpretability comparison','gene_absence':'zero contribution when not tokenized; report coverage and conditional score separately','test':'freeze gene rankings from TRAIN; selected high-activation prototypes, one donor; reused references are not independent samples','deletion':'0,1,5,10,20,40; own TRAIN coordinate ranking, TRAIN expression difference,10 joint-matched random sets, SAE nonribosomal ranking','random_match':'original bin//10, sign(reference-original), absolute difference//5; without replacement within order; attribution genes may overlap; report overlap','seed':20261019}
 OUT.mkdir(parents=True,exist_ok=True);(BASE/'cache').mkdir(parents=True,exist_ok=True)
 fixed(OUT/'protocol.json',protocol);fixed(OUT/'pairs.json',pairs)
 sources={str(p):sha(p) for p in [Path(__file__),WORK/'backend_benchmark.json',OLD/'common.py',OLD/'02_forward.py',f.sae_path,OLD/'primary_activations.npz',OLD/'pca.npz',OLD.parents[1]/'outputs/sae_interpretability/input_manifest.json']}
 # Verify every original input in previous manifest, without assuming field schema.
 manifest=json.loads((OLD.parents[1]/'outputs/sae_interpretability/input_manifest.json').read_text())
 js(OUT/'producer_manifest.json',sources)
 return f,pairs,sources

def run():
 start=time.time();f,pairs,sources=setup();assert f.gpu_model is not None,'MPS unavailable for production';total=0;reuse=0;errors=[]
 for pair in pairs:
  dest=BASE/'cache'/('pair_%02d.npz'%pair['pair_id']);binding=dest.with_suffix('.json')
  identity={'pair':pair,'sources':sources,'protocol_sha256':sha(OUT/'protocol.json')}
  if dest.exists():
   saved=json.loads(binding.read_text());assert saved['identity']==identity and saved['sha256']==sha(dest);reuse+=1;continue
  b=f.c.tokenize([pair['prototype_raw_row']]);v=b['values'].numpy()[0];pos=np.flatnonzero(~b['padding'].numpy()[0]);cols=np.array([f.vocab_to_col[int(b['genes'][0,j])] for j in pos]);ref=f.full_bins(pair['reference_raw_row'])[cols]
  baseline=scores(f,b)[0];effects=np.zeros((len(cols),4),np.float32);changed=np.flatnonzero(v[pos]!=ref)
  for offset in range(0,len(changed),16):
   ii=changed[offset:offset+16];values=np.repeat(v[None,:],len(ii),0);values[np.arange(len(ii)),pos[ii]]=ref[ii]
   effects[ii]=baseline-scores(f,f.batch(b,values))
  # Independent slow backend replay of largest SAE effect; all 4 output coordinates.
  j=int(np.argmax(np.abs(effects[:,0])));vv=v.copy();vv[pos[j]]=ref[j]
  slow=cpu_scores(f,b)[0]-cpu_scores(f,f.batch(b,vv[None,:]))[0]
  err=float(np.max(np.abs(slow-effects[j])));assert err<3e-5;errors.append(err)
  np.savez_compressed(dest,columns=cols,positions=pos,original_bins=v[pos],reference_bins=ref,baseline=baseline,effects=effects,changed=changed,replay_error=err)
  js(binding,{'identity':identity,'sha256':sha(dest)});total+=len(changed)
  print('PAIR',pair['pair_id']+1,'/',len(pairs),pair['role'],pair['donor'],'forwards',len(changed),'elapsed_s',round(time.time()-start),flush=True)
 js(OUT/'occlusion_completion.json',{'status':'complete','pairs':len(pairs),'new_forwards':total,'reused_pairs':reuse,'seconds':time.time()-start,'backend':'MPS float32; independent CPU slow replay','max_new_replay_error':max(errors) if errors else None})
if __name__=='__main__':run()
