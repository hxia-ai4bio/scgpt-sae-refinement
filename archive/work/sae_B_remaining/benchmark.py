from pathlib import Path
import sys,json,time,importlib.util
import numpy as np,torch
W=Path(__file__).resolve().parent;OLD=W.parent/'sae_interpretability';sys.path.insert(0,str(OLD));spec=importlib.util.spec_from_file_location('forward_old',OLD/'02_forward.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
@torch.inference_mode()
def main():
 torch.set_num_threads(8);f=mod.Forward();torch.set_num_threads(8);features=[137,370,146,1373,1125,1409];z=np.load(OLD/'primary_activations.npz')['kang'];ix=np.array(f.manifest['splits'][0]['indices']['train']);ix=ix[f.k['labels'][ix]==f.manifest['classes'].index('B cells')];pi=int(ix[np.argmax(z[ix,137])]);base=f.c.tokenize([int(f.k['rows'][pi])]);v=base['values'].numpy()[0];vals=np.repeat(v[None,:],32,0)
 for i in range(32):vals[i,i]=0
 batches={n:f.batch(base,vals[:n]) for n in [8,16,32]};cpu={};results=[]
 torch.backends.mha.set_fastpath_enabled(True)
 for n,b in batches.items():
  t=time.time();h=f.embedding(b).numpy();elapsed=time.time()-t;cpu[n]=h;results.append({'backend':'CPU','batch':n,'seconds':elapsed,'cells_per_second':n/elapsed})
 if torch.backends.mps.is_available():
  f.model.to('mps');torch.backends.mha.set_fastpath_enabled(False)
  for n,b in batches.items():
   bb={k:v.to('mps') for k,v in b.items()};t=time.time();h=f.embedding(bb).cpu();torch.mps.synchronize();elapsed=time.time()-t;a=f.sae.encode(h)[0][:,features].numpy();c=f.sae.encode(torch.from_numpy(cpu[n]))[0][:,features].numpy();err=float(np.max(np.abs(h.numpy()-cpu[n])));scoreerr=float(np.max(np.abs(a-c)));results.append({'backend':'MPS','batch':n,'seconds':elapsed,'cells_per_second':n/elapsed,'embedding_max_error':err,'target_score_max_error':scoreerr,'parity_PASS':err<3e-5 and scoreerr<3e-5})
   print(results[-1],flush=True)
 (W/'backend_benchmark.json').write_text(json.dumps({'prototype_index':pi,'tokens':int((~base['padding']).sum()),'features':features,'results':results},indent=2));print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
