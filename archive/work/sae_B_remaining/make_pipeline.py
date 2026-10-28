from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parent/'sae_B898_1387/F898'
s=(P/'run.py').read_text().replace('import sys,json,time,importlib.util','import sys,json,time,importlib.util,os,copy')
s=s.replace("BASE=Path(__file__).resolve().parent\nOLD=BASE.parents[1]/'sae_interpretability';OUT=BASE.parents[2]/'outputs/sae_B898_1387/F898'", "WORK=Path(__file__).resolve().parent;FEATURE=int(os.environ['SAE_FEATURE']);BASE=WORK/f'F{FEATURE}'\nOLD=WORK.parent/'sae_interpretability';OUT=WORK.parents[1]/'outputs/sae_B_remaining'/f'F{FEATURE}'")
s=s.replace("NAMES=['sae_F898','encoder_F898','embedding_neg201','pca3']", "NAMES=[f'sae_F{FEATURE}',f'encoder_F{FEATURE}','embedding_neg201','pca3']")
s=s.replace(',898]',',FEATURE]').replace("'feature':898", "'feature':FEATURE")
insert='''class DeviceForward(mod.Forward):
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
'''
s=s.replace('def fixed(path,obj):',insert+'def fixed(path,obj):')
s=s.replace('f=mod.Forward();torch.set_num_threads(8);torch.backends.mha.set_fastpath_enabled(True)', 'f=DeviceForward();torch.set_num_threads(8);torch.backends.mha.set_fastpath_enabled(False)')
s=s.replace("'score_names':NAMES,", "'score_names':NAMES,'backend':'production MPS float32 batch16; each pair largest-effect CPU slow replay; CPU verification allowed',")
s=s.replace("OUT.mkdir(parents=True,exist_ok=True);(BASE/'cache').mkdir(exist_ok=True)", "OUT.mkdir(parents=True,exist_ok=True);(BASE/'cache').mkdir(parents=True,exist_ok=True)")
s=s.replace("Path(__file__),OLD/'common.py'", "Path(__file__),WORK/'backend_benchmark.json',OLD/'common.py'")
s=s.replace('start=time.time();f,pairs,sources=setup();total=0;reuse=0;errors=[]', "start=time.time();f,pairs,sources=setup();assert f.gpu_model is not None,'MPS unavailable for production';total=0;reuse=0;errors=[]")
s=s.replace("torch.backends.mha.set_fastpath_enabled(False);slow=scores(f,b)[0]-scores(f,f.batch(b,vv[None,:]))[0];torch.backends.mha.set_fastpath_enabled(True)", "slow=cpu_scores(f,b)[0]-cpu_scores(f,f.batch(b,vv[None,:]))[0]")
s=s.replace("'seconds':time.time()-start,", "'seconds':time.time()-start,'backend':'MPS float32; independent CPU slow replay',")
(W/'run.py').write_text(s)
s=(P/'analyze.py').read_text().replace("'sae_F898'",'NAMES[0]');s=s.replace('torch.backends.mha.set_fastpath_enabled(False);slow=scores(f,f.batch(b,values[j:j+1]))[0];torch.backends.mha.set_fastpath_enabled(True)', 'slow=cpu_scores(f,f.batch(b,values[j:j+1]))[0]');s=s.replace("sha(BASE/'run.py')","sha(WORK/'run.py')");(W/'analyze.py').write_text(s)
s=(P/'verify.py').read_text().replace(',898]',',FEATURE]').replace("'sae_F898'",'NAMES[0]');(W/'verify.py').write_text(s)
s=(P/'pairs_qc.py').read_text();start=s.index('BASE=');end=s.index('pairs=json.loads');s=s[:start]+"from run import BASE,OLD,OUT\nfrom common import load_npz,dataset,sha,js\n"+s[end:];(W/'pairs_qc.py').write_text(s)
