import importlib.util,time,json
from pathlib import Path
from threadpoolctl import threadpool_limits
p=Path(__file__).resolve().parent/'pipeline.py';s=importlib.util.spec_from_file_location('masked_train_benchmark',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=m.canonical('ctrl');d,f,ix=m.fold_data(a,'ctrl',0);pn=m.panel('ctrl',0,a,ix);t=m.targets(a,pn);idx=ix['train'][:256];h=m.torch.from_numpy(a['masked_embedding'][idx]);y=m.torch.from_numpy(a['embedding'][idx]);co=m.torch.from_numpy(t['cols'][idx]);va=m.torch.from_numpy(t['valid'][idx]);ta=m.torch.from_numpy(t['target'][idx]);den=.02;out=[]
for threads in [1,4]:
 m.torch.set_num_threads(threads)
 with threadpool_limits(limits=threads):
  for arm in ['A_sae','C_sae']:
   model=m.Refiner('ctrl',0,42,arm,d['train']['h']);opt=m.torch.optim.Adam(model.parameters(),lr=.0003);start=time.time()
   for j in range(100):
    opt.zero_grad(set_to_none=True);loss=m.mask_loss(model,h,y,co,va,ta,den);loss.backward();opt.step();model.normalize()
   out.append(dict(threads=threads,arm=arm,steps=100,seconds=time.time()-start))
m.save(m.O/'TRAIN_BENCHMARK.json',out);print(out,flush=True)
