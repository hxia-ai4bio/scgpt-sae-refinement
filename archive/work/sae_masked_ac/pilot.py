"""One source-only training/VALID health block; no TEST prediction or score."""
import importlib.util
from pathlib import Path
from threadpoolctl import threadpool_limits
p=Path(__file__).resolve().parent/'pipeline.py';s=importlib.util.spec_from_file_location('masked_pilot_runtime',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.torch.set_num_threads(4);m.torch.use_deterministic_algorithms(True)
with threadpool_limits(limits=4):
 a=m.canonical('ctrl');d,f,ix=m.fold_data(a,'ctrl',0);pn=m.panel('ctrl',0,a,ix);t=m.targets(a,pn)
 for arm in m.ARMS:m.train('ctrl',0,42,arm,a,d,ix,pn,t)
 records=[]
 for arm in m.ARMS:
  q=m.W/'networks/ctrl/fold_0/seed_42'/arm/'training.json';v=m.read(q)
  assert all(m.np.isfinite(z) for g in v['gradients'] for z in g['norms'].values())
  assert all(z>0 for z in v['gradients'][1]['norms'].values()),v['gradients']
  records.append(dict(arm=arm,seconds=v['seconds'],history=v['history'],parameters=v['parameter_count'],gradient_step2=v['gradients'][1]))
 m.save(m.O/'PILOT.json',dict(status='PASS',dataset='ctrl',fold=0,seed=42,roles_seen=['TRAIN','VALID'],TEST_scored=False,training=records));print('PILOT_HEALTH_PASS',flush=True)
