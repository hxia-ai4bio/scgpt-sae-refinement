import importlib.util,numpy as np
from sklearn.linear_model import Ridge
s=importlib.util.spec_from_file_location('evalrun','work/sae_evaluation/run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rng=np.random.default_rng(7);out=[]
for n,d in [(25,40),(80,15)]:
 x=rng.normal(size=(n,d));y=rng.normal(size=(n,3));xt=rng.normal(size=(9,d));w=rng.uniform(.1,2,n)
 a=m.fit(x,y,xt,w);b=Ridge(alpha=10).fit(x,y,sample_weight=w).predict(xt);err=float(np.max(np.abs(a-b)));assert err<1e-10;out.append(dict(n=n,d=d,max_error=err))
m.save(m.O/'READOUT_VERIFICATION.json',dict(status='PASS',checks=out))
print(out)
