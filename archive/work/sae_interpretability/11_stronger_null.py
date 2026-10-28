"""Post-hoc audit-motivated control; fixed candidates and TRAIN gene orders.

Random genes match original-bin decile, substitution sign and absolute-change
5-bin stratum. Sparse strata may substantially overlap attribution genes; this
is exported rather than hidden. No inferential p value from two selected cells.
"""
import importlib.util,json,time
import numpy as np
import pandas as pd
import torch
from common import *

def main():
    t=time.time();spec=importlib.util.spec_from_file_location('forward_impl',HERE/'02_forward.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    f=mod.Forward();torch.set_num_threads(8);torch.backends.mha.set_fastpath_enabled(True)
    pairs=json.loads((OUT/'prototype_pairs.json').read_text());out=[];diagnostics=[]
    js(OUT/'strong_null_protocol.json',{'status':'post-hoc audit-motivated control','candidate_selection':'unchanged',
       'rankings':'unchanged frozen TRAIN attribution','prototypes':'same two TEST cells per feature',
       'strata':'original expression bin//10, sign(reference-original), abs(reference-original)//5',
       'randomization':'sample without replacement within joint stratum; 10 deterministic random orders per prototype',
       'caveat':'exact stratum matching may reuse many attribution genes; overlap and actual magnitude reported; one donor pilot, no significance claim'})
    for pair in [p for p in pairs if p['role']=='test']:
      feature=pair['feature'];ci=pair['candidate_index'];a=load_npz(HERE/'occlusion'/('pair_%02d.npz'%pair['pair_id']))
      df=pd.read_csv(OUT/f'gene_attribution_{feature}_train.csv');rank=dict(zip(df.gene,df.mean_attribution))
      eligible=np.array([i for i,col in enumerate(a['columns']) if f.genes[col] in rank and a['original_bins'][i]!=a['reference_bins'][i]])
      attr=np.array(sorted(eligible,key=lambda i:(-rank[f.genes[a['columns'][i]]],int(a['columns'][i]))))[:40]
      delta=a['reference_bins']-a['original_bins'];strata=[(int(v)//10,int(np.sign(d)),int(abs(d))//5) for v,d in zip(a['original_bins'],delta)]
      base=f.c.tokenize([pair['prototype_raw_row']]);orig=base['values'].numpy()[0];inputs=[];meta=[]
      for rep in range(10):
        rng=np.random.default_rng([20261018,991,pair['pair_id'],rep]);left=set(eligible.tolist());chosen=[]
        for j in attr:
          pool=sorted(i for i in left if strata[i]==strata[j]);assert pool
          i=int(rng.choice(pool));chosen.append(i);left.remove(i)
        overlap=len(set(chosen)&set(attr));diagnostics.append({'feature':feature,'pair_id':pair['pair_id'],'rep':rep,'matched_genes':len(chosen),'attribution_genes_reused':overlap,'attribution_mean_abs_bin_change':float(np.mean(np.abs(delta[attr]))),'random_mean_abs_bin_change':float(np.mean(np.abs(delta[chosen]))),'exact_joint_stratum_match':True})
        chosen=np.array(chosen)
        for n in [0,1,5,10,20,40]:
          idx=chosen[:n];vals=orig.copy();vals[a['positions'][idx]]=a['reference_bins'][idx];inputs.append(vals);meta.append((rep,n,len(idx)))
      scores=np.concatenate([f.scores(f.batch(base,np.array(inputs[i:i+16]))) for i in range(0,len(inputs),16)])
      # Save exact numerical inputs and outputs for reproducible audit of the new control.
      dest=HERE/'strong_null';dest.mkdir(exist_ok=True)
      np.savez_compressed(dest/f'pair_{pair["pair_id"]:02d}.npz',values=np.array(inputs),scores=scores,
                         order_repetition=np.array([v[0] for v in meta]),requested_genes=np.array([v[1] for v in meta]))
      for (rep,n,count),score in zip(meta,scores):
        out.append({'feature':feature,'pair_id':pair['pair_id'],'order':'strong_matched_random_'+str(rep),
          'requested_genes':n,'changed_genes':count,'raw_activation':float(score[ci]),'raw_baseline':float(a['baseline'][ci]),'raw_drop':float(a['baseline'][ci]-score[ci])})
      print('STRONG_NULL',feature,pair['pair_id'],flush=True)
    pd.DataFrame(out).to_csv(OUT/'heldout_deletion_strong_null.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'strong_null_matching_diagnostics.csv',index=False)
    js(OUT/'strong_null_completion.json',{'status':'complete','forwards':len(out),'seconds':time.time()-t,'scope':'post-hoc conditional null; same locked two prototypes per feature and one test donor'})

if __name__=='__main__':main()
