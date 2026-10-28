"""Initialization controls for dictionary stability, not biological validation."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from common import *

def dictionary(model):
    d=model.decoder.weight.detach().double();return d/d.norm(dim=0)

def main():
    torch.set_num_threads(2);final,path=load_sae();initial=TopKSAE(final.mean,32,42).eval()
    d=dictionary(final);di=dictionary(initial);candidates=json.loads((OUT/'candidates.json').read_text());out=[]
    for seed in [43,44]:
      other=TopKSAE(final.mean,32,seed).eval();dj=dictionary(other);sim=(di.T@dj).numpy()
      ri,ci=linear_sum_assignment(-sim);mapping=dict(zip(ri,ci))
      for c in candidates:
        f=c['feature'];out.append({'feature':f,'other_seed':seed,'initial_random_matched_feature':int(mapping[f]),
          'initial_random_matched_cosine':float(sim[f,mapping[f]]),'final_vs_own_initial_cosine':float(torch.dot(d[:,f],di[:,f])),
          'scope':'architecture/initialization control; final cross-fold same-seed stability may reflect shared initialization'})
    pd.DataFrame(out).to_csv(OUT/'dictionary_initialization_controls.csv',index=False)

if __name__=='__main__':main()
