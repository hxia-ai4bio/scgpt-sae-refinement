"""Post-hoc fixed-feature diagnosis of TopK suppression; no model tuning."""
import json
import numpy as np
import pandas as pd
import torch
from common import *

def main():
    torch.set_num_threads(2);k,p,m=dataset();model,path=load_sae();candidates=json.loads((OUT/'candidates.json').read_text())
    records=[]
    for domain,data,groupkey in [('kang',k,'donors'),('pbmc',p,'batches')]:
      with torch.inference_mode():
        raw=model.encoder(torch.as_tensor(data['embedding'])-model.mean).numpy()
      threshold=np.partition(raw,-32,axis=1)[:,-32]
      for c in candidates:
        f=c['feature'];ci=m['classes'].index(c['target']) if c['target']!='all' else None
        for group in np.unique(data[groupkey]):
          ix=np.flatnonzero(data[groupkey]==group);ti=ix if ci is None else ix[data['labels'][ix]==ci]
          a=raw[ix,f];v=np.where(a>=threshold[ix],a,0)
          target_margin=raw[ti,f]-threshold[ti]
          records.append({'feature':f,'target':c['target'],'domain':domain,'group':group,
              'raw_encoder_auc':float(auc_columns(a[:,None],data['labels'][ix]==ci)[0]) if ci is not None else None,
              'sparse_auc':float(auc_columns(v[:,None],data['labels'][ix]==ci)[0]) if ci is not None else None,
              'target_n':len(ti),'target_raw_mean':float(raw[ti,f].mean()),
              'target_topk_threshold_mean':float(threshold[ti].mean()),
              'target_margin_median':float(np.median(target_margin)),
              'target_retained_fraction':float((target_margin>=0).mean())})
    pd.DataFrame(records).to_csv(OUT/'topk_suppression_diagnostics.csv',index=False)
    js(OUT/'topk_diagnostic_scope.json',{'scope':'post-hoc diagnosis of already selected features; pre-TopK classifier is not an intervention or proposed improved SAE; no test-based model selection'})

if __name__=='__main__':main()
