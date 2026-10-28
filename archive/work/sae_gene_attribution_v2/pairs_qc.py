from pathlib import Path
import json,sys
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'sae_interpretability';OUT=BASE.parents[1]/'outputs/sae_gene_attribution_v2'
sys.path.insert(0,str(OLD))
from common import load_npz,dataset,sha,js
pairs=json.loads((OUT/'pairs.json').read_text());qc=load_npz(OLD/'qc.npz')['kang'];k,p,m=dataset();rows=[]
for a in pairs:
 row=dict(a)
 for role,idx in [('prototype',a['prototype_index']),('reference',a['reference_index'])]:
  row[role+'_total_counts']=float(np.expm1(qc[idx,0]));row[role+'_detected_genes']=float(np.expm1(qc[idx,1]))
 rows.append(row)
df=pd.DataFrame(rows);df.to_csv(OUT/'pairs_with_qc.csv',index=False)
cols=['prototype_total_counts','reference_total_counts','prototype_detected_genes','reference_detected_genes']
p=df.groupby(['role','donor','prototype_index'])[cols].mean().reset_index();p.groupby(['role','donor'])[cols].mean().to_csv(OUT/'pair_qc_donor_summary.csv')
# Check the exact prior input hashes and the original checkpoint/preprocessing sources now.
manifest=json.loads((OLD.parents[1]/'outputs/sae_interpretability/input_manifest.json').read_text());hashes={**manifest['sha256'],**m['sources']}
for path,h in hashes.items():assert sha(path)==h,path
js(OUT/'input_precheck.json',{'status':'PASS','checked_inputs':len(hashes),'sha256':hashes,'source':'existing historical manifests independently rehashed this run'})
print('INPUT_PRECHECK_PASS',len(hashes));print(p.groupby('role')[cols].mean().to_string())
