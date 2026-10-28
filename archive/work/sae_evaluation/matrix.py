from pathlib import Path
import pandas as pd
p=Path(__file__).resolve().parents[2]/'outputs/sae_evaluation'
a=pd.read_csv(p/'knn_summary.csv').rename(columns={'macro_f1':'knn_macro_f1','purity_macro_supported':'knn_purity_macro_supported'})
a=a.merge(pd.read_csv(p/'fewshot_AULC_summary.csv').rename(columns={'AULC':'fewshot_AULC'}),on=['dataset','method'])
a=a.merge(pd.read_csv(p/'clustering_summary.csv'),on=['dataset','method'])
f=pd.read_csv(p/'IFN_summary.csv');f=f[f.variant=='embedding']
for typ,name in [('B cells','B'),('CD14+ Monocytes','CD14')]:
 q=f[f.cell_type==typ][['dataset','method','r2','n_groups']].rename(columns={'r2':f'IFN_{name}_r2','n_groups':f'IFN_{name}_n_groups'});a=a.merge(q,on=['dataset','method'])
a.to_csv(p/'comparison_matrix.csv',index=False)
