# -*- coding: utf-8 -*-
"""Descriptive post-hoc seed rankings; no model or split selection."""
from pathlib import Path
import json,hashlib,html
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'outputs/sae_evaluation';O=ROOT/'outputs/sae_evaluation_seed_summary'
SPECS=[('knn','knn_all.csv',['k'],{'macro_f1':'max','supported_macro_f1':'max','purity_macro_supported':'max','neighbor_retention':'max'}),('IFN','IFN_all.csv',['cell_type','variant'],{'r2':'max','spearman':'max','mae':'min'}),('fewshot','fewshot_all.csv',['fraction'],{'macro_f1':'max'}),('fewshot_AULC','fewshot_AULC_all.csv',[],{'AULC':'max'}),('clustering','clustering_all.csv',['resolution'],{'ARI':'max','NMI':'max','cell_type_silhouette':'max'}),('geometry','geometry_all.csv',['role'],{'reconstruction_r2':'max','relative_displacement':'min','centered_relative_error':'min','relation_KL':'min'})]
allrows=[];group_rows=[];sources={}
for task,fn,keys,metrics in SPECS:
 p=S/fn;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();df=pd.read_csv(p);df['seed']=df.seed.astype(str)
 if task=='IFN':df=df[df.status=='COMPLETE']
 # Average subset repetitions first, then equal held-out biological-group weights.
 g=df.groupby(['dataset','method','seed','fold','group']+keys,dropna=False)[list(metrics)].mean().reset_index()
 for k in metrics:
  a=g[['dataset','method','seed','fold','group']+keys+[k]].rename(columns={k:'value'});a['task']=task;a['metric']=k;group_rows.extend(a.to_dict('records'))
 for config,q in g.groupby(['dataset','method']+keys,dropna=False):
  config=(config,) if not isinstance(config,tuple) else config;meta=dict(zip(['dataset','method']+keys,config))
  for metric,direction in metrics.items():
   vals=q.groupby('seed')[metric].mean();ng=q.groupby('seed').size();best=vals.max() if direction=='max' else vals.min();ties=vals.index[np.isclose(vals.to_numpy(),best,rtol=0,atol=1e-12)].tolist()
   for seed,v in vals.items():allrows.append(dict(**meta,task=task,metric=metric,direction=direction,seed=seed,value=float(v),n_groups=int(ng[seed]),is_best_seed=seed in ties,best_seeds=','.join(ties),best_value=float(best),method_mean=float(vals.mean())))
a=pd.DataFrame(allrows);a.to_csv(O/'all_seed_metrics.csv',index=False);pd.DataFrame(group_rows).to_csv(O/'all_group_metrics.csv',index=False)
conf=['dataset','method','task','metric','k','cell_type','variant','fraction','resolution','role'];b=a.drop_duplicates(conf).drop(columns=['seed','value','n_groups','is_best_seed']);b.to_csv(O/'best_seeds_all_configurations.csv',index=False)
PRIMARY=[('knn','macro_f1',{'k':15},'15NN Macro-F1',100),('knn','supported_macro_f1',{'k':15},'15NN supported Macro-F1',100),('knn','purity_macro_supported',{'k':15},'15NN purity',100),('IFN','r2',{'cell_type':'B cells','variant':'embedding'},'B IFN R2',1),('IFN','spearman',{'cell_type':'B cells','variant':'embedding'},'B IFN Spearman',1),('IFN','mae',{'cell_type':'B cells','variant':'embedding'},'B IFN MAE',1),('IFN','r2',{'cell_type':'CD14+ Monocytes','variant':'embedding'},'CD14 IFN R2',1),('IFN','spearman',{'cell_type':'CD14+ Monocytes','variant':'embedding'},'CD14 IFN Spearman',1),('IFN','mae',{'cell_type':'CD14+ Monocytes','variant':'embedding'},'CD14 IFN MAE',1),('fewshot_AULC','AULC',{},'Fewshot AULC',100),('clustering','ARI',{'resolution':1.},'ARI',1),('clustering','NMI',{'resolution':1.},'NMI',1),('clustering','cell_type_silhouette',{'resolution':1.},'Type silhouette',1),('geometry','reconstruction_r2',{'role':'test'},'Reconstruction R2',1),('geometry','relation_KL',{'role':'test'},'Relation KL',1),('knn','neighbor_retention',{'k':15},'15NN retention',100),('geometry','relative_displacement',{'role':'test'},'Relative displacement',1),('geometry','centered_relative_error',{'role':'test'},'Centered relative error',1)]
primary=[]
for ds in ['ctrl','stim','pbmc']:
 for task,metric,conds,label,scale in PRIMARY:
  q=a[(a.dataset==ds)&(a.task==task)&(a.metric==metric)]
  for k,v in conds.items():q=q[q[k]==v]
  r=dict(dataset=ds,metric=label,direction=q.direction.iloc[0],scale=scale)
  for method in ['original_h','mse','relation_0p3']:
   z=q[q.method==method];r[method+'_mean']=float(z.method_mean.iloc[0]*scale);r[method+'_best_value']=float(z.best_value.iloc[0]*scale);r[method+'_best_seeds']=z.best_seeds.iloc[0];r[method+'_n_groups']=int(z.n_groups.iloc[0])
  primary.append(r)
pd.DataFrame(primary).to_csv(O/'primary_metrics_best_seeds.csv',index=False)
# Compact tables for main answer; full details stay in the artifact.
for ds in ['ctrl','stim','pbmc']:
 print('\n'+ds)
 for r in primary:
  if r['dataset']!=ds:continue
  vals=[r['metric'],f"{r['original_h_mean']:.4f}"]
  for m in ['mse','relation_0p3']:vals.append(f"{r[m+'_mean']:.4f} / {r[m+'_best_value']:.4f} ({r[m+'_best_seeds']})")
  print(' | '.join(vals))
parts=['<h1>全部指标及种子排名</h1><p>所有数值来自已完成并审查的98组缓存评估，本次没有新训练或重新选模型。每个种子先在同一测试组内平均标签子集，再对留出组等权平均；Kang是供体，PBMC是批次。排名基于测试结果，仅作事后描述。指标之间不能各取最好种子再合成一个模型。</p><p>主表每项分别列三种子均值、该指标最好单种子值及种子编号；原始h没有SAE随机种子。误差小于1e-12视为并列。MAE、KL和位移/误差按最小值列，几何指标的方向只表示保真，不能解释成生物学效果更好。</p><p>少标签AULC是沿log实际标签数积分并归一化；表中乘100，属于曲线面积刻度。分类/纯度/保留率也乘100。其余指标为原始单位。ctrl B细胞IFN有效5供体，PBMC CD14 IFN仅1方向，其他Kang状态任务6供体；PBMC B细胞2方向。IFN仅同条件、同类型的RNA参考分数读出。</p>']
for ds in ['ctrl','stim','pbmc']:
 rows=[]
 for r in primary:
  if r['dataset']!=ds:continue
  z={'指标':r['metric'],'方向':r['direction'],'原始h':r['original_h_mean']}
  for m,name in [('mse','MSE'),('relation_0p3','关系λ0.3')]:z[name+'均值']=r[m+'_mean'];z[name+'最好值']=r[m+'_best_value'];z[name+'最好种子']=r[m+'_best_seeds']
  rows.append(z)
 parts.append('<h2>'+ds+'</h2>'+pd.DataFrame(rows).to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0))
parts.append('<h2>全部配置</h2><p>含k=5/15/30/50；IFN纯embedding、纯协变量、embedding+协变量；1/5/10/25/50/100%标签；Leiden分辨率0.5/1/2；TRAIN/TEST几何。纯协变量不依赖SAE种子，并列是预期结果。聚类数、样本数、目标方差及标签预算属于描述量，不指定最好方向。</p>'+b.to_html(index=False,float_format=lambda v:f'{v:.6f}',border=0))
(O/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAE全部指标及最佳种子</title><style>body{font-family:system-ui;max-width:1500px;margin:30px auto;padding:20px;line-height:1.65}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:6px 9px;border-bottom:1px solid #ddd;white-space:nowrap}</style>'+''.join(parts))
(O/'SOURCE_BINDINGS.json').write_text(json.dumps(sources,indent=2));(O/'REPRODUCIBILITY.txt').write_text('Run /opt/anaconda3/envs/scgpt/bin/python /Users/apple/Documents/Codex/2026-10-18/wo/work/sae_evaluation_seed_summary/summarize.py\nNo sealed upstream artifact was modified. Post-hoc descriptive seed rankings, not validation selection.\n')
