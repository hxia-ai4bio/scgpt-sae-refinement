# -*- coding: utf-8 -*-
"""Summarize fixed endpoints without TEST-selected methods or seeds."""
from pathlib import Path
import os,json,hashlib,html
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'mpl')
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/sae_evaluation'
METHODS=['original_h','mse','relation_0p3'];NAMES={'original_h':'Original h','mse':'MSE SAE','relation_0p3':'Relation SAE (lambda=0.3)'};COLORS=['#333333','#4477AA','#CC6677']
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def tab(df):return df.to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0)
def aggregate(df,keys,metrics):
 # Inner subset/seed repeats are averaged within each actual heldout group.
 groups=df.groupby(['dataset','fold','group','method']+keys,dropna=False)[metrics].mean().reset_index();groups.to_csv(O/('paired_'+str(aggregate.n)+'.csv'),index=False);aggregate.n+=1
 means=groups.groupby(['dataset','method']+keys,dropna=False)[metrics].mean().reset_index();return groups,means
aggregate.n=0

def main():
 kn=pd.read_csv(O/'knn_all.csv');fs=pd.read_csv(O/'fewshot_all.csv');st=pd.read_csv(O/'IFN_all.csv');cl=pd.read_csv(O/'clustering_all.csv');ge=pd.read_csv(O/'geometry_all.csv')
 groups,km=aggregate(kn[kn.k==15],[],['macro_f1','purity_macro_supported','neighbor_retention']);groups.to_csv(O/'knn_paired_groups.csv',index=False);km.to_csv(O/'knn_summary.csv',index=False)
 st=st[st.status=='COMPLETE'];sg,sm=aggregate(st,['cell_type','variant'],['r2','spearman','mae','target_variance']);sg.to_csv(O/'IFN_paired_groups.csv',index=False);sm=sm.merge(sg.groupby(['dataset','method','cell_type','variant']).size().rename('n_groups').reset_index(),on=['dataset','method','cell_type','variant']);sm.to_csv(O/'IFN_summary.csv',index=False)
 fg,fm=aggregate(fs,['fraction'],['macro_f1','labeled_n']);fg.to_csv(O/'fewshot_paired_groups.csv',index=False);fm.to_csv(O/'fewshot_summary.csv',index=False)
 ar=[]
 for keys,a in fs.groupby(['dataset','fold','group','method','seed','subset_seed']):
  a=a.sort_values('labeled_n');x=np.log(a.labeled_n.to_numpy());v=np.trapz(a.macro_f1.to_numpy(),x)/(x[-1]-x[0]);ar.append(dict(zip(['dataset','fold','group','method','seed','subset_seed'],keys),AULC=float(v)))
 au=pd.DataFrame(ar);au.to_csv(O/'fewshot_AULC_all.csv',index=False);ag,am=aggregate(au,[],['AULC']);ag.to_csv(O/'fewshot_AULC_paired_groups.csv',index=False);am.to_csv(O/'fewshot_AULC_summary.csv',index=False)
 cg,cm=aggregate(cl[cl.resolution==1.],[],['ARI','NMI','cell_type_silhouette']);cg.to_csv(O/'clustering_paired_groups.csv',index=False);cm.to_csv(O/'clustering_summary.csv',index=False)
 gg,gm=aggregate(ge,['role'],['reconstruction_r2','relative_displacement','centered_relative_error','relation_KL']);gg.to_csv(O/'geometry_paired_groups.csv',index=False);gm.to_csv(O/'geometry_summary.csv',index=False)
 pairs=[]
 for task,df,keys,metrics in [('knn',groups,[],['macro_f1','purity_macro_supported']),('IFN',sg,['cell_type','variant'],['r2','spearman']),('fewshot_AULC',ag,[],['AULC']),('clustering',cg,[],['ARI','NMI','cell_type_silhouette'])]:
  for control in ['original_h','mse']:
   kk=['dataset','fold','group']+keys;ref=df[df.method==control][kk+metrics]
   joined=df.merge(ref,on=kk,suffixes=('','_ref'))
   for metric in metrics:
    for extra,a in joined.groupby(['dataset','method']+keys,dropna=False):
     delta=a[metric]-a[metric+'_ref'];r=dict(zip(['dataset','method']+keys,extra));pairs.append(dict(task=task,control=control,metric=metric,delta_mean=float(delta.mean()),positive_groups=int((delta>1e-10).sum()),n_groups=len(a),**r))
 pd.DataFrame(pairs).to_csv(O/'planned_contrasts.csv',index=False)
 # Retain full seed-level endpoint tables and per-class metrics in source CSVs.
 fig,axes=plt.subplots(3,4,figsize=(17,10))
 for row,ds in enumerate(['ctrl','stim','pbmc']):
  for col,ct in [(1,'B cells'),(2,'CD14+ Monocytes')]:
   ax=axes[row,col]
   for method,color in zip(METHODS,COLORS):
    q=sg[(sg.dataset==ds)&(sg.method==method)&(sg.cell_type==ct)&(sg.variant=='embedding')];ax.scatter(np.full(len(q),METHODS.index(method)),q.r2,color=color,alpha=.65,s=20);ax.scatter(METHODS.index(method),q.r2.mean(),color=color,marker='_',s=300)
   ax.set_xticks(range(3),['h','MSE','Rel']);ax.set_title(f'{ds}: {ct} IFN R2');ax.axhline(0,color='#aaaaaa',lw=.7)
  ax=axes[row,0]
  for method,color in zip(METHODS,COLORS):
   q=groups[(groups.dataset==ds)&(groups.method==method)];ax.scatter(np.full(len(q),METHODS.index(method)),100*q.macro_f1,color=color,alpha=.65,s=20);ax.scatter(METHODS.index(method),100*q.macro_f1.mean(),color=color,marker='_',s=300)
  ax.set_xticks(range(3),['h','MSE','Rel']);ax.set_title(f'{ds}: 15NN Macro-F1 (%)')
  ax=axes[row,3]
  for method,color in zip(METHODS,COLORS):
   q=fm[(fm.dataset==ds)&(fm.method==method)].sort_values('fraction');ax.plot(q.fraction*100,q.macro_f1*100,'o-',color=color,label=NAMES[method])
  ax.set_xscale('log');ax.set_title(f'{ds}: fixed Ridge label efficiency');ax.set_xlabel('TRAIN labels retained (%)');ax.set_ylabel('Macro-F1 (%)')
 axes[0,3].legend(fontsize=7);fig.tight_layout();fig.savefig(O/'evaluation_overview.png',dpi=180);fig.savefig(O/'evaluation_overview.pdf');plt.close(fig)
 contrasts=pd.DataFrame(pairs);maincontrast=contrasts[(contrasts.method=='relation_0p3')&(contrasts.control=='original_h')&((contrasts.task!='IFN')|(contrasts.variant=='embedding'))]
 body='''<h1>SAE 表征评估：固定协议结果</h1><p>三种方法：原始 h、MSE SAE、关系约束 SAE（固定 λ=0.3）。本轮未重新训练 scGPT/SAE。全部 98 组已完成，未按测试结果挑选参数或种子。</p><p>统计单位：Kang 六个留出供体；PBMC 两个批次方向。先平均同组内的 SAE 种子，再对组等权平均；少标签先平均相同标签子集。ctrl/stim 来自相同供体，不是独立队列。历史数据上的新增终点属于探索性结果。</p><p>IFN 分数是同一 RNA 的八基因参考分数，不是独立实验真值。按预定样本阈值，ctrl 的 B 细胞排除 patient_1039（TEST 26&lt;30），PBMC 的 CD14 分析排除源 Batch2→目标 Batch1 方向（TRAIN 67&lt;100）；其余方法共用相同任务子集，IFN 表明确列出有效组数。分别在条件内和细胞类型内拟合固定 Ridge；没有将两套 SAE 输出混合用于 ctrl/stim 分类。少标签实验使用全部 TRAIN 无标签细胞拟合尺度；标签子集按完整 TRAIN 标签分层抽取。因此 labeled_n 是下游拟合所用标签数，不代表前瞻式标注总成本。SAE 沿用此前 VALID-MSE 选型，此无监督选择本身不消耗细胞类型标签。</p><p>本轮只完成缓存可支持的第一阶段。统一 ctrl/stim SAE 和 UMI 下采样属于后续实验，未执行。Macro-F1 固定计算八类；测试组缺失类型的 F1 记 0，支持类型 Macro-F1 另见原始 kNN 表。完整七点 λ 扫描未在这些新增终点上重复；本轮只比较预定的三种主方法。</p><img style="width:100%" src="evaluation_overview.png"><h2>固定关系 SAE 相对 h 的配对变化</h2>'''+tab(maincontrast[['dataset','task','cell_type','metric','delta_mean','positive_groups','n_groups']])+'''<p>Macro-F1、AULC、纯度和 silhouette 的 delta 为 0–1 单位；乘 100 得百分点。R²、ARI、NMI 保留原始单位。正向组数用于描述一致性，不是显著性检验。</p><h2>15NN 标签迁移</h2>'''+tab(km)+'''<h2>类型内 IFN 读出</h2>'''+tab(sm)+'''<h2>少标签曲线面积</h2>'''+tab(am)+'''<h2>聚类（固定分辨率 1.0）</h2>'''+tab(cm)+'''<p>kNN 标签迁移、聚类和分类均使用细胞类型标签，不能当作三份独立生物学证据。silhouette 仅是辅助描述，聚类数随方法变化时同时查看 ARI/NMI。</p><h2>几何诊断</h2>'''+tab(gm)+'''<p>重建 R² 的分母为相对原始 TRAIN 均值的平方偏差；KL 的教师来自原始 h，衡量教师几何保真，不证明生物学改善。统一使用原始 TRAIN 均值中心化后计算近邻。PBMC 是源批次训练、目标批次测试，没有可验证供体信息。</p><h2>审查状态</h2><p>见 EXPERIMENT_AUDIT.json 与 CLAIMS_FROM_RESULTS.json；独立复核完成前不将探索性数值写成论文结论。</p>'''
 page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAE 表征评估</title><style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:35px auto;padding:0 25px;color:#172b3a;line-height:1.7}table{font-size:12px;border-collapse:collapse;display:block;overflow:auto}td,th{padding:7px 10px;border-bottom:1px solid #ddd;white-space:nowrap}h1,h2{color:#173f5f}p{max-width:1100px}</style>'+body+'</html>'
 (O/'REPORT.html').write_text(page)
 save(O/'SUMMARY.json',dict(knn=km.to_dict('records'),IFN=sm.to_dict('records'),fewshot_AULC=am.to_dict('records'),clustering=cm.to_dict('records'),geometry=gm.to_dict('records')))
 print(maincontrast[['dataset','task','cell_type','metric','delta_mean','positive_groups','n_groups']].to_string(index=False))
if __name__=='__main__':main()
