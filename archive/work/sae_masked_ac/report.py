# -*- coding: utf-8 -*-
"""Fixed A/C endpoint summaries; never select a model from TEST."""
import os
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_masked_ac';os.environ['MPLCONFIGDIR']=str(W/'mpl')
import json
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
METHODS=['original_h','clean_mse','A_sae','C_sae','A_dense'];NAMES={'original_h':'Original h','clean_mse':'Clean SAE MSE','A_sae':'A: masked h -> clean h','C_sae':'C: masked h -> masked genes','A_dense':'A: dense control'};COLORS=['#333333','#999999','#4477AA','#CC6677','#228833']
def save(p,d):
 def finite(v):
  if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
  if isinstance(v,list):return [finite(x) for x in v]
  if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
  return v
 p.write_text(json.dumps(finite(d),ensure_ascii=False,indent=2,allow_nan=False))
def aggregate(df,keys,metrics):
 g=df.groupby(['dataset','fold','group','method']+keys,dropna=False)[metrics].mean().reset_index();s=g.groupby(['dataset','method']+keys,dropna=False)[metrics].mean().reset_index();return g,s

def main():
 df=pd.read_csv(O/'all_metrics.csv');metrics=['ridge_macro_f1','ridge_supported_macro_f1','ridge_accuracy','knn_macro_f1','knn_supported_macro_f1','knn_purity','teacher_neighbor_retention'];g,s=aggregate(df,['view'],metrics);g.to_csv(O/'paired_group_metrics.csv',index=False);s.to_csv(O/'summary.csv',index=False);df.groupby(['dataset','method','seed','view'])[metrics].mean().reset_index().to_csv(O/'seed_metrics.csv',index=False)
 ge=pd.read_csv(O/'geometry_all.csv');gg,gs=aggregate(ge,['view'],['reconstruction_r2','normalized_embedding_MSE','relative_displacement','relation_KL']);gg.to_csv(O/'geometry_paired.csv',index=False);gs.to_csv(O/'geometry_summary.csv',index=False)
 st=pd.read_csv(O/'IFN_all.csv');st=st[st.status=='COMPLETE'];ig,isum=aggregate(st,['view','cell_type','variant'],['r2','spearman','mae']);isum=isum.merge(ig.groupby(['dataset','method','view','cell_type','variant']).size().rename('n_groups').reset_index(),on=['dataset','method','view','cell_type','variant']);ig.to_csv(O/'IFN_paired.csv',index=False);isum.to_csv(O/'IFN_summary.csv',index=False)
 ex=pd.read_csv(O/'expression_all.csv');eg,es=aggregate(ex,[],['MAE','normalized_MSE','gene_macro_MAE','gene_mean_baseline_MAE','gene_mean_baseline_normalized_MSE','gene_mean_baseline_gene_macro_MAE']);eg.to_csv(O/'expression_paired.csv',index=False);es.to_csv(O/'expression_summary.csv',index=False)
 contrast=[]
 for control in ['original_h','clean_mse']:
  ref=g[g.method==control][['dataset','fold','group','view']+metrics]
  q=g.merge(ref,on=['dataset','fold','group','view'],suffixes=('','_ref'))
  for metric in metrics:
   for key,a in q.groupby(['dataset','method','view']):
    delta=a[metric]-a[metric+'_ref'];contrast.append(dict(dataset=key[0],method=key[1],view=key[2],control=control,metric=metric,delta=float(delta.mean()),positive_groups=int((delta>1e-10).sum()),groups=len(a)))
 pd.DataFrame(contrast).to_csv(O/'planned_contrasts.csv',index=False)
 dr=g.pivot(index=['dataset','fold','group','method'],columns='view',values='ridge_macro_f1').reset_index();dr['clean_minus_masked_pp']=100*(dr.clean-dr.masked);dr.to_csv(O/'paired_degradation.csv',index=False)
 sg=df.pivot(index=['dataset','fold','group','method','seed'],columns='view',values='ridge_macro_f1').reset_index();sg['clean_minus_masked_pp']=100*(sg.clean-sg.masked);sg.groupby(['dataset','method','seed'])[['clean','masked','clean_minus_masked_pp']].mean().reset_index().to_csv(O/'degradation_seed_metrics.csv',index=False)
 hist=[]
 for q in (W/'networks').glob('*/*/*/*/training.json'):
  t=json.loads(q.read_text())
  for h in t['history']:hist.append(dict(dataset=t['dataset'],fold=t['fold'],seed=t['seed'],method=t['method'],**h))
 pd.DataFrame(hist).to_csv(O/'learning_curves.csv',index=False)
 save(O/'SUMMARY.json',dict(classification=s.to_dict('records'),IFN=isum.to_dict('records'),geometry=gs.to_dict('records'),expression=es.to_dict('records'),contrasts=contrast))
 fig,axs=plt.subplots(3,3,figsize=(16,12))
 for row,ds in enumerate(['ctrl','stim','pbmc']):
  for col,view in enumerate(['clean','masked']):
   ax=axs[row,col]
   for mi,(method,color) in enumerate(zip(METHODS,COLORS)):
    q=g[(g.dataset==ds)&(g.method==method)&(g.view==view)];ax.scatter(np.full(len(q),mi),q.ridge_macro_f1*100,color=color,alpha=.6,s=20);ax.scatter(mi,q.ridge_macro_f1.mean()*100,color=color,marker='_',s=300)
   ax.set_xticks(range(5),['h','MSE','A SAE','C SAE','A dense']);ax.set_title(f'{ds}: {view} TEST Ridge Macro-F1');ax.set_ylabel('%')
  ax=axs[row,2]
  for method,color in zip(METHODS,COLORS):
   q=isum[(isum.dataset==ds)&(isum.method==method)&(isum.cell_type=='CD14+ Monocytes')&(isum.variant=='embedding')];ax.plot([0,1],[float(q[q.view==v].r2.iloc[0]) for v in ['clean','masked']],'o-',color=color,label=NAMES[method])
  ax.set_xticks([0,1],['clean','masked']);ax.set_title(f'{ds}: CD14 IFN R2');ax.axhline(0,color='#aaaaaa',lw=.7)
 axs[0,2].legend(fontsize=7);fig.tight_layout();fig.savefig(O/'overview.png',dpi=180);fig.savefig(O/'overview.pdf');plt.close(fig)
 def tab(x):return x.to_html(index=False,border=0,float_format=lambda v:f'{v:.5f}')
 parts=['<h1>Gene-masked A/C SAE refinement</h1><p>冻结 scGPT。A：从遮蔽输入的 embedding 恢复完整 embedding；C：通过 SAE 的512维重建预测被遮蔽的基因表达。两者使用相同 SAE 初始权重、相同训练批次和1000步固定预算。加入原始h、等预算干净MSE续训和参数量相近的A-dense对照；A-dense从恒等残差初始化，不能据此作严格匹配初始化的SAE机制主张。</p><p>共126个新模型，42个既有MSE模型复用，182个固定表征/读出配置，分别评估干净与遮蔽TEST共364组。Ridge仅由TRAIN-clean拟合、VALID-clean选alpha，同一分类器直接评估两种TEST视图；无回退、无TEST选checkpoint或最好种子。Kang每条件六个历史供体，PBMC两批次方向。先平均组内三个SAE种子，再对组等权平均。ctrl/stim不是独立队列，PBMC不能称跨供体。</p><p>遮蔽固定25%原选已表达基因token；隐藏位置设padding，可见表达重新分箱/归一化时排除隐藏计数。每细胞一个固定遮蔽视图，所有方法/种子共用。该任务保留了原始已表达基因的存在/选择信息，不能等同真实任意基因缺失。</p><p>C仅恢复TRAIN选出2048高变基因中被遮蔽的正表达token，不含零表达或全转录组补全。目标为原RNA logCP10k表达，经TRAIN正表达均值/标准差缩放训练；测试MAE报告原logCP10k尺度，同时报告TRAIN逐基因均值基线。C多一个训练用512→2048线性头；下游只使用512维SAE输出。</p><p>IFN为同RNA八基因参考分数。主要看embedding-only读出；协变量与embedding+协变量变体使用完整RNA技术协变量，仅是oracle敏感性分析，不能称仅依赖遮蔽输入。固定样本阈值：ctrl B细胞5供体，PBMC CD14仅1方向，其他Kang状态分析6供体。所有数值是探索性，重建/代理指标不自动证明生物学改善。</p><p>固定1000步是预算比较，不代表各目标已收敛或最优；监控中部分C与dense-A的VALID损失在更早步骤更低，因此负结果只适用于本轮训练方案。本轮未重复少标签曲线或聚类扫描，未做B特征遮蔽、mask比例搜索、动态多遮蔽视图或模型微调。</p><img style="width:100%" src="overview.png"><h2>干净/遮蔽输入的分类与近邻结果</h2>',tab(s),'<h2>关系与重建诊断</h2>',tab(gs),'<h2>IFN状态读出</h2>',tab(isum),'<h2>C表达恢复及逐基因均值基线</h2>',tab(es),'<h2>相对对照的固定比较</h2>',tab(pd.DataFrame(contrast)),'<p>独立审查与结论边界见 EXPERIMENT_AUDIT.json 和 CLAIMS_FROM_RESULTS.json。逐组、逐种子、逐类别和预测文件均已保存。</p>']
 (O/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAE masked A/C</title><style>body{font-family:system-ui;max-width:1500px;margin:30px auto;padding:20px;line-height:1.7}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{padding:6px 10px;border-bottom:1px solid #ddd;white-space:nowrap}</style>'+''.join(parts))
 print(s.to_string(index=False))
if __name__=='__main__':main()
