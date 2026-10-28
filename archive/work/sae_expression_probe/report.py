# -*- coding: utf-8 -*-
"""Paired summaries of frozen expression probes; no TEST model selection."""
from pathlib import Path
import json,html
import pandas as pd
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_expression_probe'
def main():
    d=pd.read_csv(O/'all_metrics.csv');keys=['MAE','normalized_MSE','gene_macro_MAE','gene_mean_baseline_MAE']
    g=d.groupby(['dataset','fold','group','method'])[keys].mean().reset_index();s=g.groupby(['dataset','method'])[keys].mean().reset_index()
    g.to_csv(O/'group_metrics.csv',index=False);s.to_csv(O/'summary.csv',index=False)
    seed=d.groupby(['dataset','method','seed'])[keys].mean().reset_index();seed.to_csv(O/'seed_metrics.csv',index=False)
    base=d[d.method=='original_h'];c=d[d.method=='C_sae'];p=base.merge(c,on=['dataset','fold','group','seed'],suffixes=('_original','_C'))
    p['MAE_delta_C_minus_original']=p.MAE_C-p.MAE_original;p['MAE_reduction_pct']=(1-p.MAE_C/p.MAE_original)*100;p.to_csv(O/'paired_metrics.csv',index=False)
    group=p.groupby(['dataset','fold','group'])[['MAE_original','MAE_C','MAE_delta_C_minus_original']].mean().reset_index();summary=[]
    for ds,x in group.groupby('dataset'):
        a=x.MAE_original.mean();b=x.MAE_C.mean()
        summary.append(dict(dataset=ds,original_MAE=a,C_frozen_MAE=b,delta_C_minus_original=b-a,reduction_pct=(1-b/a)*100,improved_groups=int((x.MAE_delta_C_minus_original<0).sum()),groups=len(x)))
    j=dict(primary=summary,metrics=s.to_dict('records'),seed_metrics=seed.to_dict('records'));(O/'SUMMARY.json').write_text(json.dumps(j,indent=2))
    pd.DataFrame(summary).to_csv(O/'comparison.csv',index=False)
    claims=[dict(id=f"{r['dataset']}_{k}",value=r[k],source='outputs/sae_expression_probe/SUMMARY.json') for r in summary for k in ['original_MAE','C_frozen_MAE','delta_C_minus_original']]
    (O/'EVIDENCE_CLAIMS.json').write_text(json.dumps(claims,indent=2))
    title='冻结表征的 masked gene expression 预测对照'
    body=['<h1>'+title+'</h1>','<p>两边输入均遮蔽相同25%原选已表达基因token。原始512维scGPT与既有C优化512维表征全部冻结；丢弃C旧表达头，各自重新训练相同512→2048线性头。输入按各自TRAIN-masked坐标均值/标准差缩放，目标采用相同TRAIN逐基因阳性表达统计；相同初始化、批次、1000步预算和优化器。仅按完整VALID的原logCP10k表达MAE选择预定监控点，之后锁定再统一TEST。</p>',
        '<p>每数据集三个种子，Kang各六个供体划分、PBMC两个批次方向，共84个新预测头。先平均组内种子再等权平均组。以下MAE是被遮蔽基因表达预测误差，不是表征坐标重建误差。</p>',pd.DataFrame(summary).to_html(index=False,float_format=lambda x:f'{x:.6f}'),
        '<h2>逐种子</h2>',seed.to_html(index=False,float_format=lambda x:f'{x:.6f}'),'<h2>完整辅助指标</h2>',s.to_html(index=False,float_format=lambda x:f'{x:.6f}'),
        '<h2>范围限制</h2><p>C此前已用TRAIN表达训练；本实验检验表达监督后的冻结表征在相同新读出下的可用性，不是总训练预算匹配，也不能证明SAE必要性。所有数据为历史探索性；ctrl/stim共享供体，PBMC不是跨供体验证。只预测原选且阳性的遮蔽token，不含零表达或全转录组补全，仅一种固定遮蔽率/视图。原始h与C各自的预测头均可适配遮蔽输入。固定旧C checkpoint，不据本轮TEST选择SAE种子或重新训练SAE。</p>',
        '<p>确定性复算见verification.json；独立语义审查见EXPERIMENT_AUDIT.md和CLAIMS_FROM_RESULTS.md。原始封存实验未改动。</p>']
    (O/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title><style>body{font-family:system-ui;max-width:1200px;margin:40px auto;line-height:1.7;padding:0 24px}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:8px;border:1px solid #ddd}th{background:#eef3f8}</style>'+''.join(body)+'</html>')
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
