# -*- coding: utf-8 -*-
"""Summarize fixed checkpoint selection experiment without selecting on TEST."""
from pathlib import Path
import json
import pandas as pd
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_checkpoint_selection'

def main():
    d=pd.read_csv(O/'all_metrics.csv');kn=pd.read_csv(O/'knn_metrics.csv');keys=['MAE','normalized_MSE','gene_macro_MAE','gene_mean_baseline_MAE']
    g=d.groupby(['fold','group','method'])[keys].mean().reset_index();s=g.groupby('method')[keys].mean().reset_index();g.to_csv(O/'group_metrics.csv',index=False);s.to_csv(O/'summary.csv',index=False)
    seeds=d.groupby(['method','seed'])[keys].mean().reset_index();seeds.to_csv(O/'seed_metrics.csv',index=False)
    kg=kn.groupby(['fold','group','method','view']).knn_macro_f1.mean().reset_index();ks=kg.groupby(['method','view']).knn_macro_f1.mean().reset_index();kg.to_csv(O/'knn_group_metrics.csv',index=False);ks.to_csv(O/'knn_summary.csv',index=False)
    selected=[];curves=[]
    for p in sorted((W/'bodies').glob('*/*/selection.json')):
        fi=int(p.parents[1].name.split('_')[1]);seed=int(p.parent.name.split('_')[1]);a=json.loads(p.read_text());selected.append(dict(fold=fi,seed=seed,selected_step=a['selected_step'],valid_objective=a['selected_valid_objective'],valid_MAE=a['selected_valid_MAE']))
        curves.extend(dict(fold=fi,seed=seed,**v) for v in a['history'])
    pd.DataFrame(selected).to_csv(O/'selected_checkpoints.csv',index=False);pd.DataFrame(curves).to_csv(O/'VALID_curves.csv',index=False)
    contrasts=[];pair=[]
    for ref in ['original_h','C_1000']:
        p=d[d.method=='C_selected'].merge(d[d.method==ref],on=['dataset','fold','group','seed'],suffixes=('_selected','_ref'),validate='one_to_one');assert len(p)==18
        p['reference']=ref;p['delta_MAE']=p.MAE_selected-p.MAE_ref;pair.append(p)
        group=p.groupby(['fold','group'])[['MAE_selected','MAE_ref','delta_MAE']].mean().reset_index();a=group.MAE_selected.mean();b=group.MAE_ref.mean()
        contrasts.append(dict(reference=ref,selected_MAE=a,reference_MAE=b,delta_MAE=a-b,MAE_reduction_pct=(1-a/b)*100,improved_pairs=int((p.delta_MAE<0).sum()),pairs=18,improved_donors=int((group.delta_MAE<0).sum()),donors=6))
    pd.concat(pair).to_csv(O/'paired_metrics.csv',index=False);pd.DataFrame(contrasts).to_csv(O/'contrasts.csv',index=False)
    result=dict(expression=s.to_dict('records'),knn=ks.to_dict('records'),contrasts=contrasts,selected_checkpoints=selected);(O/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    claims=[dict(id=f"{v['method']}_MAE",value=v['MAE'],source='outputs/sae_checkpoint_selection/SUMMARY.json') for v in result['expression']]
    (O/'EVIDENCE_CLAIMS.json').write_text(json.dumps(claims,indent=2))
    body=['<h1>Kang ctrl：SAE 本体 checkpoint 选择诊断</h1>','<p>保持K=32、模型、初始化、TRAIN批次与表达损失不变，重现18条1000步训练轨迹，保存0/25/50/100/200/500/1000步。仅按完整VALID的标准化表达MSE选择本体，随后冻结本体并丢弃原联合预测头。另用相同独立初始化、批次、1000步预算与VALID-MAE选择规则训练新线性表达预测头。旧原始表征与C1000的同协议预测头/结果直接复用；每条新轨迹的1000步权重必须逐张量完全等于旧模型。</p>',
        '<p>checkpoint 0继承此前已经过VALID选择的warmup SAE，不是随机或未训练的模型。主指标为同一批遮蔽阳性token的原logCP10k表达MAE，越低越好；先平均组内三个种子再等权平均六供体。没有在TEST上扫描七个checkpoint、挑种子、回退或混合原始表征。</p>',s.to_html(index=False,float_format=lambda v:f'{v:.6f}'),'<h2>配对比较</h2>',pd.DataFrame(contrasts).to_html(index=False,float_format=lambda v:f'{v:.6f}'),
        '<h2>固定15NN读出</h2>',ks.to_html(index=False,float_format=lambda v:f'{v:.6f}'),'<h2>本体选择结果</h2>',pd.DataFrame(selected).to_html(index=False,float_format=lambda v:f'{v:.6f}'),'<h2>逐种子表达指标</h2>',seeds.to_html(index=False,float_format=lambda v:f'{v:.6f}'),
        '<h2>范围与解释</h2><p>本轮只改变本体checkpoint选择，不能识别稀疏性本身的因果贡献，也不代表最优SAE/线性读出。SAE本体与新头都使用VALID选型；这是对历史反复探索Kang ctrl的诊断，不是独立盲测。只有一种固定25%原选已表达token遮蔽，不含零表达/全转录组补全。C本体有上游表达监督，原始h没有对应适配训练。VALID目标是标准化MSE，而主要TEST指标是原尺度MAE；本轮保持这个差异以单独诊断checkpoint选择。未自动扩展stim、PBMC或做K/损失搜索。</p>',
        '<p>指标复算见verification.json；独立完整性和结论边界见EXPERIMENT_AUDIT.md、CLAIMS_FROM_RESULTS.md。</p>']
    (O/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAE checkpoint选择诊断</title><style>body{font-family:system-ui;max-width:1200px;margin:40px auto;padding:0 24px;line-height:1.7}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ddd;padding:8px}th{background:#eef3f8}</style>'+''.join(body)+'</html>')
    print(s.to_string(index=False));print(ks.to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))

if __name__=='__main__':main()
