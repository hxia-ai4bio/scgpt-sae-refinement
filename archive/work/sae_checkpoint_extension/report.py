# -*- coding: utf-8 -*-
"""Summarize new stim/PBMC runs, with explicitly reused prior ctrl context."""
from pathlib import Path
import json
import pandas as pd
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_checkpoint_extension';DATASETS=['stim','pbmc']

def main():
    new=pd.concat([pd.read_csv(O/ds/'all_metrics.csv') for ds in DATASETS],ignore_index=True)
    newkn=pd.concat([pd.read_csv(O/ds/'knn_metrics.csv') for ds in DATASETS],ignore_index=True)
    assert len(new)==72 and len(newkn)==144
    new.to_csv(O/'all_metrics.csv',index=False);newkn.to_csv(O/'knn_metrics.csv',index=False)
    ctrl=pd.read_csv(ROOT/'outputs/sae_checkpoint_selection/all_metrics.csv');ctrlkn=pd.read_csv(ROOT/'outputs/sae_checkpoint_selection/knn_metrics.csv')
    ctrl['source_phase']='prior_ctrl_context';new['source_phase']='current_replication'
    ctrlkn['source_phase']='prior_ctrl_context';newkn['source_phase']='current_replication'
    allrows=pd.concat([ctrl,new],ignore_index=True);allkn=pd.concat([ctrlkn,newkn],ignore_index=True)
    allrows.to_csv(O/'combined_with_ctrl_metrics.csv',index=False)
    metrics=['MAE','normalized_MSE','gene_macro_MAE','gene_mean_baseline_MAE']
    g=allrows.groupby(['dataset','fold','group','method','source_phase'])[metrics].mean().reset_index()
    summary=g.groupby(['dataset','method','source_phase'])[metrics].mean().reset_index()
    g.to_csv(O/'group_metrics.csv',index=False);summary.to_csv(O/'summary.csv',index=False)
    seeds=allrows.groupby(['dataset','method','seed','source_phase'])[metrics].mean().reset_index();seeds.to_csv(O/'seed_metrics.csv',index=False)
    kg=allkn.groupby(['dataset','fold','group','method','view','source_phase']).knn_macro_f1.mean().reset_index()
    ks=kg.groupby(['dataset','method','view','source_phase']).knn_macro_f1.mean().reset_index();kg.to_csv(O/'knn_group_metrics.csv',index=False);ks.to_csv(O/'knn_summary.csv',index=False)
    contrasts=[];pairs=[]
    for ds,df in allrows.groupby('dataset'):
        for reference in ['original_h','C_1000']:
            a=df[df.method=='C_selected'];b=df[df.method==reference]
            p=a.merge(b,on=['dataset','fold','group','seed'],suffixes=('_selected','_ref'),validate='one_to_one');assert len(p)==(6 if ds=='pbmc' else 18)
            p['reference']=reference;p['delta_MAE']=p.MAE_selected-p.MAE_ref;pairs.append(p)
            group=p.groupby(['fold','group'])[['MAE_selected','MAE_ref','delta_MAE']].mean().reset_index()
            selected=group.MAE_selected.mean();ref=group.MAE_ref.mean()
            contrasts.append(dict(dataset=ds,reference=reference,selected_MAE=selected,reference_MAE=ref,delta_MAE=selected-ref,MAE_reduction_pct=(1-selected/ref)*100,
                improved_pairs=int((p.delta_MAE<0).sum()),pairs=len(p),improved_groups=int((group.delta_MAE<0).sum()),groups=len(group),source_phase='prior_ctrl_context' if ds=='ctrl' else 'current_replication'))
    pd.concat(pairs).to_csv(O/'paired_metrics.csv',index=False);pd.DataFrame(contrasts).to_csv(O/'contrasts.csv',index=False)
    selections=[];curves=[]
    for ds in DATASETS:
        for path in sorted((W/ds/'bodies').glob('*/*/selection.json')):
            fi=int(path.parents[1].name.split('_')[1]);seed=int(path.parent.name.split('_')[1]);d=json.loads(path.read_text())
            selections.append(dict(dataset=ds,fold=fi,seed=seed,selected_step=d['selected_step'],valid_objective=d['selected_valid_objective'],valid_MAE=d['selected_valid_MAE']))
            curves.extend(dict(dataset=ds,fold=fi,seed=seed,**h) for h in d['history'])
    assert len(selections)==24
    pd.DataFrame(selections).to_csv(O/'selected_checkpoints.csv',index=False);pd.DataFrame(curves).to_csv(O/'VALID_curves.csv',index=False)
    result=dict(new_datasets=DATASETS,new_body_trajectories=24,new_heads=24,new_expression_results=72,new_knn_results=144,
        ctrl_context='Previously sealed ctrl results, not new or independent replication.',expression=summary.to_dict('records'),knn=ks.to_dict('records'),contrasts=contrasts,selected_checkpoints=selections)
    (O/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    claims=[dict(id=f"{row['dataset']}_{row['method']}_MAE",value=row['MAE'],source='outputs/sae_checkpoint_extension/SUMMARY.json') for row in result['expression']]
    (O/'EVIDENCE_CLAIMS.json').write_text(json.dumps(claims,indent=2))
    main_table=summary.pivot(index='dataset',columns='method',values='MAE').reset_index()
    title='SAE 早期 checkpoint 规则：stim / PBMC 复现'
    body=['<h1>'+title+'</h1>',
        '<p>本轮只扩展到stim与PBMC，复用ctrl预定规则：K32、相同初始化与表达损失、1000步原训练轨迹，保存0/25/50/100/200/500/1000步；完整VALID标准化表达MSE选本体，最早最小值优先。全部24条新轨迹的1000步必须逐张量等于同数据集旧模型。丢弃联合表达头，冻结所选表征，按相同独立初始化seed+400003、TRAIN-only缩放、批次、1000步预算和VALID原尺度MAE规则重训线性头。原始h与C1000复用已封存的同协议预测头与结果。</p>',
        '<p>stim：6个供体划分×3种子；PBMC：2个批次方向×3种子，共24条新本体轨迹、24个新预测头，72行表达评价与144行kNN评价（含复用对照）。两组全部完成选型并写入GLOBAL_SELECTION_LOCK后才开始任何新TEST，未根据测试结果改变规则。</p>',
        '<p>主MAE是在同一批被遮蔽阳性基因token上的原logCP10k误差，越低越好；先平均每划分的三个种子，再等权平均供体或批次方向。ctrl行明确复用上一轮作为上下文，不计入本轮新增复现。</p>',main_table.to_html(index=False,float_format=lambda v:f'{v:.6f}'),
        '<h2>配对方向与效应</h2>',pd.DataFrame(contrasts).to_html(index=False,float_format=lambda v:f'{v:.6f}'),
        '<h2>完整表达指标</h2>',summary.to_html(index=False,float_format=lambda v:f'{v:.6f}'),
        '<h2>固定15NN读出</h2>',ks.to_html(index=False,float_format=lambda v:f'{v:.6f}'),
        '<h2>新实验选点</h2>',pd.DataFrame(selections).to_html(index=False,float_format=lambda v:f'{v:.6f}'),
        '<h2>逐种子</h2>',seeds.to_html(index=False,float_format=lambda v:f'{v:.6f}'),
        '<h2>解释边界</h2><p>stim与ctrl共享Kang供体，不能视为独立队列；PBMC仅有两个批次方向，没有可确认的供体泛化证据。数据均已历史探索，VALID在本体与读出阶段复用，结果不是独立盲测。只用一种固定25%原选正表达token遮蔽，不包含零表达或全转录组补全。checkpoint0是继承的已训练warmup SAE。本轮没有改K、结构、损失、学习率或挑选测试最好种子；也未重复Ridge分类评估。改善某个读出不等于稀疏性必要或所有生物信息更好。</p>',
        '<p>确定性验证见verification.json及各数据集verification.json；独立审计与限定结论见EXPERIMENT_AUDIT.md、CLAIMS_FROM_RESULTS.md。</p>']
    (O/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title><style>body{font-family:system-ui;max-width:1350px;margin:40px auto;padding:0 24px;line-height:1.7}table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:8px;border:1px solid #ddd}th{background:#eef3f8}</style>'+''.join(body)+'</html>')
    print(main_table.to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False));print(ks.to_string(index=False))

if __name__=='__main__':main()
