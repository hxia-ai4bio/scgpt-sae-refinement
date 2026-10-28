"""Descriptive donor-level summaries; no tuning or fitting in this file."""
import html, platform, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy, sklearn, torch
from core import *

LABELS={'original_h':'Original h','sparse_z':'SAE z only','reconstruction':'Reconstruction',
        'reconstruction_error':'Reconstruction error r','fixed_Dz':'Fixed h + alpha Dz',
        'residual_mix':'Reconstruction + lambda r','concat_SAE':'[h, SAE z]',
        'concat_random':'[h, random TopK]','concat_linear':'[h, linear pre-TopK]',
        'crossfit_residual_SAE':'Crossfit correction: SAE','crossfit_residual_random':'Crossfit correction: random',
        'crossfit_residual_linear':'Crossfit correction: linear'}
ORDER=list(LABELS)

def interval(x):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(20261019)
    samples=x[rng.integers(0,len(x),size=(20000,len(x)))].mean(1)
    return np.quantile(samples,[.025,.975]).tolist()

def main():
    assert read(O/'evaluation_completion.json')['status']=='complete'
    records=[];perclass=[]
    for p in sorted((O/'evaluations').glob('fold_*/seed_*/*.json')):
        if p.name=='completion.json':continue
        r=read(p);rr={k:v for k,v in r.items() if not isinstance(v,list)};records.append(rr)
        for j,c in enumerate(CFG['classes']):
            perclass.append({**{k:r[k] for k in ['fold','seed','test_donor','role','method']},'class':c,'f1':r['per_class_f1'][j],'support':r['support'][j]})
    df=pd.DataFrame(records);assert len(df)==18*12*2,len(df)
    base=df[df.method=='original_h'][['fold','seed','role','macro_f1']].rename(columns={'macro_f1':'baseline_F1'})
    df=df.merge(base,on=['fold','seed','role'],validate='many_to_one');df['delta_pp']=100*(df.macro_f1-df.baseline_F1)
    df.to_csv(O/'all_metrics.csv',index=False)
    donor=df[df.role=='test'].groupby(['test_donor','method'],as_index=False).agg(macro_f1=('macro_f1','mean'),delta_pp=('delta_pp','mean'),score_MSE=('score_target_MSE','mean'),corrected_errors=('corrected_errors','mean'),harmed_correct=('harmed_correct','mean'))
    donor.to_csv(O/'paired_donor_metrics.csv',index=False)
    seeds=df.groupby(['role','method','seed'],as_index=False)[['macro_f1','delta_pp']].mean();seeds.to_csv(O/'seed_means.csv',index=False)
    summaries=[]
    for method in ORDER:
        t=donor[donor.method==method];pb=df[(df.role=='pbmc')&(df.method==method)];te=df[(df.role=='test')&(df.method==method)]
        lo,hi=interval(t.delta_pp)
        summaries.append({'method':method,'TEST_macro_F1_percent':100*t.macro_f1.mean(),'TEST_delta_pp':t.delta_pp.mean(),
                          'TEST_delta_donor_bootstrap_low_pp':lo,'TEST_delta_donor_bootstrap_high_pp':hi,
                          'TEST_positive_donors':int((t.delta_pp>1e-10).sum()),'TEST_negative_donors':int((t.delta_pp< -1e-10).sum()),
                          'TEST_positive_seed_means':int((seeds[(seeds.role=='test')&(seeds.method==method)].delta_pp>1e-10).sum()),
                          'identity_selected_runs':int(te.identity_selected.sum()),'nonidentity_selected_runs':int((~te.identity_selected).sum()),
                          'PBMC_macro_F1_percent':100*pb.macro_f1.mean(),'PBMC_delta_pp':pb.delta_pp.mean(),
                          'TEST_residual_prediction_Q2':float(te.residual_prediction_Q2.mean()) if method.startswith('crossfit_') else None,
                          'PBMC_residual_prediction_Q2':float(pb.residual_prediction_Q2.mean()) if method.startswith('crossfit_') else None})
    summary=pd.DataFrame(summaries);summary.to_csv(O/'summary.csv',index=False)
    contrasts=[]
    for a,b in [('concat_SAE','original_h'),('concat_SAE','concat_random'),('concat_SAE','concat_linear'),('crossfit_residual_SAE','original_h'),('crossfit_residual_SAE','crossfit_residual_random'),('crossfit_residual_SAE','crossfit_residual_linear')]:
        aa=donor[donor.method==a].set_index('test_donor');bb=donor[donor.method==b].set_index('test_donor');delta=100*(aa.macro_f1-bb.macro_f1);lo,hi=interval(delta)
        contrasts.append({'A':a,'B':b,'TEST_mean_delta_pp':float(delta.mean()),'TEST_bootstrap_low_pp':lo,'TEST_bootstrap_high_pp':hi,'positive_donors':int((delta>1e-10).sum()),'negative_donors':int((delta< -1e-10).sum()),'PBMC_delta_pp':float(summary.set_index('method').loc[a,'PBMC_macro_F1_percent']-summary.set_index('method').loc[b,'PBMC_macro_F1_percent'])})
    pd.DataFrame(contrasts).to_csv(O/'planned_contrasts.csv',index=False)
    pc=pd.DataFrame(perclass);pcb=pc[pc.method=='original_h'][['fold','seed','role','class','f1']].rename(columns={'f1':'baseline_f1'});pc=pc.merge(pcb,on=['fold','seed','role','class'],validate='many_to_one');pc['delta_pp']=100*(pc.f1-pc.baseline_f1);pc.to_csv(O/'per_class_metrics.csv',index=False)
    pc.groupby(['role','method','class'],as_index=False)[['f1','delta_pp']].mean().to_csv(O/'per_class_summary.csv',index=False)
    selections=[]
    for fi in CFG['folds']:
        for seed in SEEDS:
            for p in (W/'models'/f'fold_{fi}'/f'seed_{seed}').glob('*/selection.json'):
                selections.append({'fold':fi,'seed':seed,**read(p)})
    pd.DataFrame(selections).to_csv(O/'selected_models.csv',index=False)
    facts={'summary':summaries,'planned_contrasts':contrasts,'scope':CFG['scope'],'label_crossfit_scope':CFG['crossfit'],
           'fixed_families_not_independent':True,'new_test_guided_tuning':False,'n_external_dataset_files':1,
           'external_independent_study_and_pretraining_overlap':'not confirmed; donor/study/pretraining overlap unknown in source data audit',
           'review_status':read(O/'EXPERIMENT_AUDIT.json')['verdict'] if (O/'EXPERIMENT_AUDIT.json').exists() else 'PENDING'}
    save(O/'facts.json',facts)
    env={'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,'platform':platform.platform(),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'};save(O/'ENVIRONMENT.json',env)
    save(O/'EXPERIMENT_TRACKER.json',{'status':'DONE' if (O/'EXPERIMENT_AUDIT.json').exists() else 'EXPERIMENTS_DONE_REVIEW_PENDING','development':'DONE','evaluation':'DONE','models':18,'methods':12,'evaluation_metric_files':432,'review':facts['review_status']})
    # Scientific figures are generated as standalone PNG/PDF artifacts.
    fig,ax=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1.1,1]});chosen=['fixed_Dz','residual_mix','concat_SAE','concat_random','concat_linear','crossfit_residual_SAE','crossfit_residual_random','crossfit_residual_linear'];s=summary.set_index('method').loc[chosen];pos=np.arange(len(chosen))
    ax[0].axvline(0,color='.6',lw=1);ax[0].barh(pos,s.TEST_delta_pp,color=['#5599aa']*2+['#4477aa','#aaaaaa','#ddaa33']*2)
    ax[0].errorbar(s.TEST_delta_pp,pos,xerr=np.vstack([s.TEST_delta_pp-s.TEST_delta_donor_bootstrap_low_pp,s.TEST_delta_donor_bootstrap_high_pp-s.TEST_delta_pp]),fmt='none',ecolor='#333333',capsize=3)
    ax[0].set_yticks(pos,[LABELS[x] for x in chosen]);ax[0].invert_yaxis();ax[0].set_xlabel('TEST Macro-F1 change (percentage points)');ax[0].set_title('Six donor means; descriptive bootstrap intervals')
    heat=donor[donor.method.isin(chosen)].pivot(index='method',columns='test_donor',values='delta_pp').loc[chosen];lim=max(.1,float(np.abs(heat.values).max()));im=ax[1].imshow(heat.values,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax[1].set_yticks(pos,[LABELS[x] for x in chosen]);ax[1].set_xticks(range(6),[x.replace('patient_','') for x in heat.columns],rotation=45)
    for i in range(8):
        for j in range(6):ax[1].text(j,i,f'{heat.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=8)
    ax[1].set_title('Paired TEST donor effects (pp)');fig.colorbar(im,ax=ax[1],shrink=.65);fig.tight_layout();fig.savefig(O/'overview.png',dpi=180,bbox_inches='tight');fig.savefig(O/'overview.pdf',bbox_inches='tight');plt.close(fig)
    display=summary[['method','TEST_macro_F1_percent','TEST_delta_pp','TEST_positive_donors','nonidentity_selected_runs','PBMC_macro_F1_percent','PBMC_delta_pp']].copy();display['method']=display.method.map(LABELS)
    text='SAE 固定分解与条件增量读出实验\n\n'+display.to_string(index=False,float_format=lambda x:f'{x:.4f}')+'\n\n预设对照：\n'+pd.DataFrame(contrasts).to_string(index=False,float_format=lambda x:f'{x:.4f}')
    caveats='''\n\n方法与结论边界：
1. 沿用历史六轮四TRAIN/一VALID/一TEST供体划分，三个SAE种子。上游SAE未重训：只用四TRAIN供体拟合，VALID重构MSE选checkpoint。全部分类超参数用VALID选定并统一锁定后才新算TEST和PBMC。
2. Kang/PBMC在此前已探索，结果是新方法在历史数据上的探索性外推评价，不是新盲测。PBMC仅一个外部数据文件，18个模型不是18份独立外部重复；源数据审计未确认其独立研究来源、供体及预训练重叠情况。
3. 原始h、重构、重构误差、稀疏z均只用TRAIN拟合通道均值/标准差和带截距的类别平衡Ridge；8类固定Macro-F1。缺失类别在主指标中记0，supported-class指标另存。单个供体极少类别支持会使F1离散。
4. 两组固定混合网格多数互为仿射缩放，在当前StandardScaler+截距下等价，不能作为独立证据。alpha=-1给出误差r（差常量），lambda=0给出重构。
5. 拼接先分别TRAIN标准化h、z，再缩放z分支，不做第二次标准化。每种分支有相同3 eta ×11 alpha预算及identity候选。random只匹配名义维度/TopK数/搜索预算，未匹配激活频率或相关性。
6. linear对照直接使用相同SAE encoder的TopK之前仿射激活，数学上仍是h的线性读出；其收益体现有效正则化/优化的变化，不能叫新增非线性结构。
7. 分类残差用TRAIN四供体LODO预测生成，每个被留出的TRAIN供体的基模型alpha在其余三供体内部再做LODO选取，因此目标细胞标签不进入其基预测拟合/选型。上游SAE是固定变换，未在内层重新拟合；仅声称标签/读出交叉拟合。
8. 残差回归目标为±1 one-hot减去Ridge分类分数；分数不是概率，MSE/Q2是平方损失代理。最终分类修正强度和正则化仍只用VALID选择，TEST残差Q2对最终h基模型误差计算。
9. TEST先按每个供体平均三个种子，再对六供体配对差异做描述性bootstrap区间。训练集彼此重叠，六供体较少；区间不能替代新供体/新研究验证。
10. z是h的确定性函数。相对h Ridge增益最多说明当前读出/正则化下存在可利用结构；只有胜过random与linear控制才更支持TopK学到的结构特异性，不证明信息论新增信息或生物因果机制。
'''
    reg=''
    if (O/'regularization_diagnostics.json').exists():
        rd=read(O/'regularization_diagnostics.json')
        reg=f"\n\n读出正则化补充：旧SAE稀疏分类读出alpha只搜索{rd['old_grid']}；本轮对所有表示统一扩展到10^-4至10^6。18组中{rd['selected_above_old_max_runs']}组稀疏z选择了高于旧上限100的alpha。重叠alpha上的VALID F1逐项一致（最大误差{rd['same_alpha_overlap_max_VALID_F1_error']:.2g}）。旧网格的历史TEST/PBMC均值为{rd['historical_old_grid_TEST_percent']:.4f}%/{rd['historical_old_grid_PBMC_percent']:.4f}%；本轮新网格的锁定结果见主表。该补充只读取已存网格及历史结果，没有追加TEST拟合或选参。较强正则化明显改变了稀疏z的可读出成绩，因此不能笼统说SAE稀疏表示在所有设置下均较弱；但本轮PBMC仍低于原h。\n"
    (O/'RESULTS_ZH.txt').write_text(text+caveats+reg)
    methods='''# Reproducibility

Run from the original task root with the interpreter in ENVIRONMENT.json:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=4
    export MPLCONFIGDIR="$PWD/work/sae_complementarity/mplcache"
    python work/sae_complementarity/run.py preflight
    python work/sae_complementarity/run.py develop
    python work/sae_complementarity/run.py lock
    python work/sae_complementarity/run.py evaluate
    python work/sae_complementarity/regularization_diagnostic.py
    python work/sae_complementarity/report.py
    python work/sae_complementarity/verify.py

Input paths/checksums are in frozen_sources.json. The original source datasets and 18 pretrained SAE checkpoint/cache sets are external prerequisites, not bundled. The full frozen downstream models, validation grids, crossfit targets, selections, locked evaluation predictions, scripts and reports are bundled. Re-running at a different root requires adjusting input roots in a NEW experiment directory and regenerating the preflight hash binding; never overwrite sealed results. No neural model is trained here.

Time: O(n*p^2+p^3) per dense Ridge decomposition, O(p^2*C) per subsequent alpha, p<=2560,C=8; memory O(n*p+p^2). Sequential CPU, float64 SciPy BLAS and LAPACK, avoids the previously observed NumPy @ warnings. Inner label-crossfit baseline work is shared across SAE seeds.
'''
    (O/'REPRODUCIBILITY.md').write_text(methods)
    review='复核尚未完成' if facts['review_status']=='PENDING' else '完整性审查：'+facts['review_status']+'；审查结论见附录。'
    body='<h1>SAE 是否提供原始线性读出之外的判别结构？</h1><p class="lead">'+html.escape(review)+'</p><p>六个历史供体轮换 × 三个 SAE 种子；预设网格统一锁定后评价。结果为探索性证据。</p>'
    body+='<h2>分类与外部评价</h2>'+display.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)+'<p>变化单位为百分点；保留次数分母为18，正向供体分母为6。</p><img src="overview.png" alt="donor effects">'
    body+='<h2>预设增量对照</h2>'+pd.DataFrame(contrasts).to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    body+='<h2>数学关系与评价边界</h2><pre>'+html.escape(caveats)+'</pre>'
    if reg:body+='<h2>读出正则化敏感性</h2><pre>'+html.escape(reg)+'</pre><p><a href="regularization_diagnostics.json">逐组重叠网格核对</a></p>'
    body+='<h2>可复查文件</h2><p>'+ ' · '.join(f'<a href="{x}">{x}</a>' for x in ['summary.csv','planned_contrasts.csv','paired_donor_metrics.csv','per_class_summary.csv','selected_models.csv','PROTOCOL.json','DATA_INVENTORY.json','MATH_NOTES_ZH.txt','verification.json','REPRODUCIBILITY.md','EXPERIMENT_AUDIT.md','CLAIMS_FROM_RESULTS.md'])+'</p>'
    for name in ['EXPERIMENT_AUDIT.md','CLAIMS_FROM_RESULTS.md']:
        if (O/name).exists():body+='<h2>'+name+'</h2><pre>'+html.escape((O/name).read_text())+'</pre>'
    css='body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:40px auto;padding:0 28px;color:#203040;background:#fafbfc}h1{font-size:30px}table{border-collapse:collapse;font-size:13px;background:white}th,td{padding:8px;border-bottom:1px solid #dde3ea;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;background:white}pre{white-space:pre-wrap;word-break:break-word;background:#eef2f6;padding:20px;border-radius:10px}.lead{color:#406080}'
    (O/'REPORT.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAE complementarity</title><style>'+css+'</style>'+body+'</html>')
    print(summary.to_string(index=False),flush=True);print('REPORT_READY',flush=True)
if __name__=='__main__':main()
