"""Complete fixed-grid report; no ranking-based selection or extra experiment dispatch."""
import html
import importlib.util
from pathlib import Path
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter

W=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lambda_experiment',W/'run.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'svg.fonttype':'none'})

def axis(ax):
    ax.set_xscale('symlog',linthresh=.01,linscale=1)
    ax.xaxis.set_major_locator(FixedLocator(e.LAMBDAS));ax.xaxis.set_major_formatter(FixedFormatter(['0','.01','.03','.1','.3','1','3']))
    ax.set_xlabel('Relation weight lambda (tau = 0.1)');ax.grid(alpha=.18)

def savefig(fig,name):
    fig.tight_layout();fig.savefig(e.O/f'{name}.png',bbox_inches='tight');fig.savefig(e.O/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def main():
    verification=e.read(e.O/'verification.json');assert verification['status']=='PASS'
    summary=e.read(e.O/'summary.json');s=pd.DataFrame(summary);grid=s[s.method!='original_h'].sort_values('lambda_KL')
    donor=pd.read_csv(e.O/'paired_donor_metrics.csv');seed=pd.read_csv(e.O/'seed_metrics.csv')
    contrasts=e.read(e.O/'planned_contrasts.json');curve=pd.read_csv(e.O/'learning_curves.csv');grad=pd.read_csv(e.O/'gradient_diagnostics.csv')
    loo=pd.read_csv(e.O/'leave_one_donor_out_sensitivity.csv')
    baseline=float(s[s.method=='original_h'].macro_f1_percent.iloc[0])
    x=grid.lambda_KL.to_numpy()
    fig,axs=plt.subplots(1,3,figsize=(16,4.5))
    for j,sd in enumerate(e.SEEDS):
        z=seed[(seed.control=='original_h')&(seed.seed==sd)].sort_values('lambda_KL')
        axs[0].plot(z.lambda_KL,z.macro_f1_percent,color='#888888',alpha=.55,lw=1,label='Individual seed means' if j==0 else None)
    axs[0].plot(x,grid.macro_f1_percent,'o-',color=COLORS[0],label='Donor-equal mean, 3 seeds')
    axs[0].axhline(baseline,color='black',ls='--',lw=1,label='Original h')
    axs[0].set_ylabel('Macro-F1 (%)');axs[0].set_title('A  Full fixed grid');axs[0].legend(fontsize=8)
    for control,color,name in [('original_h',COLORS[0],'vs original h'),('continue_mse',COLORS[1],'vs matched MSE')]:
        cs=sorted([r for r in contrasts if r['control']==control and r['method']!='original_h'],key=lambda r:r['lambda_KL'])
        yy=np.array([r['delta_pp'] for r in cs]);ci=np.array([r['descriptive_ci95_pp'] for r in cs])
        axs[1].plot(x,yy,'o-',color=color,label=name)
        axs[1].fill_between(x,ci[:,0],ci[:,1],color=color,alpha=.1)
    axs[1].axhline(0,color='black',ls='--',lw=1);axs[1].set_ylabel('Change in Macro-F1 (pp)');axs[1].set_title('B  Paired mean differences');axs[1].legend(fontsize=8)
    for j,(name,g) in enumerate(donor[donor.method!='original_h'].groupby('test_donor')):
        g=g.sort_values('lambda_KL');axs[2].plot(g.lambda_KL,g.delta_vs_h_pp,'o-',lw=1,color=COLORS[j],label=str(name))
    axs[2].axhline(0,color='black',ls='--',lw=1);axs[2].set_ylabel('Change vs original h (pp)');axs[2].set_title('C  Every held-out donor');axs[2].legend(fontsize=8,ncol=2)
    for a in axs:axis(a)
    savefig(fig,'performance_curves')
    final=curve[curve.step==1000].groupby('lambda_KL',as_index=False).mean(numeric_only=True).sort_values('lambda_KL')
    final.to_csv(e.O/'final_geometry_summary.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(15,4.2))
    for role,col in [('train',COLORS[0]),('valid',COLORS[1])]:
        axs[0].plot(final.lambda_KL,final[role+'_normalized_mse'],'o-',color=col,label=role.upper())
    axs[0].set_ylabel('Normalized reconstruction MSE');axs[0].set_title('A  Reconstruction');axs[0].legend()
    axs[1].plot(final.lambda_KL,final.train_monitor_kl,'o-',color=COLORS[0],label='TRAIN cross-donor')
    axs[1].plot(final.lambda_KL,final.valid_to_train_kl,'o-',color=COLORS[1],label='VALID to TRAIN anchors')
    axs[1].set_ylabel('Relation KL');axs[1].set_title('B  Geometric proxy');axs[1].legend(fontsize=8)
    for step,col in [(1,COLORS[0]),(1000,COLORS[1])]:
        g=grad[grad.step==step].groupby('lambda_KL').weighted_gradient_ratio.mean().sort_index()
        axs[2].plot(g.index,g.to_numpy(),'o-',color=col,label=f'Step {step}')
    axs[2].set_yscale('symlog',linthresh=.1);axs[2].set_ylabel('||lambda * grad KL|| / ||grad MSE||');axs[2].set_title('C  Actual gradient strength');axs[2].legend(fontsize=8)
    for a in axs:axis(a)
    savefig(fig,'geometry_tradeoff')
    fig,axs=plt.subplots(1,3,figsize=(15,4.2))
    for control,col,name in [('original_h',COLORS[0],'vs original h'),('continue_mse',COLORS[1],'vs matched MSE')]:
        cs=sorted([r for r in contrasts if r['control']==control and r['method']!='original_h'],key=lambda r:r['lambda_KL'])
        axs[0].plot(x,[r['positive_donors'] for r in cs],'o-',color=col,label=name)
        axs[1].plot(x,[r['positive_seed_means'] for r in cs],'o-',color=col,label=name)
        z=loo[(loo.control==control)&(loo.method!='original_h')].groupby('lambda_KL').delta_pp.agg(['min','max']).sort_index()
        axs[2].fill_between(z.index.to_numpy(),z['min'].to_numpy(),z['max'].to_numpy(),alpha=.15,color=col,label=name)
        axs[2].plot(z.index.to_numpy(),z['min'].to_numpy(),color=col,lw=1);axs[2].plot(z.index.to_numpy(),z['max'].to_numpy(),color=col,lw=1)
    axs[0].set_ylim(-.2,6.2);axs[0].set_yticks(range(7));axs[0].set_ylabel('Positive donors / 6');axs[0].set_title('A  Donor consistency')
    axs[1].set_ylim(-.1,3.1);axs[1].set_yticks(range(4));axs[1].set_ylabel('Positive seed means / 3');axs[1].set_title('B  Optimization variability')
    axs[2].axhline(0,color='black',ls='--',lw=1);axs[2].set_ylabel('Mean delta after one donor excluded (pp)');axs[2].set_title('C  All six exclusion sensitivities')
    for a in axs:axis(a);a.legend(fontsize=8)
    savefig(fig,'stability_curves')
    display=grid[['lambda_KL','macro_f1_percent','delta_pp','delta_vs_MSE_pp','positive_donors','positive_donors_vs_MSE','positive_seed_means']].copy()
    display.columns=['λ','Macro-F1 (%)','相对原始h (pp)','相对MSE (pp)','改善供体 vs h','改善供体 vs MSE','改善seed均值 vs h']
    donors=donor[donor.method!='original_h'].pivot(index='test_donor',columns='lambda_KL',values='delta_vs_h_pp')
    loorange=loo[(loo.control=='original_h')&(loo.method!='original_h')].groupby('lambda_KL').delta_pp.agg(['min','max'])
    diagnostics=[]
    for r in grid.to_dict('records'):
        lam=r['lambda_KL'];v=loorange.loc[lam]
        diagnostics.append(dict(lambda_KL=lam,delta_vs_h_pp=r['delta_pp'],delta_vs_MSE_pp=r['delta_vs_MSE_pp'],leave_one_donor_out_min_pp=float(v['min']),leave_one_donor_out_max_pp=float(v['max'])))
    e.save(e.O/'sensitivity_diagnostics.json',dict(scope='Descriptive diagnostics, not a winner selection or advance gate',grid=diagnostics))
    audit=e.O/'EXPERIMENT_AUDIT.json';claims=e.O/'CLAIMS_FROM_RESULTS.json';reviewed=audit.exists() and claims.exists()
    review='独立审查进行中；确定性核验已通过。';claim=''
    if reviewed:
        a=e.read(audit);c=e.read(claims)
        review='独立上下文、同系列模型审查（provisional）：'+html.escape(str(a.get('overall_verdict',a.get('verdict','unknown'))))
        claim=html.escape(str(c.get('suggested_claim_revision','')))
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAE关系约束 λ 敏感性分析</title>
<style>body{{font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f5f7fb;color:#24364a;margin:0}}main{{max-width:1450px;margin:auto;padding:30px}}section{{background:white;border:1px solid #dfe6ef;border-radius:12px;padding:22px;margin:20px 0;overflow:auto}}h1{{font-size:27px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #e1e7f0;text-align:left;white-space:nowrap}}th{{background:#f0f4fa}}img{{width:100%;height:auto}}a{{color:#245fc4}}.note{{color:#5c6a7b}}</style>
<main><h1>SAE 重建：关系约束权重 λ 的第一轮敏感性分析</h1><p class="note">{datetime.now().isoformat(timespec='seconds')} · Kang ctrl · 9,577细胞 / 6供体 / 8类 · 固定512维 · 历史探索性数据</p>
<section><h2>研究问题与范围</h2><p>关系约束的积极信号是否出现在相邻参数范围，并跨多个供体保持一致？这轮完整报告预先固定的七个 λ，不按TEST成绩选冠军，不修改温度、稀疏度或训练预算。</p><p>{review} · <a href="EXPERIMENT_AUDIT.md">实验审查</a> · <a href="CLAIMS_FROM_RESULTS.md">结论边界</a></p><p>{claim}</p></section>
<section><h2>全部七点结果</h2><p>原始 embedding Macro-F1 = {baseline:.4f}%。先在每个供体内部平均三个SAE种子，再对六供体等权平均。pp表示百分点。</p>{display.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)}<p>λ=0、0.03、0.3严格复用上一轮的54份训练与读出；0.01、0.1、1、3新增72份训练。相同的原始h基线只计算六个读出，不重复计为18个独立样本。</p></section>
<section><h2>性能与逐供体曲线</h2><img src="performance_curves.png" alt="完整参数曲线、配对差异和六供体变化"><p>左：灰线是三个seed各自的六供体均值，不是独立生物重复。中：阴影是六供体配对bootstrap的描述性95%区间，所有参数使用同一组20,000次抽样。训练集相互重叠且参数众多，不作逐点显著性或确认性结论。右：六个供体全部展示。</p><p><a href="performance_curves.svg">矢量图</a> · <a href="performance_curves.png">PNG</a></p></section>
<section><h2>重建与关系保持的权衡</h2><img src="geometry_tradeoff.png" alt="MSE、关系KL与梯度范数比例"><p>每个参数均为18次fit的等权监控均值。KL是原始h作为教师的自监督几何代理，不是真实生物关系；较低KL不自动等于较高分类性能。梯度比例为两个分量的范数之比，不代表梯度方向一致。</p><p><a href="geometry_tradeoff.svg">矢量图</a> · <a href="geometry_tradeoff.png">PNG</a></p></section>
<section><h2>供体与随机种子稳定性</h2><img src="stability_curves.png" alt="改善供体数量、seed数量和删去单个供体的敏感性"><p>右图报告所有六种删除一个供体后的均值范围，没有按结果删掉特定供体或改变主结果。它是描述性敏感性分析，不能替代独立数据验证。</p><p><a href="stability_curves.svg">矢量图</a> · <a href="stability_curves.png">PNG</a></p></section>
<section><h2>逐供体完整变化（相对原始h，pp）</h2>{donors.to_html(float_format=lambda x:f'{x:+.4f}',border=0)}<p>测试供体缺失的类别在固定八类Macro-F1中计0；另保存只包含实际存在类别的Macro-F1。稀有Megakaryocytes保留，不作结果驱动的类别排除。</p></section>
<section><h2>固定协议与真实训练</h2><p>L = normalized MSE + λ × mean-query KL(P_original || P_reconstructed)。温度0.1；使用固定TRAIN均值中心化的cosine；每个查询192个其他TRAIN供体锚点；教师停止梯度。字典2048、raw-value TopK32、保留所选值符号、单位范数decoder列。</p><p>每折四TRAIN、一VALID、一TEST；所有λ从同一旧20轮checkpoint出发，Adam均重置，使用相同1000个批次及余弦学习率3e−4到1e−5。固定最终step1000，未按VALID或TEST选择checkpoint。每个λ单独使用同一11档Ridge正则化及VALID标签选读出，不选择λ、不做h回退或混合搜索。</p><p>旧checkpoint曾按VALID MSE选择，本轮复用该初始化；整个流程不是全新盲测。SAE学习和关系目标不使用细胞类型标签，但Ridge选型使用VALID标签，所以不将整个流程称为完全无标签。</p></section>
<section><h2>核验与复现</h2><p>132份预测和实际512维embedding已重新读取；126份训练记录（72新增、54复用）完成权重变化、TRAIN均值、配对批次、梯度及归一化核验；132个选中读出用独立Cholesky重拟合。旧结果复现一致。</p><p>新字典更新后，旧F编号的基因归因不能自动继承。PBMC、温度τ、K、字典宽度均未新增搜索。协议继承字段的说明见PROTOCOL_NOTES.json。</p><p><a href="summary.csv">汇总CSV</a> · <a href="paired_donor_metrics.csv">逐供体CSV</a> · <a href="seed_metrics.csv">逐seed CSV</a> · <a href="learning_curves.csv">完整训练曲线</a> · <a href="gradient_diagnostics.csv">梯度诊断</a> · <a href="per_class_metrics.csv">逐类结果</a> · <a href="PROTOCOL.json">固定协议</a> · <a href="PROTOCOL_NOTES.json">协议说明</a> · <a href="verification.json">核验</a> · <a href="embedding_manifest.json">实际embedding索引</a> · <a href="REPRODUCIBILITY.txt">复现说明</a></p></section></main></html>'''
    (e.O/'REPORT.html').write_text(text)
    (e.O/'REPRODUCIBILITY.txt').write_text('''第一轮固定lambda敏感性，非最优lambda选型或独立确认实验。
环境 /opt/anaconda3/envs/scgpt/bin/python；CPU4线程。
python work/sae_lambda/run.py preflight
python work/sae_lambda/run.py develop
python work/sae_lambda/run.py lock
python work/sae_lambda/run.py evaluate
python work/sae_lambda/verify.py
python work/sae_lambda/report.py
原始缓存、旧checkpoint及冻结运行时不修改；复用路径在REUSE_MANIFEST和SELECTION_LOCK。
新增72个网络在work/sae_lambda/networks；旧54个网络在work/sae_relation/networks。
数据9577细胞6供体8类，每个测试embedding文件仅为该折留出供体，不能拼成单一全数据拟合模型。
复杂度：每fit训练O(S*(B*d*m+B^2*d))，内存O(N*d+B*m+B^2+d*m)，S1000,B256,d512,m2048。Ridge O(N*d^2+d^3)。
''')
    dest=e.O/'reproducibility';dest.mkdir(exist_ok=True)
    for p in [W/'run.py',W/'verify.py',W/'report.py']:shutil.copy2(p,dest/p.name)
    shutil.copy2(e.base.W/'run.py',dest/'frozen_relation_runtime.py')
    shutil.copy2(e.ROOT/'work/sae_refine/run.py',dest/'frozen_sae_runtime.py')
    shutil.copy2(e.ROOT/'work/sae_complementarity/core.py',dest/'frozen_readout_core.py')
    e.save(e.O/'EVIDENCE_PRECHECK.json',dict(status='verified',scope='deterministic existence only; semantic review is independent',sources={str(p):e.sha(p) for p in [e.O/'summary.json',e.O/'verification.json',e.O/'planned_contrasts.json',e.O/'sensitivity_diagnostics.json']}))
    e.tracker('COMPLETE' if reviewed else 'VERIFIED_PENDING_REVIEW',report='complete',review='same-family / provisional' if reviewed else 'pending')
    print('REPORT_WRITTEN',e.O/'REPORT.html',flush=True)

if __name__=='__main__':main()
