"""Readable fixed-grid extension report, keeping donor and batch endpoints distinct."""
import os
from pathlib import Path
W=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(W/'mplcache'))
import html,importlib.util,shutil
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter

spec=importlib.util.spec_from_file_location('extension_experiment',W/'run.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
LABELS={'stim':'Kang stim','pbmc':'PBMC'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'svg.fonttype':'none'})

def axis(ax):
    ax.set_xscale('symlog',linthresh=.01,linscale=1)
    ax.xaxis.set_major_locator(FixedLocator(e.LAMBDAS));ax.xaxis.set_major_formatter(FixedFormatter(['0','.01','.03','.1','.3','1','3']))
    ax.set_xlabel('Relation weight lambda (tau = 0.1)');ax.grid(alpha=.18)

def savefig(fig,name):
    fig.tight_layout();fig.savefig(e.O/f'{name}.png',bbox_inches='tight');fig.savefig(e.O/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def main():
    v=e.read(e.O/'verification.json');assert v['status']=='PASS'
    rows=e.read(e.O/'summary.json');summary=pd.DataFrame(rows)
    donors=pd.read_csv(e.O/'paired_group_metrics.csv');seeds=pd.read_csv(e.O/'seed_metrics.csv');curves=pd.read_csv(e.O/'learning_curves.csv');grad=pd.read_csv(e.O/'gradient_diagnostics.csv')
    contrasts=e.read(e.O/'planned_contrasts.json');preflight=e.read(e.O/'preflight.json');m=e.read(e.O/'DATA_MANIFEST.json')
    final=curves[curves.step==1000].groupby(['dataset','lambda_KL'],as_index=False).mean(numeric_only=True);final.to_csv(e.O/'final_geometry_summary.csv',index=False)
    sections=[]
    for ds in e.DATASETS:
        s=summary[(summary.dataset==ds)&(summary.method!='original_h')].sort_values('lambda_KL');x=s.lambda_KL.to_numpy();h=float(summary[(summary.dataset==ds)&(summary.method=='original_h')].macro_f1_percent.iloc[0])
        groups=donors[(donors.dataset==ds)&(donors.method!='original_h')]
        fig,axs=plt.subplots(1,3,figsize=(16,4.5))
        for j,seed in enumerate(e.SEEDS):
            ss=seeds[(seeds.dataset==ds)&(seeds.control=='original_h')&(seeds.seed==seed)].sort_values('lambda_KL')
            axs[0].plot(ss.lambda_KL,ss.macro_f1_percent,color='#888888',alpha=.5,lw=1,label='Individual seed means' if j==0 else None)
        axs[0].plot(x,s.macro_f1_percent,'o-',color=COLORS[0],label='Equal held-out group mean')
        axs[0].axhline(h,color='black',ls='--',lw=1,label='Original h');axs[0].set_ylabel('Macro-F1 (%)');axs[0].set_title(f'A  {LABELS[ds]} fixed grid');axs[0].legend(fontsize=8)
        for control,col,name in [('original_h',COLORS[0],'vs original h'),('lambda_0',COLORS[1],'vs matched MSE')]:
            cc=sorted([r for r in contrasts if r['dataset']==ds and r['control']==control and r['method']!='original_h'],key=lambda r:r['lambda_KL'])
            y=np.array([r['delta_pp'] for r in cc]);axs[1].plot(x,y,'o-',color=col,label=name)
            if ds=='stim':
                ci=np.array([r['descriptive_ci95_pp'] for r in cc]);axs[1].fill_between(x,ci[:,0],ci[:,1],color=col,alpha=.1)
        axs[1].axhline(0,color='black',ls='--',lw=1);axs[1].set_ylabel('Macro-F1 change (pp)');axs[1].set_title('B  Paired mean differences');axs[1].legend(fontsize=8)
        for j,(name,g) in enumerate(groups.groupby('test_group')):
            g=g.sort_values('lambda_KL');axs[2].plot(g.lambda_KL,g.delta_vs_h_pp,'o-',lw=1,color=COLORS[j],label=str(name))
        axs[2].axhline(0,color='black',ls='--',lw=1);axs[2].set_ylabel('Change vs original h (pp)');axs[2].set_title('C  Every held-out '+('donor' if ds=='stim' else 'batch'));axs[2].legend(fontsize=8,ncol=2 if ds=='stim' else 1)
        for ax in axs:axis(ax)
        savefig(fig,ds+'_performance_curves')
        f=final[final.dataset==ds].sort_values('lambda_KL');fig,axs=plt.subplots(1,3,figsize=(15,4.2))
        for role,col in [('train',COLORS[0]),('valid',COLORS[1])]:axs[0].plot(f.lambda_KL,f[role+'_normalized_mse'],'o-',color=col,label=role.upper())
        axs[0].set_ylabel('Normalized reconstruction MSE');axs[0].set_title('A  Reconstruction');axs[0].legend(fontsize=8)
        axs[1].plot(f.lambda_KL,f.train_monitor_kl,'o-',color=COLORS[0],label='TRAIN relation')
        axs[1].plot(f.lambda_KL,f.valid_to_train_kl,'o-',color=COLORS[1],label='VALID to TRAIN anchors');axs[1].set_ylabel('Relation KL');axs[1].set_title('B  Self-supervised geometry');axs[1].legend(fontsize=8)
        for step,col in [(1,COLORS[0]),(1000,COLORS[1])]:
            g=grad[(grad.dataset==ds)&(grad.step==step)].groupby('lambda_KL').weighted_gradient_ratio.mean().sort_index();axs[2].plot(g.index,g.to_numpy(),'o-',color=col,label=f'Step {step}')
        axs[2].set_yscale('symlog',linthresh=.1);axs[2].set_ylabel('||lambda grad KL|| / ||grad MSE||');axs[2].set_title('C  Actual gradient strength');axs[2].legend(fontsize=8)
        for ax in axs:axis(ax)
        savefig(fig,ds+'_geometry_tradeoff')
        display=s[['lambda_KL','macro_f1_percent','delta_pp','delta_vs_MSE_pp','positive_groups','positive_groups_vs_MSE','positive_seed_means']].copy()
        unit='供体' if ds=='stim' else '批次'
        display.columns=['λ','Macro-F1 (%)','相对原始h (pp)','相对MSE (pp)',f'改善{unit} vs h',f'改善{unit} vs MSE','改善seed均值 vs h']
        dsplits=pd.DataFrame([dict(fold=r['fold'],test=r['test_group'],TRAIN=r['counts']['train'],VALID=r['counts']['valid'],TEST=r['counts']['test']) for r in preflight['splits'] if r['dataset']==ds])
        matrix=groups.pivot(index='test_group',columns='lambda_KL',values='delta_vs_h_pp')
        scope=('六个原开发供体的stim细胞，4 TRAIN / 1 VALID / 1 TEST；关系目标每个查询192个其他TRAIN供体锚点。与ctrl共用参与者，不能当作独立新队列。' if ds=='stim' else '两个方向分别留出完整批次；在来源批次中固定按类别分层划出约20% VALID，剩余TRAIN。没有可靠donor标识，因此关系目标为255个其他TRAIN细胞，属于显式的批次内关系变体，不能称为跨供体约束或供体泛化。')
        sections.append(f'''<section><h2>{LABELS[ds]}：完整七点结果</h2><p>{scope}</p><p>原始 h Macro-F1 = {h:.4f}%；三个种子先在每个留出{unit}内平均，再对{e.nfold(ds)}个{unit}等权平均。pp是百分点。</p>{display.to_html(index=False,float_format=lambda z:f'{z:.4f}',border=0)}{dsplits.to_html(index=False,border=0)}<img src="{ds}_performance_curves.png" alt="{LABELS[ds]}全部参数和留出组曲线"><p>灰线是三个seed的组均值，不能当作独立生物重复。{'中图阴影仅为六供体配对bootstrap描述性区间，训练折相互重叠。' if ds=='stim' else '只有两个批次，不计算供体级置信区间或显著性；右图完整展示两种测试方向。'}</p><p><a href="{ds}_performance_curves.svg">性能矢量图</a> · <a href="{ds}_performance_curves.png">性能PNG</a></p></section>
<section><h2>{LABELS[ds]}：重建与关系保持</h2><img src="{ds}_geometry_tradeoff.png" alt="重建MSE、关系KL和梯度比例"><p>每个点为全部fold和seed监控值的等权均值。关系KL是原始h作为教师的自监督代理，不是真实生物学关系。梯度范数比反映权重实际影响，不反映两项梯度方向是否一致。</p><p><a href="{ds}_geometry_tradeoff.svg">矢量图</a> · <a href="{ds}_geometry_tradeoff.png">PNG</a></p></section>
<section><h2>{LABELS[ds]}：每个留出{unit}的变化（pp）</h2>{matrix.to_html(float_format=lambda z:f'{z:+.4f}',border=0)}</section>''')
    reference=summary[summary.method=='lambda_0p3'][['dataset','macro_f1_percent','delta_pp','delta_vs_MSE_pp','positive_groups','groups']].copy();reference.dataset=reference.dataset.map(LABELS)
    reference.columns=['固定参考λ=0.3','Macro-F1 (%)','相对h (pp)','相对MSE (pp)','改善组数','总组数']
    reviewed=(e.O/'EXPERIMENT_AUDIT.json').exists() and (e.O/'CLAIMS_FROM_RESULTS.json').exists();review='独立审查进行中；确定性数值核验已通过。';claim=''
    if reviewed:
        a=e.read(e.O/'EXPERIMENT_AUDIT.json');c=e.read(e.O/'CLAIMS_FROM_RESULTS.json');review='独立上下文、同系列模型审查（provisional）：'+html.escape(str(a.get('overall_verdict',a.get('verdict','unknown'))));claim=html.escape(str(c.get('suggested_claim_revision','')))
    mega_index=e.old.CFG['classes'].index('Megakaryocytes')
    stim_mega={r['test_group']:r['class_counts']['test'][mega_index] for r in preflight['splits'] if r['dataset']=='stim'}
    mega_note='；'.join(f'{name}: {count}' for name,count in stim_mega.items())
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAE关系重建：stim与PBMC</title>
<style>body{{font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f5f7fb;color:#24364a;margin:0}}main{{max-width:1450px;margin:auto;padding:30px}}section{{background:white;border:1px solid #dfe6ef;border-radius:12px;padding:22px;margin:20px 0;overflow:auto}}h1{{font-size:27px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #e1e7f0;text-align:left;white-space:nowrap}}th{{background:#f0f4fa}}img{{width:100%;height:auto}}a{{color:#245fc4}}.note{{color:#5c6a7b}}</style>
<main><h1>SAE关系保持重建：Kang stim 与 PBMC</h1><p class="note">{datetime.now().isoformat(timespec='seconds')} · 每个数据集独立训练SAE · scGPT冻结 · 512维重建 · 固定七点λ</p>
<section><h2>预先固定的验证问题</h2><p>在各自数据集上训练SAE之后，关系保持能否改善原始embedding的线性分类读出？λ=0.3沿用ctrl分析作为预先固定参考点，同时完整报告0、0.01、0.03、0.1、0.3、1、3，不按本轮TEST成绩选最佳λ。</p><p>{review} · <a href="EXPERIMENT_AUDIT.md">实验审查</a> · <a href="CLAIMS_FROM_RESULTS.md">结论边界</a></p><p>{claim}</p>{reference.to_html(index=False,float_format=lambda z:f'{z:.4f}',border=0)}</section>
{''.join(sections)}
<section><h2>训练与读出协议</h2><p>stim共{m['stim_cells']}细胞、PBMC共{m['pbmc_cells']}细胞，均保留八类。每个数据集、每折、每seed先独立训练一个2048宽/TopK32的MSE SAE；最多20epoch，按VALID原始MSE选择最早最佳checkpoint，patience3，epoch0可选。各λ从同一warmup checkpoint出发，重置Adam，并使用相同1000个批次和余弦学习率3e−4到1e−5。</p><p>继续训练固定最后step1000；目标为normalized MSE + λ mean-query KL(P_original||P_reconstructed)，温度0.1，TRAIN均值中心化cosine，教师停止梯度。TopK按原始值选最大32个、保留所选值符号；decoder列单位范数、无decoder bias。</p><p>每个λ直接输出512维重建；不搜索原始h混合、不回退到h。每个方法同一11档Ridge正则化，用TRAIN均值/标准差和类别均衡权重拟合，只用VALID八类Macro-F1选alpha，不重拟合TRAIN+VALID。SAE损失不使用类型标签，PBMC划分和Ridge选型仍使用标签，不能把整个流程称为完全无标签。</p></section>
<section><h2>证据范围与稀有类别</h2><p>scGPT参数完全冻结。stim执行了新的冻结前向提取；PBMC复用缓存并重新计算8个细胞核对。两个数据集的SAE均从各自TRAIN数据随机初始化和训练，未使用Kang ctrl的SAE或另一个数据集的encoder。与历史PBMC由Kang训练分类器/SAE的迁移分数不直接比较。</p><p>PBMC只有两个批次，供体和研究独立性未核实，预训练重叠未知。Batch2只有2个Megakaryocytes，反向划分后TRAIN和VALID各1个；所有类别仍保留，逐类support和F1完整保存。八类Macro-F1固定缺失类别计0，另保存仅实际存在类别的Macro-F1。</p><p>新SAE字典有各自特征编号，旧B细胞F编号与基因归因不能自动继承。几何优化和分类变化不构成因果生物学机制，且目前没有非SAE非线性基线来建立SAE特有优势。</p></section>
<section><h2>stim稀有类别分布</h2><p>Megakaryocytes各测试供体的实际细胞数：{mega_note}。该类别明显集中于一个供体，不能从总体均值推断稀有类跨供体稳定改善；不作结果驱动的类别删除。</p></section>
<section><h2>完整核验与文件</h2><p>24份warmup、168份继续训练、176份选中读出及实际512维embedding已完成回读；标签、行号、供体/批次语义、TRAIN均值、非零权重更新、配对批次和梯度均核对。选中Ridge用独立Cholesky重拟合，测试分数用显式einsum复算。</p><p><a href="summary.csv">全部汇总CSV</a> · <a href="paired_group_metrics.csv">逐供体/批次CSV</a> · <a href="per_class_metrics.csv">逐类结果与support</a> · <a href="learning_curves.csv">训练曲线</a> · <a href="gradient_diagnostics.csv">梯度</a> · <a href="verification.json">核验</a> · <a href="DATA_MANIFEST.json">数据来源</a> · <a href="PROTOCOL_NOTES.json">协议字段说明</a> · <a href="stim/PROTOCOL.json">stim协议</a> · <a href="pbmc/PROTOCOL.json">PBMC协议</a> · <a href="embedding_manifest.json">实际embedding索引</a> · <a href="REPRODUCIBILITY.txt">复现说明</a></p></section></main></html>'''
    (e.O/'REPORT.html').write_text(text)
    (e.O/'REPRODUCIBILITY.txt').write_text('''使用 /opt/anaconda3/envs/scgpt/bin/python，torch2.3.1，CPU4。
先 python work/sae_extension/prepare.py （stim分块可恢复；PBMC来自原缓存经冻结重算核对）。
已采用的流水线：
python work/sae_extension/run.py preflight --dataset pbmc
python work/sae_extension/run.py develop --dataset pbmc
等待prepare.py完成
python work/sae_extension/run.py preflight --dataset stim
python work/sae_extension/run.py develop --dataset stim
python work/sae_extension/run.py lock
python work/sae_extension/run.py evaluate
python work/sae_extension/verify.py
python work/sae_extension/report.py
数据与24个独立warmup、168个继续训练网络、固定batch索引均保存在work/sae_extension/{data,pbmc,stim}。
来源和来源哈希在sources.json、各ds/sources.json、DATA_MANIFEST和SELECTION_LOCK。
结果每个embedding文件仅为该折留出组，未把TEST用于SAE、读出或lambda选择。
训练时间复杂度warmup O(20*N*d*m)，继续训练 O(1000*(B*d*m+B^2*d))；内存O(N*d+B*m+B^2+d*m)，d512,m2048,B256。
PBMC两个批次不等于两个已验证独立供体。没有用Kang ctrl训练的SAE做PBMC迁移。
''')
    dest=e.O/'reproducibility';dest.mkdir(exist_ok=True)
    for n in ['prepare.py','run.py','verify.py','report.py']:shutil.copy2(W/n,dest/n)
    shutil.copy2(e.ROOT/'work/sae_relation/run.py',dest/'frozen_relation_runtime.py');shutil.copy2(e.ROOT/'work/sae_refine/run.py',dest/'frozen_sae_runtime.py');shutil.copy2(e.ROOT/'work/sae_complementarity/core.py',dest/'frozen_readout_core.py')
    e.save(e.O/'EVIDENCE_PRECHECK.json',dict(status='verified',scope='deterministic evidence existence only; semantic review independent',sources={str(p):e.sha(p) for p in [e.O/'summary.json',e.O/'verification.json',e.O/'planned_contrasts.json',e.O/'final_geometry_summary.csv']}))
    e.update('COMPLETE' if reviewed else 'VERIFIED_PENDING_REVIEW',report='complete',review='same-family / provisional' if reviewed else 'pending')
    print('REPORT_WRITTEN',e.O/'REPORT.html',flush=True)

if __name__=='__main__':main()
