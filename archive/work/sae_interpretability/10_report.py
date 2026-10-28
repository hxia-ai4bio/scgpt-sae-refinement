"""Readable local report and figures generated only from completed result files."""
import base64,html,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

LABEL={440:'CD14 candidate',160:'B-cell candidate',120:'NK candidate',1724:'Depth-associated comparator'}
ZH={440:'CD14单核细胞候选',160:'B细胞候选',120:'NK细胞候选',1724:'深度关联对照'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140,'savefig.bbox':'tight'})

def embed(path):return '<img src="data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'">'
def table(df):return df.to_html(index=False,escape=True,float_format=lambda x:f'{x:.3f}',border=0,na_rep='未计算')

def main():
    rep=pd.read_csv(OUT/'candidate_replication.csv');fold=pd.read_csv(OUT/'donor_equal_single_feature_readouts.csv')
    deletion=pd.read_csv(OUT/'heldout_deletion_curves.csv');deletion['method']=deletion.order.str.replace(r'_\d+$','',regex=True)
    strong_count=0
    if (OUT/'heldout_deletion_strong_null.csv').exists():
      strong=pd.read_csv(OUT/'heldout_deletion_strong_null.csv');strong['method']='strong_matched_random'
      strong_count=len(strong);deletion=pd.concat([deletion,strong],ignore_index=True)
    enrich=pd.read_csv(OUT/'pathway_enrichment_all.csv');st=pd.read_csv(OUT/'dictionary_stability.csv')
    occ=json.loads((OUT/'occlusion_completion.json').read_text());verify=json.loads((OUT/'deterministic_verification.json').read_text())
    candidates=json.loads((OUT/'candidates.json').read_text());gate=pd.read_csv(OUT/'topk_suppression_diagnostics.csv')
    cols=['test_auc','Batch1_3k_auc','Batch2_68k_r_auc']
    means=fold.groupby(['target','representation'])[cols].mean().reset_index()
    means.to_csv(OUT/'summary_single_feature_auc.csv',index=False)
    curves=deletion[deletion.requested_genes==40].groupby(['feature','method'])[['raw_baseline','raw_activation','raw_drop']].mean().reset_index()
    curves.to_csv(OUT/'summary_40gene_perturbation.csv',index=False)
    # Six donor values are shown; these are not 18 independent model replications.
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    colors={'embedding':'#8797a4','pca':'#b6a177','sae':'#247c7b'}
    for ax,target in zip(axes,['B cells','CD14+ Monocytes','NK cells']):
      for j,method in enumerate(['embedding','pca','sae']):
        vals=fold[(fold.target==target)&(fold.representation==method)].test_auc.values
        ax.bar(j,vals.mean(),color=colors[method],width=.55,alpha=.8)
        ax.scatter(j+np.linspace(-.16,.16,len(vals)),vals,c='#263541',s=16,zorder=3)
      ax.set_title(target);ax.set_xticks([0,1,2],['Embedding','PCA','SAE']);ax.set_ylim(.75,1.005)
      ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Held-out donor one-v-rest AUROC')
    fig.suptitle('TRAIN-selected single coordinate | 6 donors, seed-averaged dots',y=1.04)
    fig.tight_layout();fig.savefig(OUT/'single_feature_readouts.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,f in zip(axes.flat,[440,160,120,1724]):
      sub=deletion[deletion.feature==f]
      for method,label,color in [('attribution','TRAIN attribution','#247c7b'),('expression_difference','TRAIN expression difference','#b07b38'),('matched_random','Expression-bin matched random','#8797a4'),('strong_matched_random','Also matched sign + change size','#9c668a')]:
        dd=sub[sub.method==method];mu=dd.groupby('requested_genes').raw_activation.mean()
        if len(dd)==0:continue
        ax.plot(mu.index,mu.values,'o-',label=label,color=color,markersize=3)
        if method=='matched_random':
          band=dd.groupby('requested_genes').raw_activation.agg(['min','max'])
          ax.fill_between(band.index,band['min'],band['max'],color=color,alpha=.12)
      ax.set_title(f'{LABEL[f]} (F{f})');ax.set_xlabel('Number of substituted genes');ax.set_ylabel('Raw SAE activation');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Two selected TEST cells per feature; one donor | Shading: random/cell range, not CI')
    fig.tight_layout();fig.savefig(OUT/'heldout_deletion.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,f in zip(axes.flat,[440,160,120,1724]):
      d=pd.read_csv(OUT/f'gene_attribution_{f}_train.csv').head(8).iloc[::-1]
      ax.barh(d.gene,d.mean_attribution,color='#247c7b');ax.set_title(f'{LABEL[f]} (F{f})');ax.set_xlabel('Mean raw activation decrease (4 TRAIN cells)')
    fig.tight_layout();fig.savefig(OUT/'gene_attributions.png');plt.close(fig)
    cards=[];textcards=[]
    for c in candidates:
      f=c['feature'];assoc=pd.read_csv(OUT/f'gene_association_{f}_train.csv').head(6)
      attr=pd.read_csv(OUT/f'gene_attribution_{f}_train.csv').head(6)
      rr=rep[(rep.feature==f)&(~rep.role.isin(['train','valid']))][['domain','group','auc','nonzero','n']]
      e=enrich[(enrich.source=='attribution')&(enrich['split']=='train')&(enrich.feature==f)].sort_values('q_hypergeom').head(3)
      ee=e[['pathway','overlap','q_hypergeom','q_matched']].copy()
      for key in ['q_hypergeom','q_matched']:ee[key]=ee[key].map(lambda v:f'{v:.3g}' if np.isfinite(v) else '未计算')
      assoc_label='跨类型表达关联（加性控制细胞类型与供体）' if f==1724 else '同类型内表达关联'
      cards.append(f'<section><h2>F{f} · {ZH[f]}</h2><p><b>{assoc_label}：</b>{html.escape("、".join(assoc.gene))}</p><p><b>输入扰动贡献较高：</b>{html.escape("、".join(attr.gene))}</p>{table(rr)}<details><summary>训练归因的通路注释（完整结果另附）</summary>{table(ee)}</details></section>')
      textcards.append(f'F{f} {ZH[f]}\n表达关联前6：'+', '.join(assoc.gene)+'\n归因前6：'+', '.join(attr.gene))
    cross=st[st.other_test_fold=='patient_1016'].groupby('feature')[['decoder_cosine','activation_spearman_primary_train']].agg(['min','max'])
    consistency=pd.read_csv(OUT/'attribution_consistency.csv').groupby(['feature','role']).agg(mean_gene_spearman=('spearman','mean'),min_shared_genes=('shared_genes','min'),cell_pairs=('spearman','count')).reset_index()
    ribo=[]
    for f in [440,160,120,1724]:
      d=pd.read_csv(OUT/f'gene_attribution_{f}_train.csv');top=d[d.mean_attribution>0].head(40)
      ribo.append({'feature':f,'top40_RPL_RPS_genes':int(top.gene.str.match(r'^RP[LS]').sum()),'top_positive_genes':len(top)})
    js(OUT/'ribosomal_gene_summary.json',ribo)
    audit_path=OUT/'EXPERIMENT_AUDIT.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {'verdict':'审查进行中'}
    auditlabel=str(audit.get('verdict',audit.get('overall_verdict','见审计文件')))
    text=(
      'SAE生物可解释性：首轮探索结果（2026-10-18）\n\n'
      f'完成：18个K32模型的类型读出诊断；主模型2048特征概览；4个候选；24个原型/参照对；{occ["single_gene_changed_forwards"]}次单基因替换前向与{occ["deletion_forwards"]+strong_count}次组合替换前向（含审计后补充的{strong_count}次更严格随机对照）。\n'
      'SAE/scGPT没有重新训练。主归因使用4个训练供体各1细胞及1个测试供体的2细胞，每个细胞1个参照；PBMC没有运行输入归因。\n\n'
      '六供体平均单坐标AUROC（先对种子平均；不是八类Macro-F1）：\n'+means.to_string(index=False,float_format=lambda x:f'{x:.4f}')+'\n\n'
      '固定训练基因排序，在测试原型细胞替换40个基因后的原始激活下降：\n'+curves.to_string(index=False,float_format=lambda x:f'{x:.4f}')+'\n\n'
      + '\n\n'.join(textcards)+'\n\n'
      '解释边界：B细胞相关信号具有较好的跨文件保留；CD14特征在PBMC中被TopK大幅压低；NK跨文件结果弱于原embedding单坐标对照。不能声称SAE整体优于原表征。\n'
      '四个特征的扰动重要基因都混入大量RPL/RPS基因。不能把类型关联直接命名为独立功能程序；同源核糖体基因造成多个重叠富集条目，不代表发现多个独立机制。\n'
      '归因排序在本轮每类两个测试细胞上，40基因替换平均下降大于差异表达与随机排序；这是复合输入→scGPT→SAE特征的局部敏感性，不是原scGPT依赖SAE特征的证据，更不是真实细胞因果验证。\n'
      '加强随机对照是事后审计驱动：同时匹配原表达分箱、替换方向及变化量分箱；有限匹配池会复用部分归因基因，详见strong_null_matching_diagnostics.csv，不能据此作供体级显著性推断。\n'
      '不同种子解码方向余弦较低；同种子跨折共享初始化与大量训练供体，不能当独立概念稳定性证明。\n'
      '基因归因跨原型一致性：\n'+consistency.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n这些细胞对共享原型，不能当独立重复；B细胞身份信号可读出不等于归因基因稳定。\n'
      '限制：历史探索数据；只一个主归因模型/测试供体；每细胞一个参照；一对NK训练参照跨细胞类型；固定token/bin干预不是重新处理raw-count干预；Kang没有MT-基因，线粒体混杂不可评估；SAE采用20轮预算，未验证充分收敛；对照候选维数不等；目前没有盲评整体可解释比例、细胞功能干预或SAE内部消融。\n'
      '通路：训练集表达关联基因的富集在匹配表达量与检出率的随机对照下，没有条目达到FDR<0.05（最小q约0.0703）；归因基因的显著条目主要集中于相互重叠的翻译/核糖体相关通路。采用Hallmark/Reactome ORA，非论文GSEA。各split独立挑前40基因的富集仅作描述；真正固定TRAIN前40基因的迁移诊断见expression_association_summary.json。匹配基因随机检验只在TRAIN运行，TEST/PBMC为未计算。\n'
      f'确定性核验：{verify["status"]}；语义审计：{auditlabel}（same-family/provisional）。\n'
      '建议下一步：以B细胞候选为首个案例，扩展同类型多参照、更多供体原型，并对核糖体与细胞身份信号做有针对性的分离；暂不把这些特征命名为功能模块。\n')
    (OUT/'RESULTS_ZH.txt').write_text(text)
    stats=f'<div class="stats"><div><b>18</b>K=32模型</div><div><b>4</b>固定候选</div><div><b>{occ["single_gene_changed_forwards"]:,}</b>单基因替换前向</div><div><b>{occ["deletion_forwards"]+strong_count}</b>组合替换前向</div></div>'
    summary='<p class="lead">已找到能重复读出细胞身份的候选特征，但输入扰动揭示了明显的核糖体与通用表达信号。当前证据支持“生物相关表征”，尚不足以命名独立功能模块。</p>'
    body='<h1>SAE生物可解释性 · 首轮探索</h1><p class="muted">2026-10-18 · Kang ctrl + PBMC · 冻结模型 · 所有结果可追溯到随附CSV/JSON</p>'+summary+stats
    body+='<section><h2>读出信号能否跨供体保留</h2><p>每折仅用训练供体筛选单坐标；点为六个测试供体各自的三种子平均。SAE、原embedding和PCA候选维数不同，因此不能归因于稀疏性或宣布整体可解释性优势。</p>'+embed(OUT/'single_feature_readouts.png')+table(means)+'</section>'
    body+='<section><h2>输入归因是否比表达差异更忠实</h2><p>基因排序固定于训练原型；在一个留出供体、每特征两个高激活细胞上检验。曲线显示原始激活，随机阴影不是置信区间。紫色曲线是事后审计补充的更严格对照，同时匹配原表达分箱、替换方向和变化量分箱；有限匹配池复用了部分归因基因，完整诊断另附。结果只覆盖选定局部区域。</p>'+embed(OUT/'heldout_deletion.png')+table(curves)+'</section>'
    body+='<section><h2>关联基因与真正影响激活的基因不同</h2><p>下图为固定token的单基因表达分箱替换。不存在于该细胞输入中的基因按零贡献计入跨原型平均；每基因输入覆盖次数保存在CSV中。多个候选共同依赖RPL/RPS基因，不应把富集条目直接解释为独立机制。</p>'+embed(OUT/'gene_attributions.png')+'</section>'
    body+=''.join(cards)
    body+='<section><h2>TopK竞争与稳定性</h2>'+table(gate[(gate.feature==440)&(gate.domain=='pbmc')][['group','raw_encoder_auc','sparse_auc','target_retained_fraction']])+'<p>CD14候选的TopK前分数保留较多类型信息，进入TopK后大部分被置零。这是事后诊断，未据此调模型。跨种子解码余弦范围见下表；同种子跨折的高相似也受到共同初始化影响。</p>'+cross.to_html(float_format=lambda v:f'{v:.3f}')+'<h3>基因归因跨原型一致性偏弱</h3><p>相关只在两细胞共同输入的基因上计算。细胞对共享原型，不是独立生物重复；B细胞身份信号可读出不等于归因基因稳定。</p>'+table(consistency)+'</section>'
    body+='<section><h2>范围与未解决问题</h2><p>'+html.escape(text[text.index('限制：'):text.index('通路：')])+'</p><p>'+html.escape(text[text.index('通路：'):text.index('确定性核验：')])+'</p><p>算术及重放核验：'+verify['status']+'；语义审计：'+html.escape(auditlabel)+'（同模型家族，暂定判断）。</p><p><b>下一步：</b>优先扩展B细胞候选的多参照、跨供体归因，并检验核糖体信号与细胞身份信号是否可分离。</p></section>'
    body+='<section><h2>结果文件</h2><ul>'+''.join(f'<li><a href="{p.name}">{html.escape(p.name)}</a></li>' for p in sorted(OUT.glob('*')) if p.suffix in ['.csv','.json','.txt','.md'])+'</ul></section>'
    css='body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f4f6f7;color:#24343d;line-height:1.7}main{max-width:1120px;margin:40px auto;padding:0 22px}h1{font-size:34px;margin-bottom:4px}h2{font-size:21px}section{background:white;padding:24px;margin:20px 0;border-radius:14px;overflow:auto}.lead{font-size:20px;max-width:980px}.muted{color:#687a83}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stats div{background:#e2eeec;border-radius:10px;padding:18px}.stats b{display:block;font-size:28px;color:#176463}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #dfe6e9;text-align:left;white-space:nowrap}th{background:#edf3f4}a{color:#176463}details{margin-top:20px}summary{cursor:pointer}@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}}'
    (OUT/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAE生物可解释性</title><style>'+css+'</style><main>'+body+'</main></html>')
    js(OUT/'EXPERIMENT_TRACKER.json',{'status':'completed','primary_atlas':'complete','all18_readouts':'complete','expression_proxy':'complete','occlusion':'complete','deletion_controls':'complete','enrichment':'complete','deterministic_verification':verify['status'],'semantic_audit':auditlabel,'scope':'bounded exploratory pilot; no scGPT/SAE retraining or stimulation analysis'})
    print('REPORT_WRITTEN',OUT/'REPORT.html')

if __name__=='__main__':main()
