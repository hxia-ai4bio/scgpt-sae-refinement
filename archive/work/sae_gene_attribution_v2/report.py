"""Read-only rendering of completed, verified gene attribution results."""
from pathlib import Path
import os,json,base64,html
BASE=Path(__file__).resolve().parent;OUT=BASE.parents[1]/'outputs/sae_gene_attribution_v2'
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'matplotlib_cache'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'savefig.bbox':'tight'})
def table(d):return d.to_html(index=False,float_format=lambda v:f'{v:.4g}',na_rep='未评估',border=0)
def img(p):return '<img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">'
def main():
 tr=pd.read_csv(OUT/'gene_rankings_train.csv');te=pd.read_csv(OUT/'gene_rankings_test.csv');pp=pd.read_csv(OUT/'per_prototype_gene_attribution.csv');ref=pd.read_csv(OUT/'reference_sensitivity.csv');don=pd.read_csv(OUT/'train_donor_consistency.csv');de=pd.read_csv(OUT/'deletion_summary.csv');ribo=pd.read_csv(OUT/'ribosomal_summary.csv');pairs=json.loads((OUT/'pairs.json').read_text());occ=json.loads((OUT/'occlusion_completion.json').read_text());ver=json.loads((OUT/'verification.json').read_text());completion=json.loads((OUT/'analysis_completion.json').read_text())
 qc=pd.read_csv(OUT/'pair_qc_donor_summary.csv');qcm=qc.groupby('role')[['prototype_detected_genes','reference_detected_genes']].mean()
 gate_single=0;gate_combo=0;combo_nonzero=0;single_diff=0.;combo_diff=0.
 for path in sorted((BASE/'cache').glob('*.npz')):
  with np.load(path) as a:
   z=a['baseline'][0]-a['effects'][:,0];aa=a['baseline'][1]-a['effects'][:,1];changed=a['changed'];gate_single+=int(((z[changed]==0)&(np.abs(aa[changed])>1e-8)).sum());single_diff=max(single_diff,float(np.abs(a['effects'][:,0]-a['effects'][:,1]).max()))
 for path in sorted((BASE/'deletion').glob('*.npz')):
  with np.load(path) as a:
   valid=a['actual']>0;ss=a['scores'];gate_combo+=int(((ss[valid,0]==0)&(np.abs(ss[valid,1])>1e-8)).sum());combo_nonzero+=int(valid.sum());combo_diff=max(combo_diff,float(np.abs(ss[:,0]-ss[:,1]).max()))
 (OUT/'topk_observed_diagnostics.json').write_text(json.dumps({'single_gene_topk_exits':gate_single,'single_gene_sae_vs_encoder_effect_max_difference':single_diff,'combination_topk_exits':gate_combo,'nonzero_combination_inputs':combo_nonzero,'combination_sae_vs_encoder_score_max_difference':combo_diff,'scope':'observed selected local inputs only'},indent=2))
 score='sae_F160';merged=tr.merge(te[['gene',score,'input_prototypes']],on='gene',how='left',suffixes=('_train','_test'));merged['ribosomal_prefix']=merged.gene.str.startswith(('RPL','RPS'))
 old=OUT.parent/'sae_interpretability/gene_association_160_train.csv'
 if old.exists():merged=merged.merge(pd.read_csv(old)[['gene','partial_r']],on='gene',how='left')
 merged.to_csv(OUT/'gene_evidence_combined.csv',index=False)
 stable=merged[(merged.sae_F160_positive_donors>=3)&(merged.input_donors>=3)&(~merged.ribosomal_prefix)&(merged.sae_F160_train>0)].copy();stable.to_csv(OUT/'descriptive_consistent_nonribosomal.csv',index=False)
 # TRAIN points are donor averages; absent token genes contribute zero as specified.
 top=tr.head(20);ids=pp[pp.role=='train'].prototype_index.unique();dense=pp[pp.role=='train'].pivot(index='prototype_index',columns='gene',values=score).reindex(index=ids,columns=top.gene).fillna(0);donmap=pp[pp.role=='train'].drop_duplicates('prototype_index').set_index('prototype_index').donor;dense['donor']=donmap;bydon=dense.groupby('donor').mean()
 assert np.allclose(bydon[top.gene].mean(0).values,top[score].values,atol=1e-10)
 fig,ax=plt.subplots(figsize=(9,7));order=np.arange(len(top));colors=['#a17c50' if g.startswith(('RPL','RPS')) else '#247c7b' for g in top.gene]
 ax.barh(order,top[score],color=colors,alpha=.75)
 for i,(name,row) in enumerate(bydon.iterrows()):ax.scatter(row[top.gene],order+(i-1.5)*.07,s=13,label=name,alpha=.8)
 ax.set_yticks(order,top.gene);ax.invert_yaxis();ax.set_xlabel('Raw SAE activation drop: equal-weight donor mean');ax.set_title('B-cell F160 | TRAIN top20 genes\nDots: 4 donors; brown: RPL/RPS prefix');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'train_gene_attribution.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 methods=[('sae_F160','SAE TRAIN ranking','#247c7b'),('encoder_F160','Pre-TopK TRAIN ranking','#6884ac'),('embedding_neg201','Embedding TRAIN ranking','#b39563'),('expression_difference','Expression difference','#bf734c'),('matched_random','Joint-matched random','#8d8e94'),('sae_nonribosomal','SAE excluding RPL/RPS','#9c668a')]
 for ax,sc in zip(axes,['sae_F160','encoder_F160']):
  for method,label,color in methods:
   a=de[(de.score==sc)&(de.method==method)];ax.plot(a.requested_genes,a.activation,'o-',markersize=3,label=label,color=color)
  ax.set_title(sc);ax.set_xlabel('Requested substituted genes');ax.set_ylabel('Raw activation');ax.grid(alpha=.15)
 axes[0].legend(fontsize=7);fig.suptitle('TEST: 4 selected B cells, one donor, 3 reused references each');fig.tight_layout();fig.savefig(OUT/'heldout_deletion.png',dpi=160);plt.close(fig)
 cols=['gene','sae_F160_train','sae_F160_test','sae_F160_positive_donors','input_donors','input_prototypes_train','ribosomal_prefix']
 short=merged[cols].head(20).rename(columns={'sae_F160_train':'训练归因','sae_F160_test':'测试归因','sae_F160_positive_donors':'正向训练供体数','input_donors':'覆盖训练供体数','input_prototypes_train':'覆盖训练原型数','ribosomal_prefix':'RPL/RPS前缀'})
 refsum=ref.groupby(['role','score'])[['spearman','top20_jaccard']].median().reset_index();ds=don.groupby('score').spearman.agg(['median','min','max']).reset_index();at40=de[(de.score=='sae_F160')&(de.requested_genes==40)][['method','baseline','activation','drop']]
 rnd=pd.read_csv(OUT/'matched_random_diagnostics.csv');rr=ribo[(ribo.role=='train')&(ribo.score=='sae_F160')].iloc[0]
 audit=OUT/'EXPERIMENT_AUDIT.json';auditlabel='待审查'
 if audit.exists():
  aa=json.loads(audit.read_text());auditlabel=str(aa.get('verdict',aa.get('overall_verdict','见审计记录')))
 text=f'''B细胞 SAE F160 多参照基因归因（2026-10-18）

范围：固定上一轮TRAIN选定的主模型F160；8个训练原型（4供体各2个），4个测试原型（1供体）；{len(pairs)}个同供体同类型原型/参照对。patient_107仅有2个可用不激活B细胞，其余最多3个。不使用跨类型补足。
覆盖：训练/测试分别包含{len(tr)}/{len(te)}个不同输入基因；未被输入覆盖不等于没有生物学作用。
完成：{occ['new_forwards']}次新单基因替换前向；{completion['deletion_forwards']}次组合替换前向。参照并非独立生物重复。
配对QC：训练原型/参照平均检出基因数为{qcm.loc['train','prototype_detected_genes']:.0f}/{qcm.loc['train','reference_detected_genes']:.0f}；测试为{qcm.loc['test','prototype_detected_genes']:.0f}/{qcm.loc['test','reference_detected_genes']:.0f}。同类型配对仍有表达复杂度差异，可能影响归因。
方法：固定token和padding，逐基因替换成参照细胞的分箱值；计算原始激活下降。先平均参照，再平均原型，再等权平均供体。未进入原型token的基因贡献记0，另列覆盖次数和条件均值。

TRAIN前20基因：
{short.to_string(index=False,na_rep="未评估")}

TRAIN正向前{int(rr.positive_top_n)}基因中，{int(rr.ribosomal_top_n)}个带RPL/RPS前缀。
训练供体两两基因归因一致性（共同输入基因上Spearman）：
{ds.to_string(index=False)}
参照敏感性中位数（相同原型的不同参照，细胞对不独立）：
{refsum.to_string(index=False)}

训练排序固定后，TEST替换40个基因的SAE原始激活下降：
{at40.to_string(index=False)}
随机对照联合匹配原表达分箱、替换方向及变化幅度分箱（非数值完全相等），10次随机排列；与SAE前40基因重叠范围为{int(rnd.overlap_sae40.min())}–{int(rnd.overlap_sae40.max())}，均值{rnd.overlap_sae40.mean():.2f}。不同表示的原始激活单位不能直接比较。非RPL/RPS排序是探索性诊断，没有针对它单独匹配的随机对照，不能据此宣称非核糖体特异性。

解释边界：只研究B细胞候选F160，不表示整个字典的可解释率；所有原型/参照由SAE选择，因此原始embedding与PCA只是同细胞诊断对照。TEST是此前已分析过的历史测试集，不是新的盲测；仅一个供体的4个高激活细胞，3个参照会反复使用；不作细胞级显著性或人群因果推断。本轮单基因替换观察到{gate_single}次F160退出TopK，SAE与筛选前归因最大差异为{single_diff:.3g}；{combo_nonzero}次非空组合替换中观察到{gate_combo}次退出。因此应依据实测而非默认把差异归于TopK跳变。核糖体前缀只是操作性分类，非完整基因功能注释。原型表达分箱替换不是湿实验敲除。
“跨至少3个训练供体正向、非RPL/RPS”的表格为描述性后筛选，不用于改变已固定的测试排序；不能据此命名新功能程序。测试缺失基因显示未评估而非零。
确定性核验：{ver['status']}；科研审计：{auditlabel}（same-family/provisional）。
'''
 (OUT/'RESULTS_ZH.txt').write_text(text)
 body='<h1>B细胞 SAE F160：多参照基因归因</h1><p>固定模型与特征 · 4个训练供体 + 1个测试供体 · 单基因输入敏感性</p>'
 body+='<section><h2>实际做了什么</h2><p>'+html.escape(text[text.index('范围：'):text.index('TRAIN前20基因：')]).replace('\n','<br>')+'</p></section>'
 body+='<section><h2>配对仍有技术差异</h2><p>最近的同类型不激活细胞不等于技术因素匹配的细胞。两组检出基因数的差异需要与归因结果一起解读。</p>'+table(qc)+'</section>'
 body+='<section><h2>哪些基因影响激活</h2>'+img(OUT/'train_gene_attribution.png')+table(short)+f'<p>训练正向前40基因中，{int(rr.ribosomal_top_n)}个带RPL/RPS前缀。点表示供体，不是置信区间。</p></section>'
 body+='<section><h2>换参照、换供体后是否一致</h2><p>以下相关系数只在共同输入基因上计算。参照和原型重复使用，不能把配对数量当生物重复。</p>'+table(refsum)+table(ds)+'</section>'
 body+='<section><h2>训练排序能否影响测试细胞</h2>'+img(OUT/'heldout_deletion.png')+table(at40)+'<p>随机对照针对SAE排序匹配；非RPL/RPS排序没有单独匹配的随机对照，不能据此宣称非核糖体特异性。各条曲线仅作当前局部输入的诊断。参照重复和有限匹配池限制独立性。</p></section>'
 body+='<section><h2>供后续核查的非核糖体候选</h2><p>描述性后筛选：至少3个训练供体正向、覆盖至少3个供体、非RPL/RPS前缀。不等于新功能模块。</p>'+table(stable[cols].head(15))+'</section>'
 body+='<section><h2>解释范围与核验</h2><p>'+html.escape(text[text.index('解释边界：'):]).replace('\n','<br>')+'</p></section>'
 body+='<section><h2>完整数据</h2><ul>'+''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(OUT.iterdir()) if p.suffix in ['.csv','.json','.txt','.md'])+'</ul></section>'
 css='body{font-family:-apple-system,sans-serif;background:#f4f6f7;color:#26363e;line-height:1.7}main{max-width:1100px;margin:35px auto;padding:0 20px}section{background:white;border-radius:12px;padding:22px;margin:20px 0;overflow:auto}h1{font-size:30px}h2{font-size:21px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dce5e5;white-space:nowrap;text-align:left}th{background:#edf3f3}a{color:#247c7b}'
 (OUT/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B细胞基因归因</title><style>'+css+'</style><main>'+body+'</main></html>')
 (OUT/'EXPERIMENT_TRACKER.json').write_text(json.dumps({'status':'complete','occlusion':'complete','aggregation':'complete','deletion':'complete','verification':ver['status'],'semantic_review':auditlabel,'scope':'F160 B cells local sensitivity; no new training, pathway tests or expert blind evaluation'},indent=2,ensure_ascii=False))
 print('REPORT_WRITTEN',OUT/'REPORT.html')
if __name__=='__main__':main()
