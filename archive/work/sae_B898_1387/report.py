from pathlib import Path
import os,json,html,base64
import numpy as np,pandas as pd
from scipy.stats import spearmanr
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B898_1387';os.environ.setdefault('MPLCONFIGDIR',str(W.parent/'sae_gene_attribution_v2/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}',na_rep='—')
def js(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def main():
 paths=pd.read_csv(O/'all_pathway_results.csv');readout=pd.read_csv(O/'feature_readouts.csv');summaries=[];tops=[];gate=[];combined={};picked=[]
 for fid in [898,1387]:
  FO=O/f'F{fid}';FW=W/f'F{fid}';g=pd.read_csv(FO/'gene_evidence_combined.csv');combined[fid]=g.set_index('gene');score=f'sae_F{fid}_train';test=f'sae_F{fid}_test'
  for role in ['train','test']:
   ap=FO/f'expression_association_{role}.csv'
   if ap.exists():g=g.merge(pd.read_csv(ap).rename(columns={'partial_r':f'RNA_partial_r_{role}'}),on='gene',how='left')
  combined[fid]=g.set_index('gene')
  for mode,a in [('all',g),('non_RPL_RPS',g[~g.gene.str.startswith(('RPL','RPS'))])]:
   top=a[a[score]>0].head(40).copy();top['feature']=fid;top['list']=mode;tops.append(pd.DataFrame({'feature':fid,'list':mode,'gene':top.gene,'train_rank':top.train_rank,'train_contribution':top[score],'test_contribution':top[test],'positive_TRAIN_donors':top[f'sae_F{fid}_positive_donors'],'input_TRAIN_donors':top.input_donors,'input_TRAIN_prototypes':top.input_prototypes_train,'conditional_train_mean':top[f'sae_F{fid}_conditional_mean'],'train_donor_SD':top[f'sae_F{fid}_donor_sd'],'RNA_partial_r_train':top.RNA_partial_r_train,'RNA_partial_r_test':top.RNA_partial_r_test}))
  train40=g[g[score]>0].head(40);present=train40[test].notna();don=pd.read_csv(FO/'train_donor_consistency.csv');ref=pd.read_csv(FO/'reference_sensitivity.csv');rep=readout[readout.feature==fid];tr=rep[rep.role=='train'];te=rep[rep.role=='test'];pb=rep[rep.domain=='pbmc'];corr=don[don.score==f'sae_F{fid}'].spearman.median();rr=ref[(ref.score==f'sae_F{fid}')&(ref.role=='train')].top20_jaccard.median()
  qc=pd.read_csv(FO/'pair_qc_donor_summary.csv').groupby('role')[['prototype_detected_genes','reference_detected_genes']].mean()
  summaries.append({'feature':fid,'train_mean_donor_AUC':tr.auc_B_vs_other.mean(),'train_min_donor_AUC':tr.auc_B_vs_other.min(),'historical_test_AUC':te.auc_B_vs_other.iloc[0],'PBMC_batch1_AUC':pb[pb.group=='Batch1_3k'].auc_B_vs_other.iloc[0],'PBMC_batch2_AUC':pb[pb.group=='Batch2_68k_r'].auc_B_vs_other.iloc[0],'TRAIN_covered_genes':len(g),'TRAIN_prototype_detected_genes':qc.loc['train','prototype_detected_genes'],'TRAIN_reference_detected_genes':qc.loc['train','reference_detected_genes'],'TEST_prototype_detected_genes':qc.loc['test','prototype_detected_genes'],'TEST_reference_detected_genes':qc.loc['test','reference_detected_genes'],'TRAIN_top40_RPL_RPS':int(train40.gene.str.startswith(('RPL','RPS')).sum()),'TRAIN_top40_TEST_covered':int(present.sum()),'TRAIN_top40_TEST_positive':int((train40.loc[present,test]>0).sum()),'TRAIN_donor_pair_Spearman_median':corr,'TRAIN_reference_top20_Jaccard_median':rr})
  for p in sorted((FW/'cache').glob('pair_*.npz')):
   z=np.load(p);effects=z['effects'];pert=z['baseline'][None,:]-effects;changed=z['changed'];unequal=np.abs(effects[changed,0]-effects[changed,1])>3e-5;gate.append({'feature':fid,'kind':'single_gene','pair':p.stem,'single_gene_inputs':len(changed),'TopK_encoder_effect_unequal':int(unequal.sum()),'TopK_zero_after_change':int((np.abs(pert[changed,0])<1e-8).sum()),'max_effect_difference':float(np.abs(effects[:,0]-effects[:,1]).max())})
  for p in sorted((FW/'deletion').glob('pair_*.npz')):
   z=np.load(p);ss=z['scores'];changed=z['actual']>0;gate.append({'feature':fid,'kind':'combination','pair':p.stem,'single_gene_inputs':int(changed.sum()),'TopK_encoder_effect_unequal':int((np.abs(ss[changed,0]-ss[changed,1])>3e-5).sum()),'TopK_zero_after_change':int((np.abs(ss[changed,0])<1e-8).sum()),'max_effect_difference':float(np.abs(ss[:,0]-ss[:,1]).max())})
  subset=paths[paths.feature==fid]
  # Selection for display is explicit and includes nonsignificant immune annotations.
  for analysis,part in subset.groupby('analysis'):
   show=set(part.nsmallest(5,'q_ora_global').term)|set(part.sort_values(['q_matched_global','p_matched','p_ora','term']).head(5).term)
   immune=['GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION_OF_ENDOGENOUS_ANTIGEN','GOBP_B_CELL_ACTIVATION','GOBP_B_CELL_RECEPTOR_SIGNALING_PATHWAY','HALLMARK_INTERFERON_GAMMA_RESPONSE']
   show.update(t for t in immune if t in set(part.term));view=part[part.term.isin(show)].copy();view['display_reason']=np.where(view.term.isin(immune),'predefined immune annotation lookup; may be nonsignificant','top5 ordinary or matched q (matched p breaks q ties); exploratory display');picked.append(view)
 summary=pd.DataFrame(summaries);summary.to_csv(O/'feature_summary.csv',index=False);topdf=pd.concat(tops,ignore_index=True);topdf.to_csv(O/'top40_gene_contributions.csv',index=False);pd.DataFrame(gate).rename(columns={'single_gene_inputs':'changed_inputs'}).to_csv(O/'TopK_gate_diagnostics.csv',index=False)
 candidates=pd.concat(picked).drop_duplicates(['feature','analysis','term']);candidates.to_csv(O/'candidate_pathway_summary.csv',index=False)
 links=[]
 for row in candidates.to_dict('records'):
  g=combined[row['feature']];fid=row['feature']
  for gene in str(row['overlap_genes']).split(';'):
   if not gene or gene=='nan':continue
   v=g.loc[gene];links.append({key:row[key] for key in ['feature','analysis','collection','term','GO_ID','q_ora_global','q_matched_global','q_matched_exact_global','display_reason']}|{'gene':gene,'train_rank':int(v.train_rank),'train_contribution':v[f'sae_F{fid}_train'],'test_contribution':v[f'sae_F{fid}_test'],'positive_TRAIN_donors':int(v[f'sae_F{fid}_positive_donors']),'input_TRAIN_prototypes':int(v.input_prototypes_train)})
 pd.DataFrame(links).to_csv(O/'candidate_pathway_genes.csv',index=False)
 old=pd.read_csv(O.parent/'sae_gene_attribution_v2/gene_evidence_combined.csv');combined[160]=old.set_index('gene');overlaps=[]
 for a,b in [(160,898),(160,1387),(898,1387)]:
  ga=combined[a];gb=combined[b]
  for mode in ['all','non_RPL_RPS']:
   aa=ga if mode=='all' else ga[~ga.index.str.startswith(('RPL','RPS'))];bb=gb if mode=='all' else gb[~gb.index.str.startswith(('RPL','RPS'))];ta=set(aa[aa[f'sae_F{a}_train']>0].sort_values(f'sae_F{a}_train',ascending=False).head(40).index);tb=set(bb[bb[f'sae_F{b}_train']>0].sort_values(f'sae_F{b}_train',ascending=False).head(40).index);common=aa.index.intersection(bb.index)
   overlaps.append({'feature_a':a,'feature_b':b,'list':mode,'top40_overlap':len(ta&tb),'top40_jaccard':len(ta&tb)/len(ta|tb),'shared_genes':';'.join(sorted(ta&tb)),'common_background_genes':len(common),'rank_Spearman_common':float(spearmanr(aa.loc[common,f'sae_F{a}_train'],bb.loc[common,f'sae_F{b}_train']).statistic),'scope':'different feature-selected prototypes/references; descriptive, not paired feature effect'})
 pd.DataFrame(overlaps).to_csv(O/'feature_gene_overlap.csv',index=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(2,2,figsize=(14,11))
 for j,fid in enumerate([898,1387]):
  g=combined[fid].reset_index();cols=[f'sae_F{fid}_train',f'sae_F{fid}_test'];
  for i,mode in enumerate(['All genes','Non-RPL/RPS']):
   a=g if i==0 else g[~g.gene.str.startswith(('RPL','RPS'))];a=a[a[cols[0]]>0].sort_values(cols[0],ascending=False).head(12).iloc[::-1];ax=axs[i,j];y=np.arange(len(a));ax.barh(y-.18,a[cols[0]],height=.35,label='TRAIN donor-balanced',color='#247c7b');ax.barh(y+.18,a[cols[1]],height=.35,label='Historical TEST',color='#b58d5b');ax.set_yticks(y,a.gene);ax.axvline(0,c='#999',lw=.6);ax.set_title(f'F{fid} | {mode} | TRAIN top12');ax.set_xlabel('Activation drop after single-gene replacement');ax.legend(fontsize=8)
   for yy,vv in zip(y,a[cols[1]]):
    if not np.isfinite(vv):ax.text(0,yy+.18,' NA',va='center',fontsize=8,color='#777')
 fig.tight_layout();fig.savefig(O/'gene_contributions.png',dpi=160);plt.close(fig)
 a898=paths[(paths.feature==898)&(paths.analysis=='nonribosomal_top40')&(paths.term=='GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION')].iloc[0];a1387=paths[(paths.feature==1387)&(paths.analysis=='nonribosomal_top40')&(paths.term=='REACTOME_INTERFERON_GAMMA_SIGNALING')].iloc[0]
 count=(paths.q_matched_global<.05).sum();ordinary=(paths.q_ora_global<.05).sum();exact=(paths.q_matched_exact_global<.05).sum();audit='pending';ap=O/'EXPERIMENT_AUDIT.json'
 if ap.exists():
  z=json.loads(ap.read_text());audit=str(z.get('verdict',z.get('overall_verdict','see audit')))
 text=f'''F898 / F1387 B细胞基因贡献与GO/pathway分析
结果要点：F898的前列贡献包括YBX1、TPT1、NACA；去RPL/RPS后，抗原加工与呈递条目重叠{int(a898.overlap)}个基因（{a898.overlap_genes}），普通全局q={a898.q_ora_global:.4g}，匹配全局q={a898.q_matched_global:.4g}。F1387的前列贡献包括TMSB4X、BTG1、NPM1、SH3BGRL3；去RPL/RPS后，Reactome IFN-γ信号条目重叠{int(a1387.overlap)}个基因（{a1387.overlap_genes}），普通q={a1387.q_ora_global:.4g}，匹配q={a1387.q_matched_global:.4g}。不能把前列贡献基因全部归入某一通路，需看实际成员对应表。HLA-DPA1（F898）和HLA-DPB1（F1387）训练与测试贡献方向不一致，不能列为已验证稳定核心。
已完成两个冻结特征，TRAIN每供体2个高激活B细胞，共8个；TEST单供体4个；每个原型最多3个同供体同类型零激活近邻参照。原型按各自F选择，不能将两组当作配对实验。
贡献=原输入激活−替换单个基因输入值后的激活；固定tokens/padding，替换成参照的离散表达bin，不重新归一化raw counts。先平均参照，再平均原型，再平均供体。未token覆盖基因在该原型贡献为0，另报覆盖和条件均值。高排名不等于跨供体稳定，需同时查看正向供体数与原型覆盖次数。最近同类型零激活参照不等于技术匹配，原型/参照的检出基因数差异见特征汇总，可能影响贡献。
训练贡献决定基因排序；测试只检验冻结训练排序。组合替换的测试原型是预选高激活细胞，且参照重复使用，结果不是供体级推断或真实细胞干预。
GO BP / Reactome / Hallmark使用MSigDB2025.1.Hs，普通ORA与10000次随机基因集检验。每特征两组：全部正向top40；去RPL/RPS前缀后重选正向top40，背景同步移除该前缀。背景是该特征真正TRAIN输入覆盖的基因，未注释基因保留。仅检验背景成员5..500的条目。
两个特征×两个列表×三个库合并BH校正，共{len(paths)}项。普通ORA全局FDR<0.05：{ordinary}项；主要匹配随机检验：{count}项；预先规定的精确条件分布敏感性分析：{exact}项。匹配基于941个TRAIN B细胞的平均raw count/检出率五分位及原型覆盖分箱；匹配仅在分箱层面，可能残留连续指标差异。MonteCarlo p=(1+超越次数)/10001，非零最小分辨率1/10001；精确条件检验仍依赖同一分箱可交换性假设。
普通ORA显著但匹配随机未显著的条目只作候选注释。明确的免疫条目也保留未显著结果，不凭基因名称补入不属于该通路的基因。去RPL/RPS只是一项前缀敏感性分析，不能移除全部翻译相关基因。本轮是ORA，不是GSEA。感染相关Reactome条目可能由共享宿主基因带来，不能据此推断样本感染。
另提供同一RNA内B细胞表达与F激活的偏相关，控制总counts、检出基因数和供体；它不是输入贡献，也不是独立验证，不用于本轮通路选择。
同一RNA的表型标签/特征读出和输入敏感性，不构成独立生物学机制验证。TopK门控与encoder贡献分别保存；原embedding/PCA只是在这些SAE原型上的诊断坐标，不代表公平的整体可解释性比较。一个历史TEST供体，已多次探索；无独立盲测，不能从正向贡献推断通路激活或因果机制。
最终审查：{audit}（same-family/provisional）。
'''
 (O/'RESULTS_ZH.txt').write_text(text)
 body='<h1>F898 与 F1387：基因贡献和通路</h1><p class="lead">直接检查两个 B 细胞特征依赖哪些输入基因，以及这些基因对应哪些 GO / pathway 注释。</p><section><h2>结果要点</h2><p>'+html.escape(text.split('结果要点：')[1].split('已完成两个')[0])+'</p></section><section><h2>特征与迁移</h2>'+table(summary)+'</section><section><h2>主要基因贡献</h2><img src="data:image/png;base64,'+base64.b64encode((O/'gene_contributions.png').read_bytes()).decode()+'"><p>TRAIN top12；测试贡献缺失表示输入未覆盖；RNA偏相关缺失表示未纳入相关分析或不可计算。正贡献表示替换后特征下降，不能解释为真实细胞内因果作用。</p></section>'
 for fid in [898,1387]:
  g=combined[fid].reset_index();show=g.head(25)[['gene','train_rank',f'sae_F{fid}_train',f'sae_F{fid}_test',f'sae_F{fid}_positive_donors','input_prototypes_train','RNA_partial_r_train','RNA_partial_r_test']];body+=f'<section><h2>F{fid} 前25基因</h2>'+table(show)+'</section>'
  subset=paths[paths.feature==fid].sort_values(['q_matched_global','p_matched','q_ora_global','term']);body+=f'<section><h2>F{fid} 通路：按匹配随机全局q排序前15</h2>'+table(subset.head(15)[['analysis','collection','term','GO_ID','overlap','q_ora_global','q_matched_global','q_matched_exact_global','overlap_genes']])+'</section>'
  de=pd.read_csv(O/f'F{fid}/deletion_summary.csv');body+=f'<section><h2>F{fid} 测试原型40基因组合替换</h2>'+table(de[(de.score==f'sae_F{fid}')&(de.requested_genes==40)][['method','baseline','activation','drop']])+'<p>按训练分数排序，再在每个测试输入里取可替换基因；这不是把一份全局40基因清单强制加入所有细胞。不同方法的raw drop仅在同一特征内比较；仅一个测试供体。随机对照针对SAE全部基因排序匹配，非RPL/RPS排序没有专用匹配随机对照，不能据此宣称非核糖体特异性。</p></section>'
 body+='<section><h2>与F160的关系</h2>'+table(pd.DataFrame(overlaps))+'<p>各特征原型/参照不同，仅作描述；F160的旧富集校正家族也不同，不能直接比较q值强弱。</p></section><section><h2>方法、检验与限制</h2><p>'+html.escape(text).replace('\n','<br>')+'</p>'+table(pd.read_csv(O/'pathway_summary.csv'))+'</section>'
 body+='<section><h2>文件</h2><ul>'+''.join(f'<li><a href="{p.relative_to(O)}">{p.relative_to(O)}</a></li>' for p in sorted(O.rglob('*')) if p.is_file() and p.suffix in ['.csv','.json','.md','.txt'])+'</ul><p><a href="https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/">官方MSigDB基因集</a></p></section>'
 css='body{font-family:-apple-system,sans-serif;background:#f4f6f7;color:#273b40;line-height:1.7}main{max-width:1250px;margin:35px auto;padding:0 20px}section{background:white;border-radius:12px;margin:20px 0;padding:24px;overflow:auto}h1{font-size:29px}h2{font-size:21px}.lead{font-size:19px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid #dce5e5;padding:8px;text-align:left}th{background:#edf3f3}img{width:100%;height:auto}a{color:#247c7b}'
 (O/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B细胞F898和F1387分析</title><style>'+css+'</style><main>'+body+'</main></html>')
 js(O/'EXPERIMENT_TRACKER.json',{'attribution':'complete','heldout_deletion':'complete','GO_pathways':'complete','gene_mapping':'complete','attribution_verification':{str(f):json.loads((O/f'F{f}/verification.json').read_text())['status'] for f in [898,1387]},'pathway_verification':json.loads((O/'pathway_verification.json').read_text())['status'],'scientific_audit':audit,'scope':'2frozen B-cell features, historical selected prototypes, no biological causal validation'})
 print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
