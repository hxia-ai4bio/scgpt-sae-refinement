"""Assemble frozen-TRAIN gene profiles and pathway evidence; no new selection tests."""
from pathlib import Path
import os,json,html,base64,itertools
import numpy as np,pandas as pd
from scipy.stats import spearmanr
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_B_remaining';FEATURES=[137,370,146,1373,1125,1409];ALL=[160,898,1387]+FEATURES
os.environ.setdefault('MPLCONFIGDIR',str(W.parent/'sae_gene_attribution_v2/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(d):return d.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}',na_rep='—')
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def feature_dir(fid):
 if fid==160:return O.parent/'sae_gene_attribution_v2'
 if fid in [898,1387]:return O.parent/'sae_B898_1387'/f'F{fid}'
 return O/f'F{fid}'
def read_genes(fid):
 d=pd.read_csv(feature_dir(fid)/'gene_evidence_combined.csv').sort_values([f'sae_F{fid}_train','gene'],ascending=[False,True]).reset_index(drop=True)
 if 'train_rank' not in d:d['train_rank']=np.arange(1,len(d)+1)
 return d

def main():
 paths=pd.read_csv(O/'all_pathway_results.csv');readout=pd.read_csv(O/'feature_readouts.csv');genes={fid:read_genes(fid) for fid in ALL};selected=pd.read_csv(O/'selected_features.csv').set_index('feature');summary=[];tops=[];gates=[];candidates=[];deletion=[]
 for fid in FEATURES:
  FO=feature_dir(fid);FW=W/f'F{fid}';g=genes[fid];score=f'sae_F{fid}_train';test=f'sae_F{fid}_test'
  for role in ['train','test']:
   g=g.merge(pd.read_csv(FO/f'expression_association_{role}.csv').rename(columns={'partial_r':f'RNA_partial_r_{role}'}),on='gene',how='left')
  genes[fid]=g
  for mode,a in [('all_input_top40',g),('nonribosomal_top40',g[~g.gene.str.startswith(('RPL','RPS'))])]:
   top=a[a[score]>0].head(40);tops.append(pd.DataFrame({'feature':fid,'analysis':mode,'gene':top.gene,'train_rank':top.train_rank,'train_contribution':top[score],'test_contribution':top[test],'positive_TRAIN_donors':top[f'sae_F{fid}_positive_donors'],'input_TRAIN_donors':top.input_donors,'input_TRAIN_prototypes':top.input_prototypes_train,'conditional_train_mean':top[f'sae_F{fid}_conditional_mean'],'train_donor_SD':top[f'sae_F{fid}_donor_sd'],'RNA_partial_r_train':top.RNA_partial_r_train,'RNA_partial_r_test':top.RNA_partial_r_test}))
  tr40=g[g[score]>0].head(40);present=tr40[test].notna();don=pd.read_csv(FO/'train_donor_consistency.csv');ref=pd.read_csv(FO/'reference_sensitivity.csv');qc=pd.read_csv(FO/'pair_qc_donor_summary.csv').groupby('role')[['prototype_detected_genes','reference_detected_genes']].mean();r=readout[readout.feature==fid];pb=r[r.domain=='pbmc'];non_all=g[(~g.gene.str.startswith(('RPL','RPS')))&(g[score]>0)].head(40);non=non_all.head(8);consistent=non_all[(non_all[f'sae_F{fid}_positive_donors']>=3)&(non_all.input_donors>=3)&(non_all[test]>0)]
  summary.append({'feature':fid,'scope_group':selected.loc[fid,'scope_group'],'atlas_max_abs_technical_partial':selected.loc[fid,'max_abs_technical_partial'],'mean_TRAIN_donor_AUC':r[r.role=='train'].auc_B_vs_other.mean(),'min_TRAIN_donor_AUC':r[r.role=='train'].auc_B_vs_other.min(),'historical_TEST_AUC':r[r.role=='test'].auc_B_vs_other.iloc[0],'PBMC_batch1_AUC':pb[pb.group=='Batch1_3k'].auc_B_vs_other.iloc[0],'PBMC_batch2_AUC':pb[pb.group=='Batch2_68k_r'].auc_B_vs_other.iloc[0],'TRAIN_covered_genes':len(g),'TRAIN_top40_RPL_RPS':int(tr40.gene.str.startswith(('RPL','RPS')).sum()),'TRAIN_top40_TEST_covered':int(present.sum()),'TRAIN_top40_TEST_positive':int((tr40.loc[present,test]>0).sum()),'TRAIN_donor_pair_Spearman_median':don[don.score==f'sae_F{fid}'].spearman.median(),'TRAIN_reference_top20_Jaccard_median':ref[(ref.score==f'sae_F{fid}')&(ref.role=='train')].top20_jaccard.median(),'TRAIN_prototype_detected_genes':qc.loc['train','prototype_detected_genes'],'TRAIN_reference_detected_genes':qc.loc['train','reference_detected_genes'],'TEST_prototype_detected_genes':qc.loc['test','prototype_detected_genes'],'TEST_reference_detected_genes':qc.loc['test','reference_detected_genes'],'TRAIN_non_RPL_RPS_top8':';'.join(non.gene),'descriptive_same_direction_non_RPL_top8':';'.join(consistent.head(8).gene),'descriptive_same_direction_non_RPL_n':len(consistent)})
  for p in sorted((FW/'cache').glob('pair_*.npz')):
   z=np.load(p);ii=z['changed'];after=z['baseline'][None,:]-z['effects'];gates.append({'feature':fid,'kind':'single_gene','pair':p.stem,'changed_inputs':len(ii),'SAE_encoder_unequal':int((np.abs(z['effects'][ii,0]-z['effects'][ii,1])>3e-5).sum()),'TopK_exits':int(((after[ii,0]==0)&(np.abs(after[ii,1])>1e-8)).sum()),'max_effect_difference':float(np.abs(z['effects'][:,0]-z['effects'][:,1]).max())})
  for p in sorted((FW/'deletion').glob('pair_*.npz')):
   z=np.load(p);ii=z['actual']>0;ss=z['scores'];gates.append({'feature':fid,'kind':'combination','pair':p.stem,'changed_inputs':int(ii.sum()),'SAE_encoder_unequal':int((np.abs(ss[ii,0]-ss[ii,1])>3e-5).sum()),'TopK_exits':int(((ss[ii,0]==0)&(np.abs(ss[ii,1])>1e-8)).sum()),'max_effect_difference':float(np.abs(ss[:,0]-ss[:,1]).max())})
  de=pd.read_csv(FO/'deletion_summary.csv');de=de[(de.score==f'sae_F{fid}')&(de.requested_genes==40)].copy();de['feature']=fid;deletion.append(de)
  for analysis,part in paths[paths.feature==fid].groupby('analysis'):
   chosen=set(part.sort_values(['q_ora_global','p_ora','term']).head(5).term)|set(part.sort_values(['q_matched_global','p_matched','p_ora','term']).head(5).term)
   immune=['GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION','GOBP_B_CELL_ACTIVATION','GOBP_B_CELL_RECEPTOR_SIGNALING_PATHWAY','REACTOME_INTERFERON_GAMMA_SIGNALING','HALLMARK_TNFA_SIGNALING_VIA_NFKB']
   chosen.update(t for t in immune if t in set(part.term));part=part[part.term.isin(chosen)].copy();part['display_reason']=np.where(part.term.isin(immune),'predefined immune annotation lookup; may be nonsignificant','top5 ordinary or matched q; p breaks ties; exploratory display');candidates.append(part)
 summary=pd.DataFrame(summary);summary.to_csv(O/'feature_summary.csv',index=False);tops=pd.concat(tops,ignore_index=True);tops['descriptive_direction_consistent']=(tops.positive_TRAIN_donors>=3)&(tops.input_TRAIN_donors>=3)&(tops.test_contribution>0);tops.to_csv(O/'top40_gene_contributions.csv',index=False);tops[(tops.analysis=='nonribosomal_top40')&tops.descriptive_direction_consistent].to_csv(O/'descriptive_followup_genes.csv',index=False);gates=pd.DataFrame(gates);gates.to_csv(O/'TopK_gate_diagnostics.csv',index=False);deletion=pd.concat(deletion,ignore_index=True);deletion.to_csv(O/'test_40gene_replacement.csv',index=False)
 candidates=pd.concat(candidates).drop_duplicates(['feature','analysis','term']);candidates.to_csv(O/'candidate_pathway_summary.csv',index=False);links=[]
 for row in candidates.to_dict('records'):
  fid=row['feature'];g=genes[fid].set_index('gene')
  for gene in str(row['overlap_genes']).split(';'):
   if not gene or gene=='nan':continue
   v=g.loc[gene];links.append({k:row[k] for k in ['feature','analysis','collection','term','GO_ID','q_ora_global','q_matched_global','q_matched_exact_global','display_reason']}|{'gene':gene,'train_rank':int(v.train_rank),'train_contribution':v[f'sae_F{fid}_train'],'test_contribution':v[f'sae_F{fid}_test'],'positive_TRAIN_donors':int(v[f'sae_F{fid}_positive_donors']),'input_TRAIN_prototypes':int(v.input_prototypes_train)})
 pd.DataFrame(links).to_csv(O/'candidate_pathway_genes.csv',index=False)
 overlap=[];alltop=[]
 for fid,g in genes.items():
  for mode,a in [('all',g),('non_RPL_RPS',g[~g.gene.str.startswith(('RPL','RPS'))])]:
   for row in a[a[f'sae_F{fid}_train']>0].head(40).to_dict('records'):alltop.append({'feature':fid,'list':mode,'gene':row['gene'],'train_contribution':row[f'sae_F{fid}_train'],'test_contribution':row[f'sae_F{fid}_test'],'positive_TRAIN_donors':row[f'sae_F{fid}_positive_donors'],'input_TRAIN_prototypes':row.get('input_prototypes_train',np.nan)})
 for a,b in itertools.combinations(ALL,2):
  aa=genes[a].set_index('gene');bb=genes[b].set_index('gene')
  for mode in ['all','non_RPL_RPS']:
   ga=aa if mode=='all' else aa[~aa.index.str.startswith(('RPL','RPS'))];gb=bb if mode=='all' else bb[~bb.index.str.startswith(('RPL','RPS'))];ta=set(ga[ga[f'sae_F{a}_train']>0].head(40).index);tb=set(gb[gb[f'sae_F{b}_train']>0].head(40).index);common=ga.index.intersection(gb.index);overlap.append({'feature_a':a,'feature_b':b,'list':mode,'top40_overlap':len(ta&tb),'top40_Jaccard':len(ta&tb)/len(ta|tb),'common_background':len(common),'shared_genes':';'.join(sorted(ta&tb)),'Spearman_common_genes':float(spearmanr(ga.loc[common,f'sae_F{a}_train'],gb.loc[common,f'sae_F{b}_train']).statistic),'scope':'different feature-selected prototypes/references; descriptive only'})
 pd.DataFrame(overlap).to_csv(O/'all9_feature_gene_overlap.csv',index=False);pd.DataFrame(alltop).to_csv(O/'all9_top40_gene_contributions.csv',index=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(3,2,figsize=(14,16))
 for ax,fid in zip(axs.ravel(),FEATURES):
  g=genes[fid];a=g[(~g.gene.str.startswith(('RPL','RPS')))&(g[f'sae_F{fid}_train']>0)].head(12).iloc[::-1];y=np.arange(len(a));ax.barh(y-.18,a[f'sae_F{fid}_train'],height=.35,label='TRAIN donor-balanced',color='#247c7b');ax.barh(y+.18,a[f'sae_F{fid}_test'],height=.35,label='Historical TEST',color='#b58d5b');ax.set_yticks(y,a.gene);ax.axvline(0,c='#999',lw=.6);ax.set_title(f'F{fid} | TRAIN non-RPL/RPS top12');ax.set_xlabel('Activation drop after single-gene replacement');ax.legend(fontsize=8)
  for yy,v in zip(y,a[f'sae_F{fid}_test']):
   if not np.isfinite(v):ax.text(0,yy+.18,' NA',va='center',fontsize=8,color='#777')
 fig.tight_layout();fig.savefig(O/'gene_contributions.png',dpi=160);plt.close(fig)
 audit='pending';ap=O/'EXPERIMENT_AUDIT.json'
 if ap.exists():audit=str(json.loads(ap.read_text()).get('verdict','see audit'))
 n=len(paths);ordinary=int((paths.q_ora_global<.05).sum());matched=int((paths.q_matched_global<.05).sum());exact=int((paths.q_matched_exact_global<.05).sum());gate_summary=gates.groupby(['feature','kind'])[['changed_inputs','SAE_encoder_unequal','TopK_exits']].sum().reset_index()
 short=summary[['feature','mean_TRAIN_donor_AUC','min_TRAIN_donor_AUC','historical_TEST_AUC','TRAIN_non_RPL_RPS_top8','TRAIN_top40_TEST_covered','TRAIN_top40_TEST_positive','TRAIN_donor_pair_Spearman_median']]
 text=f'''其余高AUROC B细胞特征的基因贡献分析
范围：全部合格特征中TRAIN供体平均B-vs-other AUROC>=0.70，排除此前已完成160/898/1387，得到137/370/146/1373/1125/1409。后两者最差供体AUROC<0.70，单独标注为跨供体较弱。高AUROC是描述性筛选，不等于统计显著的生物学功能。
固定同一个历史K32、seed42模型；本轮不重训，也未检验这些编号的跨种子对应关系。每特征4个TRAIN供体各2个高激活B细胞、历史单TEST供体4个；各配最多3个同供体同类型零激活近邻。各特征选择自己的原型/参照，跨特征不是配对实验。原型/参照的QC差异、复用和覆盖稀疏性均保留。
输入贡献=原始激活−单基因输入值替换后的激活，固定原gene tokens和padding，取参照全细胞的离散bin；不是原始counts敲除或真实细胞干预；这里只测给定gene-token集合下的表达值敏感性，未测删除/加入gene-token身份本身的贡献。先等权参照，再原型，再供体；未进入某原型tokens的基因该原型贡献记0，另报覆盖与条件均值。高排名不等于稳定，需看正向供体数和覆盖次数；TEST缺失表示未覆盖，不是0。
生产MPS float32 batch16，CPU独立慢后端重放每对最大贡献与最大组合替换；数值/输入/来源hash核验独立保存。TopK实际得分与preTopK encoder另存，已发生门控变化时不能把全部效应解释为平滑编码变化，详见TopK_gate_diagnostics.csv。embedding/PCA只是同一批SAE原型上的诊断坐标，非全局可解释性优势基准。
基因排序仅由TRAIN决定。组合替换是在每个TEST原型中按训练排序选可替换的基因，不是对所有细胞强制同一个40基因清单；与TRAIN表达差异和10个联合分箱匹配随机顺序比较。非RPL/RPS组合排序没有专用随机对照。历史测试已多次探索，仅1供体，不能作盲测或供体级显著性推断。
GO BP / Reactome / Hallmark：MSigDB2025.1.Hs，标准ORA。各特征全部正向top40及移除RPL/RPS后重新选top40，背景同步移除前缀；背景限TRAIN实际输入覆盖的基因，未注释基因保留，条目背景成员5..500。每列表10000个随机基因集，联合匹配全部941个TRAIN B细胞中的平均raw count/检出率五分位及原型覆盖1-2/3-5/6-8分箱。仅分箱匹配，存在连续指标残余差异；选择基因仍可进入随机集。MC p=(超越数+1)/10001，精确条件超几何卷积单列作预定敏感性分析，消除MC误差但不消除分箱假设。
全局BH跨本轮全部6特征×2列表×3库，共{n}项；普通ORA显著{ordinary}，主匹配检验显著{matched}，精确敏感性显著{exact}。这是本轮统一家族；旧3特征保留历史校正，不直接比较q强弱。显著数为分析-条目行数，重叠GO/Reactome不是独立机制；感染条目也不证明样本感染。去RPL/RPS是前缀敏感性分析，不会移除所有翻译基因。
普通富集而匹配未通过的条目仅作候选注释，不推断通路活性、因果或独立功能模块。次要RNA偏相关在B细胞内控制logcounts/loggenes/供体，属于同RNA关联而非输入贡献或独立验证；它不用于选本轮通路。Kang缺少MT-前缀编码基因，无法用该QC指标评估线粒体比例；核编码的线粒体功能基因仍可存在。另列描述性后续候选：冻结TRAIN非RPL/RPS前40中，至少3个TRAIN供体正向且TEST均值正向；此为探索性展示，不改变任何基因/通路检验，也不是独立验证。
汇总：
{short.to_string(index=False)}
数值复核PASS；科研审查{audit}（same-family/provisional）。
'''
 (O/'RESULTS_ZH.txt').write_text(text)
 body='<h1>B细胞：其余6个高AUROC特征的基因分析</h1><p class="lead">固定既有模型，以TRAIN筛选和排序，保留测试反向、参照敏感性及匹配随机阴性结果。</p><section><h2>主要结果</h2>'+table(short)+f'<p>本轮{n}项：普通ORA FDR&lt;0.05为{ordinary}项；匹配背景为{matched}项；精确条件敏感性为{exact}项。普通富集不等于通过匹配背景的富集。</p></section><section><h2>贡献排名靠前的非RPL/RPS基因</h2><img src="data:image/png;base64,'+base64.b64encode((O/'gene_contributions.png').read_bytes()).decode()+'"><p>每特征TRAIN前12；各自原型不同，raw贡献量不直接比较特征优劣；NA表示测试未输入覆盖。</p></section>'
 for fid in FEATURES:
  g=genes[fid];show=g.head(25)[['gene','train_rank',f'sae_F{fid}_train',f'sae_F{fid}_test',f'sae_F{fid}_positive_donors','input_prototypes_train','RNA_partial_r_train','RNA_partial_r_test']];part=paths[(paths.feature==fid)&(paths.analysis=='nonribosomal_top40')];view=part.sort_values(['q_ora_global','p_ora','term']).head(8)
  body+=f'<section><h2>F{fid}：TRAIN贡献前25</h2>'+table(show)+'<p>测试贡献缺失是未输入覆盖；RNA偏相关缺失表示未纳入该分析或不可计算。正贡献供体数分母4，原型覆盖分母8。</p>'+f'<h2>F{fid}：去RPL/RPS后的普通ORA前8</h2>'+table(view[['collection','term','GO_ID','overlap','q_ora_global','p_matched','q_matched_global','q_matched_exact_global','overlap_genes']])+'</section>'
 body+='<section><h2>TEST每原型按训练排序取40个可替换基因</h2>'+table(deletion[['feature','method','baseline','activation','drop']])+'</section><section><h2>TopK门控诊断</h2>'+table(gate_summary)+'</section><section><h2>QC与稳定性完整统计</h2>'+table(summary)+'</section><section><h2>方法和解释边界</h2><p>'+html.escape(text).replace('\n','<br>')+'</p></section><section><h2>来源与下载</h2><p><a href="https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/">官方MSigDB基因集</a>；全部成员和源文件hash保留。all9文件为不同特征所选原型间的描述比较，旧3特征未重算归因。</p><ul>'+''.join(f'<li><a href="{p.relative_to(O)}">{p.relative_to(O)}</a></li>' for p in sorted(O.rglob('*')) if p.is_file() and p.suffix in ['.csv','.json','.md','.txt'])+'</ul></section>'
 css='body{font-family:-apple-system,sans-serif;background:#f4f6f7;color:#273b40;line-height:1.7}main{max-width:1250px;margin:35px auto;padding:0 20px}section{background:white;border-radius:12px;margin:20px 0;padding:24px;overflow:auto}h1{font-size:29px}h2{font-size:21px}.lead{font-size:19px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid #dce5e5;padding:8px;text-align:left}th{background:#edf3f3}img{width:100%;height:auto}a{color:#247c7b}'
 (O/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B细胞其余高AUROC特征基因分析</title><style>'+css+'</style><main>'+body+'</main></html>')
 save(O/'EXPERIMENT_TRACKER.json',{'status':'complete' if audit!='pending' else 'awaiting_review','features':FEATURES,'attribution':'complete','deletion':'complete','pathways':'complete','source_numeric_verification':'PASS','scientific_review':audit,'scope':'frozen B-cell prototypes; input sensitivity/annotation association, not causal biology'})
 print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
