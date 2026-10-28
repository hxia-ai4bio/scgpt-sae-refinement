from pathlib import Path
import os,json,html,base64,textwrap
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_pathways_v3';os.environ.setdefault('MPLCONFIGDIR',str(W.parent/'sae_gene_attribution_v2/matplotlib_cache'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def table(d):return d.to_html(index=False,float_format=lambda x:f'{x:.4g}',na_rep='未覆盖',border=0)
def main():
 d=pd.read_csv(O/'all_pathway_results.csv');summary=pd.read_csv(O/'summary.csv');genes=pd.read_csv(O/'gene_background_metrics.csv');genes['train_rank']=np.arange(1,len(genes)+1)
 chosen=[('all_input_top40','GOBP_TRANSLATION','翻译'),('nonribosomal_top40','GOBP_MRNA_CATABOLIC_PROCESS','mRNA分解'),('nonribosomal_top40','GOBP_POST_TRANSCRIPTIONAL_REGULATION_OF_GENE_EXPRESSION','转录后调控'),('nonribosomal_top40','GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION_OF_ENDOGENOUS_ANTIGEN','内源性抗原加工与呈递'),('nonribosomal_top40','HALLMARK_INTERFERON_GAMMA_RESPONSE','IFN-γ响应')]
 goids=pd.read_csv(O/'go_term_metadata.csv').set_index('term').GO_ID.to_dict()
 rows=[];gr=[]
 for analysis,term,label in chosen:
  v=d[(d.analysis==analysis)&(d.term==term)];assert len(v)==1;v=v.iloc[0];hit=str(v.overlap_genes).split(';') if pd.notna(v.overlap_genes) else []
  rows.append({'分析':analysis,'条目':label,'数据库条目':term,'GO编号':goids.get(term,'不适用'),'重叠基因数':v.overlap,'普通ORA全局q':v.q_ora_global,'匹配随机全局q':v.q_matched_global,'对应入选基因':'、'.join(hit),'解释':'候选注释；未通过匹配随机FDR','url':v.url})
  for g in hit:
   a=genes[genes.gene==g].iloc[0];gr.append({'条目':label,'gene':g,'TRAIN排名':int(a.train_rank),'训练归因':a.sae_F160_train,'测试归因':a.sae_F160_test,'正向训练供体数':int(a.sae_F160_positive_donors),'输入覆盖原型数':int(a.input_prototypes_train),'普通ORA全局q':v.q_ora_global,'匹配随机全局q':v.q_matched_global})
 candidate=pd.DataFrame(rows);candidate.to_csv(O/'candidate_pathway_summary.csv',index=False);gene_table=pd.DataFrame(gr);gene_table.to_csv(O/'candidate_pathway_genes.csv',index=False)
 # No assertion of significance for hand-highlighted annotation cases.
 focusgenes=['CD74','HLA-A','HLA-B','CD79A','YBX1','NPM1','PABPC1','SERBP1','EIF3E']
 focus=genes[genes.gene.isin(focusgenes)][['gene','train_rank','sae_F160_train','sae_F160_test','sae_F160_positive_donors','input_prototypes_train','partial_r']].copy();focus.to_csv(O/'gene_followup_candidates.csv',index=False)
 view=pd.concat([v.sort_values('q_ora_global').head(6) for _,v in d.groupby('analysis')]).iloc[::-1]
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(11,8));y=np.arange(len(view));ax.barh(y-.18,-np.log10(np.maximum(view.q_ora_global,1e-300)),height=.34,label='Ordinary ORA, global BH q',color='#247c7b');ax.barh(y+.18,-np.log10(np.maximum(view.q_matched_global,1e-300)),height=.34,label='Matched gene null, global BH q',color='#b58d5b');ax.axvline(-np.log10(.05),ls='--',c='#8d4b4b',label='q = 0.05')
 ax.set_yticks(y,['\n'.join(textwrap.wrap(('All | ' if row.analysis=='all_input_top40' else 'Non-RPL/RPS | ')+row.term.replace('GOBP_','GO: ').replace('REACTOME_','Reactome: ').replace('_',' '),48)) for row in view.itertuples()]);ax.set_xlabel('-log10(global adjusted q)');ax.set_title('F160 pathway overrepresentation | top6 ordinary-ORA terms per analysis\nAll matched-null terms fail global FDR 0.05');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(O/'pathway_enrichment.png',dpi=160);plt.close(fig)
 verified=json.loads((O/'verification.json').read_text());audit='待审查'
 if (O/'EXPERIMENT_AUDIT.json').exists():
  a=json.loads((O/'EXPERIMENT_AUDIT.json').read_text());audit=str(a.get('verdict',a.get('overall_verdict','见审计文件')))
 completion=json.loads((O/'completion.json').read_text());terms=int(len(d));ordinary=int((d.q_ora_global<.05).sum());matched=int((d.q_matched_global<.05).sum())
 text=f'''F160 GO / pathway分析（2026-10-18）

已完成：GO Biological Process、Reactome、Hallmark（MSigDB2025.1.Hs）。共{terms}个分析-条目检验；普通ORA全局FDR<0.05共{ordinary}项，联合匹配随机对照全局FDR<0.05共{matched}项，最小匹配q={d.q_matched_global.min():.4f}。
本轮是超几何过度代表分析ORA，不是GSEA。原始基因排序来自刚完成的F160多参照TRAIN归因。两组预定分析：全部正向前40；移除RPL/RPS前缀后重新选正向前40。第二组的背景也同步去掉RPL/RPS。背景只使用训练原型真正输入覆盖的1838个基因，而非全基因组；未注释基因保留在背景。
每组10000个随机基因集，联合匹配全部941个TRAIN B细胞中的平均raw count/检出率五分位，以及原型输入覆盖率分箱；分箱匹配不是数值完全相等，检出率仍存在残余差异；一组分析中 JUN 所在层只有一个候选基因，因此在该组随机集中固定出现。两个分析和三个数据库合并校正，共6462项；匹配p最小分辨率1/10001，零超越不是p=0。

候选条目与对应基因（全部未通过匹配随机全局FDR）：
{candidate.drop(columns=['url']).to_string(index=False)}

可优先核查的基因及原归因：
{focus.to_string(index=False,na_rep='未覆盖')}

解读：完整前40主要反映翻译/核糖体条目；移除RPL/RPS后，普通ORA出现mRNA分解和转录后调控线索。CD74/HLA-A/HLA-B可映射到抗原加工呈递，CD74/HLA-A/HLA-B/PSME2/BTG1可映射到IFN-γ响应；两项均未通过普通ORA全局FDR，更未通过匹配随机检验。CD79A虽为后续关注基因，不能将其添加到不包含它的上述通路重叠基因中。
严格结论：本轮获得了对应基因和候选生物过程注释，没有发现通过当前匹配随机检验及全局FDR的通路。不能称F160已被证明编码某一独立功能程序，也不能从正向归因推断通路被激活。
限制：背景是被模型输入覆盖的有限基因；匹配检验依赖分箱选择和有限置换分辨率；同一基因出现在重叠GO/Reactome条目中不代表多个独立机制。移除RPL/RPS只是前缀规则，仍可能保留其他翻译相关基因。感染相关Reactome条目的出现不证明样本感染。归因仍受上一轮原型/参照表达复杂度差异、4个TRAIN供体及单个历史TEST供体限制。TEST仅展示冻结训练候选基因的效应，不另行选通路或作为独立机制验证。
数值核验：{verified['status']}；科研审查：{audit}（same-family/provisional）。
'''
 (O/'RESULTS_ZH.txt').write_text(text)
 body='<h1>F160：GO 与通路对应基因</h1><p class="lead">有可追踪的生物过程注释，但匹配表达量、检出率与输入覆盖率后，没有条目通过全局 FDR 0.05。</p>'
 body+='<section><h2>做了哪些检验</h2><p>'+html.escape(text[text.index('已完成：'):text.index('候选条目')]).replace('\n','<br>')+'</p>'+table(summary)+'</section>'
 body+='<section><h2>普通富集与匹配对照</h2><img src="data:image/png;base64,'+base64.b64encode((O/'pathway_enrichment.png').read_bytes()).decode()+'"></section>'
 body+='<section><h2>对应的基因是什么</h2><p>以下是便于后续核查的候选注释，不是已确认功能模块。完整检验结果均保留。</p>'+table(candidate.drop(columns=['url']))+'</section>'
 body+='<section><h2>重点基因的归因证据</h2><p>TRAIN排名基于原1838基因；测试缺失表示未输入覆盖，不能当作零效应。</p>'+table(focus)+'</section>'
 body+='<section><h2>可以得出什么结论</h2><p>'+html.escape(text[text.index('解读：'):]).replace('\n','<br>')+'</p></section>'
 body+='<section><h2>数据与来源</h2><p><a href="https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/">官方MSigDB2025.1.Hs基因集</a>；具体文件来源和SHA256见gene_set_provenance.json。</p><ul>'+''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(O.iterdir()) if p.suffix in ['.csv','.json','.txt','.md'])+'</ul></section>'
 css='body{font-family:-apple-system,sans-serif;background:#f4f6f7;color:#273b40;line-height:1.7}main{max-width:1150px;margin:35px auto;padding:0 20px}section{background:white;border-radius:12px;margin:20px 0;padding:24px;overflow:auto}h1{font-size:30px}h2{font-size:21px}.lead{font-size:19px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid #dce5e5;padding:8px;text-align:left}th{background:#edf3f3}img{width:100%;height:auto}a{color:#247c7b}'
 (O/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F160通路分析</title><style>'+css+'</style><main>'+body+'</main></html>')
 (O/'EXPERIMENT_TRACKER.json').write_text(json.dumps({'status':'complete','ORA':'complete','matched_null':'complete','gene_mapping':'complete','verification':verified['status'],'semantic_review':audit,'scope':'positive attribution ORA, two prespecified gene lists; no GSEA/model inference'},indent=2))
 print('REPORT_WRITTEN',O/'REPORT.html')
if __name__=='__main__':main()
