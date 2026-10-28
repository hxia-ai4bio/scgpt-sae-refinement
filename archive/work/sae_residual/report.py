"""Complete descriptive paired reporting; no model selection from TEST metrics."""
from pathlib import Path
import json,html,base64,re,os
import numpy as np,pandas as pd
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/sae_residual'
ARMS=['identity','gate','rotated_gate','lowrank','mlp','gate_supcon','rotated_gate_supcon','mlp_supcon']
CLASS_NAMES=['B cells','CD14+ Monocytes','CD4 T cells','CD8 T cells','Dendritic cells','FCGR3A+ Monocytes','Megakaryocytes','NK cells']
LABELS={'identity':'Original embedding','gate':'SAE gate','rotated_gate':'Rotated gate','lowrank':'SAE low-rank','mlp':'Dense MLP','gate_supcon':'SAE gate + SupCon','rotated_gate_supcon':'Rotated gate + SupCon','mlp_supcon':'Dense MLP + SupCon'}
def table(x):return x.to_html(index=False,border=0,float_format=lambda a:f'{a:.4f}',na_rep='—')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 complete=json.loads((O/'completion.json').read_text());assert complete['status']=='complete';verify=json.loads((O/'verification.json').read_text());assert verify['status']=='PASS';rows=[];classes=[]
 for p in sorted(O.glob('fold_*/seed_*/*/result.json')):
  r=json.loads(p.read_text())
  for role in ['valid','test','pbmc']:
   for readout in ['ridge','knn','trained_head']:
    m=r[role][readout];rows.append({'fold':r['fold'],'test_donor':r['test_donor'],'seed':r['seed'],'arm':r['arm'],'role':role,'readout':readout,'macro_f1':m['macro_f1'],'supported_macro_f1':m['supported_macro_f1'],'accuracy':m['accuracy'],'identity_selected':r['identity_selected'],'beta':r['selected_beta'],'ridge_alpha':r['probe']['alpha'],'residual_relative_norm':r[role]['residual_relative_norm_mean'],'residual_relative_norm_max':r[role]['residual_relative_norm_max'],'original_cosine':r[role]['original_cosine_mean']})
    for c,(score,support) in enumerate(zip(m['per_class_f1'],m['support'])):classes.append({'fold':r['fold'],'test_donor':r['test_donor'],'seed':r['seed'],'arm':r['arm'],'role':role,'readout':readout,'class_index':c,'F1':score,'support':support})
 save(O/'CLASS_LABELS.json',{str(i):name for i,name in enumerate(CLASS_NAMES)})
 d=pd.DataFrame(rows);d.to_csv(O/'all_metrics.csv',index=False);cl=pd.DataFrame(classes);cl['class_name']=cl.class_index.map(dict(enumerate(CLASS_NAMES)));cl.to_csv(O/'per_class_metrics.csv',index=False);base=d[d.arm=='identity'][['fold','seed','role','readout','macro_f1','supported_macro_f1']].rename(columns={'macro_f1':'baseline_macro_f1','supported_macro_f1':'baseline_supported_macro_f1'});d=d.merge(base,on=['fold','seed','role','readout'],validate='many_to_one');d['delta_pp']=100*(d.macro_f1-d.baseline_macro_f1);d['supported_delta_pp']=100*(d.supported_macro_f1-d.baseline_supported_macro_f1);d.to_csv(O/'paired_seed_metrics.csv',index=False)
 donor=d.groupby(['test_donor','arm','role','readout'],sort=False)[['macro_f1','supported_macro_f1','delta_pp','supported_delta_pp','residual_relative_norm','original_cosine']].mean().reset_index();donor.to_csv(O/'paired_donor_metrics.csv',index=False);summ=[];rng=np.random.default_rng(20261019)
 for arm in ARMS:
  a=donor[(donor.arm==arm)&(donor.role=='test')&(donor.readout=='ridge')];z=a.delta_pp.to_numpy();boot=z[rng.integers(0,len(z),size=(10000,len(z)))].mean(1);se=d[(d.arm==arm)&(d.role=='test')&(d.readout=='ridge')].groupby('seed').delta_pp.mean();p=d[(d.arm==arm)&(d.role=='pbmc')&(d.readout=='ridge')];k=donor[(donor.arm==arm)&(donor.role=='test')&(donor.readout=='knn')];identity=d[(d.arm==arm)&(d.role=='test')&(d.readout=='ridge')].identity_selected.sum();v=dict(arm=arm,TEST_Ridge_macro_F1=100*a.macro_f1.mean(),TEST_donor_SD=100*a.macro_f1.std(ddof=1),TEST_delta_pp=float(z.mean()),delta_bootstrap_low=float(np.quantile(boot,.025)),delta_bootstrap_high=float(np.quantile(boot,.975)),positive_donors=int((z>1e-10).sum()),positive_seed_means=int((se>1e-10).sum()),PBMC_Ridge_macro_F1=100*p.macro_f1.mean(),PBMC_delta_pp=p.delta_pp.mean(),TEST_kNN_delta_pp=k.delta_pp.mean(),identity_selected_runs=int(identity),selected_nonidentity_runs=int(18-identity),TEST_mean_relative_residual=a.residual_relative_norm.mean());summ.append(v)
 summary=pd.DataFrame(summ);summary.to_csv(O/'summary.csv',index=False)
 control_pairs=[('gate','rotated_gate'),('lowrank','mlp'),('gate_supcon','rotated_gate_supcon'),('gate_supcon','gate'),('rotated_gate_supcon','rotated_gate'),('mlp_supcon','mlp')];contrasts=[]
 for left,right in control_pairs:
  a=donor[(donor.arm==left)&(donor.role=='test')&(donor.readout=='ridge')].set_index('test_donor');b=donor[(donor.arm==right)&(donor.role=='test')&(donor.readout=='ridge')].set_index('test_donor')
  for dd in a.index:contrasts.append({'contrast':left+' minus '+right,'test_donor':dd,'delta_pp':100*(a.loc[dd,'macro_f1']-b.loc[dd,'macro_f1'])})
 pd.DataFrame(contrasts).to_csv(O/'planned_contrasts.csv',index=False)
 gate=summary.set_index('arm').loc['gate'];rot=summary.set_index('arm').loc['rotated_gate'];criteria={'mean_gain_at_least_1pp':bool(gate.TEST_delta_pp>=1.),'at_least_4_positive_donors':bool(gate.positive_donors>=4),'at_least_2_positive_seed_means':bool(gate.positive_seed_means>=2),'PBMC_loss_no_more_than_0_5pp':bool(gate.PBMC_delta_pp>=-.5),'better_than_rotated_control':bool(gate.TEST_delta_pp>rot.TEST_delta_pp+1e-10)};save(O/'continuation_checks.json',{'target':'predeclared primary SAE gate','criteria':criteria,'all_met':all(criteria.values()),'scope':'development continuation threshold, not significance or confirmation'})
 selections=[];hist=[];weights=[]
 for sp in sorted(W.glob('fold_*/seed_*/*/selection.json')):
  s=json.loads(sp.read_text());selections.append({k:s[k] for k in ['fold','test_donor','seed','arm','selected_beta','identity_selected','valid_macro_f1','identity_valid_macro_f1']})
  for t in s['trials']:
   meta=json.loads(Path(t['metadata']).read_text());hist.append({'fold':s['fold'],'seed':s['seed'],'arm':s['arm'],'beta':t['beta'],'best_epoch':meta['best_epoch'],'last_epoch':meta['last_epoch'],'seconds':meta['seconds'],'max_train_residual_ratio_squared':meta['max_train_residual_ratio_squared'],'head_valid_macro_f1':meta['valid_head_macro_f1'],'fresh_Ridge_valid_macro_f1':meta['fresh_ridge']['valid_macro_f1']})
 for p in sorted(O.glob('fold_*/seed_*/*/gate.npz')):
  r=json.loads((p.parent/'result.json').read_text());g=np.load(p)['g']
  for fid in np.argsort(-abs(g))[:40]:weights.append({'fold':r['fold'],'test_donor':r['test_donor'],'seed':r['seed'],'arm':r['arm'],'feature':int(fid),'gate_weight':float(g[fid]),'scope':'feature ID belongs only to this checkpoint; no cross-seed alignment; weight is not causal importance'})
 pd.DataFrame(selections).to_csv(O/'selected_models.csv',index=False);pd.DataFrame(hist).to_csv(O/'training_diagnostics.csv',index=False);pd.DataFrame(weights,columns=['fold','test_donor','seed','arm','feature','gate_weight','scope']).to_csv(O/'largest_gate_weights.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(15,5.8),gridspec_kw={'width_ratios':[1.1,1]});sub=summary[summary.arm!='identity'].iloc[::-1];y=np.arange(len(sub));axs[0].barh(y,sub.TEST_delta_pp,color='#298787',alpha=.75)
 for yy,arm in zip(y,sub.arm):
  a=donor[(donor.arm==arm)&(donor.role=='test')&(donor.readout=='ridge')];axs[0].scatter(a.delta_pp,np.full(len(a),yy),s=25,color='#183d50',zorder=3)
 axs[0].set_yticks(y,[LABELS[a] for a in sub.arm]);axs[0].axvline(0,c='#777',lw=1);axs[0].set_xlabel('Fresh Ridge Macro-F1 change (percentage points)');axs[0].set_title('Bar: mean; dots: six held-out donors')
 mat=donor[(donor.arm!='identity')&(donor.role=='test')&(donor.readout=='ridge')].pivot(index='arm',columns='test_donor',values='delta_pp').reindex(ARMS[1:]);lim=max(1.,float(np.abs(mat.to_numpy()).max()));im=axs[1].imshow(mat,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);axs[1].set_yticks(np.arange(len(mat)),[LABELS[a] for a in mat.index]);axs[1].set_xticks(np.arange(len(mat.columns)),[d.replace('patient_','') for d in mat.columns]);axs[1].set_xlabel('Held-out donor');axs[1].set_title('Three-seed mean change per donor')
 for i in range(len(mat)):
  for j in range(len(mat.columns)):axs[1].text(j,i,f'{mat.iloc[i,j]:+.1f}',ha='center',va='center',fontsize=9)
 fig.colorbar(im,ax=axs[1],label='Percentage points',shrink=.7);fig.tight_layout();fig.savefig(O/'performance.png',dpi=170);plt.close(fig)
 warning_count={}
 for name in ['development.log','evaluation.log','evaluation_slow_initial.log']:
  p=W/name
  if p.exists():
   for line in p.read_text(errors='replace').splitlines():
    if 'RuntimeWarning:' in line:key=line.split('RuntimeWarning:',1)[1].strip();warning_count[key]=warning_count.get(key,0)+1
 save(O/'numerical_warning_summary.json',{'log_warning_counts':warning_count,'verification':verify,'note':'NumPy/BLAS matrix-product warnings retained; all saved coefficients/scores checked finite, normal equations and independent outputs checked. Warnings are not silently discarded.'})
 audit='pending';ap=O/'EXPERIMENT_AUDIT.json'
 if ap.exists():
  aa=json.loads(ap.read_text());audit=str(aa.get('verdict',aa.get('overall_verdict','see audit')))
 methods='''固定历史scGPT masked-mean 512维表征及每轮TRAIN拟合的K32、2048字典SAE；6轮4TRAIN/1VALID/1TEST供体，3个SAE/训练耦合种子42/43/44。原始h完整保留，raw signed TopK，不额外ReLU或标准化z。CPU float32稀疏训练，float64新Ridge。每组使用相同类别均匀、供体平衡的有放回批次；稀有类别重复采样，不能增加生物学样本量。基线CE头按VALID选weight decay；其后残差和头联合训练，从同一CE头和零残差开始。checkpoint/早停按训练头VALID Macro-F1；beta与新Ridge alpha再按VALID新Ridge选择，identity优先平局。每臂3个beta，包含identity；不是穷尽优化，也不能用阴性结果排除其他训练目标。所有126臂的选择锁定后才计算TEST。SupCon臂独立从零残差开始，不是继续训练已选CE适配器。SupCon只把同类型不同TRAIN供体作为正样本，忽略同类型同供体及重复行的分母配对，无正样本anchor不参与该损失。
gate/rotated_gate各2048参数；lowrank/MLP各40960参数。gate与MLP不是精确参数匹配。旋转保持字典列范数和Gram，但不匹配原h各坐标的方差、归一化后的修正分布；实际残差范数单列，不把控制解释为完美匹配。低秩方案借鉴表征修改思路，不是原版LoReFT。
主要评价为冻结h-prime后重新拟合TRAIN标准化、类别平衡Ridge，由VALID选11个alpha，另报固定k=15的标准化余弦kNN和训练时分类头。固定八类Macro-F1中缺失类别记0，另报实际存在类型的Macro-F1与所有支持数。评价单位为供体，先平均三个种子，再比较六供体；同一供体种子不是独立重复，SAE种子与适配器种子耦合。各轮训练供体重叠，bootstrap区间仅为六个观测供体差值的描述区间，不作独立折或总体显著性保证。PBMC是同一4638细胞文件的多模型迁移结果，不是18个独立数据集。
Kang和PBMC均已历史探索，本轮是开发性验证，不是新的盲测。既有B细胞候选和TEST归因未用于特征选择；所有特征保留。没有重训scGPT/SAE，没有证明新的信息或生物机制。权重按checkpoint独立记录，未做跨种子特征编号对齐，不能直接比较不同SAE的同编号权重。门控大小不等于功能重要性或因果作用。K64仅为后续可选敏感性，本轮未执行。378个开发候选+18个基线头；早期无完整hash绑定的部分运行已中断保留于recovery，不复用，未产生TEST分数。
'''
 text='SAE残差优化实验\n\n'+summary.to_string(index=False)+'\n\n预定主要门槛：'+json.dumps(criteria,ensure_ascii=False)+'\n\n方法与边界：\n'+methods+f'\n数值核验PASS；独立科研审查{audit}（same-family/provisional）。\n';(O/'RESULTS_ZH.txt').write_text(text)
 short=summary[['arm','TEST_Ridge_macro_F1','TEST_delta_pp','positive_donors','positive_seed_means','PBMC_delta_pp','TEST_kNN_delta_pp','identity_selected_runs']]
 body='<h1>SAE残差是否改善细胞表征？</h1><p>六轮供体、三个种子；全部选择仅用TRAIN/VALID，统一锁定后评分TEST。</p><section><h2>全部预定组别</h2>'+table(short)+'</section><section><h2>供体差异</h2><img src="data:image/png;base64,'+base64.b64encode((O/'performance.png').read_bytes()).decode()+'"></section><section><h2>完整描述统计</h2>'+table(summary)+'</section><section><h2>预定对照</h2>'+table(pd.DataFrame(contrasts).groupby('contrast',sort=False).delta_pp.agg(['mean','min','max']).reset_index())+'</section><section><h2>方法与证据边界</h2><p>'+html.escape(methods).replace('\n','<br>')+f'</p><p>数值核验：PASS；科研审查：{html.escape(audit)}，same-family/provisional。</p></section><section><h2>结果文件</h2><ul>'+''.join(f'<li><a href="{p.relative_to(O)}">{p.relative_to(O)}</a></li>' for p in sorted(O.glob('*')) if p.is_file() and p.suffix in ['.csv','.json','.md','.txt'])+'</ul></section>'
 css='body{font-family:-apple-system,sans-serif;background:#f1f5f5;color:#213c42;line-height:1.7}main{max-width:1350px;margin:32px auto;padding:0 20px}section{background:white;border-radius:12px;padding:22px;margin:22px 0;overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dae3e3;text-align:left}th{background:#edf3f3}img{width:100%;height:auto}a{color:#217d7d}'
 (O/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAE残差优化实验</title><style>'+css+'</style><main>'+body+'</main></html>')
 save(O/'EXPERIMENT_TRACKER.json',{'status':'awaiting_review' if audit=='pending' else 'complete','training':'complete','evaluation':'complete','numerical_verification':'PASS','scientific_review':audit,'scope':'historically explored data; descriptive development comparison'})
 print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
