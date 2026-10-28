# -*- coding: utf-8 -*-
"""Replay all cached SAE/probe outputs and produce complete eight-class reports."""
from pathlib import Path
import json,csv,warnings,hashlib,statistics as st
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sae import TopKSAE
from core import R,write,sha,metrics,reconstruction
from run import sae_eval,load_kang,METHODS,SEEDS

def main():
 torch.set_num_threads(4);x,y,donors,rows,folds,proto=load_kang();j=json.loads((R/'summary.json').read_text());assert j['status']=='complete' and len(j['results'])==54
 with np.load(R/'pbmc_embeddings.npz') as a:px=a['embedding'];py=a['labels'];pr=a['rows'];pb=a['batches']
 max_sae_error=0.;max_score_error=0.;max_equation=0.;parts=0;probe_count=0;seenoriginal=set();identities=set()
 for r in j['results']:
  identity=(r['fold']['test'],r['seed'],r['k']);assert identity not in identities;identities.add(identity)
  fold,ix=next((f,i) for f,i in folds if f['test']==r['fold']['test']);out=Path(r['path']);state=torch.load(out/'sae.pt',map_location='cpu');mean=x[ix['train']].astype(float).mean(0).astype(np.float32);assert np.array_equal(mean,state['mean'].numpy());m=TopKSAE(state['mean'],r['k'],r['seed']);m.load_state_dict(state);assert (m.decoder.weight.double().norm(dim=0)-1).abs().max()<2e-6
  h={s:x[ix[s]] for s in ['train','valid','test']};h['pbmc']=px;z={};recs={}
  with np.load(out/'sae_outputs.npz') as a:
   for s in h:
    _,zz,re,ii,vv=sae_eval(m,h[s]);z[s]=zz;recs[s]=re;err=float(np.max(np.abs(re-a[s+'_reconstruction'])));max_sae_error=max(max_sae_error,err);assert err<1e-6;assert np.array_equal(ii,a[s+'_indices']) and np.array_equal(vv,a[s+'_values']);assert ((zz!=0).sum(1)<=r['k']).all();labels=py if s=='pbmc' else y[ix[s]];assert np.array_equal(labels,a[s+'_labels']);rec=reconstruction(h[s],re,mean)
    for key,v in rec.items():assert abs(v-r['reconstruction'][s][key])<1e-9*max(1,abs(v))
    freq=(zz!=0).mean(0);assert np.array_equal(freq,r['reconstruction'][s]['feature_activation_frequency']);assert int((freq==0).sum())==r['reconstruction'][s]['dead_features'];parts+=1
  hist=r['sae']['history'];minimum=min(h['valid_mse'] for h in hist);assert r['sae']['best_epoch']==next(h['epoch'] for h in hist if h['valid_mse']==minimum)
  for method in METHODS:
   path=R/('test_'+fold['test'])/'original' if method=='original' else out/method
   if method=='original' and fold['test'] in seenoriginal:continue
   if method=='original':seenoriginal.add(fold['test'])
   features=h if method=='original' else z if method=='sparse' else recs if method=='reconstruction' else {s:np.concatenate([h[s],z[s]],1) for s in h}
   q=r['probes'][method];alpha=q['selected_alpha'];expected=next(t['alpha'] for t in q['trials'] if t['valid']['macro_f1']==max(v['valid']['macro_f1'] for v in q['trials']));assert alpha==expected
   with np.load(path/'linear_probe.npz') as a,np.load(path/'predictions.npz') as b:
    sm=a['scaler_mean'];sc=a['scaler_scale'];active=a['active_columns'];coef=a['coef'];inter=a['intercept'];assert np.max(np.abs(sm-features['train'].astype(float).mean(0)))<1e-10
    for s in features:
     ss=((features[s].astype(float)-sm)/sc)[:,active]@coef+inter;err=float(np.max(np.abs(ss-b[s+'_scores'])));max_score_error=max(max_score_error,err);assert err<1e-9;labels=py if s=='pbmc' else y[ix[s]];assert np.array_equal(labels,b[s+'_label']);p=ss.argmax(1);cm=np.zeros((8,8),int);np.add.at(cm,(labels,p),1);den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(8,float),where=den!=0);assert np.array_equal(cm,q[s]['confusion_matrix']);assert np.max(np.abs(f1-q[s]['per_class_f1']))<1e-12;assert abs(f1.mean()-q[s]['macro_f1'])<1e-12;assert abs((labels==p).mean()-q[s]['accuracy'])<1e-12
    tx=((features['train'].astype(float)-sm)/sc)[:,active];yy=y[ix['train']];w=len(yy)/(8*np.bincount(yy,minlength=8)[yy]);xm=np.average(tx,axis=0,weights=w);target=2*np.eye(8)[yy]-1.;ym=np.average(target,axis=0,weights=w);xc=tx-xm;yc=target-ym;rhs=xc.T@(w[:,None]*yc);g=xc.T@(w[:,None]*(xc@coef-yc))+alpha*coef;err=float(np.max(np.abs(g))/max(1,float(np.max(np.abs(rhs)))));max_equation=max(max_equation,err);assert err<1e-7;assert np.max(np.abs(inter-(ym-xm@coef)))<1e-9
   probe_count+=1
 print('VERIFICATION_PASS',len(identities),parts,probe_count,flush=True);write(R/'deterministic_verification.json',dict(status='PASS',sae_checkpoints=54,sae_split_replays=parts,unique_linear_probe_families=probe_count,probe_evaluation_partitions=probe_count*4,max_sae_output_error=max_sae_error,max_linear_scores_error=max_score_error,max_weighted_normal_equation_relative_error=max_equation,manual_confusion_f1_accuracy=True,TRAINmean_and_scaler_verified=True,VALIDcheckpoint_and_alpha_selection=True,scope='cached embeddings full SAE/probe replay; original scGPT sourcecache bound and8cellforward check, not full raw-count replay'))
 report(j,proto,px,py,pb)
 write(R/'RUN_STATUS.json',dict(stage='computational_complete',completed_sae=54,probe_families=probe_count,review='fresh integrity and claim review pending'))

def report(j,proto,px,py,pb):
 names=dict(original='Original Embedding',sparse='SAE Sparse Features',reconstruction='SAE Reconstruction',concat='Original + SAE Features');classes=proto['classes'];ag=j['aggregate'];rr=j['results'];d=load_manifest()
 lines=['# Cell-level TopK SAE for frozen scGPT','', '## 1. 实验目的与研究假设','H1：512维cell embedding能由少量稀疏字典特征有效重建。H2：稀疏特征改善留出供体及外部PBMC分类。只做SAE，无MCR²、MLP分类器或scGPT微调。','', '## 2. 数据与模型设置',f"Whole-human严格加载checkpoint；最终12层masked Mean Pooling、512gene-token上下文，51bin。Kang ctrl六历史供体共{d['reused_kang_cells']+d['new_kang_cells']}细胞，复用{d['reused_kang_cells']}向量、补提取{d['new_kang_cells']}巨核细胞。PBMC实际{len(px)}细胞；两个批次{d['pbmc_batches']}。保持八类字符串映射："+'；'.join(classes)+'。', '原donor TEST/VALID轮换；四TRAIN供体全量合格细胞，不复用之前64donor/type上限。因此不与此前七类分数直接比较。原七类NK id6在八类中为id7，标签从源字符串重映射。QC保留既有固定词表/≥16表达词表基因规则，与性能无关；未按测试分数筛样本。', 'SAE只用TRAIN估计均值/拟合，VALID MSE选最早最佳；probe只用TRAIN拟合StandardScaler与balanced Ridge，alpha=.01,.1,1,10,100同预算，VALID宏F1选最早候选，不做TRAIN+VALID再拟合。PBMC只在最终推理和评分出现。三个固定种子42/43/44。', '8类macroF1固定labels=0..7、zero_division=0；patient107没有Megakaryocytes，该类贡献0，另保存supported-type macro作为诊断。各类支持数在逐类表。', '运行设备CPU float32 SAE，CPUfloat64线性读出，4threads。3090两台配置服务器不可达；MPS冻结PBMC提取曾发生command-buffer失败并丢弃输出，CPU重跑。失败日志保留，不用于模型比较。','', '## 3. SAE数学定义','a=W_enc(h−mu_TRAIN)+b_enc；z=TopK_raw(a,K)，保留K个最大的原始激活值（非绝对值、无ReLU），其余置零；h_hat=W_dec z+mu_TRAIN。512→2048→512，decoder无bias。每次Adam更新后字典列单位范数。encoder初始为unitdecoder转置，之后独立训练。MSE、lr3e−4、batch256、最多20epochs、VALID patience3，无L1、辅助损失或dead-feature重采样。', '参考2603.02952的方法迁移到cell-level，非gene-token严格复现。论文原batch4096/5epochs，本轮按任务指定预算及VALID早停。', '重建R²=1−Σ||h−h_hat||²/Σ||h−mean_evalsplit||²；evalsplit均值仅作指标分母，不进入模型/预处理。补充Q²采用TRAINmean分母，跨数据集均值偏移会使R²与Q²不同。MSE按所有cell×512元素平均。','', '## 4. K=32分类对照', '均值±SD采用先对每seed平均六fold，再对三个seed计算样本SD；SD反映优化种子，不是供体总体不确定性。原始Ridge确定性，无SAE优化重复，其seed SD为0。PBMC始终是同一组细胞，被不同TRAINfold模型重复预测，不是18个独立外部数据集。','', '|Method|Kang Macro-F1 (%)|PBMC Macro-F1 (%)|ΔKang vs Original (pp)|ΔPBMC vs Original (pp)|','|---|---:|---:|---:|---:|']
 for m in METHODS:
  a=ag['32'][m];b=ag['32']['original'];lines.append(f"|{names[m]}|{a['test']['mean']*100:.3f}±{a['test']['sd_three_seed_means']*100:.3f}|{a['pbmc']['mean']*100:.3f}±{a['pbmc']['sd_three_seed_means']*100:.3f}|{100*(a['test']['mean']-b['test']['mean']):+.3f}|{100*(a['pbmc']['mean']-b['pbmc']['mean']):+.3f}|")
 lines+=['','## 5. K消融与重建质量','主K表以Sparse Features分类为统一终点；Reconstruction和Concat也完整保留，不能按TEST挑最优K。Dead比例主表用TRAIN，另报告TEST/PBMC未激活比例。','', '|K|Kang TEST重建R²|PBMC重建R²|Kang Sparse F1%|PBMC Sparse F1%|TRAIN dead比例|实际活跃特征均值|','|---|---:|---:|---:|---:|---:|---:|']
 for k in [16,32,64]:
  a=ag[str(k)];v=a['sae_metrics'];rs=[r for r in rr if r['k']==k];rstd=lambda split,key:float(np.std([np.mean([r['reconstruction'][split][key] for r in rs if r['seed']==seed]) for seed in SEEDS],ddof=1));lines.append(f"|{k}|{v['test']['r2_test_centered']:.4f}±{rstd('test','r2_test_centered'):.4f}|{v['pbmc']['r2_test_centered']:.4f}±{rstd('pbmc','r2_test_centered'):.4f}|{a['sparse']['test']['mean']*100:.3f}±{a['sparse']['test']['sd_three_seed_means']*100:.3f}|{a['sparse']['pbmc']['mean']*100:.3f}±{a['sparse']['pbmc']['sd_three_seed_means']*100:.3f}|{v['train']['dead_feature_ratio']*100:.2f}±{rstd('train','dead_feature_ratio')*100:.2f}%|{v['test']['mean_active_features']:.2f}|")
 lines+=['','完整K×方法结果：','','|K|Method|Kang F1%|PBMC F1%|Kang 18fold/seed SD(pp)|PBMC 18model SD(pp)|','|---|---|---:|---:|---:|---:|']
 for k in [16,32,64]:
  for m in METHODS:
   a=ag[str(k)][m];lines.append(f"|{k}|{names[m]}|{100*a['test']['mean']:.3f}|{100*a['pbmc']['mean']:.3f}|{100*a['test']['sd_18_fold_seed']:.3f}|{100*a['pbmc']['sd_18_fold_seed']:.3f}|")
 lines+=['','重建指标：','','|K|split|MSE|R²|TRAINmean Q²|dead比例|实际活跃特征数|','|---|---|---:|---:|---:|---:|---:|']
 for k in [16,32,64]:
  for s,a in ag[str(k)]['sae_metrics'].items():lines.append(f"|{k}|{s}|{a['mse']:.7f}|{a['r2_test_centered']:.5f}|{a['q2_train_mean']:.5f}|{a['dead_feature_ratio']*100:.2f}%|{a['mean_active_features']:.2f}|")
 lines+=['','## 6. 逐类和配对结果','完整每fold/seed/K/方法的support、F1及混淆矩阵见result.json与per_class_metrics.csv；主K32逐类F1平均：','','|类型|'+ '|'.join(names[m] for m in METHODS)+'|','|---|'+'---:|'*4]
 for i,c in enumerate(classes):lines.append('|'+c+'|'+'|'.join(f"{100*ag['32'][m]['test']['per_class_f1'][i]:.2f}" for m in METHODS)+'|')
 lines+=['','PBMC K32逐类F1平均：','','|类型|'+ '|'.join(names[m] for m in METHODS)+'|','|---|'+'---:|'*4]
 for i,c in enumerate(classes):lines.append('|'+c+'|'+'|'.join(f"{100*ag['32'][m]['pbmc']['per_class_f1'][i]:.2f}" for m in METHODS)+'|')
 pairs=[]
 for r in rr:
  for m in METHODS[1:]:
   for s in ['test','pbmc']:pairs.append(dict(k=r['k'],test_donor=r['fold']['test'],seed=r['seed'],method=m,endpoint=s,delta_pp=100*(r['probes'][m][s]['macro_f1']-r['probes']['original'][s]['macro_f1'])))
 for k in [16,32,64]:
  for m in METHODS[1:]:
   for s in ['test','pbmc']:
    vv=[a['delta_pp'] for a in pairs if a['k']==k and a['method']==m and a['endpoint']==s];lines.append(f"- K{k}/{m}/{s}：正/负/平 {sum(v>0 for v in vv)}/{sum(v<0 for v in vv)}/{sum(v==0 for v in vv)}，均值{st.mean(vv):+.3f}pp。")
 with (R/'paired_differences.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=pairs[0].keys());w.writeheader();w.writerows(pairs)
 with (R/'per_run_metrics.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['K','test_donor','valid_donor','seed','method','endpoint','macro_f1','accuracy','selected_alpha','sae_best_epoch'])
  for r in rr:
   for m in METHODS:
    for endpoint in ['test','pbmc']:
     q=r['probes'][m];w.writerow([r['k'],r['fold']['test'],r['fold']['valid'],r['seed'],m,endpoint,q[endpoint]['macro_f1'],q[endpoint]['accuracy'],q['selected_alpha'],r['sae']['best_epoch']])
 with (R/'per_class_metrics.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['K','test_donor','seed','method','endpoint','class','support','f1'])
  for r in rr:
   for m in METHODS:
    for s in ['test','pbmc']:
     for i,c in enumerate(classes):a=r['probes'][m][s];w.writerow([r['k'],r['fold']['test'],r['seed'],m,s,c,a['support'][i],a['per_class_f1'][i]])
 lines+=['','## 7. 主要发现',f"54个SAE中{sum(r['sae']['best_epoch']==20 for r in rr)}个最佳epoch在20上限；当前预算不代表已收敛，不在看到TEST后延长。",'语义判断由新审查在CLAIMS_FROM_RESULTS.md中给出；数值表不按TEST/PBMC选择K。H1没有事先设定“有效”R²阈值，报告连续质量而非事后造成功阈值。','', '## 8. 实验局限','六个历史探索Kang供体，巨核细胞60个中52个在1039；某些TRAINfold只有少数该类，TEST有1个的类别F1高度离散。PBMC本地文件来源/研究独立性、预训练覆盖尚未核实，称跨文件探索测试，不声称独立新研究盲测。与此前七类/样本上限不同，不能直接横向比历史绝对分数。', '稀疏2048D与Concat2560D线性分类器参数量大于原512D；即使有收益，也不能单凭四组归因于稀疏性，尚缺等维随机/其他字典控制。SAE Reconstruction是Sparse Features的线性解码，二者分类差异还涉及读出维数、标准化及正则化，不是重建凭空增加信息。', 'noPBMCfitting/selection，重复使用同一个外部集合不增加独立样本数。没有通路、基因程序或干预证据，不将稀疏特征直接命名生物机制。MPS失败日志、NumPy/BLAS运行告警完整保留；核验通过不能证明每次历史运算不存在异常。','', '## 9. 是否值得第二阶段MCR²','本轮没有实现MCR²。只有当SAE稀疏特征或拼接体现可重复的判别/外部读出价值，才有理由另立方案。高重建R²本身不保证分类提升；无增益时应先检查读出、容量和稳定性，而不是自动叠加MCR²。最终具体建议见CLAIMS_FROM_RESULTS.md。']
 (R/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 fig,axs=plt.subplots(2,2,figsize=(11,8));colors=dict(original='#666666',sparse='#0072B2',reconstruction='#D55E00',concat='#009E73');xx=[16,32,64]
 for m in METHODS:
  for ax,s in [(axs[0,0],'test'),(axs[0,1],'pbmc')]:ax.errorbar(xx,[100*ag[str(k)][m][s]['mean'] for k in xx],yerr=[100*ag[str(k)][m][s]['sd_three_seed_means'] for k in xx],fmt='o-',color=colors[m],label=names[m],lw=1)
 for s,c in [('test','#0072B2'),('pbmc','#D55E00')]:axs[1,0].plot(xx,[ag[str(k)]['sae_metrics'][s]['r2_test_centered'] for k in xx],'o-',color=c,label=s)
 for r in rr:
  if r['k']==32:axs[1,1].plot([h['epoch'] for h in r['sae']['history']],[h['valid_mse'] for h in r['sae']['history']],alpha=.4,lw=.8)
 for ax,title in zip(axs.flat,['Kang8-class macro-F1 (%)','PBMC8-class macro-F1 (%)','Reconstruction R²','K32 VALID MSE trajectories']):ax.set_title(title);ax.grid(alpha=.15)
 for ax in axs.flat[:3]:ax.set_xlabel('K');ax.set_xticks(xx)
 axs[0,0].legend(fontsize=7);axs[1,0].legend();axs[1,1].set_xlabel('Epoch');fig.tight_layout();fig.savefig(R/'sae_summary.png',dpi=300);fig.savefig(R/'sae_summary.pdf');plt.close(fig)
 # Confusion matrices: sum counts for display, not pooledF1; same PBMC evaluated18 times.
 fig,axs=plt.subplots(2,4,figsize=(16,8))
 for row,s in enumerate(['test','pbmc']):
  for col,m in enumerate(METHODS):
   cm=np.sum([np.array(r['probes'][m][s]['confusion_matrix']) for r in rr if r['k']==32],0);den=cm.sum(1,keepdims=True);a=np.divide(cm,den,out=np.zeros_like(cm,dtype=float),where=den!=0);axs[row,col].imshow(a,cmap='Blues',vmin=0,vmax=1);axs[row,col].set_title(s+' / '+m);axs[row,col].set_xticks(range(8),['B','CD14','CD4','CD8','DC','FCGR3A','Mega','NK'],rotation=60);axs[row,col].set_yticks(range(8),['B','CD14','CD4','CD8','DC','FCGR3A','Mega','NK'])
 fig.suptitle('K32 row-normalized confusion; repeated-model count sums for visualization only',fontsize=11);fig.tight_layout();fig.savefig(R/'confusion_matrices.png',dpi=200);fig.savefig(R/'confusion_matrices.pdf');plt.close(fig)
 (R/'README.md').write_text('# TopK SAE\n\nOnly SAE, frozen wholehuman512Dpool and linear balancedRidge. See protocol.json and data_manifest.json for source/splits. Run prepare.py (CPU frozen missingfeatures), smoke.py, run.py (K32 then16,64), finish.py using /opt/anaconda3/envs/scgpt/bin/python; fourCPUthreads. Sources/oldexperiments unchanged. Completed SAE/candidate files reused on same immutableprotocol. No in-epochresume. Mean±SD definitions and absent-class rule in RESULTS_ZH.md. Fullweights/predictions/featurefrequencies/logs retained.\n')

def load_manifest():return json.loads((R/'data_manifest.json').read_text())
if __name__=='__main__':
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always')
  try:main()
  finally:write(R/'finish_warnings.json',dict(count=len(ws),messages=sorted(set(str(w.message) for w in ws))))
