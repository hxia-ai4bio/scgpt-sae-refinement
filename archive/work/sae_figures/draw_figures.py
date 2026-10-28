# -*- coding: utf-8 -*-
"""Plot existing audited SAE results only; no model fitting or TEST selection.
Cached activation aggregation O(N*2048); other work is table aggregation.
"""
import os
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/sae_figures'
os.environ['MPLCONFIGDIR']=str(W/'mpl');os.environ['MPLBACKEND']='Agg'
import json,hashlib,textwrap
import numpy as np,pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata

I=ROOT/'outputs/sae_interpretability';G=ROOT/'outputs/sae_gene_attribution_v2';P=ROOT/'outputs/sae_pathways_v3'
OLD=Path('/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014')
INPUTS={}
BLUE='#0072B2';ORANGE='#E69F00';GREEN='#009E73';PURPLE='#CC79A7';GRAY='#87929B';INK='#23313D'
TARGETS=['B cells','CD14+ Monocytes','NK cells'];COLORS={TARGETS[0]:BLUE,TARGETS[1]:ORANGE,TARGETS[2]:GREEN}
SHORT=['B','CD14','CD4 T','CD8 T','DC','FCGR3A','Mega','NK'];TARGET_LABELS=['B cells','CD14+\nmonocytes','NK cells']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(p):INPUTS[str(p)]=sha(p)
def frame(p):record(p);return pd.read_csv(p)
def js(p):record(p);return json.loads(p.read_text())
def npz(p):
 record(p)
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def style():
 mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
  'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7.5,'axes.titlesize':9,
  'axes.linewidth':.65,'axes.edgecolor':'#51606B','axes.labelcolor':INK,'text.color':INK,
  'xtick.color':'#51606B','ytick.color':'#51606B','xtick.major.size':3,'ytick.major.size':3,
  'legend.fontsize':7,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
  'savefig.facecolor':'white','figure.facecolor':'white'})

def panel(ax,letter,title,subtitle=None):
 ax.text(-.15,1.095,letter,transform=ax.transAxes,fontsize=12,fontweight='bold',va='bottom',color=INK)
 ax.set_title(title,loc='left',fontweight='semibold',pad=23 if subtitle else 15)
 if subtitle:ax.text(0,1.027,subtitle,transform=ax.transAxes,fontsize=6.8,color='#66737D',va='bottom')
 ax.spines[['top','right']].set_visible(False)

def export(fig,stem):
 for suffix in ['png','pdf','svg']:fig.savefig(O/f'{stem}.{suffix}',dpi=350)
 # A direct monochrome re-render from the same plotting artists for accessibility QA.
 return fig

def prepare():
 manifest=js(OLD/'data_manifest.json');data=npz(OLD/'kang_embeddings.npz');split=manifest['splits'][0]
 assert split['test']=='patient_1016' and split['valid']=='patient_1039'
 act=npz(ROOT/'work/sae_interpretability/primary_activations.npz')['kang'];assert act.shape==(len(data['rows']),2048) and np.isfinite(act).all()
 cache=npz(OLD/'test_patient_1016/seed_42/k_32/sae_outputs.npz')
 for role in ['train','valid','test']:
  ii=np.asarray(split['indices'][role]);np.testing.assert_array_equal(cache[role+'_rows'],data['rows'][ii]);expected=np.zeros((len(ii),2048),np.float32);np.put_along_axis(expected,cache[role+'_indices'],cache[role+'_values'],axis=1);np.testing.assert_allclose(act[ii],expected,rtol=0,atol=2e-5)
 assert np.all((act!=0).sum(1)<=32)
 atlas=frame(I/'feature_atlas_train.csv');donor=frame(I/'donor_equal_single_feature_readouts.csv');candidates=js(I/'candidates.json')
 # Match existing TRAIN means/nonzero counts, hence atlas and activation row identity.
 tr=np.array(split['indices']['train']);np.testing.assert_allclose(act[tr].mean(0),atlas.train_mean,rtol=0,atol=1e-7);np.testing.assert_array_equal((act[tr]!=0).sum(0),atlas.train_nonzero)
 return manifest,data,split,act,atlas,donor,candidates

def figure2(manifest,data,split,z,atlas,donor,candidates):
 classes=manifest['classes'];tr=np.array(split['indices']['train']);labels=data['labels'];groups=data['donors'];eligible=atlas[atlas.eligible].copy()
 # Descriptive heatmap candidates selected only from TRAIN, with a minimum target coverage.
 selected=[];reasons={}
 for ci,name in enumerate(classes):
  if (labels[tr]==ci).sum()<20:continue
  order=eligible.sort_values([name+'_min_donor_auc',name+'_mean_donor_auc','feature'],ascending=[False,False,True],kind='stable')
  for f in order.feature.iloc[:2].astype(int):
   if f not in selected:selected.append(f)
   reasons.setdefault(f,[]).append(name)
 for c in candidates:
  f=int(c['feature'])
  if f not in selected:selected.append(f)
  reasons.setdefault(f,[]).append(c['role'])
 support=[];means=[]
 for ci,name in enumerate(classes):
  per=[]
  for d in split['train']:
   ii=tr[(groups[tr]==d)&(labels[tr]==ci)]
   if len(ii):per.append(z[ii][:,selected].mean(0))
  means.append(np.mean(per,axis=0));support.append(dict(cell_type=name,TRAIN_cells=int((labels[tr]==ci).sum()),contributing_TRAIN_donors=len(per)))
 means=np.array(means).T;scaled=(means-means.mean(1,keepdims=True))/np.maximum(means.std(1,keepdims=True),1e-12)
 pd.DataFrame(support).to_csv(O/'Figure2_A_cell_support.csv',index=False)
 rows=[]
 for i,f in enumerate(selected):
  for ci,name in enumerate(classes):rows.append(dict(feature=f,selected_for=';'.join(reasons[f]),cell_type=name,donor_balanced_mean=means[i,ci],row_zscore=scaled[i,ci]))
 pd.DataFrame(rows).to_csv(O/'Figure2_A_heatmap.csv',index=False)
 a=donor[donor.representation=='sae'].copy();assert set(a.target)==set(TARGETS) and len(a)==18
 a.to_csv(O/'Figure2_B_D_AUROC.csv',index=False)
 perf=a.groupby('target')[['test_auc','Batch1_3k_auc','Batch2_68k_r_auc']].mean()
 technical=eligible[['feature','max_abs_technical_partial']].copy();technical['max_TRAIN_type_AUROC']=eligible[[cl+'_mean_donor_auc' for cl in classes]].max(axis=1);technical.to_csv(O/'Figure2_C_associations.csv',index=False)
 fig=plt.figure(figsize=(7.6,6.9));gs=fig.add_gridspec(2,2,width_ratios=[1.18,1],left=.09,right=.975,bottom=.13,top=.87,hspace=.64,wspace=.43)
 axA=fig.add_subplot(gs[0,0]);axB=fig.add_subplot(gs[0,1]);axC=fig.add_subplot(gs[1,0]);axD=fig.add_subplot(gs[1,1])
 fig.suptitle('SAE coordinates associate with cell identity and technical variation',x=.09,y=.975,ha='left',fontsize=12,fontweight='bold')
 fig.text(.09,.938,'Frozen scGPT embedding  |  2,048 SAE coordinates  |  TopK = 32',fontsize=8.2,color='#66737D')
 panel(axA,'A','Cell-type activation atlas','One fixed dictionary; candidates selected on TRAIN')
 image=axA.imshow(scaled,aspect='auto',cmap='cividis',vmin=-2.5,vmax=2.5,interpolation='nearest')
 axA.set_xticks(range(8),[SHORT[i]+f"\n{support[i]['TRAIN_cells']}" for i in range(8)],fontsize=6.7)
 axA.set_yticks(range(len(selected)),[f'F{f}' for f in selected],fontsize=7)
 for tick,f in zip(axA.get_yticklabels(),selected):
  if f==160:tick.set_color(BLUE);tick.set_fontweight('bold')
 axA.tick_params(length=0);axA.set_xlabel('Cell type / TRAIN cell count',labelpad=5)
 cax=axA.inset_axes([.02,-.28,.65,.055]);cb=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=[-2,0,2]);cb.ax.tick_params(labelsize=6.5,length=2);cb.set_label('Row z-score of donor-balanced mean activation',fontsize=6.5,labelpad=2)
 panel(axB,'B','Held-out cell-type readout','One dot per donor; three seeds averaged within donor')
 for i,name in enumerate(TARGETS):
  aa=a[a.target==name].sort_values('test_donor');assert len(aa)==6
  yy=i+np.linspace(-.10,.10,6);axB.scatter(aa.test_auc,yy,s=20,color=COLORS[name],alpha=.75,edgecolor='white',linewidth=.35,zorder=3)
  value=aa.test_auc.mean();axB.plot(value,i,marker='|',ms=16,mew=2,color=INK,zorder=4)
  axB.text(.805,i-.25,f'mean {value:.3f}',fontsize=7.2,color=COLORS[name])
 axB.set_yticks(range(3),TARGET_LABELS);axB.set_ylim(2.48,-.48);axB.set_xlim(min(.8,float(a.test_auc.min())-.02),1.005);axB.set_xticks([.8,.85,.9,.95,1.]);axB.set_xlabel('Single-coordinate AUROC');axB.grid(axis='x',color='#E5E9ED',lw=.6);axB.tick_params(axis='y',length=0)
 panel(axC,'C','Cell-type and QC associations',f'{len(technical)} coverage-filtered coordinates; TRAIN only')
 axC.scatter(technical.max_TRAIN_type_AUROC,technical.max_abs_technical_partial,s=8,color='#C1CAD2',alpha=.65,linewidths=0)
 feature_targets={160:'B cells',440:'CD14+ Monocytes',120:'NK cells',1724:'QC comparator'}
 offsets={160:(-43,9),440:(-55,-10),120:(-43,-16),1724:(-57,8)}
 for f,label in feature_targets.items():
  rr=technical[technical.feature==f].iloc[0];color=COLORS.get(label,PURPLE);axC.scatter(rr.max_TRAIN_type_AUROC,rr.max_abs_technical_partial,s=29,color=color,edgecolor='white',lw=.45,zorder=3)
  axC.annotate(f'F{f}',(rr.max_TRAIN_type_AUROC,rr.max_abs_technical_partial),xytext=offsets[f],textcoords='offset points',fontsize=7.2,color=color,arrowprops=dict(arrowstyle='-',lw=.5,color=color))
 axC.set_xlim(.49,1.01);axC.set_ylim(-.015,max(.60,technical.max_abs_technical_partial.max()+.05));axC.set_xlabel('Maximum mean donor AUROC across cell types');axC.set_ylabel('Maximum |partial r| with QC');axC.grid(color='#EDF0F3',lw=.5)
 panel(axD,'D','Exploratory PBMC readout','Mean across TRAIN-selected fold / seed dictionaries')
 for i,name in enumerate(TARGETS):
  color=COLORS[name]
  axD.scatter(perf.loc[name,'Batch1_3k_auc'],i-.09,s=31,color=color,marker='o',edgecolor='white',lw=.4,zorder=3)
  axD.scatter(perf.loc[name,'Batch2_68k_r_auc'],i+.09,s=35,facecolor='white',edgecolor=color,marker='^',lw=1.1,zorder=3)
 axD.axvline(.5,color='#9AA6AF',lw=.8,ls='--');axD.set_xlim(.45,1.015);axD.set_ylim(2.45,-.45);axD.set_yticks(range(3),TARGET_LABELS);axD.set_xlabel('Single-coordinate AUROC');axD.grid(axis='x',color='#E5E9ED',lw=.6);axD.tick_params(axis='y',length=0)
 handles=[Line2D([],[],marker='o',color='none',mfc=INK,mec=INK,ms=4,label='Batch 1'),Line2D([],[],marker='^',color='none',mfc='white',mec=INK,ms=4,label='Batch 2')]
 axD.legend(handles=handles,loc='upper center',bbox_to_anchor=(.50,-.25),ncol=2,handletextpad=.5,columnspacing=1.4)
 fig.text(.09,.035,'A, C: primary TRAIN dictionary. B: six held-out Kang ctrl donors. D: two PBMC batches, not donor replicates.',fontsize=7,color='#66737D')
 export(fig,'Figure2_SAE_overview');plt.close(fig)
 return dict(heatmap_features=selected,eligible_coordinates=len(technical),heldout_mean_AUROC={k:float(v) for k,v in perf.test_auc.items()},PBMC_mean_AUROC=perf.to_dict('index'))

def figure3(manifest,data,split,z):
 classes=manifest['classes'];test=np.array(split['indices']['test']);assert len(test)==1723
 tr=frame(G/'gene_rankings_train.csv');ranked=tr[tr.sae_F160>0].sort_values('sae_F160',ascending=False,kind='stable').head(15).copy();ranked['RPL_RPS_prefix']=ranked.gene.str.startswith(('RPL','RPS'))
 ranked.to_csv(O/'Figure3_B_gene_sensitivity.csv',index=False)
 curves=frame(G/'deletion_per_prototype.csv');curves=curves[(curves.score=='sae_F160')&curves.method.isin(['sae_F160','matched_random'])].copy();assert curves.prototype_index.nunique()==4
 summary=frame(G/'deletion_summary.csv');ss=summary[(summary.score=='sae_F160')&summary.method.isin(['sae_F160','matched_random'])]
 for (method,n),xx in curves.groupby(['method','requested_genes']):assert abs(xx['drop'].mean()-float(ss[(ss.method==method)&(ss.requested_genes==n)]['drop'].iloc[0]))<1e-12
 curves.to_csv(O/'Figure3_C_replacement_curves.csv',index=False)
 pathways=frame(P/'candidate_pathway_summary.csv');pathways.to_csv(O/'Figure3_D_candidate_annotations.csv',index=False)
 assert (pathways['匹配随机全局q']>=.05).all()
 activ=pd.DataFrame(dict(canonical_index=test,raw_row=data['rows'][test],cell_id=data['cell_ids'][test],cell_type=np.array(classes)[data['labels'][test]],donor=data['donors'][test],F160=z[test,160]));activ.to_csv(O/'Figure3_A_TEST_activation.csv',index=False)
 fig=plt.figure(figsize=(7.6,7.0));gs=fig.add_gridspec(2,2,left=.09,right=.975,bottom=.12,top=.87,hspace=.68,wspace=.43)
 axA=fig.add_subplot(gs[0,0]);axB=fig.add_subplot(gs[0,1]);axC=fig.add_subplot(gs[1,0]);axD=fig.add_subplot(gs[1,1])
 fig.suptitle('F160: a B-cell-associated coordinate with local gene sensitivity',x=.09,y=.975,ha='left',fontsize=12,fontweight='bold')
 fig.text(.09,.938,'One fixed SAE dictionary  |  Reference-bin substitution  |  Candidate biological annotations',fontsize=8.2,color='#66737D')
 panel(axA,'A','Activation across cell types','All cells from the primary held-out donor')
 arrays=[z[test[data['labels'][test]==i],160] for i in range(8)]
 box=axA.boxplot(arrays,positions=np.arange(8),widths=.52,showfliers=False,whis=(5,95),patch_artist=True,medianprops=dict(color=INK,lw=1.05),boxprops=dict(lw=.65),whiskerprops=dict(color=GRAY,lw=.65),capprops=dict(color=GRAY,lw=.65))
 rng=np.random.default_rng(20261020)
 for i,(values,patch) in enumerate(zip(arrays,box['boxes'])):
  color=BLUE if i==0 else '#B3BDC5';patch.set_facecolor('#B6D9EC' if i==0 else '#E2E7EB');patch.set_edgecolor(color)
  selected=rng.choice(len(values),min(50,len(values)),replace=False);jitter=rng.uniform(-.15,.15,len(selected));axA.scatter(i+jitter,values[selected],s=5,color=color,alpha=.45,linewidths=0,zorder=2)
 axA.set_xticks(range(8),[SHORT[i]+f'\n{len(arrays[i])}' for i in range(8)],fontsize=6.6);axA.set_ylabel('F160 activation');axA.set_xlabel('Cell type / TEST cell count');axA.set_xlim(-.6,7.6);axA.grid(axis='y',color='#EDF0F3',lw=.5)
 panel(axB,'B','TRAIN gene sensitivity','Mean ± SD over 4 TRAIN donors; coverage at right')
 y=np.arange(len(ranked));colors=[GRAY if flag else BLUE for flag in ranked.RPL_RPS_prefix]
 axB.barh(y,ranked.sae_F160,color=colors,height=.65,xerr=ranked.sae_F160_donor_sd,error_kw=dict(ecolor='#556571',elinewidth=.65,capsize=1.7,capthick=.65))
 axB.set_yticks(y,ranked.gene,fontsize=7);axB.invert_yaxis();maximum=float((ranked.sae_F160+ranked.sae_F160_donor_sd).max())*1.27;axB.set_xlim(0,maximum)
 for i,row in enumerate(ranked.itertuples()):axB.text(maximum*.985,i,f'{row.input_prototypes}/{row.total_prototypes}',ha='right',va='center',fontsize=6.7,color='#66737D')
 axB.set_xlabel('F160 activation drop after one gene-bin replacement',fontsize=7.1);axB.grid(axis='x',color='#EDF0F3',lw=.5);axB.set_axisbelow(True);axB.ticklabel_format(axis='x',style='plain')
 panel(axC,'C','TRAIN ranking transfers locally','4 selected TEST B cells from 1 historical donor')
 for method,color,ls,label in [('sae_F160',BLUE,'-','TRAIN-ranked genes'),('matched_random',GRAY,'--','Joint-matched random')]:
  x=curves[curves.method==method]
  for pi,a in x.groupby('prototype_index'):
   a=a.sort_values('requested_genes');axC.plot(a.requested_genes,a['drop'],color=color,lw=.6,alpha=.23,ls=ls)
  mean=x.groupby('requested_genes')['drop'].mean();axC.plot(mean.index,mean.values,color=color,ls=ls,lw=1.9,marker='o' if method=='sae_F160' else 's',markersize=3.1,label=label,zorder=4)
 axC.set_xlabel('Requested gene-bin replacements');axC.set_ylabel('F160 activation drop');axC.set_xticks([0,10,20,30,40]);axC.set_xlim(-.6,41);axC.grid(axis='y',color='#EDF0F3',lw=.5);axC.legend(loc='upper left',handlelength=2)
 panel(axD,'D','Candidate pathway annotations','Fixed GO / Hallmark terms; no matched-null discovery')
 axD.axis('off');axD.set_xlim(0,1);axD.set_ylim(0,1)
 axD.text(0,.97,'Candidate annotation',fontsize=7.1,fontweight='bold',va='top');axD.text(.76,.97,'Ordinary\nglobal q',ha='right',va='top',fontsize=7.1,fontweight='bold');axD.text(1,.97,'Matched-null\nglobal q',ha='right',va='top',fontsize=7.1,fontweight='bold');axD.axhline(.835,color='#CCD4DB',lw=.7)
 labels=['Translation','mRNA catabolism','Post-transcriptional regulation','Endogenous antigen presentation','IFN-γ response']
 for i,(_,row) in enumerate(pathways.iterrows()):
  yy=.77-i*.14;full=row['分析']=='all_input_top40';ordinary=float(row['普通ORA全局q']);matched=float(row['匹配随机全局q'])
  axD.text(0,yy,labels[i],fontsize=7.2,va='center');axD.text(0,yy-.047,f"{int(row['重叠基因数'])} overlaps · "+('full top40' if full else 'non-RPL/RPS top40'),fontsize=6.2,color='#74808A',va='center')
  axD.text(.76,yy,format(ordinary,'.1e') if ordinary<.001 else f'{ordinary:.3f}',ha='right',va='center',fontsize=7.1);axD.text(1,yy,f'{matched:.3f}',ha='right',va='center',fontsize=7.1)
 axD.text(0,-.025,'0 discoveries at matched-null global FDR < 0.05',fontsize=6.8,color='#66737D')
 fig.text(.09,.035,'TRAIN sensitivity: 8 prototypes / 4 donors. TEST replacement: 4 selected cells / 1 donor. Gray gene bars: RPL/RPS.',fontsize=7,color='#66737D')
 export(fig,'Figure3_Bcell_F160');plt.close(fig)
 return dict(feature=160,primary_test_donor=split['test'],all_TEST_cells=len(test),TEST_B_cells=len(arrays[0]),TRAIN_attribution_prototypes=8,TRAIN_attribution_donors=4,TEST_replacement_prototypes=4,TEST_replacement_donors=1,
  drop_at40={method:float(ss[(ss.method==method)&(ss.requested_genes==40)]['drop'].iloc[0]) for method in ['sae_F160','matched_random']},matched_null_significant_candidates=int((pathways['匹配随机全局q']<.05).sum()))

def main():
 O.mkdir(parents=True,exist_ok=True);style();data=prepare();a=figure2(*data);b=figure3(*data[:4]);record(Path(__file__))
 for p in [I/'CLAIMS_FROM_RESULTS.md',G/'CLAIMS_FROM_RESULTS.md',P/'CLAIMS_FROM_RESULTS.md']:record(p)
 (O/'FIGURE_DATA_AND_CHECKS.json').write_text(json.dumps(dict(Figure2=a,Figure3=b,activation_identity_checks='All TRAIN/VALID/TEST role rows and cached TopK activations agree; existing TRAIN atlas reproduced.',normalization='Figure2A only: row z-score of donor-balanced cell-type mean activation. AUROCs, sensitivities and q-values remain in original reported units.',statistical_scope='Descriptive displays. No new tests, significance stars, models, training or TEST-based candidate selection.',sources=INPUTS),indent=2))
 print('FIGURES_COMPLETE',len(INPUTS),'bound inputs',flush=True)
if __name__=='__main__':main()
