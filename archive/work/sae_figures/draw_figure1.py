from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path('work/sae_figures/mpl').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
import numpy as np
import hashlib,json
O=Path('outputs/sae_figures');O.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none'})
INK='#23323D';MUTED='#667782';BLUE='#0072B2';GREEN='#009E73';ORANGE='#E69F00';GRAY='#87929B'
fig=plt.figure(figsize=(12,10.5),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1200);ax.set_ylim(0,1050);ax.axis('off')
def text(x,y,s,size=10,color=INK,ha='left',va='center',weight='normal',**kw):
 return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,fontweight=weight,**kw)
def box(x,y,w,h,fc='white',ec='#D8E1E7',lw=1,r=12):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec,lw=lw,zorder=1);ax.add_patch(p);return p
def arrow(a,b,color=GRAY,lw=1.5,style='-|>',rad=0,ls='-'):
 p=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=12,color=color,lw=lw,connectionstyle=f'arc3,rad={rad}',linestyle=ls,zorder=2);ax.add_patch(p)
def node(x,w,title,formula,sub,color,fill):
 box(x,795,w,106,fill,color,lw=1.05)
 text(x+w/2,877,title,10,ha='center',weight='bold')
 text(x+w/2,845,formula,9.2 if title=='SAE encoder' else (11 if title in ['Sparse features','SAE decoder'] else 12),ha='center')
 text(x+w/2,815,sub,8.2,MUTED,ha='center')
text(42,1014,'Interpretable SAE decomposition and relation-preserving reconstruction',19,weight='bold')
text(42,983,'Frozen scGPT representations  |  Sparse dictionary learning  |  Cell–cell geometry regularization',11,MUTED)
box(27,650,1146,299,'#F7FAFC','#E3EAF0',r=17)
text(44,922,'A',18,weight='bold');text(77,922,'From gene expression to a sparse cell representation',13,weight='bold')
xs=[50,207,379,511,693,865,1032];ws=[125,140,100,150,140,135,115]
nodes=[('scRNA-seq',r'$x_i$','Gene + value tokens',GRAY,'#FFFFFF'),('Frozen scGPT',r'$f_{\mathrm{scGPT}}$','Final-layer mean pooling',BLUE,'#EAF4FA'),('Embedding',r'$h_i\in\mathbb{R}^{512}$','Frozen cache',BLUE,'#EAF4FA'),('SAE encoder',r'$a_i=W_e(h_i-\mu)+b_e$','Trainable',GREEN,'#EAF7F2'),('Sparse features',r'$z_i=\mathrm{TopK}_{32}(a_i)$','2,048 coordinates',ORANGE,'#FFF6E2'),('SAE decoder',r'$\hat h_i=\mu+Dz_i$','Trainable dictionary',GREEN,'#EAF7F2'),('Reconstruction',r'$\hat h_i\in\mathbb{R}^{512}$','512-dimensional',BLUE,'#EAF4FA')]
for x,w,(title,formula,sub,c,fc) in zip(xs,ws,nodes):node(x,w,title,formula,sub,c,fc)
for i in range(6):arrow((xs[i]+ws[i]+5,848),(xs[i+1]-5,848))
# Sparse activation glyph: diagrammatic, with retained signs.
x0=702;y0=736
for j in range(21):
 v={2:24,7:37,11:-12,16:29}.get(j,0)
 ax.add_patch(Rectangle((x0+j*6,y0+min(v,0)),3,max(abs(v),2),facecolor=ORANGE if v else '#D9E1E6',edgecolor='none'))
arrow((763,793),(763,777),ORANGE,lw=1)
text(685,696,'Sparse coordinates → cell-type associations, gene sensitivity and candidate annotations',10,MUTED,ha='center')
text(685,675,'Interpretability analyses: Figures 2–3; feature identities refer to the fixed dictionary.',8.7,MUTED,ha='center')
text(66,750,r'$h_i-\mu\;\approx\;\sum_{k=1}^{2048}z_{ik}d_k$',14)
text(66,716,'At most 32 retained coordinates per cell',9,MUTED)
text(66,692,'TopK selects largest raw values; retained signs are kept.',8.1,MUTED)
# Panel B
box(27,46,1146,581,'#FFFFFF','#D9E2E8',r=17)
text(44,597,'B',18,weight='bold');text(77,597,'Reconstruct each cell while preserving cross-donor relations',13,weight='bold')
text(77,568,'Frozen original embedding as teacher; update SAE encoder and decoder using TRAIN cells only.',10,MUTED)
box(80,450,307,80,'#EAF4FA',BLUE)
text(233.5,506,'Original embedding batch',11,ha='center',weight='bold')
text(233.5,477,r'$H=\{h_i\}_{i=1}^{B}$'+'  ·  frozen teacher',12,ha='center')
box(813,450,307,80,'#EAF7F2',GREEN)
text(966.5,506,'SAE reconstruction batch',11,ha='center',weight='bold')
text(966.5,477,r'$\hat H=\{\hat h_i\}_{i=1}^{B}$'+'  ·  student',12,ha='center')
box(457,454,286,72,'#FFF6E2','#D9AD44')
text(600,503,'Cell-wise reconstruction',10,ha='center',weight='bold')
text(600,476,r'$\mathcal{L}_{\mathrm{rec}}=\mathrm{MSE}(H,\hat H)/c_{\mathrm{TRAIN}}$',12,ha='center')
arrow((389,490),(451,490),BLUE);arrow((811,490),(749,490),GREEN)
# Relation cards
box(67,260,420,151,'#F2F7FB','#C7DBE8');box(713,260,420,151,'#F0F8F5','#C7E3D8')
arrow((233.5,447),(233.5,415),BLUE);arrow((966.5,447),(966.5,415),GREEN)
text(82,391,'Teacher relations P',11,weight='bold',color=BLUE)
text(728,391,'Student relations Q',11,weight='bold',color=GREEN)
def matrix(x,y,shift=0):
 # Illustrative relations only, explicitly declared in caption.
 base=np.array([[0,.8,.2,.5,.1],[.8,0,.4,.7,.2],[.2,.4,0,.3,.8],[.5,.7,.3,0,.4],[.1,.2,.8,.4,0]])
 base=np.clip(base+shift*np.array([[0,1,0,-1,1],[1,0,1,-1,0],[0,1,0,0,-1],[-1,-1,0,0,1],[1,0,-1,1,0]]),0,1)
 for i in range(5):
  for j in range(5):
   color='#E1E7EB' if i==j else plt.colormaps['cividis'](base[i,j])
   ax.add_patch(Rectangle((x+16*j,y-16*i),15,15,facecolor=color,edgecolor='white',lw=.35,zorder=3))
matrix(86,351);matrix(732,351,.055)
text(186,345,r'$P_{ij}=\mathrm{softmax}_{j\in\mathcal{A}_i}\!\left(u_i^\top u_j/\tau\right)$',13)
text(186,309,r'$u_i=(h_i-\mu)/\|h_i-\mu\|_2$',11)
text(186,280,'Stop gradient through P',9,BLUE,weight='bold')
text(832,345,r'$Q_{ij}=\mathrm{softmax}_{j\in\mathcal{A}_i}\!\left(v_i^\top v_j/\tau\right)$',13)
text(832,309,r'$v_i=(\hat h_i-\mu)/\|\hat h_i-\mu\|_2$',11)
text(832,280,'Gradients flow through Q to SAE',9,GREEN,weight='bold')
box(526,301,148,66,'#FFF6E2','#D9AD44')
text(600,340,'Relation matching',9,ha='center',weight='bold');text(600,315,r'$D_{\mathrm{KL}}(P\,\|\,Q)$',13,ha='center')
arrow((491,333),(522,333),BLUE);arrow((709,333),(678,333),GREEN)
text(600,231,r'$\mathcal{A}_i=\{j\text{ in batch}:\ \mathrm{donor}(j)\neq\mathrm{donor}(i)\}$',11,ha='center')
text(600,207,r'$B=256$  ·  64 cells per TRAIN donor  ·  192 anchors per query  ·  $\tau=0.1$',9.5,MUTED,ha='center')
box(77,100,1046,79,'#F7FAFC','#DCE5EB',r=10)
text(600,150,r'$\mathcal{L}_{\mathrm{total}}=\frac{\mathrm{MSE}(H,\hat H)}{c_{\mathrm{TRAIN}}}+\lambda\,\frac{1}{B}\sum_i\sum_{j\in\mathcal{A}_i}P_{ij}\log\frac{P_{ij}}{Q_{ij}}$',16,ha='center')
text(600,77,r'$\mu=\mathrm{mean}_{\mathrm{TRAIN}}(h)$;  $c_{\mathrm{TRAIN}}=\mathrm{mean}_{i\in\mathrm{TRAIN},\,d}(h_{id}-\mu_d)^2$;  $\lambda=0$ gives the matched MSE-only control.',9,MUTED,ha='center')
text(42,24,'Schematic only. KL matches cell–cell relations; it is not a latent Gaussian prior or a cell-type classification loss.',9,MUTED)
for fmt in ['png','pdf','svg']:fig.savefig(O/f'Figure1_method_overview.{fmt}',dpi=350,facecolor='white')
plt.close(fig)
sources=['outputs/sae_lambda/reproducibility/frozen_relation_runtime.py','outputs/sae_lambda/reproducibility/frozen_sae_runtime.py']
(O/'Figure1_PROVENANCE.json').write_text(json.dumps({'type':'Method schematic; no quantitative experimental results','source_checks':{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},'verified_details':['512D frozen scGPT final-layer mean pooling','2048-coordinate raw-value TopK32 with retained signs','fixed TRAIN mean and no extra decoder bias','TRAIN normalized MSE','centered cosine relations at temperature 0.1','KL teacher P to student Q, teacher detached','cross-TRAIN-donor in-batch anchors, batch256, four donors x64,192 anchors per query'],'relation_matrix_glyphs':'Illustrative patterns, not measured values','scope':'Relation-MSE/KL branch only; separate from later masked-gene-expression C optimization.'},ensure_ascii=False,indent=2))
print('Figure1 exported PNG PDF SVG')
