"""Compose Figure 4 from verified source tables; heatmap is the full-width focal panel."""
from pathlib import Path
import os,csv,json,hashlib,shutil
ROOT=Path("/Users/apple/Documents/Codex/2026-10-18/wo")
os.environ["MPLCONFIGDIR"]=str(ROOT/"work/sae_figures/mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.cm import ScalarMappable

OUT=ROOT/"outputs/sae_figures/Figure4_complete"
OUT.mkdir(exist_ok=True)
PREV=ROOT/"outputs/sae_figures/bcell_clean"
HEAT=ROOT/"outputs/sae_figures/bcell_functional_heatmap_v2"
hashes={}
def read(path):
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:return list(csv.DictReader(f))
activ=read(PREV/"panel_A_activation_all_cells.csv")
curves=read(PREV/"panel_C_per_cell_replacement.csv")
summary=read(PREV/"panel_C_mean_replacement.csv")
heat=read(HEAT/"heatmap_source_data.csv")
annotations=read(HEAT/"gene_groups_and_evidence.csv")
genes=[r["gene"] for r in annotations]
groups=list(dict.fromkeys(r["display_group"] for r in annotations))
assignment={r["gene"]:r["display_group"] for r in annotations}
donors=list(dict.fromkeys(r["row"] for r in heat))
lookup={(r["row"],r["gene"]):r for r in heat}
assert len(genes)==40 and len(donors)==5 and donors[-1]=="Mean" and len(lookup)==200
values=np.array([[float(lookup[d,g]["contribution"]) for g in genes] for d in donors])
missing=np.array([[lookup[d,g]["not_tokenized"]=="True" for g in genes] for d in donors])
np.testing.assert_allclose(values[-1],values[:4].mean(0),rtol=0,atol=1e-12)
prototypes=sorted({r["prototype_index"] for r in curves})
assert len(prototypes)==4
for r in summary:
    vv=[float(q["drop"]) for q in curves if q["method"]==r["method"] and q["requested_genes"]==r["requested_genes"]]
    np.testing.assert_allclose(np.mean(vv),float(r["drop"]),rtol=0,atol=1e-12)

INK="#222222";GRAY="#989FA5";BLUE="#4185AB"
plt.rcParams.update({"font.family":"Arial","font.size":11,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.edgecolor":"#333333","axes.linewidth":.65,"pdf.fonttype":42,"svg.fonttype":"none"})
fig=plt.figure(figsize=(14.8,7.4),facecolor="white")
axa=fig.add_axes([.082,.625,.395,.278])
axb=fig.add_axes([.635,.625,.347,.278])
axc=fig.add_axes([.082,.215,.9,.212])
for x,y,letter,title in [
    (.082,.943,"a","F160 activation across cell types"),
    (.635,.943,"b","F160 response to gene replacement"),
    (.082,.505,"c","F160 gene contributions by function"),
]:
    fig.text(x-.032,y,letter,fontsize=15,weight="bold",va="center")
    fig.text(x,y,title,fontsize=12,va="center")

classes=["B cells","CD14+ Monocytes","CD4 T cells","CD8 T cells","Dendritic cells","FCGR3A+ Monocytes","NK cells"]
short=["B cells","CD14+ Mono","CD4 T","CD8 T","DC","FCGR3A+ Mono","NK"]
arrays=[np.array([float(r["F160"]) for r in activ if r["cell_type"]==c]) for c in classes]
box=axa.boxplot(arrays,positions=np.arange(7),widths=.46,showfliers=False,whis=(5,95),
    patch_artist=True,boxprops=dict(linewidth=.75),medianprops=dict(color=INK,linewidth=1),
    whiskerprops=dict(color=GRAY,linewidth=.7),capprops=dict(color=GRAY,linewidth=.7))
rng=np.random.default_rng(20261021)
for i,(v,p) in enumerate(zip(arrays,box["boxes"])):
    p.set_facecolor("#D3E5EF" if i==0 else "#E9ECEF")
    p.set_edgecolor(BLUE if i==0 else GRAY)
    ix=rng.choice(len(v),min(40,len(v)),replace=False)
    axa.scatter(i+rng.uniform(-.16,.16,len(ix)),v[ix],s=8,c=BLUE if i==0 else "#AFB5BA",
                alpha=.6,edgecolors="none",zorder=3)
axa.set_xticks(range(7),short,fontsize=10,rotation=30,ha="right",rotation_mode="anchor")
axa.tick_params(axis="x",pad=7)
axa.set_ylabel("F160 activation",labelpad=7)
axa.set_xlim(-.55,6.55)
axa.set_ylim(min(0,min(v.min() for v in arrays))-.045,max(v.max() for v in arrays)*1.06)
axa.set_yticks([0,.2,.4,.6,.8,1.0])

for method,col,ls,label in [("sae_F160",BLUE,"-","Attribution-ranked"),("matched_random",GRAY,"--","Matched random")]:
    for pid in prototypes:
        rr=sorted([r for r in curves if r["method"]==method and r["prototype_index"]==pid],key=lambda r:int(r["requested_genes"]))
        axb.plot([int(r["requested_genes"]) for r in rr],[float(r["drop"]) for r in rr],color=col,alpha=.27,lw=.7,ls=ls)
    rr=sorted([r for r in summary if r["method"]==method],key=lambda r:int(r["requested_genes"]))
    axb.plot([int(r["requested_genes"]) for r in rr],[float(r["drop"]) for r in rr],
        color=col,lw=1.8,marker="o" if method=="sae_F160" else "s",ms=3.5,ls=ls,label=label)
axb.set_xlim(-1,42)
axb.set_ylim(-.01,max(float(r["drop"]) for r in curves)*1.14)
axb.set_xticks([0,10,20,30,40])
axb.set_xlabel("Genes selected for replacement",labelpad=7)
axb.set_ylabel("F160 activation drop",labelpad=7)
axb.legend(loc="upper left",frameon=False,fontsize=10,handlelength=2.2,borderaxespad=.2)
for ax in [axa,axb]:
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(False);ax.tick_params(width=.65,length=3,direction="out")

cmap=LinearSegmentedColormap.from_list("contribution",["#3979A7","#94BBD4","#FAFAF8","#E8A276","#B94E32"])
norm=Normalize(vmin=-.04,vmax=.04,clip=False)
positions={};bounds={};cursor=0
for group in groups:
    gg=[g for g in genes if assignment[g]==group]
    bounds[group]=(cursor,cursor+len(gg))
    for g in gg:positions[g]=cursor;cursor+=1
    cursor+=.48
extent=cursor-.48;ypos=[0,1,2,3,4.32]
for i,y in enumerate(ypos):
    for j,g in enumerate(genes):
        axc.add_patch(Rectangle((positions[g],y),1,1,
            fc="#D8D8D8" if missing[i,j] else cmap(norm(values[i,j])),ec="white",lw=.4))
for group in groups:
    left,right=bounds[group]
    axc.plot([left+.02,right-.02],[-.19,-.19],color="#777777",lw=.65,solid_capstyle="butt",clip_on=False)
    label=group.replace("Antigen presentation","Antigen\npresentation")
    axc.text((left+right)/2,-.64,label,ha="center",va="center",fontsize=10.5,linespacing=1.05,clip_on=False)
axc.set_xlim(-.02,extent+.02);axc.set_ylim(5.34,-.02)
axc.set_xticks([positions[g]+.5 for g in genes],genes,rotation=90,fontsize=10.5,fontstyle="italic")
axc.set_yticks([y+.5 for y in ypos],[d.replace("patient_","Donor ") for d in donors],fontsize=10.5)
axc.get_yticklabels()[-1].set_weight("bold")
axc.tick_params(axis="both",length=0,pad=6)
for s in axc.spines.values():s.set_visible(False)
cax=fig.add_axes([.39,.075,.24,.014])
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal",ticks=[-.04,-.02,0,.02,.04])
cb.solids.set_rasterized(False);cb.outline.set_visible(False)
cb.ax.tick_params(length=2,width=.6,labelsize=10)
cb.set_label("Contribution to F160",fontsize=11,labelpad=5)
fig.legend(handles=[Patch(fc="#D8D8D8",ec="none",label="Not in input")],
    loc="center left",bbox_to_anchor=(.67,.08),frameon=False,fontsize=10)

stem=OUT/"Figure4_Bcell_F160"
for ext in ["png","pdf","svg"]:
    fig.savefig(stem.with_suffix("."+ext),dpi=400,facecolor="white")
fig.savefig(ROOT/"work/sae_figures/Figure4_complete_preview.png",dpi=140)
plt.close(fig)
for path in [PREV/"panel_A_activation_all_cells.csv",PREV/"panel_C_per_cell_replacement.csv",
             PREV/"panel_C_mean_replacement.csv",HEAT/"heatmap_source_data.csv",HEAT/"gene_groups_and_evidence.csv"]:
    shutil.copy2(path,OUT/path.name)
old_caption=(PREV/"CAPTION_ZH.txt").read_text()
parts=old_caption.splitlines()
activation_caption=next(x for x in parts if x.startswith("A："))
replacement_caption=next(x for x in parts if x.startswith("C："))
replacement_caption="B："+replacement_caption[2:]
heat_caption=(HEAT/"CAPTION_ZH.txt").read_text()
(OUT/"CAPTION_ZH.txt").write_text("Figure 4 | F160的细胞类型关联、基因替换响应与功能分组贡献。\n\n"+
    activation_caption+"\n\n"+replacement_caption+"\n\nC："+heat_caption+
    "\n三面板中的F160均来自同一历史主模型；A/B使用历史留出供体，C为TRAIN四供体。面板B只有同一供体的4个选定高激活细胞，不作为四个供体重复。所有样本量、分组与来源限制保留于图注和数据文件，未在图内重复列小字。\n")
hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/"PROVENANCE.json").write_text(json.dumps({"layout":"a activation top left; b replacement top right; c full-width grouped heatmap below",
    "changes":"layout and typography only; original data, gene set, donor weighting and raw contribution scale retained",
    "sources_sha256":hashes,"checks":"donor mean identities and replacement averages verified to 1e-12"},indent=2))
print("Saved",stem)
