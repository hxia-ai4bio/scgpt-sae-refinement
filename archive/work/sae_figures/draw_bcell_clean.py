"""Replot existing F160 attribution evidence as a concise three-panel figure."""
from pathlib import Path
import os, csv, json, hashlib
ROOT=Path("/Users/apple/Documents/Codex/2026-10-18/wo")
os.environ["MPLCONFIGDIR"]=str(ROOT/"work/sae_figures/mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT=ROOT/"outputs/sae_figures/bcell_clean"
OUT.mkdir(exist_ok=True)
INPUTS={}
def read(path):
    path=Path(path)
    INPUTS[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:return list(csv.DictReader(f))
def write(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

act=read(ROOT/"outputs/sae_figures/Figure3_A_TEST_activation.csv")
assert len(act)==1723 and {r["donor"] for r in act}=={"patient_1016"}
cachepath=ROOT/"work/sae_interpretability/primary_activations.npz"
INPUTS[str(cachepath)]=hashlib.sha256(cachepath.read_bytes()).hexdigest()
with np.load(cachepath,allow_pickle=False) as z:
    np.testing.assert_allclose([float(r["F160"]) for r in act],
                               z["kang"][[int(r["canonical_index"]) for r in act],160],atol=1e-7,rtol=0)
classes=["B cells","CD14+ Monocytes","CD4 T cells","CD8 T cells","Dendritic cells","FCGR3A+ Monocytes","NK cells"]
short=["B cells","CD14+ Mono","CD4 T","CD8 T","DC","FCGR3A+ Mono","NK"]
arrays=[np.array([float(r["F160"]) for r in act if r["cell_type"]==c]) for c in classes]
assert all(len(x)>=20 for x in arrays)
write("panel_A_activation_all_cells.csv",act)

genes=read(ROOT/"outputs/sae_gene_attribution_v2/gene_rankings_train.csv")
ranked=sorted((r for r in genes if float(r["sae_F160"])>0),key=lambda r:-float(r["sae_F160"]))[:12]
for r in ranked:
    r["RPL_RPS_prefix"]=r["gene"].startswith(("RPL","RPS"))
write("panel_B_top12_gene_contributions.csv",ranked)

curves=[r for r in read(ROOT/"outputs/sae_gene_attribution_v2/deletion_per_prototype.csv")
        if r["score"]=="sae_F160" and r["method"] in ["sae_F160","matched_random"]]
summary=[r for r in read(ROOT/"outputs/sae_gene_attribution_v2/deletion_summary.csv")
         if r["score"]=="sae_F160" and r["method"] in ["sae_F160","matched_random"]]
prototypes=sorted({r["prototype_index"] for r in curves})
assert len(prototypes)==4
for r in summary:
    actual=[float(q["drop"]) for q in curves if q["method"]==r["method"] and q["requested_genes"]==r["requested_genes"]]
    assert len(actual)==4
    np.testing.assert_allclose(np.mean(actual),float(r["drop"]),atol=1e-12,rtol=0)
write("panel_C_per_cell_replacement.csv",curves)
write("panel_C_mean_replacement.csv",summary)
paths=read(ROOT/"outputs/sae_figures/Figure3_D_candidate_annotations.csv")
assert all(float(r["匹配随机全局q"])>=.05 for r in paths)
write("candidate_pathway_annotations.csv",paths)

INK="#202020";BODY="#202020";GRAY="#989FA5";BLUE="#4185AB";PALE="#E9ECEF"
plt.rcParams.update({
    "font.family":"Arial","font.size":10,"text.color":INK,"axes.labelcolor":BODY,
    "axes.edgecolor":"#303030","axes.linewidth":.65,"xtick.color":BODY,"ytick.color":BODY,
    "pdf.fonttype":42,"svg.fonttype":"none",
})
fig=plt.figure(figsize=(11.8,3.8),facecolor="white")
axa=fig.add_axes([.060,.26,.237,.60])
axb=fig.add_axes([.405,.26,.243,.60])
axc=fig.add_axes([.751,.26,.233,.60])
for x,letter,title in [
    (.06,"a","F160 activation across cell types"),
    (.405,"b","Genes contributing to F160"),
    (.751,"c","F160 response to gene replacement"),
]:
    fig.text(x-.036,.926,letter,fontsize=14,weight="bold")
    fig.text(x+.118,.926,title,fontsize=10.5,ha="center")

box=axa.boxplot(arrays,positions=np.arange(7),widths=.52,showfliers=False,
    whis=(5,95),patch_artist=True,
    boxprops=dict(linewidth=.75),medianprops=dict(color=INK,linewidth=1),
    whiskerprops=dict(color=GRAY,linewidth=.7),capprops=dict(color=GRAY,linewidth=.7))
rng=np.random.default_rng(20261021)
for i,(values,patch) in enumerate(zip(arrays,box["boxes"])):
    patch.set_facecolor("#D3E5EF" if i==0 else PALE)
    patch.set_edgecolor(BLUE if i==0 else GRAY)
    chosen=rng.choice(len(values),min(40,len(values)),replace=False)
    axa.scatter(i+rng.uniform(-.16,.16,len(chosen)),values[chosen],s=6,
                c=BLUE if i==0 else "#AFB5BA",alpha=.6,edgecolors="none",zorder=3)
axa.set_xticks(range(7),short,fontsize=9,rotation=45,ha="right",rotation_mode="anchor")
axa.set_ylabel("F160 activation")
axa.set_xlim(-.55,6.55)
# Cover every observed value, including values absent from the sampled overlay.
axa.set_ylim(min(0,min(v.min() for v in arrays))-.045,max(v.max() for v in arrays)*1.06)
axa.tick_params(axis="x",length=0,pad=7)

y=np.arange(len(ranked))
vals=np.array([float(r["sae_F160"]) for r in ranked])
sds=np.array([float(r["sae_F160_donor_sd"]) for r in ranked])
colors=[BLUE for r in ranked]
axb.barh(y,vals,color=colors,height=.65,xerr=sds,
         error_kw=dict(ecolor="#404040",elinewidth=.75,capsize=1.6,capthick=.65))
axb.set_yticks(y,[r["gene"] for r in ranked],fontsize=9.5,fontstyle="italic")
axb.invert_yaxis()
low=min(-.001,float(np.min(vals-sds))-.001)
high=float(np.max(vals+sds))+.003
axb.set_xlim(low,high)
axb.axvline(0,color="#A0A0A0",lw=.55)
axb.set_xticks([0,.01,.02,.03])
axb.set_xlabel("F160 activation drop",labelpad=7)

for method,col,ls,label in [
    ("sae_F160",BLUE,"-","Attribution-ranked"),
    ("matched_random",GRAY,"--","Matched random"),
]:
    for pid in prototypes:
        rows=sorted((r for r in curves if r["method"]==method and r["prototype_index"]==pid),
                    key=lambda r:int(r["requested_genes"]))
        axc.plot([int(r["requested_genes"]) for r in rows],[float(r["drop"]) for r in rows],
                 color=col,alpha=.27,lw=.65,ls=ls)
    means=sorted((r for r in summary if r["method"]==method),key=lambda r:int(r["requested_genes"]))
    xx=[int(r["requested_genes"]) for r in means]; yy=[float(r["drop"]) for r in means]
    axc.plot(xx,yy,color=col,lw=1.7,marker="o" if method=="sae_F160" else "s",ms=3,ls=ls,label=label)
axc.set_xlim(-1,42)
axc.set_ylim(-.01,max(float(r["drop"]) for r in curves)*1.14)
axc.set_xticks([0,10,20,30,40])
axc.set_xlabel("Genes selected for replacement",fontsize=10,labelpad=7)
axc.set_ylabel("F160 activation drop",labelpad=6)
axc.legend(loc="upper left",frameon=False,fontsize=9,handlelength=2.2,borderaxespad=.1)

for ax in [axa,axb,axc]:
    ax.spines[["top","right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.tick_params(width=.65,length=3,direction="out")
stem=OUT/"Bcell_F160_clean"
for ext in ["png","pdf","svg"]:
    fig.savefig(stem.with_suffix("."+ext),dpi=400,facecolor="white")
fig.savefig(ROOT/"work/sae_figures/bcell_clean_preview.png",dpi=150)
plt.close(fig)

(OUT/"CAPTION_ZH.txt").write_text("""F160：B 细胞相关的 SAE 特征及其局部基因敏感性。
A：Kang ctrl 主模型（seed42，TopK32）在历史留出供体 patient_1016 中的 F160 激活。箱体为四分位距，中线为中位数，须按第5和第95百分位设置；箱线统计使用各类全部细胞，散点仅用固定随机种子抽取每类至多40个细胞展示。展示至少20个TEST细胞的类别，省略仅1个细胞的巨核细胞。B、CD14+ Mono、CD4 T、CD8 T、DC、FCGR3A+ Mono、NK的细胞数依次为125、393、502、504、27、81、90。无显著性检验。
B：TRAIN 上按照F160平均激活下降排序的前12个基因。每次固定基因token和padding，将一个基因的表达分箱值替换为同供体同类型低激活参照细胞的分箱值，测量替换前后的F160激活差。8个高激活B细胞原型来自4个TRAIN供体，每原型使用至多3个近邻零激活参照；先平均参照、再平均原型、最后按供体等权平均。误差线为4个供体间SD，而非SEM或置信区间。未进入某原型输入token的基因计零；每个基因的覆盖度见配套数据，例如VIM仅在3/8个TRAIN原型输入中出现。条形统一使用蓝色，不将核糖体类别作为额外图例。排序未按TEST结果调整。
C：固定TRAIN基因排序，在同一历史TEST供体内选择的4个高激活B细胞中，累计替换请求的基因分箱值。蓝色为TRAIN归因排序，灰色虚线为联合匹配随机对照（原始分箱、参照替换方向、替换幅度分层匹配，每原型10个随机集平均）。浅色细线是单个原型曲线，粗线为4个原型等权平均。横轴为请求替换数，非所有原型实际替换的非零输入数。40个请求替换时，归因排序与匹配随机的平均F160下降分别为0.248289与0.135295。4个细胞来自同一供体，不能视为4个独立供体重复。未添加统计显著性。
C支持当前所选细胞和参照规则下的局部模型输入敏感性排序；不证明基因的生物因果作用、稳定独立的B细胞程序或SAE的全局解释性优势。当前替换输入下F160仍处于TopK内，不能将效应归因于TopK门控变化。
候选功能另附candidate_pathway_annotations.csv；5个固定候选条目均未通过匹配随机全局FDR<0.05，不能标注为已验证功能。其中抗原呈递和IFN-γ响应连普通ORA全局q也未达0.05。主图不使用显著性气泡或机制图标包装这些候选注释。
本图重绘已有实验结果，不增加模型训练、统计检验或TEST结果驱动的基因筛选。
""")
INPUTS[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/"PROVENANCE.json").write_text(json.dumps({
    "feature":160,"dataset":"Kang ctrl","primary_seed":42,"test_donor":"patient_1016",
    "training_prototypes":8,"training_donors":4,"test_prototypes":4,"test_donors":1,
    "genes":[r["gene"] for r in ranked],
    "verification":["all plotted activation values match primary cached F160","curve means reproduce original summary to 1e-12",
                    "top12 ranked using TRAIN only","all 5 candidate pathway matched-null q >= .05"],
    "source_sha256":INPUTS},indent=2))
print("Saved",stem)
