"""Plot fixed TRAIN top40 attribution genes with source-backed functional grouping."""
from pathlib import Path
import os, csv, json, hashlib
ROOT=Path("/Users/apple/Documents/Codex/2026-10-18/wo")
os.environ["MPLCONFIGDIR"]=str(ROOT/"work/sae_figures/mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

OUT=ROOT/"outputs/sae_figures/bcell_functional_heatmap"
OUT.mkdir(exist_ok=True)
SOURCE=ROOT/"outputs/sae_gene_attribution_v2"
SETS=ROOT/"work/sae_pathways_v3/gene_sets"
hashes={}
def rec(path):
    path=Path(path);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return path
def read(path):
    with rec(path).open() as f:return list(csv.DictReader(f))
def write(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sets={}
for filename in ["c5.go.bp.v2025.1.Hs.symbols.gmt","c2.cp.reactome.v2025.1.Hs.symbols.gmt"]:
    with rec(SETS/filename).open() as f:
        for line in f:
            name,url,*genes=line.rstrip("\n").split("\t")
            sets[name]={"url":url,"genes":set(genes)}
all_ranked=sorted(read(SOURCE/"gene_rankings_train.csv"),key=lambda r:-float(r["sae_F160"]))
top=all_ranked[:40]
assert len(top)==40 and all(float(r["sae_F160"])>0 for r in top)
topgenes=[r["gene"] for r in top]
rankmap={r["gene"]:i+1 for i,r in enumerate(all_ranked)}
meta={r["gene"]:r for r in top}
terms={
    "Antigen presentation":["GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION"],
    "Metabolism":["REACTOME_GLUCOSE_METABOLISM","REACTOME_PYRUVATE_METABOLISM"],
    "Cytoskeleton":["GOBP_CYTOSKELETON_ORGANIZATION","GOBP_INTERMEDIATE_FILAMENT_ORGANIZATION"],
    "RNA / translation":["GOBP_TRANSLATION","GOBP_POST_TRANSCRIPTIONAL_REGULATION_OF_GENE_EXPRESSION"],
}
assert all(term in sets for values in terms.values() for term in values)
# One visual group per gene. Multiple memberships are recorded, not inferred from F160.
# Prioritize metabolism before cytoskeleton/RNA for the multifunctional gene GAPDH.
priority=["Antigen presentation","Metabolism","Cytoskeleton","RNA / translation"]
groups=["Ribosomal proteins","RNA / translation","Antigen presentation","Cytoskeleton","Metabolism","Other"]
assignment={}
membership={}
for gene in topgenes:
    membership[gene]=[term for group in priority for term in terms[group] if gene in sets[term]["genes"]]
    if gene.startswith(("RPL","RPS")):
        assignment[gene]="Ribosomal proteins"
    else:
        assignment[gene]=next((group for group in priority if any(gene in sets[t]["genes"] for t in terms[group])),"Other")
# NPM1 is annotated to both RNA/translation and cytoskeleton. Retain its planned
# RNA/translation display group; this annotation-only override does not select genes.
assert "GOBP_TRANSLATION" in membership["NPM1"]
assignment["NPM1"]="RNA / translation"
ordered=[g for group in groups for g in topgenes if assignment[g]==group]
assert len(ordered)==40 and set(ordered)==set(topgenes)

proto=[r for r in read(SOURCE/"per_prototype_gene_attribution.csv") if r["role"]=="train"]
donors=sorted({r["donor"] for r in proto})
assert len(donors)==4
ids={d:sorted({r["prototype_index"] for r in proto if r["donor"]==d}) for d in donors}
assert all(len(v)==2 for v in ids.values())
lookup={(r["prototype_index"],r["gene"]):float(r["sae_F160"]) for r in proto}
assert len(lookup)==len(proto)
values=np.array([[np.mean([lookup.get((pi,g),0.) for pi in ids[d]]) for g in ordered] for d in donors])
coverage=np.array([[sum((pi,g) in lookup for pi in ids[d]) for g in ordered] for d in donors])
mean=values.mean(axis=0)
sd=values.std(axis=0,ddof=1)
np.testing.assert_allclose(mean,[float(meta[g]["sae_F160"]) for g in ordered],rtol=0,atol=1e-12)
np.testing.assert_allclose(sd,[float(meta[g]["sae_F160_donor_sd"]) for g in ordered],rtol=0,atol=1e-12)
np.testing.assert_array_equal(coverage.sum(axis=0),[int(meta[g]["input_prototypes"]) for g in ordered])
matrix=np.vstack([values,mean])
plot_cover=np.vstack([coverage,coverage.sum(axis=0)])

annotations=[]
for g in ordered:
    annotations.append({
        "gene":g,"TRAIN_rank":rankmap[g],"display_group":assignment[g],
        "matched_annotation_terms":";".join(membership[g]),
        "annotation_urls":";".join(sets[t]["url"] for t in membership[g]),
        "ribosomal_prefix":g.startswith(("RPL","RPS")),
        "display_override":"NPM1: RNA / translation (overlapping annotations)" if g=="NPM1" else "",
        "TRAIN_mean":float(meta[g]["sae_F160"]),"TRAIN_donor_SD":float(meta[g]["sae_F160_donor_sd"]),
        "input_prototypes":int(meta[g]["input_prototypes"]),"total_prototypes":8,
    })
write("gene_groups_and_evidence.csv",annotations)
rows=[]
for i,d in enumerate(donors+["Mean"]):
    for j,g in enumerate(ordered):
        rows.append({"row":d,"gene":g,"display_group":assignment[g],"contribution":matrix[i,j],
                     "input_prototypes":int(plot_cover[i,j]),"total_prototypes":2 if i<4 else 8,
                     "not_tokenized":bool(plot_cover[i,j]==0)})
write("heatmap_source_data.csv",rows)
write("fixed_TRAIN_top40.csv",top)

plt.rcParams.update({
    "font.family":"Arial","font.size":10,"text.color":"#222222","axes.labelcolor":"#222222",
    "xtick.color":"#222222","ytick.color":"#222222","pdf.fonttype":42,"svg.fonttype":"none",
})
# Raw values, symmetric fixed scale; neither row nor column z-scoring.
cmap=LinearSegmentedColormap.from_list("contribution",["#3979A7","#94BBD4","#FAFAF8","#E8A276","#B94E32"])
norm=Normalize(vmin=-.04,vmax=.04,clip=False)
missing="#D8D8D8"
colors=["#9AAEBB","#B3A4C4","#A4BCA5","#C2B397","#9BBEBD","#BDBDBD"]
labels=["Ribosomal proteins","RNA / translation","Antigen\npresentation","Cytoskeleton","Metabolism","Other"]
fig=plt.figure(figsize=(14.8,4.05),facecolor="white")
ax=fig.add_axes([.082,.355,.9,.41])
positions={};bounds={};cursor=0.
for group in groups:
    gg=[g for g in ordered if assignment[g]==group]
    bounds[group]=(cursor,cursor+len(gg))
    for g in gg:
        positions[g]=cursor;cursor+=1
    cursor+=.62
extent=cursor-.62
ypos=[0,1,2,3,4.45]
for i,y in enumerate(ypos):
    for j,g in enumerate(ordered):
        x=positions[g]
        ax.add_patch(Rectangle((x,y),1,1,facecolor=missing if plot_cover[i,j]==0 else cmap(norm(matrix[i,j])),
                               edgecolor="white",linewidth=.6))
# No rainbow categories in the cells: the narrow strip only indicates functional blocks.
for group,label,col in zip(groups,labels,colors):
    start,end=bounds[group]
    ax.plot([start+.02,end-.02],[-.20,-.20],color=col,lw=3,solid_capstyle="butt",clip_on=False)
    ax.text((start+end)/2,-.40,label,ha="center",va="bottom",fontsize=10,linespacing=1.1,clip_on=False)
ax.set_xlim(-.02,extent+.02)
ax.set_ylim(5.47,-.02)
ax.set_xticks([positions[g]+.5 for g in ordered],ordered,rotation=90,fontsize=9,fontstyle="italic")
ax.set_yticks([y+.5 for y in ypos],[d.replace("patient_","Donor ") for d in donors]+["Mean"],fontsize=10)
ax.get_yticklabels()[-1].set_weight("bold")
ax.tick_params(axis="both",length=0,pad=6)
for s in ax.spines.values():s.set_visible(False)
fig.text(.5,.958,"Functional grouping of F160 gene contributions",ha="center",fontsize=13)

cax=fig.add_axes([.364,.125,.29,.028])
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal",ticks=[-.04,-.02,0,.02,.04])
cb.outline.set_visible(False)
cb.ax.tick_params(length=2,width=.6,labelsize=9)
cb.set_label("Contribution to F160",fontsize=10,labelpad=5)
if (plot_cover==0).any():
    fig.legend(handles=[Patch(facecolor=missing,edgecolor="none",label="Not in input")],
               loc="center left",bbox_to_anchor=(.70,.135),frameon=False,fontsize=9)
stem=OUT/"F160_functional_gene_contributions"
for ext in ["png","pdf","svg"]:
    fig.savefig(stem.with_suffix("."+ext),dpi=400,facecolor="white")
fig.savefig(ROOT/"work/sae_figures/grouped_heatmap_preview.png",dpi=140)
plt.close(fig)

caption="""F160 基因贡献的功能分组热图。
数据：Kang ctrl 主模型（seed42，TopK32）；固定原有TRAIN平均归因排名前40个正贡献基因，没有根据功能类别、TEST表现或本图颜色重新挑选。组内仍按原始TRAIN平均贡献递减排序，完整名次见gene_groups_and_evidence.csv。
行：四个TRAIN供体及其等权均值。每个供体包含两个预先选定的F160高激活B细胞，每个原型使用至多三个同供体同细胞类型、F160零激活的近邻参照。各原型的归因已对参照等权平均，再在供体内对两个原型等权平均，最后对四供体等权平均。
颜色：原始有符号贡献，即将单个基因的分箱输入替换为参照值后，F160激活的下降量（原始激活减替换后激活）。橙色为正贡献，蓝色为负贡献，接近白色为零；色标对称固定为[-0.04,0.04]，未截断数据、未作逐基因或逐供体z-score。颜色不是基因表达量，也不是通路激活分数。
未进入原型输入token的基因按照原始分析计零再参与供体和全体均值。若一个基因在某供体的两个原型中都未被token化，则该供体格子显示灰色（Not in input），以区别于可观测的零响应；Mean仍遵循原始零填充规则。一个原型中存在而另一个不存在时，格子为含零填充的供体平均；覆盖度见源数据。灰格不表示基因没有生物学作用。
分组：核糖体蛋白组依据RPL/RPS前缀；其他组使用本项目已保存的MSigDB 2025.1 human GO BP / Reactome成员关系。为了每基因只显示一次，默认赋组优先级为：核糖体蛋白、抗原呈递、代谢、细胞骨架、RNA/翻译、Other；唯一人工展示例外为同时具有RNA/翻译与细胞骨架注释的NPM1，依据其GOBP_TRANSLATION成员关系按预先讨论的方案放入RNA/翻译组。显示顺序为核糖体蛋白、RNA/翻译、抗原呈递、细胞骨架、代谢、Other。一个基因可能具有多种功能，所有匹配的候选注释都保留在配套CSV。例如GAPDH同时有代谢、细胞骨架、RNA相关注释，此处根据上述规则展示在代谢组。所有分组均不改变原始top40基因集及其贡献值。
RNA/翻译：GOBP_TRANSLATION与GOBP_POST_TRANSCRIPTIONAL_REGULATION_OF_GENE_EXPRESSION；抗原呈递：GOBP_ANTIGEN_PROCESSING_AND_PRESENTATION；细胞骨架：GOBP_CYTOSKELETON_ORGANIZATION与GOBP_INTERMEDIATE_FILAMENT_ORGANIZATION；代谢：REACTOME_GLUCOSE_METABOLISM与REACTOME_PYRUVATE_METABOLISM。Other仅表示未落入本图选择的注释组，不代表功能未知。
这些分区是已有基因功能注释，不是新发现的独立功能模块，也不是显著富集证据。本项目之前的五个候选功能条目未通过匹配随机全局FDR<0.05。本图不进行新的富集、训练或显著性检验，不声称生物因果关系。
绘制前将由原型表重算的40个基因供体等权均值、供体间SD和输入覆盖度与原始gene_rankings_train.csv逐一核对，误差在1e-12以内。
"""
(OUT/"CAPTION_ZH.txt").write_text(caption)
rec(Path(__file__))
result={
    "groups":{group:[g for g in ordered if assignment[g]==group] for group in groups},
    "selection":"fixed original TRAIN top40; all retained; reordered by documented annotation",
    "assignment_priority":["Ribosomal proteins"]+priority+["Other"],
    "annotation_only_override":{"NPM1":"RNA / translation; GOBP_TRANSLATION membership; overlapping annotations"},
    "donors":donors,"raw_value_range":[float(matrix.min()),float(matrix.max())],
    "color_range":[-.04,.04],"missing_donor_gene_cells":int((coverage==0).sum()),
    "normalization":"none; raw signed contributions",
    "checks":"all mean, donor SD and coverage match original table; no changed gene set",
    "sources_sha256":hashes,
}
(OUT/"PROVENANCE.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!="sources_sha256"},indent=2))
