"""Existing task-specific results; method identities remain explicit in every panel."""
from pathlib import Path
import os,csv,json,hashlib
ROOT=Path("/Users/apple/Documents/Codex/2026-10-18/wo")
os.environ["MPLCONFIGDIR"]=str(ROOT/"work/sae_figures/mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=ROOT/"outputs/sae_figures/performance"
sources={}
def read(path):
    sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:return list(csv.DictReader(f))
def write(name,rows):
    with (OUT/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
datasets=["ctrl","stim","pbmc"];names=["Kang ctrl","Kang stim","PBMC"]
expr=read(ROOT/"outputs/sae_checkpoint_extension/combined_with_ctrl_metrics.csv")
knn=read(ROOT/"outputs/sae_checkpoint_selection/knn_metrics.csv")+read(ROOT/"outputs/sae_checkpoint_extension/knn_metrics.csv")
cluster=read(ROOT/"outputs/sae_evaluation/clustering_all.csv")
selections=read(OUT/"selection_registry.csv")
expr_seeds={}
for ds in datasets:
    scores={seed:np.mean([float(r["MAE"]) for r in expr if r["dataset"]==ds and r["method"]=="C_selected" and r["seed"]==str(seed)]) for seed in [42,43,44]}
    assert all(np.isfinite(v) for v in scores.values())
    expr_seeds[ds]=min(scores,key=scores.get)
data_a,data_b,data_c=[],[],[]
for ds in datasets:
    n=2 if ds=="pbmc" else 6
    for method in ["original_h","C_selected"]:
        rr=[r for r in expr if r["dataset"]==ds and r["method"]==method and r["seed"]==str(expr_seeds[ds])]
        assert len(rr)==n
        for r in rr:
            data_a.append(dict(dataset=ds,group=r["group"],method=method,seed=r["seed"],MAE=float(r["MAE"])))
        for view in ["clean","masked"]:
            rr=[r for r in knn if r["dataset"]==ds and r["method"]==method and r["seed"]==str(expr_seeds[ds]) and r["view"]==view]
            assert len(rr)==n
            for r in rr:
                data_b.append(dict(dataset=ds,group=r["group"],method=method,seed=r["seed"],view=view,macro_f1_percent=float(r["knn_macro_f1"])*100))
    for method,display in [("original_h",None),("mse","MSE"),("relation_0p3","MSE + KL (lambda=0.3)")]:
        seed="shared" if display is None else next(r["seed"] for r in selections if r["dataset"]==ds and r["display_row"]==display)
        rr=[r for r in cluster if r["dataset"]==ds and r["method"]==method and r["seed"]==seed and float(r["resolution"])==1.]
        assert len(rr)==n,(ds,method,seed,len(rr))
        for r in rr:data_c.append(dict(dataset=ds,group=r["group"],method=method,seed=seed,ARI=float(r["ARI"])))
write("secondary_A_expression.csv",data_a);write("secondary_B_robustness.csv",data_b);write("secondary_C_clustering.csv",data_c)
plt.rcParams.update({"font.family":"Arial","font.size":10,"text.color":"#252525","axes.labelcolor":"#252525",
                     "axes.edgecolor":"#555555","axes.linewidth":.65,"pdf.fonttype":42,"svg.fonttype":"none"})
fig=plt.figure(figsize=(14.5,4.8),facecolor="white")
axes=[fig.add_axes([.065+i*.325,.285,.26,.52]) for i in range(3)]
for ax,i,title in zip(axes,range(3),["Expression prediction","Masked-input robustness","Clustering structure"]):
    fig.text(.028+i*.325,.905,"abc"[i],fontsize=13,weight="bold")
    fig.text(.065+i*.325,.905,title,fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xticks(range(3),names,fontsize=9)
    ax.set_xlim(-.45,2.45)
    ax.tick_params(length=3,labelsize=9)
colors={"original_h":"#92999E","C_selected":"#397C9E","mse":"#9BC4D8","relation_0p3":"#397C9E"}
summaries=[]
ax=axes[0]
for i,ds in enumerate(datasets):
    rr=[r for r in data_a if r["dataset"]==ds]
    for g in sorted(set(r["group"] for r in rr)):
        vals=[next(r["MAE"] for r in rr if r["group"]==g and r["method"]==m) for m in ["original_h","C_selected"]]
        ax.plot([i-.12,i+.12],vals,color="#D8DDE0",lw=.7,zorder=1)
    for method,off in [("original_h",-.12),("C_selected",.12)]:
        vals=[r["MAE"] for r in rr if r["method"]==method]
        ax.scatter(np.full(len(vals),i+off),vals,s=12,color=colors[method],alpha=.45,edgecolors="none",zorder=2)
        avg=float(np.mean(vals))
        ax.scatter(i+off,avg,s=46,color=colors[method],edgecolors="white",linewidth=.7,zorder=3)
        summaries.append(dict(panel="a",dataset=ds,method=method,condition="masked",value=avg))
ax.set_ylabel("Masked-gene MAE",labelpad=7)
ax.set_ylim(.285,.41)

ax=axes[1]
for i,ds in enumerate(datasets):
    for method,off in [("original_h",-.12),("C_selected",.12)]:
        vals=[float(np.mean([r["macro_f1_percent"] for r in data_b if r["dataset"]==ds and r["method"]==method and r["view"]==view])) for view in ["clean","masked"]]
        ax.plot([i+off,i+off],vals,lw=1.6,color=colors[method])
        ax.scatter(i+off,vals[0],facecolors="white",edgecolors=colors[method],s=45,linewidths=1.2,zorder=3)
        ax.scatter(i+off,vals[1],facecolors=colors[method],edgecolors="white",s=45,linewidths=.6,zorder=4)
        for view,v in zip(["clean","masked"],vals):
            summaries.append(dict(panel="b",dataset=ds,method=method,condition=view,value=v))
ax.set_ylabel("15-NN Macro-F1 (%)",labelpad=7)
ax.set_ylim(65,85)

ax=axes[2]
for i,ds in enumerate(datasets):
    rr=[r for r in data_c if r["dataset"]==ds]
    for g in sorted(set(r["group"] for r in rr)):
        vals=[next(r["ARI"] for r in rr if r["group"]==g and r["method"]==m) for m in ["original_h","mse","relation_0p3"]]
        ax.plot([i-.18,i,i+.18],vals,color="#D8DDE0",lw=.65,zorder=1)
    for method,off in [("original_h",-.18),("mse",0),("relation_0p3",.18)]:
        vals=[r["ARI"] for r in rr if r["method"]==method]
        ax.scatter(np.full(len(vals),i+off),vals,s=12,color=colors[method],alpha=.45,edgecolors="none",zorder=2)
        avg=float(np.mean(vals))
        ax.scatter(i+off,avg,s=46,color=colors[method],edgecolors="white",linewidth=.7,zorder=3)
        summaries.append(dict(panel="c",dataset=ds,method=method,condition="clean",value=avg))
ax.set_ylabel("ARI",labelpad=7)
ax.set_ylim(0,.85)
def handle(c,label):
    return Line2D([],[],ls="",marker="o",color=c,markersize=5,label=label)
for ax in axes[:2]:
    ax.legend(handles=[handle(colors["original_h"],"Original"),handle(colors["C_selected"],"Expression SAE")],
              loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,fontsize=9,handletextpad=.3,columnspacing=1)
extra=axes[1].legend(handles=[
    Line2D([],[],ls="",marker="o",markerfacecolor="white",markeredgecolor="#555555",markersize=5,label="Clean"),
    Line2D([],[],ls="",marker="o",color="#555555",markersize=5,label="25% masked")
],loc="upper center",bbox_to_anchor=(.5,-.32),ncol=2,frameon=False,fontsize=8.5,handletextpad=.3,columnspacing=1)
# Preserve both method and view legends.
from matplotlib.legend import Legend
method_legend=Legend(axes[1],[handle(colors["original_h"],"Original"),handle(colors["C_selected"],"Expression SAE")],
                     ["Original","Expression SAE"],loc="upper center",bbox_to_anchor=(.5,-.19),
                     ncol=2,frameon=False,fontsize=9,handletextpad=.3,columnspacing=1)
axes[1].add_artist(method_legend)
axes[2].legend(handles=[handle(colors["original_h"],"Original"),handle(colors["mse"],"MSE"),
                        handle(colors["relation_0p3"],r"KL ($\lambda=0.3$)")],
               loc="upper center",bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=9,handletextpad=.3,columnspacing=.8)
stem=OUT/"Secondary_performance_three_panels"
for ext in ["png","pdf","svg"]:fig.savefig(stem.with_suffix("."+ext),dpi=400,facecolor="white")
fig.savefig(ROOT/"work/sae_figures/secondary_performance_preview.png",dpi=150)
plt.close(fig)
write("secondary_panel_means.csv",summaries)
caption="""三项下游评估（已有任务特定实验的排版稿）。
重要方法区分：a、b 使用表达监督 SAE（C_selected，按 VALID 表达重建 MSE 选择 body checkpoint）；c 使用纯 embedding 重建 SAE（MSE / MSE+KL，λ=0.3）。a、b 的改善不能归因于 c 或分类表中的 KL 正则，不代表一个模型同时改善全部任务。图面直接使用不同方法名称，未统称为 Ours。
a：在相同遮蔽输入上，由冻结表征训练的独立线性探针预测被遮蔽阳性 gene tokens。指标为 token-level MAE。每个数据集选 C_selected TEST MAE 最低的种子，ctrl=44、stim=43、PBMC=43，并取原始 h 的同种子探针对照。淡点为各留出组，灰线连接同组，实心大点为组等权均值；Kang 各六供体，PBMC 两批次方向。该选种子方式是历史 TEST 最佳展示，有选择偏差，不是无偏泛化估计。
b：沿用 a 的同一个表达 SAE 种子，不再为鲁棒性另选种子。比较 clean 输入与预先固定的 25% 阳性 token 遮蔽输入的 15NN Macro-F1；空心点为 clean，实心点为 masked，短线连接二者的组等权均值。无额外遮蔽比例，无 UMI 下采样实验，不把两个条件补成多强度曲线。评价使用已有固定参考和读出协议；较低下降量不等于较高绝对性能，因此同时展示 clean/masked 的绝对分数。无误差条，组级原始数值在 secondary_B_robustness.csv。
c：TEST 内 cosine 15NN 图、Leiden resolution=1.0 的 ARI，聚类算法固定随机状态 20261019。使用 λ=0.3 的已有模型，而非 stim/PBMC 分类表中 λ=0.03 的最佳模型。MSE 和 KL 各按对应配置的分类 Macro-F1 选择种子，不根据 ARI 再选：ctrl MSE42/KL42、stim MSE43/KL42、PBMC MSE43/KL43；Original 为共享原始表征。淡点为组，连线连接同组，大点为组等权平均。未用 TEST 标签优化 resolution。
本图仅汇总已有结果，没有新训练。表达模型与 KL 模型的目标不同，本版用于讨论两类已有分析；若要证明同一 MSE+KL 模型的多任务收益，必须补充相应表达预测和遮蔽鲁棒性评估。各任务都保留未改善结果，不把不同模型和不同种子的最优指标拼成不存在的单个模型。种子未标在图面，但在图注与源数据中完整披露。没有显著性标记。
"""
(OUT/"SECONDARY_CAPTION_ZH.txt").write_text(caption)
sources[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/"SECONDARY_PROVENANCE.json").write_text(json.dumps({"sources_sha256":sources,"expression_seeds":expr_seeds,
    "status":"Task-specific existing-results draft; not a unified-model evaluation","panel_means":summaries},indent=2))
print(json.dumps({"expression_seeds":expr_seeds,"means":summaries},indent=2))
