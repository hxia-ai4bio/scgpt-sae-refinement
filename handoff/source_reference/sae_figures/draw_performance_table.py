"""Descriptive best-seed classification table and lambda sweep, no new fitting."""
from pathlib import Path
import os, csv, json, hashlib
ROOT = Path("/Users/apple/Documents/Codex/2026-10-18/wo")
os.environ["MPLCONFIGDIR"] = str(ROOT / "work/sae_figures/mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "outputs/sae_figures/performance"
OUT.mkdir(parents=True, exist_ok=True)
sources = {}
def read(path):
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        return list(csv.DictReader(f))
def write(name, data):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

raw = read(ROOT / "outputs/sae_lambda/all_metrics.csv")
for r in raw:
    r["dataset"] = "ctrl"
    r["group"] = r["test_donor"]
extra = read(ROOT / "outputs/sae_extension/all_metrics.csv")
for r in extra:
    r["group"] = r["test_group"]
raw += extra
datasets = ["ctrl", "stim", "pbmc"]
names = {"ctrl":"Kang ctrl", "stim":"Kang stim", "pbmc":"PBMC"}
lambdas = [0, .01, .03, .1, .3, 1, 3]
selected, table, curves, stats = [], [], [], {}
seed_summary = read(ROOT / "outputs/sae_lambda/seed_metrics.csv")
for r in seed_summary: r["dataset"] = "ctrl"
seed_summary += read(ROOT / "outputs/sae_extension/seed_metrics.csv")
for ds in datasets:
    rr = [r for r in raw if r["dataset"] == ds]
    n_groups = 2 if ds == "pbmc" else 6
    original = [r for r in rr if r["method"] == "original_h"]
    assert len(original) == n_groups
    baseline = np.mean([float(r["macro_f1"]) for r in original]) * 100
    winners = []
    for lam in lambdas:
        candidates = []
        for seed in [42,43,44]:
            entries = [r for r in rr if r["method"] != "original_h" and
                       float(r["lambda_KL"]) == lam and r["seed"] == str(seed)]
            assert len(entries) == n_groups
            assert len(set(r["group"] for r in entries)) == n_groups
            score = float(np.mean([float(r["macro_f1"]) for r in entries]) * 100)
            check = [r for r in seed_summary if r["dataset"] == ds and
                     r["control"] == "original_h" and float(r["lambda_KL"]) == lam and r["seed"] == str(seed)]
            assert len(check) == 1
            np.testing.assert_allclose(score, float(check[0]["macro_f1_percent"]), rtol=0, atol=1e-10)
            candidates.append(dict(dataset=ds, lambda_KL=lam, seed=seed, macro_f1_percent=score,
                                   n_groups=n_groups, method=entries[0]["method"]))
        best = max(candidates, key=lambda r:(r["macro_f1_percent"],-r["seed"]))
        winners.append(best); curves.append(best)
    mse = winners[0]
    fixed = next(r for r in winners if r["lambda_KL"] == .3)
    best = max(winners[1:], key=lambda r:(r["macro_f1_percent"],-r["lambda_KL"],-r["seed"]))
    stats[ds] = dict(original=baseline, mse=mse, fixed=fixed, best=best)
    for name, r in [("MSE",mse),("MSE + KL (lambda=0.3)",fixed),("MSE + KL (best lambda)",best)]:
        selected.append(dict(display_row=name, **r))
    table.append(dict(dataset=ds, Original=baseline, MSE=mse["macro_f1_percent"],
                      KL_fixed_0p3=fixed["macro_f1_percent"], KL_best=best["macro_f1_percent"],
                      best_lambda=best["lambda_KL"], delta_best_pp=best["macro_f1_percent"]-baseline))
write("classification_table.csv",table)
write("lambda_best_seed_curve.csv",curves)
write("selection_registry.csv",selected)

plt.rcParams.update({"font.family":"Arial","font.size":9,"text.color":"#252525",
                     "axes.labelcolor":"#252525","axes.edgecolor":"#555555","axes.linewidth":.65,
                     "pdf.fonttype":42,"svg.fonttype":"none"})
fig = plt.figure(figsize=(11.5,7.0),facecolor="white")
ax = fig.add_axes([.065,.565,.90,.365]);ax.set_axis_off()
ax.set_xlim(0,1);ax.set_ylim(0,1)
colx = [.53,.735,.94]
ax.text(0,1.09,"Cell-type classification",fontsize=13,va="bottom")
ax.plot([0,1],[.98,.98],color="#252525",lw=1)
ax.text(.01,.895,"Representation",fontsize=10)
for x,ds in zip(colx,datasets):
    ax.text(x,.895,names[ds],fontsize=10,ha="center")
ax.plot([0,1],[.83,.83],color="#252525",lw=.65)
rowdefs=[("Original embedding","Original"),("SAE · MSE","MSE"),
         (r"SAE · MSE + KL ($\lambda=0.3$)","KL_fixed_0p3"),
         (r"SAE · MSE + KL (best $\lambda$)","KL_best")]
for y,(label,key) in zip([.72,.57,.42,.27],rowdefs):
    ax.text(.01,y,label,va="center",fontsize=10)
    for x,record in zip(colx,table):
        val=record[key]
        winner=max(record[k] for _,k in rowdefs)
        ax.text(x,y,f"{val:.2f}",va="center",ha="center",fontsize=11,
                weight="bold" if abs(val-winner)<1e-10 else "normal")
ax.plot([0,1],[.18,.18],color="#8D8D8D",lw=.55)
ax.text(.01,.085,r"$\Delta$ vs Original (best $\lambda$)",va="center",fontsize=10)
for x,record in zip(colx,table):
    ax.text(x,.085,f"{record['delta_best_pp']:+.2f} pp",ha="center",va="center",fontsize=10)
ax.plot([0,1],[.015,.015],color="#252525",lw=1)
fig.text(.065,.515,r"Sensitivity to $\lambda$",fontsize=12)
for i,ds in enumerate(datasets):
    axes=fig.add_axes([.075+i*.315,.135,.255,.285])
    vals=[r["macro_f1_percent"] for r in curves if r["dataset"]==ds]
    axes.plot(range(7),vals,marker="o",markersize=4,lw=1.35,color="#397C9E")
    axes.axhline(stats[ds]["original"],color="#969696",lw=1,ls=(0,(4,3)))
    axes.set_title(names[ds],fontsize=10,pad=9)
    axes.set_xticks(range(7),["0","0.01","0.03","0.1","0.3","1","3"],fontsize=8)
    axes.set_xlabel(r"$\lambda$",labelpad=4)
    if i==0: axes.set_ylabel("Macro-F1 (%)",labelpad=7)
    axes.set_ylim(min(vals+[stats[ds]["original"]])-.35,max(vals+[stats[ds]["original"]])+.35)
    axes.spines[["top","right"]].set_visible(False)
    axes.tick_params(length=3,labelsize=8)
    axes.set_xlim(-.25,6.25)
from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([],[],color="#397C9E",marker="o",lw=1.35,markersize=4,label="SAE reconstruction"),
    Line2D([],[],color="#969696",lw=1,ls=(0,(4,3)),label="Original embedding")
],loc="lower center",bbox_to_anchor=(.52,.015),ncol=2,frameon=False,fontsize=9)
stem=OUT/"Classification_and_lambda"
for ext in ["png","pdf","svg"]: fig.savefig(stem.with_suffix("."+ext),dpi=400,facecolor="white")
fig.savefig(ROOT/"work/sae_figures/performance_table_preview.png",dpi=150)
plt.close(fig)
caption="""分类性能与 λ 扫描。上表为冻结表征 Ridge 读出的 Macro-F1（%）。Kang ctrl、Kang stim 各在六个留出供体上等权平均；PBMC 在两个批次方向上等权平均，批次不等于已验证独立供体。原始 embedding 对照沿用相同划分与读出协议。
按用户要求，表中每个数据集、每个 SAE 配置展示三个种子中 TEST 组均值最高者；种子编号不在图面上标注，完整记录在 selection_registry.csv。固定 λ=0.3 与扫描最优 λ 分开列示。扫描最优 λ 分别为 ctrl 0.3、stim 0.03、PBMC 0.03；对应最佳种子分别为 42、43、43。MSE 最佳种子分别为 42、43、43。固定 λ=0.3 最佳种子分别为 42、42、43。每列数值最高者加粗。Δ 行是扫描最优结果减去原始表征，单位为百分点（pp）。
下方每个 λ 的点也取该配置三个种子中的最高 TEST 组均值，因此是最佳种子包络，不是固定种子的纯 λ 敏感性曲线；同时包含种子选择影响。λ=0 为纯 MSE。横轴是离散参数类别，不表示等距数值尺度；直线仅连接已测参数，无插值拟合。灰色虚线为原始 embedding。纵轴在三个数据集间不同，按其绝对 Macro-F1 值读数。
这些是基于历史 TEST 的探索性最佳结果，不是验证集选择后的无偏泛化估计，不能据此宣称稳定提升。表注直接披露选择规则，Discussion 可进一步讨论种子波动与选择偏差。各供体训练集有重叠。此轮只汇总已有结果，没有再训练或改动原始指标。逐配置结果已从各组原始记录重算，并与历史 seed_metrics.csv 核对至 1e-10。
"""
(OUT/"CLASSIFICATION_CAPTION_ZH.txt").write_text(caption)
sources[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/"CLASSIFICATION_PROVENANCE.json").write_text(json.dumps({"sources_sha256":sources,"table":table},indent=2))
print(json.dumps(table,indent=2))
