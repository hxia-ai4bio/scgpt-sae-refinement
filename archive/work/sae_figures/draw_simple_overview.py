"""Two-panel replot of existing exploratory SAE results; no training or selection on TEST."""
import os
from pathlib import Path
ROOT = Path("/Users/apple/Documents/Codex/2026-10-18/wo")
os.environ["MPLCONFIGDIR"] = str(ROOT / "work/sae_figures/mpl")
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

OUT = ROOT / "outputs/sae_figures/overview_simple"
OUT.mkdir(exist_ok=True)
OLD = Path("/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014")
INPUTS = {}

def record(path):
    path = Path(path)
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read_csv(path):
    with record(path).open() as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

manifest = json.loads(record(OLD / "data_manifest.json").read_text())
split = manifest["splits"][0]
assert split["test"] == "patient_1016"
classes = manifest["classes"]
with np.load(record(OLD / "kang_embeddings.npz"), allow_pickle=False) as data:
    labels, donors, rawrows = data["labels"], data["donors"], data["rows"]
with np.load(record(ROOT / "work/sae_interpretability/primary_activations.npz"), allow_pickle=False) as cached:
    z = cached["kang"]
test = np.array(split["indices"]["test"])
assert set(donors[test]) == {split["test"]}
assert z.shape == (len(labels), 2048)
assert np.isfinite(z).all() and np.max(np.count_nonzero(z, axis=1)) <= 32
with np.load(record(OLD / "test_patient_1016/seed_42/k_32/sae_outputs.npz"), allow_pickle=False) as cached:
    np.testing.assert_array_equal(rawrows[test], cached["test_rows"])
    dense = np.zeros((len(test), 2048), dtype=np.float32)
    np.put_along_axis(dense, cached["test_indices"], cached["test_values"], axis=1)
    np.testing.assert_allclose(z[test], dense, rtol=0, atol=2e-5)

# Reuse the original TRAIN-only candidate list and its order. Remove QC-only F1724.
existing = read_csv(ROOT / "outputs/sae_figures/Figure2_A_heatmap.csv")
features = list(dict.fromkeys(int(r["feature"]) for r in existing if int(r["feature"]) != 1724))
reasons = {int(r["feature"]): r["selected_for"] for r in existing}
assert len(features) == 13
counts = [int(np.sum(labels[test] == ci)) for ci in range(len(classes))]
assert min(counts) > 0
display_indices = [ci for ci, n in enumerate(counts) if n >= 20]
display_classes = [classes[ci] for ci in display_indices]
means = np.stack([z[test[labels[test] == ci]][:, features].mean(0) for ci in display_indices], axis=1)
row_sd = means.std(axis=1, keepdims=True)
scaled = (means - means.mean(axis=1, keepdims=True)) / np.maximum(row_sd, 1e-12)
heatmap_rows = [
    dict(feature=f, selected_for=reasons[f], cell_type=classes[ci], test_donor=split["test"],
         n_cells=counts[ci], mean_activation=float(means[fi, di]), row_zscore=float(scaled[fi, di]))
    for fi, f in enumerate(features) for di, ci in enumerate(display_indices)
]
write_csv(OUT / "panel_A_heldout_activations.csv", heatmap_rows)

# Verify the saved donor mean against all three seed results. Feature IDs differ by fitted dictionary.
targets = ["B cells", "CD14+ Monocytes", "NK cells"]
all_runs = [r for r in read_csv(ROOT / "outputs/sae_interpretability/all_fold_single_feature_readouts.csv") if r["representation"] == "sae"]
donor_rows = [r for r in read_csv(ROOT / "outputs/sae_interpretability/donor_equal_single_feature_readouts.csv") if r["representation"] == "sae"]
assert len(all_runs) == 54 and len(donor_rows) == 18
for r in donor_rows:
    subset = [q for q in all_runs if q["target"] == r["target"] and q["test_donor"] == r["test_donor"]]
    assert {int(q["seed"]) for q in subset} == {42, 43, 44}
    np.testing.assert_allclose(float(r["test_auc"]), np.mean([float(q["test_auc"]) for q in subset]), atol=1e-12, rtol=0)
write_csv(OUT / "panel_B_donor_AUROC.csv", donor_rows)
write_csv(OUT / "panel_B_selected_features_by_fold_seed.csv", all_runs)

INK, BODY, GRAY = "#20344A", "#38434F", "#72808C"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "text.color": INK,
    "axes.labelcolor": BODY, "xtick.color": BODY, "ytick.color": BODY,
    "axes.edgecolor": "#B5C0C9", "axes.linewidth": .65,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
fig = plt.figure(figsize=(10.4, 4.85), facecolor="white")
ax_a = fig.add_axes([.088, .245, .465, .55])
ax_b = fig.add_axes([.695, .245, .28, .55])
fig.text(.5, .958, "Cell-type associations of SAE features", ha="center", fontsize=16, weight="bold")
fig.text(.5, .903, "Kang ctrl  ·  Frozen scGPT  ·  TopK SAE", ha="center", fontsize=9.5, color=GRAY)
for x, letter, title, subtitle in [
    (.088, "A", "Cell-type activation", "13 TRAIN-selected features · one held-out donor"),
    (.695, "B", "Held-out discrimination", "6 donors · 3 seeds averaged within each donor"),
]:
    fig.text(x-.035, .84, letter, fontsize=14, weight="bold")
    fig.text(x, .84, title, fontsize=11, weight="bold")
    fig.text(x, .805, subtitle, fontsize=8, color=GRAY)

cmap = LinearSegmentedColormap.from_list("blue_white_amber", ["#699ABA", "#F6F7F8", "#EDB34C"])
image = ax_a.imshow(scaled, aspect="auto", interpolation="nearest", cmap=cmap,
                    norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5))
ax_a.set_yticks(np.arange(len(features)), [f"F{f}" for f in features], fontsize=8.5)
short = ["B", "CD14+\nMono", "CD4 T", "CD8 T", "DC", "FCGR3A+\nMono", "Mega", "NK"]
ax_a.set_xticks(np.arange(len(display_indices)), [f"{short[ci]}\n(n={counts[ci]})" for ci in display_indices], fontsize=8)
ax_a.tick_params(axis="both", length=0, pad=5)
ax_a.set_xticks(np.arange(-.5,len(display_indices),1), minor=True)
ax_a.set_yticks(np.arange(-.5,len(features),1), minor=True)
ax_a.grid(which="minor", color="white", linewidth=.55)
ax_a.tick_params(which="minor", length=0)
for spine in ax_a.spines.values():
    spine.set_visible(False)
cax = fig.add_axes([.21, .095, .23, .018])
cb = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[-2,0,2], extend="both")
cb.outline.set_visible(False)
cb.ax.tick_params(length=2, labelsize=8)
cb.set_label("Mean activation (row z-score)", fontsize=8, labelpad=4)

colors = ["#269BC0", "#B391C9", "#63AF91"]
summary = []
for i,(target,color) in enumerate(zip(targets, colors)):
    rows = sorted([r for r in donor_rows if r["target"] == target], key=lambda r:r["test_donor"])
    assert len(rows) == 6
    values = np.array([float(r["test_auc"]) for r in rows])
    # Stable donor-order offsets, not random replications.
    jitter = np.linspace(-.16,.16,6)
    ax_b.scatter(i+jitter, values, s=31, facecolor=color, edgecolor="white", linewidth=.6, zorder=4)
    mean = values.mean()
    ax_b.plot([i-.23,i+.23], [mean,mean], lw=1.3, color=INK, zorder=3)
    ax_b.scatter([i], [mean], marker="D", s=33, facecolor=INK, edgecolor="white", linewidth=.6, zorder=5)
    ax_b.text(i,1.024,f"{mean:.3f}",ha="center",fontsize=10,weight="bold",color=color)
    summary.append(dict(cell_type=target, donor_mean_AUROC=float(mean), donor_SD=float(values.std(ddof=1)), n_donors=6))
ax_b.axhline(.5, color=GRAY, lw=.85, ls=(0,(4,3)))
ax_b.set_ylim(.47,1.06)
ax_b.set_xlim(-.5,2.5)
ax_b.set_yticks([.5,.6,.7,.8,.9,1.0])
ax_b.set_xticks([0,1,2], ["B cells","CD14+\nmonocytes","NK cells"], fontsize=9)
ax_b.set_ylabel("Single-feature AUROC",labelpad=8)
ax_b.spines[["top","right"]].set_visible(False)
ax_b.grid(axis="y", color="#EDF0F2", lw=.65)
ax_b.set_axisbelow(True)
ax_b.tick_params(axis="x",length=0,pad=7)
ax_b.legend(handles=[
    Line2D([],[],marker="o",color="none",markerfacecolor=GRAY,markeredgecolor="white",markersize=5,label="Held-out donor"),
    Line2D([],[],marker="D",color="none",markerfacecolor=INK,markeredgecolor="white",markersize=5,label="Donor mean"),
],loc="upper center",bbox_to_anchor=(.5,-.20),ncol=2,frameon=False,fontsize=8,handletextpad=.4,columnspacing=1)
write_csv(OUT / "panel_B_summary.csv", summary)

stem = OUT / "SAE_celltype_overview_simple"
for ext in ["png","svg","pdf"]:
    fig.savefig(stem.with_suffix("."+ext), dpi=400, facecolor="white")
fig.savefig(ROOT/"work/sae_figures/simple_preview.png", dpi=140)
plt.close(fig)
caption = f"""SAE 特征的细胞类型关联（探索性分析）。
A：Kang ctrl 主模型（seed 42；TEST donor {split['test']}）的留出数据激活热图。沿用原 Figure2 中仅根据 TRAIN 数据选择的细胞类型相关特征，移除 QC 对照 F1724，剩余 13 个特征；未按 TEST 表现重新挑选或排序。每格为对应细胞类型的平均有符号激活，再在每个特征内部对图示七类均值做 z-score。颜色不用于比较不同特征的绝对激活强度。仅展示至少有 20 个 TEST 细胞的类别，故省略只有 1 个 TEST 细胞的巨核细胞；横轴列出 TEST 细胞数。该热图使用一个固定字典。
B：三类预先指定的细胞类型（B cells、CD14+ Monocytes、NK cells）的单特征 one-vs-rest AUROC。每个供体轮换、每个种子均仅在其 TRAIN 数据上根据覆盖度和最小供体 AUROC 规则独立选择特征；六个小点对应六个留出供体，每点先平均三个种子（42、43、44）。深色菱形及横线表示六供体等权平均，虚线为 AUROC=0.5。数字为供体平均 AUROC。
A 和 B 的字典范围不同：B 不是 A 中同一个特征 ID 在六供体上的重复测量，不证明特征身份跨模型稳定。六供体训练集存在重叠；此图不是独立外部验证，也不证明生物机制或相较其他表征的解释性优势。当前六供体分析仅覆盖上述三类，未补造其他类型的 AUROC。
只重新汇总和绘制已有输出，没有训练新模型、增加统计检验或改变原始结果。原始多面板图保留。
"""
(OUT/"CAPTION_ZH.txt").write_text(caption)
record(Path(__file__))
(OUT/"PROVENANCE.json").write_text(json.dumps({
    "feature_ids":features,"test_donor":split["test"],"test_cell_counts":dict(zip(classes,counts)),
    "heatmap_displayed_cell_types":display_classes,"display_filter":"at least 20 TEST cells; omit Megakaryocytes (n=1), not selected on activation",
    "checks":["TEST row identities match original cache","sparse activations match original cache",
              "at most 32 nonzero entries per cell","donor averages match all three seed records"],
    "selection":"Reuse TRAIN-only original Figure2 selection; omit QC F1724; no TEST-driven selection",
    "panel_B_summary":summary,"sources_sha256":INPUTS,
},indent=2))
print(json.dumps({"saved":str(stem),"features":features,"test_counts":counts,"AUROC":summary},indent=2))
