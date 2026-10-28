"""Clean two-panel overview, using verified cached readouts for all eight types."""
import os
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT = Path("/Users/apple/Documents/Codex/2026-10-18/wo")
os.environ["MPLCONFIGDIR"] = str(ROOT / "work/sae_figures/mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

OUT = ROOT / "outputs/sae_figures/overview_all_celltypes"
SOURCE = ROOT / "outputs/sae_figures/overview_simple/panel_A_heldout_activations.csv"


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


heat = read(SOURCE)
rows = read(OUT / "panel_B_donor_AUROC.csv")
summary = read(OUT / "panel_B_summary.csv")
features = list(dict.fromkeys(r["feature"] for r in heat))
types_a = list(dict.fromkeys(r["cell_type"] for r in heat))
# Preserve biological order from panel a, then include the rare eighth class.
types_b = types_a + ["Megakaryocytes"]
matrix = np.array([[float(next(r["row_zscore"] for r in heat
                               if r["feature"] == f and r["cell_type"] == c))
                    for c in types_a] for f in features])
labels = {"B cells": "B cells", "CD14+ Monocytes": "CD14+ Mono", "CD4 T cells": "CD4 T",
          "CD8 T cells": "CD8 T", "Dendritic cells": "DC", "FCGR3A+ Monocytes": "FCGR3A+ Mono",
          "NK cells": "NK", "Megakaryocytes": "Megakaryocytes"}
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "text.color": "#252525",
    "axes.labelcolor": "#252525", "xtick.color": "#252525", "ytick.color": "#252525",
    "axes.edgecolor": "#555555", "axes.linewidth": .65,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
fig = plt.figure(figsize=(12.0, 5.4), facecolor="white")
axa = fig.add_axes([.075, .255, .48, .625])
axb = fig.add_axes([.725, .255, .245, .625])
fig.text(.025, .93, "a", fontsize=13, weight="bold")
fig.text(.075, .93, "Cell-type activation", fontsize=12)
fig.text(.675, .93, "b", fontsize=13, weight="bold")
fig.text(.725, .93, "Cell-type discrimination", fontsize=12)
cmap = LinearSegmentedColormap.from_list("blue_white_orange", ["#3D7FA8", "#FAF9F6", "#D78252"])
im = axa.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5),
                interpolation="nearest", aspect="auto")
axa.set_yticks(range(len(features)), [f"F{f}" for f in features])
axa.set_xticks(range(len(types_a)), [labels[c] for c in types_a], rotation=35, ha="right")
axa.tick_params(axis="both", length=0, pad=5)
axa.set_xticks(np.arange(-.5, len(types_a), 1), minor=True)
axa.set_yticks(np.arange(-.5, len(features), 1), minor=True)
axa.grid(which="minor", color="white", linewidth=.55)
axa.tick_params(which="minor", length=0)
for s in axa.spines.values():
    s.set_visible(False)
cax = fig.add_axes([.20, .102, .23, .017])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-2, 0, 2])
cb.solids.set_rasterized(False)
cb.outline.set_visible(False)
cb.ax.tick_params(length=2, labelsize=8)
cb.set_label("Activation (row z-score)", fontsize=9, labelpad=4)

blue = "#4389AC"
for i, target in enumerate(types_b):
    rr = sorted([r for r in rows if r["target"] == target], key=lambda r: r["test_donor"])
    for r, off in zip(rr, np.linspace(-.20, .20, 6)):
        if r["test_auc"] == "":
            continue
        sparse = int(r["test_target_n"]) < 5
        axb.scatter(float(r["test_auc"]), i + off, s=23,
                    facecolors="white" if sparse else blue,
                    edgecolors=blue if sparse else "white", linewidths=.8, zorder=3)
    mean = float(next(r["donor_mean_AUROC"] for r in summary if r["cell_type"] == target))
    axb.plot([mean, mean], [i - .29, i + .29], color="#252525", lw=1.4, zorder=4)
axb.set_yticks(range(len(types_b)), [labels[c] for c in types_b])
axb.set_ylim(len(types_b) - .5, -.5)
axb.set_xlim(.48, 1.015)
axb.set_xticks([.5, .6, .7, .8, .9, 1.0])
axb.axvline(.5, color="#AAAEB1", linestyle=(0, (3, 3)), lw=.7, zorder=0)
axb.set_xlabel("Single-feature AUROC", labelpad=7)
axb.spines[["top", "right", "left"]].set_visible(False)
axb.tick_params(axis="y", length=0, pad=8)
axb.tick_params(axis="x", length=3)
legend = [
    Line2D([], [], marker="o", ls="", color=blue, markersize=4, label="Donor"),
    Line2D([], [], marker="|", ls="", color="#252525", markersize=9, label="Mean"),
    Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=blue,
           markersize=4, label="<5 cells"),
]
axb.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.245), ncol=3,
           frameon=False, fontsize=8, handlelength=.7, handletextpad=.4, columnspacing=1)
stem = OUT / "SAE_celltype_overview_all"
for ext in ["png", "pdf", "svg"]:
    fig.savefig(stem.with_suffix("." + ext), dpi=400, facecolor="white")
fig.savefig(ROOT / "work/sae_figures/all_celltypes_preview.png", dpi=150)
plt.close(fig)
shutil.copy2(SOURCE, OUT / SOURCE.name)

caption = """SAE 特征的细胞类型关联（Kang ctrl，探索性分析）。
a：主模型 seed 42、TEST donor patient_1016 的单供体激活热图。沿用原先仅由 TRAIN 选择的 13 个特征及其顺序，没有根据 TEST 结果重新选择。每格是该类细胞的平均有符号激活，在每个特征内部对图示七类均值做 z-score（总体标准差，ddof=0）。该色标不表示跨特征的绝对激活强度。与原图相同，仅显示该 TEST 供体中至少 20 个细胞的类别；巨核细胞只有 1 个，故 a 中省略。各类别的确切细胞数保留在源数据中。
b：全部 8 类细胞的单特征 one-vs-rest AUROC。沿用原先 6 个供体划分和 3 个种子（42、43、44）。每次仅使用 TRAIN 数据筛选特征：总非零数至少 20、非零频率不超过 0.5、每个 TRAIN 供体非零数至少 3；最大化 TRAIN 供体 AUROC 的最小值，平局时依次取平均 AUROC 更高及编号更小者。使用原始有符号激活的正向读出，不在 TEST 上翻转符号或选择特征。稀有类别扩展规则：某 TRAIN 供体没有目标细胞时，其 AUROC 未定义，不参与最小值和均值；但仍须满足该供体的非零覆盖阈值。其余类别在全部四个 TRAIN 供体上选特征。
每个点是一个 TEST 供体的三个种子 AUROC 平均值，黑色短线是有效供体的等权平均；竖虚线表示 AUROC=0.5。点的轻微纵向偏移只为减少重叠，没有额外含义。空心点表示该供体目标类型细胞数少于 5，结果非常不稳定，应谨慎解释。前七类均有 6 个有效供体；巨核细胞在 patient_107 中为 0 个，AUROC 不可计算，该供体不画点、不替换为 0.5；剩余 5 个供体中 4 个只有 1–4 个巨核细胞，另一个有 52 个，因此巨核细胞仅作描述性补充。
a 是一个固定模型字典的示例；b 的特征在每个划分与种子下独立选出，特征编号不能跨字典直接比较。b 不代表 a 中固定编号特征的跨供体重复验证。新增五类是本轮扩展分析，并非原先三类预指定目标。供体训练集重叠，不把 18 次拟合视为 18 个独立数据集；无显著性检验，也不作机制、因果或表征优越性声明。
本轮只复用冻结模型的缓存激活，无模型再训练。原先三类的全部 54 条 SAE 记录已核对所选编号和 TEST AUROC；全部有效 TEST AUROC 已用含平局处理的秩公式与正负样本两两比较独立交叉验证。完整缺失状态、供体细胞数、种子结果、选择规则及校验见同目录 CSV/JSON。
"""
(OUT / "CAPTION_ZH.txt").write_text(caption)
(OUT / "PLOT_PROVENANCE.json").write_text(json.dumps({
    "panel_a_source": str(SOURCE), "panel_a_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "type_order_a": types_a, "type_order_b": types_b, "font": "Arial",
    "x_axis_b": [.48, 1.015], "heatmap_limits": [-2.5, 2.5],
    "modification": "Original panel a values unchanged; panel b expanded to all types; consistent clean layout.",
}, indent=2))
print(stem)
