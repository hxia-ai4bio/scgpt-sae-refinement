"""Redraw existing downstream results without refitting or changing selections."""
from pathlib import Path
import csv
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sae_paper_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ["ctrl", "stim", "pbmc"]
NAMES = ["Kang ctrl", "Kang stim", "PBMC"]
COLORS = {"original_h": "#858C91", "C_selected": "#0072B2",
          "mse": "#83BAD5", "relation_0p3": "#0072B2"}


def read(name):
    with (ROOT / "data" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def handle(method, label):
    return Line2D([], [], linestyle="none", marker="o", markersize=4,
                  color=COLORS[method], label=label)


def main():
    a = read("secondary_A_expression.csv")
    b = read("secondary_B_robustness.csv")
    c = read("secondary_C_clustering.csv")
    expected = read("secondary_panel_means.csv")
    calculated = []
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7.5,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "axes.edgecolor": "#444444", "axes.linewidth": .65,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.95))
    fig.subplots_adjust(left=.08, right=.992, bottom=.32, top=.85, wspace=.42)
    for i, (ax, title) in enumerate(zip(axes, [
            "Expression prediction", "Masked-input robustness", "Clustering structure"])):
        ax.set_title(title, pad=11)
        ax.text(-.25, 1.11, f"({chr(97+i)})", transform=ax.transAxes,
                fontsize=8.5, fontweight="bold")
        ax.set_xticks(range(3), NAMES)
        ax.set_xlim(-.47, 2.47)
        ax.spines[["top", "right"]].set_visible(False)

    def paired(ax, rows, methods, offsets, metric, panel):
        for i, ds in enumerate(DATASETS):
            rr = [r for r in rows if r["dataset"] == ds]
            groups = sorted(set(r["group"] for r in rr))
            assert len(groups) == (2 if ds == "pbmc" else 6)
            for group in groups:
                values = []
                for method in methods:
                    entries = [r for r in rr if r["group"] == group and r["method"] == method]
                    assert len(entries) == 1
                    values.append(float(entries[0][metric]))
                ax.plot(i + np.array(offsets), values, color="#B9C4CB", lw=.55,
                        alpha=.85, zorder=1)
            for method, off in zip(methods, offsets):
                vals = [float(r[metric]) for r in rr if r["method"] == method]
                ax.scatter(np.full(len(vals), i+off), vals, s=9, color=COLORS[method],
                           alpha=.55, edgecolors="none", zorder=2)
                avg = float(np.mean(vals))
                ax.scatter(i+off, avg, s=29, color=COLORS[method], edgecolors="white",
                           linewidths=.65, zorder=3)
                calculated.append((panel, ds, method, "masked" if panel == "a" else "clean", avg))

    paired(axes[0], a, ["original_h", "C_selected"], [-.14, .14], "MAE", "a")
    axes[0].set_ylabel("Masked-gene MAE", labelpad=5)
    axes[0].set_ylim(.295, .39)
    axes[0].yaxis.set_major_locator(MultipleLocator(.02))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    for i, ds in enumerate(DATASETS):
        for method, off in [("original_h", -.14), ("C_selected", .14)]:
            vals = []
            for view in ["clean", "masked"]:
                rr = [r for r in b if r["dataset"] == ds and r["method"] == method and r["view"] == view]
                assert len(rr) == (2 if ds == "pbmc" else 6)
                mean = float(np.mean([float(r["macro_f1_percent"]) for r in rr]))
                vals.append(mean)
                calculated.append(("b", ds, method, view, mean))
            axes[1].plot([i+off]*2, vals, color=COLORS[method], lw=1.4, zorder=1)
            axes[1].scatter(i+off, vals[0], facecolors="white", edgecolors=COLORS[method],
                            s=30, linewidths=1, zorder=2)
            axes[1].scatter(i+off, vals[1], facecolors=COLORS[method], edgecolors="white",
                            s=25, linewidths=.5, zorder=3)
    axes[1].set_ylabel("15-NN Macro-F1 (%)", labelpad=5)
    axes[1].set_ylim(65, 85)
    axes[1].yaxis.set_major_locator(MultipleLocator(5))

    paired(axes[2], c, ["original_h", "mse", "relation_0p3"], [-.21, 0, .21], "ARI", "c")
    axes[2].set_ylabel("ARI", labelpad=5)
    axes[2].set_ylim(0, .72)
    axes[2].yaxis.set_major_locator(MultipleLocator(.2))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    for ax in axes[:2]:
        leg = ax.legend(handles=[handle("original_h", "Original"), handle("C_selected", "Expression SAE")],
                        loc="upper center", bbox_to_anchor=(.5, -.19), frameon=False,
                        ncol=2, fontsize=7, handletextpad=.35, handlelength=.65,
                        columnspacing=.85, borderaxespad=0)
        if ax is axes[1]:
            ax.add_artist(leg)
    axes[1].legend(handles=[
        Line2D([], [], ls="none", marker="o", markerfacecolor="white", markeredgecolor="#555555",
               markersize=4, label="Clean"),
        Line2D([], [], ls="none", marker="o", color="#555555", markersize=4, label="25% masked"),
    ], loc="upper center", bbox_to_anchor=(.5, -.35), ncol=2, frameon=False,
       fontsize=7, handletextpad=.35, handlelength=.65, columnspacing=.85, borderaxespad=0)
    axes[2].legend(handles=[handle("original_h", "Original"), handle("mse", "MSE"),
                            handle("relation_0p3", r"KL ($\lambda=0.3$)")],
                   loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3, frameon=False,
                   fontsize=7, handletextpad=.3, handlelength=.6, columnspacing=.6, borderaxespad=0)
    for panel, ds, method, condition, value in calculated:
        match = [r for r in expected if (r["panel"], r["dataset"], r["method"], r["condition"]) ==
                 (panel, ds, method, condition)]
        assert len(match) == 1
        np.testing.assert_allclose(value, float(match[0]["value"]), atol=1e-12, rtol=0)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT / f"images/secondary_performance.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Verified all {len(calculated)} displayed means against existing results; exported PDF/SVG/PNG.")


if __name__ == "__main__":
    main()
