"""Rebuild the editable table and vector figure from the supplied summary data.

Run: python plot_results.py  (requires matplotlib and numpy).
Selection provenance is documented in README.txt and data/selection_registry.csv.
"""
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
LAMBDAS = [0, .01, .03, .1, .3, 1, 3]


def read_csv(name):
    with (ROOT / "data" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    table = {r["dataset"]: r for r in read_csv("classification_table.csv")}
    curve = read_csv("lambda_best_seed_curve.csv")
    for folder in ["tables", "figures", "images"]:
        (ROOT / folder).mkdir(exist_ok=True)

    lines = [
        r"\begin{table}[!htbp]",
        r"  \caption{Cell-type classification performance (Macro-F1, \%). The best value in each column is shown in bold. $\Delta$ is measured in percentage points (pp).}",
        r"  \label{tab:classification}",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \renewcommand{\arraystretch}{1.15}",
        r"  \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}",
        r"    \toprule",
        r"    Representation & Kang ctrl & Kang stim & PBMC \\",
        r"    \midrule",
    ]
    rows = [
        ("Original embedding", "Original"),
        ("SAE: MSE", "MSE"),
        (r"SAE: MSE + KL ($\lambda=0.3$)", "KL_fixed_0p3"),
        (r"SAE: MSE + KL (best $\lambda$)", "KL_best"),
    ]
    for label, key in rows:
        values = []
        for ds in DATASETS:
            value = float(table[ds][key])
            maximum = max(float(table[ds][k]) for _, k in rows)
            formatted = f"{value:.2f}"
            if abs(value - maximum) < 1e-10:
                formatted = r"\textbf{" + formatted + "}"
            values.append(formatted)
        lines.append("    " + label + " & " + " & ".join(values) + r" \\")
    lines += [r"    \midrule", r"    $\Delta$ vs. original (best $\lambda$)"]
    deltas = [float(table[ds]["KL_best"]) - float(table[ds]["Original"]) for ds in DATASETS]
    lines[-1] += " & " + " & ".join(f"$+{d:.2f}$" for d in deltas) + r" \\"
    lines += [r"    \bottomrule", r"  \end{tabular*}", r"\end{table}"]
    (ROOT / "tables/classification.tex").write_text("\n".join(lines) + "\n")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7.5,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "axes.edgecolor": "#444444", "axes.linewidth": .65,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
    fig.subplots_adjust(left=.079, right=.991, bottom=.27, top=.85, wspace=.38)
    limits = [(82.5, 84.5), (83.25, 85.55), (81.25, 83.6)]
    blue, gray = "#0072B2", "#858585"
    for i, (ds, name, ax) in enumerate(zip(DATASETS, NAMES, axes)):
        records = [r for r in curve if r["dataset"] == ds]
        records.sort(key=lambda r: float(r["lambda_KL"]))
        assert [float(r["lambda_KL"]) for r in records] == LAMBDAS
        y = np.array([float(r["macro_f1_percent"]) for r in records])
        baseline = float(table[ds]["Original"])
        np.testing.assert_allclose(y[0], float(table[ds]["MSE"]), atol=1e-10)
        np.testing.assert_allclose(y[4], float(table[ds]["KL_fixed_0p3"]), atol=1e-10)
        np.testing.assert_allclose(max(y[1:]), float(table[ds]["KL_best"]), atol=1e-10)
        ax.axhline(baseline, color=gray, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.plot(range(7), y, color=blue, lw=1.35, marker="o", markersize=3.6,
                markeredgewidth=.45, markeredgecolor="white", zorder=3)
        ax.set_title(name, pad=9)
        ax.text(-.16, 1.09, f"({chr(97+i)})", transform=ax.transAxes,
                fontweight="bold", fontsize=8.5)
        ax.set_xticks(range(7), ["0", "0.01", "0.03", "0.1", "0.3", "1", "3"])
        ax.set_xlabel(r"$\lambda$", labelpad=4)
        ax.set_xlim(-.25, 6.25)
        ax.set_ylim(*limits[i])
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro-F1 (%)", labelpad=4)
    fig.legend(handles=[
        Line2D([], [], color=blue, marker="o", markersize=3.6, lw=1.35,
               label="SAE reconstruction"),
        Line2D([], [], color=gray, lw=1, ls=(0, (4, 3)), label="Original embedding"),
    ], loc="lower center", bbox_to_anchor=(.53, .015), ncol=2,
       frameon=False, fontsize=8, handlelength=2.5, columnspacing=2)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(ROOT / f"images/lambda_sensitivity.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    (ROOT / "figures/lambda_sensitivity.tex").write_text(r"""\begin{figure}[!htbp]
  \centering
  \includegraphics[width=\linewidth]{images/lambda_sensitivity.pdf}
  \caption{Sensitivity to the relational regularization weight $\lambda$. Blue curves show the Macro-F1 of SAE-reconstructed embeddings; gray dashed lines indicate the original embeddings. $\lambda=0$ corresponds to MSE-only training. The horizontal axis lists the tested values as discrete settings.}
  \label{fig:lambda-sensitivity}
\end{figure}
""")
    print("Generated LaTeX table and PDF/SVG/600-dpi PNG figure; numerical consistency checks passed.")


if __name__ == "__main__":
    main()
