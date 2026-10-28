"""Verify the share package with Python 3 standard library only.

Run from any directory: python3 /path/to/SAE_group_handoff/verify_handoff.py
This checks files and summary arithmetic, not model fitting or biological truth.
"""
from pathlib import Path
from statistics import mean, stdev
import csv
import hashlib
import json
import math
import re
import sys

BASE = Path(__file__).resolve().parent
checks = []


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, message, tol=1e-9):
    require(math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol), message)


def read(relative):
    with (BASE / relative).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    records = json.loads((BASE / "SOURCE_MANIFEST.json").read_text())
    for r in records:
        p = BASE / r["relative_path"]
        require(p.is_file(), f"Missing source copy: {p}")
        require(p.stat().st_size == r["bytes"], f"File size changed: {p}")
        require(hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"], f"Hash changed: {p}")
    checks.append({"check": "copied_source_hashes", "files": len(records), "status": "PASS"})

    raw = read("evidence/sae_lambda/all_metrics.csv")
    for r in raw:
        r["dataset"] = "ctrl"
        r["group"] = r["test_donor"]
    extra = read("evidence/sae_extension/all_metrics.csv")
    for r in extra:
        r["group"] = r["test_group"]
    raw += extra
    table = read("data/performance/classification_table.csv")
    curves = read("data/performance/lambda_best_seed_curve.csv")
    selections = read("data/performance/selection_registry.csv")
    require(len(table) == 3 and len(curves) == 21 and len(selections) == 9, "Unexpected table size")
    for ds in ["ctrl", "stim", "pbmc"]:
        n = 2 if ds == "pbmc" else 6
        rr = [r for r in raw if r["dataset"] == ds]
        baseline = [r for r in rr if r["method"] == "original_h"]
        require(len(baseline) == n, "Baseline group count")
        baseline_score = mean(float(r["macro_f1"]) for r in baseline) * 100
        t = next(r for r in table if r["dataset"] == ds)
        close(t["Original"], baseline_score, f"{ds} Original")
        for point in [r for r in curves if r["dataset"] == ds]:
            candidates = []
            for seed in [42, 43, 44]:
                group = [r for r in rr if r["method"] != "original_h" and
                         r["seed"] == str(seed) and float(r["lambda_KL"]) == float(point["lambda_KL"])]
                require(len(group) == n and len({r["group"] for r in group}) == n, "Group uniqueness")
                candidates.append((mean(float(r["macro_f1"]) for r in group) * 100, seed))
            winner = max(candidates, key=lambda a: (a[0], -a[1]))
            close(point["macro_f1_percent"], winner[0], f"{ds} lambda curve")
            require(int(point["seed"]) == winner[1], "Best seed identity")
        ds_curve = [r for r in curves if r["dataset"] == ds]
        zero = next(r for r in ds_curve if float(r["lambda_KL"]) == 0)
        fixed = next(r for r in ds_curve if float(r["lambda_KL"]) == .3)
        best = max((r for r in ds_curve if float(r["lambda_KL"]) > 0),
                   key=lambda r: (float(r["macro_f1_percent"]), -float(r["lambda_KL"]), -int(r["seed"])))
        for key, r in [("MSE", zero), ("KL_fixed_0p3", fixed), ("KL_best", best)]:
            close(t[key], r["macro_f1_percent"], f"{ds} table {key}")
        close(t["delta_best_pp"], float(best["macro_f1_percent"]) - baseline_score, "Delta pp")
        for label, r in [("MSE", zero), ("MSE + KL (lambda=0.3)", fixed), ("MSE + KL (best lambda)", best)]:
            chosen = next(q for q in selections if q["dataset"] == ds and q["display_row"] == label)
            require(chosen["seed"] == r["seed"], "Selection seed")
            close(chosen["lambda_KL"], r["lambda_KL"], "Selection lambda")
    checks.append({"check": "table_and_lambda_recomputed_from_group_records", "status": "PASS"})

    runs = read("data/figure3/panel_B_selected_features_by_fold_seed.csv")
    donors = read("data/figure3/panel_B_donor_AUROC.csv")
    summary = read("data/figure3/panel_B_summary.csv")
    require(len(runs) == 144 and len(donors) == 48 and len(summary) == 8, "Figure3 record counts")
    for d in donors:
        rr = [r for r in runs if r["target"] == d["target"] and r["test_donor"] == d["test_donor"]]
        require({r["seed"] for r in rr} == {"42", "43", "44"}, "Figure3 seed set")
        valid = [float(r["test_auc"]) for r in rr if r["test_auc"]]
        if valid:
            close(d["test_auc"], mean(valid), "Donor AUROC")
        else:
            require(not d["test_auc"], "Undefined AUC must remain missing")
    for s in summary:
        vals = [float(d["test_auc"]) for d in donors if d["target"] == s["cell_type"] and d["test_auc"]]
        close(s["donor_mean_AUROC"], mean(vals), "AUROC mean")
        close(s["donor_SD"], stdev(vals), "AUROC donor SD")
        require(int(s["n_donors"]) == len(vals), "AUROC donor count")
    checks.append({"check": "figure3_all_types_donor_and_seed_aggregation", "status": "PASS"})

    sources = {
        "a": ("data/performance/secondary_A_expression.csv", "MAE"),
        "b": ("data/performance/secondary_B_robustness.csv", "macro_f1_percent"),
        "c": ("data/performance/secondary_C_clustering.csv", "ARI"),
    }
    panel_means = read("data/performance/secondary_panel_means.csv")
    for panel, (filename, metric) in sources.items():
        rows = read(filename)
        for s in [r for r in panel_means if r["panel"] == panel]:
            rr = [r for r in rows if r["dataset"] == s["dataset"] and r["method"] == s["method"]]
            if panel == "b":
                rr = [r for r in rr if r["view"] == s["condition"]]
            n = 2 if s["dataset"] == "pbmc" else 6
            require(len(rr) == n and len({r["group"] for r in rr}) == n, "Secondary group support")
            close(s["value"], mean(float(r[metric]) for r in rr), "Secondary means")
    checks.append({"check": "figure5_group_means", "status": "PASS"})

    document = (BASE / "PROJECT_HANDOFF.md").read_text(encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document)
    local_links = [s for s in links if not s.startswith(("http:", "https:", "#"))]
    for target in local_links:
        require((BASE / target).is_file(), f"Broken local link: {target}")
    require(len(list((BASE / "figures").glob("*.png"))) == 6, "Expected six shared PNG figures")
    require(document.count("```") % 2 == 0, "Unbalanced fenced code blocks")
    checks.append({"check": "markdown_relative_file_links_and_figures", "links": len(local_links), "status": "PASS"})

    report = {
        "status": "PASS", "checks": checks,
        "document_sha256": hashlib.sha256((BASE / "PROJECT_HANDOFF.md").read_bytes()).hexdigest(),
        "scope": "Copied-source integrity and reported summary arithmetic only; no model refitting, no raw-RNA reanalysis, no independent biological validation.",
    }
    (BASE / "VERIFICATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
