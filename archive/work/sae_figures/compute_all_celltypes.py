"""Replay frozen SAE caches; TRAIN-only feature selection, donor-level TEST AUROC.

Sorting cost O(seeds * folds * features * cells * log(cells)); memory O(cells *
features). Ranks are reused across all cell-type targets; no model training.
"""
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/Users/apple/Documents/Codex/2026-10-18/wo")
OLD = Path("/Users/apple/.codex/.chatgpt-projects/g-p-6abd227c77ec8191b96f942248d505b7/experiments/sae_topk_20261014")
OUT = ROOT / "outputs/sae_figures/overview_all_celltypes"
OUT.mkdir(parents=True, exist_ok=True)
hashes = {}


def record(path):
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def write(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


manifest = json.loads(record(OLD / "data_manifest.json").read_text())
classes = manifest["classes"]
with np.load(record(OLD / "kang_embeddings.npz"), allow_pickle=False) as f:
    labels, donors, raw_rows = f["labels"], f["donors"], f["rows"]
counts = [
    dict(donor=d, cell_type=c, n_cells=int(np.sum((donors == d) & (labels == ci))))
    for d in sorted(set(donors)) for ci, c in enumerate(classes)
]
write("cell_counts_by_donor.csv", counts)
protocol = {
    "dataset": "Kang ctrl", "targets": classes, "seeds": [42, 43, 44],
    "selection": "TRAIN only; nonzero >=20, frequency <=0.5, nonzero >=3 in every TRAIN donor; maximize minimum donor AUROC, then mean donor AUROC, then lower feature index. Signed activation retained, positive direction only.",
    "rare_class_extension": "An undefined TRAIN donor AUROC (no positives or negatives) is excluded from the min/mean; all valid TRAIN donors are used. Coverage rule still includes EVERY TRAIN donor. No valid TRAIN donor or no eligible feature => no selection. Undefined TEST AUROC is missing, never replaced with 0.5.",
    "summary": "Average seeds 42/43/44 within each held-out donor, then equal-weight valid donor means; sample SD across donors. Report valid donor count and TEST positive counts; <5 positives flagged as very sparse.",
    "limitations": "Historical exploratory analysis; feature IDs differ across SAE fits. TRAIN sets overlap. Not independent external cohorts. No significance tests.",
    "source": "Frozen saved sae_outputs.npz; no retraining, no VALID/TEST-driven selection.",
}
(OUT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
runs = []
start = time.time()
for split in manifest["splits"]:
    ix = {role: np.array(split["indices"][role]) for role in ["train", "valid", "test"]}
    assert not (set(ix["train"]) & set(ix["test"]))
    assert set(donors[ix["train"]]) == set(split["train"])
    assert set(donors[ix["test"]]) == {split["test"]}
    assert set(split["train"]).isdisjoint({split["test"], split["valid"]})
    for seed in [42, 43, 44]:
        path = OLD / f"test_{split['test']}/seed_{seed}/k_32/sae_outputs.npz"
        z = np.zeros((len(labels), 2048), dtype=np.float32)
        with np.load(record(path), allow_pickle=False) as f:
            for role, ids in ix.items():
                np.testing.assert_array_equal(f[role + "_rows"], raw_rows[ids])
                np.testing.assert_array_equal(f[role + "_labels"], labels[ids])
                part = np.zeros((len(ids), 2048), dtype=np.float32)
                np.put_along_axis(part, f[role + "_indices"], f[role + "_values"], axis=1)
                z[ids] = part
        assert np.isfinite(z).all() and np.max(np.count_nonzero(z, axis=1)) <= 32
        tr, te = ix["train"], ix["test"]
        eligible = ((z[tr] != 0).sum(0) >= 20) & ((z[tr] != 0).mean(0) <= .5)
        train_auc = []
        valid_donors = []
        for donor in split["train"]:
            ids = tr[donors[tr] == donor]
            eligible &= (z[ids] != 0).sum(0) >= 3
            ranks = rankdata(z[ids], axis=0, method="average").astype(np.float64)
            a = np.full((len(classes), 2048), np.nan)
            for ci in range(len(classes)):
                pos = labels[ids] == ci
                n1, n0 = int(pos.sum()), int((~pos).sum())
                if n1 and n0:
                    a[ci] = (ranks[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            train_auc.append(a)
        train_auc = np.array(train_auc)
        for ci, target in enumerate(classes):
            valid = np.isfinite(train_auc[:, ci, 0])
            rec = dict(test_donor=split["test"], seed=seed, target=target,
                       representation="sae", coordinate="", sign=1,
                       train_min_donor_auc="", train_mean_donor_auc="",
                       train_evaluable_donors=int(valid.sum()),
                       train_evaluable_donor_ids=";".join(np.array(split["train"])[valid]),
                       eligible_features=int(eligible.sum()), test_auc="",
                       test_target_n=int(np.sum(labels[te] == ci)),
                       test_other_n=int(np.sum(labels[te] != ci)), status="ok")
            candidates = np.flatnonzero(eligible)
            if not valid.any() or not len(candidates):
                rec["status"] = "no_train_selection"
            else:
                lo = train_auc[valid, ci].min(0)
                avg = train_auc[valid, ci].mean(0)
                feature = int(candidates[np.lexsort((candidates, -avg[candidates], -lo[candidates]))[0]])
                rec.update(coordinate=feature, train_min_donor_auc=float(lo[feature]),
                           train_mean_donor_auc=float(avg[feature]))
                pos = labels[te] == ci
                if not pos.any() or pos.all():
                    rec["status"] = "undefined_test_auc"
                else:
                    # Independent pairwise definition, including half credit for ties.
                    pv, nv = z[te[pos], feature], z[te[~pos], feature]
                    auc = float(((pv[:, None] > nv).sum() + .5 * (pv[:, None] == nv).sum()) / (len(pv) * len(nv)))
                    ranks = rankdata(z[te, feature]).astype(np.float64)
                    rank_auc = (ranks[pos].sum() - len(pv) * (len(pv) + 1) / 2) / (len(pv) * len(nv))
                    np.testing.assert_allclose(auc, rank_auc, rtol=0, atol=1e-14)
                    rec["test_auc"] = auc
            runs.append(rec)
        print(split["test"], seed, "seconds", round(time.time() - start, 1), flush=True)
write("panel_B_selected_features_by_fold_seed.csv", runs)

# Regression check against all 54 previously reported SAE records.
old_path = ROOT / "outputs/sae_interpretability/all_fold_single_feature_readouts.csv"
with record(old_path).open() as f:
    previous = [r for r in csv.DictReader(f) if r["representation"] == "sae"]
lookup = {(r["test_donor"], str(r["seed"]), r["target"]): r for r in runs}
max_diff = 0.
for old in previous:
    new = lookup[(old["test_donor"], old["seed"], old["target"])]
    assert int(old["coordinate"]) == new["coordinate"], (old, new)
    diff = abs(float(old["test_auc"]) - float(new["test_auc"]))
    max_diff = max(max_diff, diff)
    assert diff < 1e-6, (old, new)
assert len(previous) == 54

donor_rows, summary = [], []
for target in classes:
    for donor in sorted(set(donors)):
        rr = [r for r in runs if r["target"] == target and r["test_donor"] == donor]
        assert {r["seed"] for r in rr} == {42, 43, 44}
        valid = [r for r in rr if r["test_auc"] != ""]
        assert len(valid) in [0, 3]
        donor_rows.append(dict(target=target, test_donor=donor,
                               test_auc=float(np.mean([r["test_auc"] for r in valid])) if valid else "",
                               n_valid_seeds=len(valid), test_target_n=rr[0]["test_target_n"],
                               test_other_n=rr[0]["test_other_n"],
                               very_sparse_test=rr[0]["test_target_n"] < 5,
                               status="ok" if valid else rr[0]["status"]))
    rr = [r for r in donor_rows if r["target"] == target and r["test_auc"] != ""]
    vals = np.array([r["test_auc"] for r in rr])
    summary.append(dict(cell_type=target, donor_mean_AUROC=float(vals.mean()),
                        donor_SD=float(vals.std(ddof=1)), n_donors=len(vals),
                        min_test_positive_n=min(r["test_target_n"] for r in rr),
                        max_test_positive_n=max(r["test_target_n"] for r in rr),
                        n_very_sparse_donors=sum(r["very_sparse_test"] for r in rr)))
write("panel_B_donor_AUROC.csv", donor_rows)
write("panel_B_summary.csv", summary)
record(Path(__file__))
(OUT / "COMPUTATION_CHECKS.json").write_text(json.dumps({
    "previous_54_selected_features_match": True, "previous_54_max_AUROC_difference": max_diff,
    "pairwise_vs_rank_AUC_verified": True, "cache_rows_labels_and_sparsity_verified": True,
    "train_test_donor_disjoint_verified": True, "records": len(runs),
    "seconds": time.time() - start, "sources_sha256": hashes,
}, indent=2))
print(json.dumps(summary, indent=2), flush=True)
