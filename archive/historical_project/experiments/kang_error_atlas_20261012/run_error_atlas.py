"""Donor-held-out Kang error atlas for frozen scGPT representations.

The script deliberately separates measurement quality, representation loss,
linear-head failure, marker-supported errors, and unresolved/ambiguous errors.
The configured atlas-excluded donors are present in the loaded source file but
are never tokenized, fitted, or evaluated in this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import sys
import warnings
import platform

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
import seaborn as sns
import scipy
import sklearn
import torch
from torch import nn


UPSTREAM = Path(
    "/Users/apple/Documents/Codex/2026-10-10/"
    "scgpt-1-users-apple-codex-chatgpt/outputs/scgpt_stable_module_mvp"
)
sys.path.insert(0, str(UPSTREAM))
from train_sparse_modules import load_dense  # noqa: E402


LAYER_KEYS = ("layer0", "layer3", "layer6", "layer9", "layer12")
FEATURE_PIPELINE_VERSION = "kang-error-atlas-features-v1"
MARKERS = {
    "B cells": ("MS4A1", "CD79A", "CD37", "CD74", "HLA-DRA"),
    "CD4 T cells": ("IL7R", "CCR7", "LTB", "MAL", "LEF1"),
    "CD8 T cells": ("CD8A", "CD8B", "CCL5", "LCK", "TRAC"),
    "CD14+ Monocytes": ("S100A8", "S100A9", "CTSS", "LILRB1", "CTSD"),
    "Dendritic cells": ("FCER1A", "CD1C", "CST3", "CLEC10A", "HLA-DQA1"),
    "FCGR3A+ Monocytes": ("FCGR3A", "MS4A7", "LST1", "IFITM3", "LGALS3BP"),
    "NK cells": ("NKG7", "GNLY", "PRF1", "KLRD1", "GZMB"),
}


def decode(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        raise RuntimeError("MPS requested but unavailable")
    return torch.device("cpu")


class KangDevelopmentCohort:
    """Load raw counts but expose only pre-registered development donors."""

    def __init__(self, config: dict, vocab: dict):
        self.config = config
        with h5py.File(config["dataset"], "r") as handle:
            obs, var = handle["obs"][:], handle["var"][:]
            self.meta = {
                name: decode(handle["uns"][name + "_categories"][:])[obs[name]]
                for name in ("replicate", "label", "cell_type")
            }
            self.cell_ids = decode(obs["index"])
            self.gene_names = decode(var["name"])
            self.x = sparse.csc_matrix(
                (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
                shape=tuple(handle["X"].attrs["h5sparse_shape"]),
            ).tocsr()
        self.x.eliminate_zeros()
        if len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError("Duplicate genes require an explicit aggregation rule")
        if (self.x.data < 0).any() or not np.isfinite(self.x.data).all():
            raise ValueError("Counts must be finite and nonnegative")

        all_donors = set(self.meta["replicate"])
        development = list(config["development_donors"])
        untouched = set(config["untouched_donors"])
        if set(development) & untouched:
            raise ValueError("Development and untouched donors overlap")
        if not set(development).issubset(all_donors) or not untouched.issubset(all_donors):
            raise ValueError("Unknown donor in split configuration")

        eligible = (
            np.isin(self.meta["replicate"], development)
            & (self.meta["label"] == config["condition"])
            & ~np.isin(self.meta["cell_type"], config["excluded_cell_types"])
        )
        self.vocab_ids = np.full(len(self.gene_names), -1, dtype=np.int64)
        in_vocab = np.array([name in vocab for name in self.gene_names])
        self.vocab_ids[in_vocab] = [vocab[name] for name in self.gene_names[in_vocab]]
        candidates = np.flatnonzero(eligible)
        minimum = int(config["min_expressed_vocab_genes"])
        usable = []
        for row in candidates:
            source_columns = self.x[int(row)].indices
            if np.sum(self.vocab_ids[source_columns] >= 0) >= minimum:
                usable.append(int(row))
        self.rows = np.asarray(usable, dtype=np.int64)
        self.pad_id = int(vocab["<pad>"])
        self.classes = sorted(set(self.meta["cell_type"][self.rows]))
        self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}
        if set(self.classes) != set(MARKERS):
            raise ValueError(f"Marker programs do not match cohort classes: {self.classes}")
        self.gene_to_column = {name: idx for idx, name in enumerate(self.gene_names)}

    def tokenize(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        seq_len = int(self.config["seq_len"])
        genes = np.full((len(rows), seq_len), self.pad_id, dtype=np.int64)
        values = np.full((len(rows), seq_len), -2, dtype=np.float32)
        labels = np.empty(len(rows), dtype=np.int64)
        for batch_idx, row in enumerate(map(int, rows)):
            raw = self.x[row]
            columns = raw.indices
            counts = raw.data.astype(np.float64, copy=False)
            total = float(counts.sum())
            retained = self.vocab_ids[columns] >= 0
            columns, counts = columns[retained], counts[retained]
            if total <= 0 or len(columns) < self.config["min_expressed_vocab_genes"]:
                raise ValueError(f"Ineligible row reached tokenizer: {row}")
            normalized = np.log1p(counts / total * 1e4)
            edges = np.quantile(normalized, np.linspace(0, 1, self.config["n_bins"] - 1))
            left = np.digitize(normalized, edges)
            right = np.digitize(normalized, edges, right=True)
            tie_rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], row, 11]))
            ties = tie_rng.random(len(self.gene_names))
            bins = np.ceil(ties[columns] * (right - left) + left).astype(np.float32)
            pick = np.arange(len(columns))
            if len(pick) > seq_len:
                priority_rng = np.random.default_rng(
                    np.random.SeedSequence([self.config["seed"], row, 12])
                )
                priority = priority_rng.random(len(self.gene_names))
                pick = np.sort(np.argsort(priority[columns], kind="stable")[:seq_len])
            count = len(pick)
            genes[batch_idx, :count] = self.vocab_ids[columns[pick]]
            values[batch_idx, :count] = bins[pick]
            labels[batch_idx] = self.class_to_id[str(self.meta["cell_type"][row])]
        return {
            "genes": torch.from_numpy(genes),
            "values": torch.from_numpy(values),
            "padding": torch.from_numpy(genes == self.pad_id),
            "labels": torch.from_numpy(labels),
        }

    def normalized_expression(self, rows: np.ndarray, columns=None):
        matrix = self.x[rows]
        if columns is not None:
            matrix = matrix[:, columns]
        matrix = matrix.astype(np.float64).tocsr()
        totals = np.asarray(self.x[rows].sum(1)).ravel()
        scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
        matrix = sparse.diags(scale) @ matrix
        matrix.data = np.log1p(matrix.data)
        return matrix

    def audit(self) -> dict:
        return {
            "source_shape": list(self.x.shape),
            "development_cells": int(len(self.rows)),
            "development_donors": list(self.config["development_donors"]),
            "atlas_excluded_donors": list(self.config["untouched_donors"]),
            "atlas_excluded_donors_loaded_with_source_file": True,
            "atlas_excluded_donors_tokenized_fitted_or_evaluated": False,
            "atlas_excluded_donors_were_historically_evaluated_elsewhere": True,
            "classes": self.classes,
            "counts_by_donor": {
                donor: int(np.sum(self.meta["replicate"][self.rows] == donor))
                for donor in self.config["development_donors"]
            },
            "counts_by_class": {
                name: int(np.sum(self.meta["cell_type"][self.rows] == name))
                for name in self.classes
            },
            "gene_filter": "fixed checkpoint vocabulary membership only; no donor-derived filter",
        }


class LayerFeatureExtractor(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.last_ffn = None
        self.handle = self.base.transformer_encoder.layers[-1].linear2.register_forward_pre_hook(
            self._capture
        )

    def _capture(self, _module, inputs):
        self.last_ffn = inputs[0]

    @staticmethod
    def pool(hidden, valid):
        denominator = valid.sum(1, keepdim=True).clamp_min(1)
        return (hidden * valid.unsqueeze(-1)).sum(1) / denominator

    @torch.inference_mode()
    def forward(self, batch):
        genes, values, padding = batch["genes"], batch["values"], batch["padding"]
        valid = (~padding).to(torch.float32)
        hidden = self.base.encoder(genes) + self.base.value_encoder(values)
        outputs = {"layer0": self.pool(hidden, valid)}
        self.last_ffn = None
        for index, block in enumerate(self.base.transformer_encoder.layers, start=1):
            hidden = block(hidden, src_key_padding_mask=padding)
            if index in {3, 6, 9, 12}:
                outputs[f"layer{index}"] = self.pool(hidden, valid)
        if self.last_ffn is None:
            raise RuntimeError("Last-layer FFN hook did not run")
        outputs["ffn12"] = self.pool(self.last_ffn.abs(), valid)
        return {key: value.float() for key, value in outputs.items()}


def feature_manifest(config, cohort, dataset_hash, checkpoint_hash):
    return {
        "pipeline_version": FEATURE_PIPELINE_VERSION,
        "dataset_sha256": dataset_hash,
        "checkpoint_sha256": checkpoint_hash,
        "rows_sha256": array_sha256(cohort.rows),
        "development_donors": list(config["development_donors"]),
        "condition": config["condition"],
        "excluded_cell_types": list(config["excluded_cell_types"]),
        "seq_len": int(config["seq_len"]),
        "n_bins": int(config["n_bins"]),
        "min_expressed_vocab_genes": int(config["min_expressed_vocab_genes"]),
        "seed": int(config["seed"]),
        "layers": list(LAYER_KEYS) + ["ffn12"],
    }


def extract_features(config, cohort, model, device, destination: Path, expected_manifest: dict):
    manifest_path = destination.parent / "feature_manifest.json"
    if destination.exists():
        if not manifest_path.exists():
            raise ValueError("Feature cache exists without a binding manifest")
        if json.loads(manifest_path.read_text()) != expected_manifest:
            raise ValueError("Feature cache manifest does not match current inputs")
        with np.load(destination) as saved:
            if np.array_equal(saved["rows"], cohort.rows):
                return
        raise ValueError("Existing feature cache does not match the current cohort")
    torch.backends.mha.set_fastpath_enabled(False)
    extractor = LayerFeatureExtractor(model).to(device)
    collected = {key: [] for key in (*LAYER_KEYS, "ffn12")}
    labels = []
    batch_size = int(config["feature_batch_size"])
    started = time.time()
    for start in range(0, len(cohort.rows), batch_size):
        rows = cohort.rows[start : start + batch_size]
        batch = {key: value.to(device) for key, value in cohort.tokenize(rows).items()}
        features = extractor(batch)
        for key, value in features.items():
            collected[key].append(value.cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        completed = min(start + batch_size, len(cohort.rows))
        if start == 0 or completed % 400 < batch_size:
            print(f"features {completed}/{len(cohort.rows)} elapsed={time.time()-started:.1f}s", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        rows=cohort.rows,
        labels=np.concatenate(labels),
        **{key: np.concatenate(parts) for key, parts in collected.items()},
    )
    atomic_json(manifest_path, expected_manifest)
    extractor.handle.remove()


def probabilities(model, x):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
        )
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(x)
        else:
            score = model.decision_function(x)
            score = score - score.max(1, keepdims=True)
            probability = np.exp(score)
            probability /= probability.sum(1, keepdims=True)
    if not np.isfinite(probability).all():
        raise FloatingPointError("Readout produced non-finite probabilities")
    return probability, probability.argmax(1)


def entropy(probability):
    return -(probability * np.log(np.clip(probability, 1e-12, 1))).sum(1)


def margin(probability):
    ordered = np.sort(probability, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def marker_scores(cohort: KangDevelopmentCohort):
    available = {
        cell_type: [gene for gene in genes if gene in cohort.gene_to_column]
        for cell_type, genes in MARKERS.items()
    }
    missing = {cell_type: sorted(set(MARKERS[cell_type]) - set(genes)) for cell_type, genes in available.items()}
    score = np.zeros((len(cohort.rows), len(cohort.classes)), dtype=np.float64)
    for class_id, cell_type in enumerate(cohort.classes):
        columns = [cohort.gene_to_column[gene] for gene in available[cell_type]]
        score[:, class_id] = np.asarray(
            cohort.normalized_expression(cohort.rows, columns).mean(1)
        ).ravel()
    return score, available, missing


def make_logistic(c, seed, penalty="l2"):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            penalty=penalty,
            # On this Apple/NumPy stack LBFGS emits hundreds of transient
            # overflow warnings even when its final coefficients are finite.
            # SAGA follows a stable path for both L2 and L1 readouts.
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=seed,
            n_jobs=1,
        ),
    )


def make_ridge(alpha):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=alpha, class_weight="balanced"),
    )


def fit_readout(model, x, y):
    """Fit while rejecting real non-finite results from the Apple BLAS warning path."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
        )
        model.fit(x, y)
    estimator = model[-1] if hasattr(model, "__getitem__") else model
    if not np.isfinite(estimator.coef_).all():
        raise FloatingPointError("Readout produced non-finite coefficients")
    if hasattr(estimator, "intercept_") and not np.isfinite(estimator.intercept_).all():
        raise FloatingPointError("Readout produced non-finite intercept")
    if hasattr(model, "named_steps") and "standardscaler" in model.named_steps:
        scaler = model.named_steps["standardscaler"]
        if not np.isfinite(scaler.scale_).all():
            raise FloatingPointError("Readout scaler produced non-finite scales")
    return model


def make_sparse_expression_logistic(c, seed):
    """Linear baseline that preserves the sparse log-expression matrix."""
    return LogisticRegression(
        C=c,
        penalty="l2",
        solver="saga",
        class_weight="balanced",
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
        n_jobs=1,
    )


def evaluate(config, cohort, features_path: Path, output_root: Path):
    with np.load(features_path) as saved:
        # sklearn preserves float32 through StandardScaler.  In rare directions
        # this made the optimizer's matrix products overflow even though the
        # original features were finite and modest.  The readout stage is small,
        # so use float64 explicitly; the frozen scGPT cache remains float32.
        features = {
            key: saved[key].astype(np.float64, copy=True)
            for key in (*LAYER_KEYS, "ffn12")
        }
        labels = saved["labels"].copy()
        rows = saved["rows"].copy()
    donors = cohort.meta["replicate"][rows]
    cell_types = cohort.meta["cell_type"][rows]
    if set(donors) != set(config["development_donors"]):
        raise ValueError("Feature cache donor mismatch")

    vocab_columns = np.flatnonzero(cohort.vocab_ids >= 0)
    raw = cohort.normalized_expression(rows, vocab_columns)
    marker, marker_genes, missing_markers = marker_scores(cohort)
    n = len(rows)
    class_count = len(cohort.classes)
    predictions = {key: np.full(n, -1, dtype=int) for key in LAYER_KEYS}
    probabilities_by_layer = {
        key: np.full((n, class_count), np.nan, dtype=float) for key in LAYER_KEYS
    }
    raw_prediction = np.full(n, -1, dtype=int)
    raw_probability = np.full((n, class_count), np.nan, dtype=float)
    sparse_prediction = np.full(n, -1, dtype=int)
    sparse_probability = np.full((n, class_count), np.nan, dtype=float)
    sparse64_prediction = np.full(n, -1, dtype=int)
    sparse64_probability = np.full((n, class_count), np.nan, dtype=float)
    random64_prediction = np.full(n, -1, dtype=int)
    random64_probability = np.full((n, class_count), np.nan, dtype=float)
    random_repeats = int(config["random_control_repeats"])
    random_predictions = np.full((random_repeats, n), -1, dtype=int)
    marker_prediction = np.full(n, -1, dtype=int)
    marker_probability = np.full((n, class_count), np.nan, dtype=float)
    knn_prediction = np.full(n, -1, dtype=int)
    knn_true_purity = np.full(n, np.nan, dtype=float)
    sparse_nonzero = []
    selected_neuron_rows = []
    random_neuron_rows = []
    selected_sets = []
    fold_rows = []
    fit_status_rows = []

    for held_out in config["development_donors"]:
        train = donors != held_out
        test = donors == held_out
        print(f"fit held_out={held_out} train={train.sum()} test={test.sum()}", flush=True)
        for key in LAYER_KEYS:
            classifier = make_ridge(config["ridge_alpha"])
            fit_readout(classifier, features[key][train], labels[train])
            probability, prediction = probabilities(classifier, features[key][test])
            predictions[key][test] = prediction
            probabilities_by_layer[key][test] = probability

        raw_classifier = make_sparse_expression_logistic(
            config["logistic_c"], config["seed"]
        )
        fit_readout(raw_classifier, raw[train], labels[train])
        probability, prediction = probabilities(raw_classifier, raw[test])
        raw_prediction[test], raw_probability[test] = prediction, probability

        sparse_classifier = make_logistic(
            config["sparse_ffn_c"], config["seed"], penalty="l1"
        )
        fit_readout(sparse_classifier, features["ffn12"][train], labels[train])
        probability, prediction = probabilities(sparse_classifier, features["ffn12"][test])
        sparse_prediction[test], sparse_probability[test] = prediction, probability
        coefficients = sparse_classifier.named_steps["logisticregression"].coef_
        fit_status_rows.append(
            {
                "held_out_donor": held_out,
                "raw_logistic_max_n_iter": int(np.max(raw_classifier.n_iter_)),
                "raw_logistic_max_iter_limit": int(raw_classifier.max_iter),
                "sparse_ffn_max_n_iter": int(np.max(
                    sparse_classifier.named_steps["logisticregression"].n_iter_
                )),
                "sparse_ffn_max_iter_limit": int(
                    sparse_classifier.named_steps["logisticregression"].max_iter
                ),
            }
        )
        sparse_nonzero.append(
            {
                "held_out_donor": held_out,
                "nonzero_coefficients": int(np.count_nonzero(coefficients)),
                "selected_neurons": int(np.sum(np.any(coefficients != 0, axis=0))),
                "total_neurons": int(coefficients.shape[1]),
            }
        )

        budget = int(config["sparse_neuron_budget"])
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            f_scores, _ = f_classif(features["ffn12"][train], labels[train])
        f_scores = np.nan_to_num(f_scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        selected = np.sort(np.argsort(f_scores, kind="stable")[-budget:])
        selected_sets.append(set(map(int, selected)))
        selected_classifier = make_ridge(config["ridge_alpha"])
        fit_readout(selected_classifier, features["ffn12"][train][:, selected], labels[train])
        probability, prediction = probabilities(
            selected_classifier, features["ffn12"][test][:, selected]
        )
        sparse64_prediction[test], sparse64_probability[test] = prediction, probability

        donor_index = list(config["development_donors"]).index(held_out)
        training_variance = features["ffn12"][train].var(0)
        random_pool = np.flatnonzero(training_variance > 1e-12)
        if len(random_pool) < budget:
            raise ValueError("Fewer active neurons than the registered random budget")
        for repeat in range(random_repeats):
            random_rng = np.random.default_rng(
                np.random.SeedSequence([config["seed"], donor_index, budget, repeat])
            )
            random_selected = np.sort(
                random_rng.choice(random_pool, size=budget, replace=False)
            )
            random_classifier = make_ridge(config["ridge_alpha"])
            fit_readout(
                random_classifier,
                features["ffn12"][train][:, random_selected],
                labels[train],
            )
            probability, prediction = probabilities(
                random_classifier, features["ffn12"][test][:, random_selected]
            )
            random_predictions[repeat, test] = prediction
            if repeat == 0:
                random64_prediction[test], random64_probability[test] = prediction, probability
            for neuron in random_selected:
                random_neuron_rows.append(
                    {
                        "held_out_donor": held_out,
                        "repeat": repeat,
                        "neuron": int(neuron),
                        "training_variance": float(training_variance[neuron]),
                    }
                )

        selected_matrix = features["ffn12"][train][:, selected]
        marker_matrix = marker[train]
        selected_z = selected_matrix - selected_matrix.mean(0, keepdims=True)
        marker_z = marker_matrix - marker_matrix.mean(0, keepdims=True)
        selected_z /= np.linalg.norm(selected_z, axis=0, keepdims=True).clip(min=1e-12)
        marker_z /= np.linalg.norm(marker_z, axis=0, keepdims=True).clip(min=1e-12)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
            )
            correlations = selected_z.T @ marker_z
        if not np.isfinite(correlations).all():
            raise FloatingPointError("Neuron-marker correlations are non-finite")
        top_program = np.abs(correlations).argmax(1)
        for position, neuron in enumerate(selected):
            selected_neuron_rows.append(
                {
                    "held_out_donor": held_out,
                    "neuron": int(neuron),
                    "training_f_score": float(f_scores[neuron]),
                    "top_marker_program": cohort.classes[int(top_program[position])],
                    "marker_correlation": float(correlations[position, top_program[position]]),
                }
            )

        marker_classifier = make_ridge(config["ridge_alpha"])
        fit_readout(marker_classifier, marker[train], labels[train])
        probability, prediction = probabilities(marker_classifier, marker[test])
        marker_prediction[test], marker_probability[test] = prediction, probability

        scaler = StandardScaler().fit(features["layer12"][train])
        train_scaled = scaler.transform(features["layer12"][train])
        test_scaled = scaler.transform(features["layer12"][test])
        neighbor_count = min(int(config["knn_neighbors"]), train.sum())
        neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine", n_jobs=1)
        neighbors.fit(train_scaled)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
            )
            distances, indices = neighbors.kneighbors(test_scaled, return_distance=True)
        if not np.isfinite(distances).all() or not np.isfinite(indices).all():
            raise FloatingPointError("kNN produced non-finite distances or indices")
        train_labels = labels[train]
        neighbor_labels = train_labels[indices]
        knn_prediction[test] = np.apply_along_axis(
            lambda values: np.bincount(values, minlength=class_count).argmax(), 1, neighbor_labels
        )
        test_labels = labels[test]
        knn_true_purity[test] = (neighbor_labels == test_labels[:, None]).mean(1)

        fold_rows.append(
            {
                "held_out_donor": held_out,
                "cells": int(test.sum()),
                "macro_f1_layer12": f1_score(labels[test], predictions["layer12"][test], average="macro"),
                "macro_f1_raw_linear": f1_score(labels[test], raw_prediction[test], average="macro"),
                "macro_f1_sparse_ffn": f1_score(labels[test], sparse_prediction[test], average="macro"),
                "macro_f1_sparse64": f1_score(labels[test], sparse64_prediction[test], average="macro"),
                "macro_f1_random64_repeat0": f1_score(labels[test], random64_prediction[test], average="macro"),
                "macro_f1_marker": f1_score(labels[test], marker_prediction[test], average="macro"),
            }
        )

    if any((value < 0).any() for value in predictions.values()) or (raw_prediction < 0).any():
        raise RuntimeError("Out-of-fold prediction coverage is incomplete")

    total_counts = np.asarray(cohort.x[rows].sum(1)).ravel()
    detected_genes = np.diff(cohort.x[rows].indptr)
    mitochondrial_columns = np.flatnonzero(np.char.startswith(cohort.gene_names.astype(str), "MT-"))
    if len(mitochondrial_columns):
        mito_counts = np.asarray(cohort.x[rows][:, mitochondrial_columns].sum(1)).ravel()
        mito_fraction = np.divide(
            mito_counts, total_counts, out=np.zeros_like(total_counts, dtype=float), where=total_counts > 0
        )
    else:
        mito_fraction = np.zeros(n, dtype=float)

    low_quality = np.zeros(n, dtype=bool)
    for donor in config["development_donors"]:
        take = donors == donor
        count_low = np.quantile(total_counts[take], 0.10)
        gene_low = np.quantile(detected_genes[take], 0.10)
        mito_high = np.quantile(mito_fraction[take], 0.90)
        low_quality[take] = (
            (total_counts[take] <= count_low)
            | (detected_genes[take] <= gene_low)
            | ((mito_fraction[take] >= mito_high) & (mito_fraction[take] > 0))
        )

    final_prediction = predictions["layer12"]
    final_wrong = final_prediction != labels
    early_correct = (predictions["layer0"] == labels) | (predictions["layer3"] == labels)
    late_layer_readout_loss = final_wrong & early_correct & ~low_quality
    neighborhood_support = final_wrong & (knn_prediction == labels) & (knn_true_purity >= 0.5) & ~low_quality
    marker_supported = final_wrong & (marker_prediction == labels) & ~low_quality
    raw_supported = final_wrong & (raw_prediction == labels) & ~low_quality
    sparse_ffn_rescue = final_wrong & (sparse_prediction == labels) & ~low_quality
    sparse64_rescue = final_wrong & (sparse64_prediction == labels) & ~low_quality
    random64_rescue = final_wrong & (random64_prediction == labels) & ~low_quality
    sparse_candidate = (
        (late_layer_readout_loss | neighborhood_support | marker_supported | sparse64_rescue)
        & raw_supported
        & ~low_quality
    )
    ambiguity_candidate = (
        final_wrong
        & (raw_prediction != labels)
        & (marker_prediction != labels)
        & (knn_true_purity < 0.3)
        & ~low_quality
    )

    category = np.full(n, "correct", dtype=object)
    category[final_wrong] = "unresolved_model_error"
    category[final_wrong & low_quality] = "low_quality_associated"
    category[ambiguity_candidate] = "ambiguous_or_overlap"
    category[sparse_candidate] = "posthoc_sparse_diagnostic_candidate"

    final_probability = probabilities_by_layer["layer12"]
    records = pd.DataFrame(
        {
            "source_row": rows,
            "cell_id": cohort.cell_ids[rows],
            "donor": donors,
            "true_cell_type": cell_types,
            "predicted_cell_type": [cohort.classes[value] for value in final_prediction],
            "correct": ~final_wrong,
            "error_category": category,
            "total_counts": total_counts,
            "detected_genes": detected_genes,
            "mitochondrial_fraction": mito_fraction,
            "final_confidence": final_probability.max(1),
            "final_margin": margin(final_probability),
            "final_entropy": entropy(final_probability),
            "knn_true_label_purity": knn_true_purity,
            "knn_predicted_cell_type": [cohort.classes[value] for value in knn_prediction],
            "raw_expression_correct": raw_prediction == labels,
            "raw_expression_predicted_cell_type": [cohort.classes[value] for value in raw_prediction],
            "marker_correct": marker_prediction == labels,
            "marker_predicted_cell_type": [cohort.classes[value] for value in marker_prediction],
            "sparse_ffn_correct": sparse_prediction == labels,
            "sparse_ffn_predicted_cell_type": [cohort.classes[value] for value in sparse_prediction],
            "sparse64_correct": sparse64_prediction == labels,
            "sparse64_predicted_cell_type": [cohort.classes[value] for value in sparse64_prediction],
            "random64_correct": random64_prediction == labels,
            "random64_repeat0_predicted_cell_type": [cohort.classes[value] for value in random64_prediction],
            "early_layer_correct": early_correct,
            "late_layer_readout_loss_flag": late_layer_readout_loss,
            "neighborhood_support_flag": neighborhood_support,
            "marker_support_flag": marker_supported,
            "raw_support_flag": raw_supported,
            "sparse_ffn_rescue_flag": sparse_ffn_rescue,
            "sparse64_rescue_flag": sparse64_rescue,
            "random64_rescue_flag": random64_rescue,
            "sparse_candidate_flag": sparse_candidate,
            "ambiguity_candidate_flag": ambiguity_candidate,
            "low_quality_flag": low_quality,
        }
    )
    for key in LAYER_KEYS:
        records[f"{key}_predicted_cell_type"] = [
            cohort.classes[value] for value in predictions[key]
        ]
        records[f"{key}_correct"] = predictions[key] == labels
        records[f"{key}_margin"] = margin(probabilities_by_layer[key])
    for repeat in range(random_repeats):
        records[f"random64_repeat{repeat}_predicted_cell_type"] = [
            cohort.classes[value] for value in random_predictions[repeat]
        ]

    layer_metrics = pd.DataFrame(
        [
            {
                "representation": key,
                "macro_f1": f1_score(labels, predictions[key], average="macro"),
                "accuracy": float(np.mean(labels == predictions[key])),
            }
            for key in LAYER_KEYS
        ]
        + [
            {
                "representation": "raw_expression_linear",
                "macro_f1": f1_score(labels, raw_prediction, average="macro"),
                "accuracy": float(np.mean(labels == raw_prediction)),
            },
            {
                "representation": "sparse_ffn_l1",
                "macro_f1": f1_score(labels, sparse_prediction, average="macro"),
                "accuracy": float(np.mean(labels == sparse_prediction)),
            },
            {
                "representation": "selected_ffn_64",
                "macro_f1": f1_score(labels, sparse64_prediction, average="macro"),
                "accuracy": float(np.mean(labels == sparse64_prediction)),
            },
            {
                "representation": "random_ffn_64_repeat0",
                "macro_f1": f1_score(labels, random64_prediction, average="macro"),
                "accuracy": float(np.mean(labels == random64_prediction)),
            },
            {
                "representation": "marker_scores",
                "macro_f1": f1_score(labels, marker_prediction, average="macro"),
                "accuracy": float(np.mean(labels == marker_prediction)),
            },
        ]
    )
    fold_metrics = pd.DataFrame(fold_rows)
    sparse_stats = pd.DataFrame(sparse_nonzero)
    fit_status = pd.DataFrame(fit_status_rows)
    selected_neurons = pd.DataFrame(selected_neuron_rows)
    random_neurons = pd.DataFrame(random_neuron_rows)

    wrong_records = records.loc[~records["correct"]].copy()
    pair_counts = (
        wrong_records.groupby(["true_cell_type", "predicted_cell_type"], observed=True)
        .agg(
            errors=("cell_id", "size"),
            sparse_candidates=("sparse_candidate_flag", "sum"),
            low_quality=("low_quality_flag", "sum"),
            ambiguous=("ambiguity_candidate_flag", "sum"),
            median_knn_true_purity=("knn_true_label_purity", "median"),
        )
        .reset_index()
        .sort_values("errors", ascending=False)
    )
    pair_counts["sparse_candidate_fraction"] = (
        pair_counts["sparse_candidates"] / pair_counts["errors"]
    )

    records.to_csv(output_root / "per_cell_errors.csv", index=False)
    layer_metrics.to_csv(output_root / "layer_metrics.csv", index=False)
    fold_metrics.to_csv(output_root / "fold_metrics.csv", index=False)
    sparse_stats.to_csv(output_root / "sparse_ffn_sparsity.csv", index=False)
    fit_status.to_csv(output_root / "fit_status.csv", index=False)
    selected_neurons.to_csv(output_root / "selected_neurons_64.csv", index=False)
    random_neurons.to_csv(output_root / "random_neurons_64.csv", index=False)
    random_metrics = pd.DataFrame(
        [
            {
                "repeat": repeat,
                "macro_f1": f1_score(labels, random_predictions[repeat], average="macro"),
                "accuracy": float(np.mean(labels == random_predictions[repeat])),
                "rescues_baseline_errors": int(
                    np.sum(final_wrong & (random_predictions[repeat] == labels))
                ),
                "harms_baseline_correct": int(
                    np.sum((~final_wrong) & (random_predictions[repeat] != labels))
                ),
            }
            for repeat in range(random_repeats)
        ]
    )
    random_metrics.to_csv(output_root / "random64_repeats.csv", index=False)
    pair_counts.to_csv(output_root / "confusion_pairs.csv", index=False)

    make_figures(output_root, cohort.classes, labels, final_prediction, features["layer12"], records,
                 layer_metrics, fold_metrics)
    summary = {
        "protocol": cohort.audit(),
        "overall": {
            "cells": int(n),
            "errors": int(final_wrong.sum()),
            "accuracy": float(np.mean(~final_wrong)),
            "macro_f1_layer12": float(f1_score(labels, final_prediction, average="macro")),
            "macro_f1_raw_linear": float(f1_score(labels, raw_prediction, average="macro")),
            "macro_f1_sparse_ffn": float(f1_score(labels, sparse_prediction, average="macro")),
            "macro_f1_sparse64": float(f1_score(labels, sparse64_prediction, average="macro")),
            "macro_f1_random64_repeat0": float(f1_score(labels, random64_prediction, average="macro")),
            "macro_f1_marker": float(f1_score(labels, marker_prediction, average="macro")),
            "mean_donor_macro_f1_layer12": float(fold_metrics["macro_f1_layer12"].mean()),
        },
        "error_taxonomy": {
            key: int(value) for key, value in records["error_category"].value_counts().items()
        },
        "diagnostic_flags_among_errors": {
            "low_quality": int((final_wrong & low_quality).sum()),
            "late_layer_readout_loss": int(late_layer_readout_loss.sum()),
            "neighborhood_support": int(neighborhood_support.sum()),
            "raw_support": int(raw_supported.sum()),
            "marker_support": int(marker_supported.sum()),
            "sparse_ffn_rescue": int(sparse_ffn_rescue.sum()),
            "sparse64_rescue": int(sparse64_rescue.sum()),
            "random64_rescue": int(random64_rescue.sum()),
            "sparse_candidate": int(sparse_candidate.sum()),
            "ambiguous_or_overlap": int(ambiguity_candidate.sum()),
        },
        "marker_genes_used": marker_genes,
        "missing_marker_genes": missing_markers,
        "sparse_ffn": {
            "mean_selected_neurons": float(sparse_stats["selected_neurons"].mean()),
            "min_selected_neurons": int(sparse_stats["selected_neurons"].min()),
            "max_selected_neurons": int(sparse_stats["selected_neurons"].max()),
        },
        "fixed_budget_sparse_test": {
            "budget": int(config["sparse_neuron_budget"]),
            "mean_pairwise_selected_jaccard": float(np.mean([
                len(first & second) / len(first | second)
                for index, first in enumerate(selected_sets)
                for second in selected_sets[index + 1:]
            ])),
            "selected_rescues_baseline_errors": int(
                np.sum(final_wrong & (sparse64_prediction == labels))
            ),
            "selected_harms_baseline_correct": int(
                np.sum((~final_wrong) & (sparse64_prediction != labels))
            ),
            "random_repeats": random_repeats,
            "random_macro_f1_mean": float(random_metrics["macro_f1"].mean()),
            "random_macro_f1_sample_sd": float(random_metrics["macro_f1"].std(ddof=1)),
            "random_rescues_baseline_errors_mean": float(
                random_metrics["rescues_baseline_errors"].mean()
            ),
            "random_harms_baseline_correct_mean": float(
                random_metrics["harms_baseline_correct"].mean()
            ),
        },
        "qc_limitations": {
            "mitochondrial_genes_found": int(len(mitochondrial_columns)),
            "definition": "donor-relative bottom decile of counts or detected genes; mitochondrial criterion unavailable when zero MT- genes are present"
        },
    }
    atomic_json(output_root / "SUMMARY.json", summary)
    write_report(output_root, summary, layer_metrics, fold_metrics, pair_counts)
    return summary


def make_figures(root, classes, labels, prediction, embedding, records, layer_metrics, fold_metrics):
    figure_root = root / "figures"
    figure_root.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    matrix = confusion_matrix(labels, prediction, labels=np.arange(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes,
                yticklabels=classes, vmin=0, vmax=1, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title="Donor-held-out normalized confusion matrix")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(figure_root / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=fold_metrics, x="held_out_donor", y="macro_f1_layer12", color="#4C78A8", ax=ax)
    ax.set(ylim=(0, 1), xlabel="Held-out donor", ylabel="Macro-F1", title="Cross-donor performance")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figure_root / "donor_macro_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = list(LAYER_KEYS) + ["raw_expression_linear", "sparse_ffn_l1",
                               "selected_ffn_64", "random_ffn_64_repeat0", "marker_scores"]
    sns.barplot(data=layer_metrics, x="representation", y="macro_f1", order=order,
                color="#59A14F", ax=ax)
    ax.set(ylim=(0, 1), xlabel="Representation/readout", ylabel="OOF Macro-F1",
           title="Where predictive information is retained")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(figure_root / "representation_macro_f1.png", dpi=180)
    plt.close(fig)

    errors = records.loc[~records["correct"], "error_category"].value_counts().reset_index()
    errors.columns = ["category", "cells"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=errors, x="category", y="cells", color="#F28E2B", ax=ax)
    ax.set(xlabel="Primary diagnostic category", ylabel="Error cells", title="Error taxonomy")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(figure_root / "error_taxonomy.png", dpi=180)
    plt.close(fig)

    qc = records.assign(status=np.where(records["correct"], "correct", "error"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, column, title in zip(
        axes,
        ("total_counts", "detected_genes", "mitochondrial_fraction"),
        ("Total counts", "Detected genes", "Mitochondrial fraction"),
    ):
        sns.boxplot(data=qc, x="status", y=column, showfliers=False, ax=ax)
        ax.set(xlabel="", ylabel=title)
    fig.suptitle("QC distributions of correct and incorrect predictions")
    fig.tight_layout()
    fig.savefig(figure_root / "qc_correct_vs_error.png", dpi=180)
    plt.close(fig)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
        )
        coordinates = PCA(n_components=2, random_state=0).fit_transform(
            StandardScaler().fit_transform(embedding)
        )
    if not np.isfinite(coordinates).all():
        raise FloatingPointError("PCA visualization produced non-finite coordinates")
    plot = pd.DataFrame({"PC1": coordinates[:, 0], "PC2": coordinates[:, 1],
                         "cell_type": records["true_cell_type"],
                         "status": np.where(records["correct"], "correct", "error")})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(data=plot, x="PC1", y="PC2", hue="cell_type", s=9, alpha=0.55,
                    linewidth=0, ax=axes[0])
    sns.scatterplot(data=plot, x="PC1", y="PC2", hue="status", palette={"correct": "#BDBDBD", "error": "#E15759"},
                    s=10, alpha=0.65, linewidth=0, ax=axes[1])
    axes[0].set_title("Final embedding by true cell type")
    axes[1].set_title("Errors in final embedding")
    for ax in axes:
        ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "embedding_pca.png", dpi=180)
    plt.close(fig)


def write_report(root, summary, layer_metrics, fold_metrics, pairs):
    overall = summary["overall"]
    flags = summary["diagnostic_flags_among_errors"]
    fixed = summary["fixed_budget_sparse_test"]
    errors = overall["errors"]
    top_pairs = pairs.head(8)
    lines = [
        "# Kang 全量开发集错例图谱",
        "",
        "## 结论",
        "",
        f"本分析使用 6 个 donor 的 {overall['cells']:,} 个 ctrl PBMC 细胞做逐 donor 留出预测；"
        "`patient_101` 与 `patient_1244` 不进入本次 atlas 的 tokenization、拟合或评估，但它们已在历史实验中评估过，"
        "因此不是新的盲测集。没有人为降低测序深度。",
        "",
        f"最终层 scGPT embedding 的 OOF Macro-F1 为 **{overall['macro_f1_layer12']:.4f}**，"
        f"共错 {errors:,} 个细胞；六个 donor 等权平均 Macro-F1 为 **{overall['mean_donor_macro_f1_layer12']:.4f}**。"
        f"依赖真实标签的事后规则定义出 **{flags['sparse_candidate']:,}** 个稀疏诊断候选"
        f"（占全部错误 {flags['sparse_candidate']/max(errors,1):.1%}）。它们只用于提出假设，"
        "不能作为部署时 gate，也不是已经证明可修复。",
        "",
        "## 哪些错误可能受益于稀疏可解释性",
        "",
        f"- **后层线性可读性下降**：{flags['late_layer_readout_loss']:,} 个。早层 readout 分对、最终层 readout 分错；"
        "这支持尝试多层稀疏残差，但不证明信息已被模型删除。",
        f"- **最终空间的局部邻域仍支持真值**：{flags['neighborhood_support']:,} 个。"
        "这支持测试 prototype/局部稀疏 head，但不单独证明现有线性头是错误根因。",
        f"- **marker 支持真值**：{flags['marker_support']:,} 个。"
        "适合测试显式 marker-program score 或 pathway bottleneck；marker 来自同一表达矩阵，不是独立生物验证。",
        f"- **固定 64 神经元能救回**：排除相对低QC proxy 后为 {flags['sparse64_rescue']:,} 个。"
        "这仍是事后统计，不能直接成为可部署规则。",
        "",
        "## 固定稀疏预算的直接检验",
        "",
        f"训练折内按 F-score 选择 64/512 个 FFN 神经元，跨折选择集合平均 Jaccard 为 "
        f"**{fixed['mean_pairwise_selected_jaccard']:.3f}**，Macro-F1 为 "
        f"**{overall['macro_f1_sparse64']:.4f}**。相对最终层基线，它救回 "
        f"**{fixed['selected_rescues_baseline_errors']}** 个错例，同时破坏 "
        f"**{fixed['selected_harms_baseline_correct']}** 个原本正确样本。"
        f"10 个从训练折非恒定神经元池抽取的随机对照，Macro-F1 为 "
        f"**{fixed['random_macro_f1_mean']:.4f} +/- {fixed['random_macro_f1_sample_sd']:.4f}**，"
        f"平均救回 **{fixed['random_rescues_baseline_errors_mean']:.1f}** 个、破坏 "
        f"**{fixed['random_harms_baseline_correct_mean']:.1f}** 个。",
        "",
        "## 哪些错误不应首先靠稀疏模块解决",
        "",
        f"- **相对低 counts/低检测基因相关**：{flags['low_quality']:,} 个。数据无 `MT-` 基因，"
        "所以这只是 donor 内部低尾 proxy，不能称已经确认低质量。",
        f"- **原始表达、marker 与邻域都不支持真值**：{flags['ambiguous_or_overlap']:,} 个。"
        "可能是连续细胞状态、标签噪声或真边界，强行稀疏化容易制造貌似清晰的解释。",
        "- 系统性 donor 偏移应靠归一化、domain adaptation 或更广训练 donor 解决，而不是给单个错误强加生物模块解释。",
        "",
        "## 对照结果",
        "",
        "| Representation/readout | OOF Macro-F1 | Accuracy |",
        "|---|---:|---:|",
    ]
    for row in layer_metrics.itertuples(index=False):
        lines.append(f"| {row.representation} | {row.macro_f1:.4f} | {row.accuracy:.4f} |")
    lines += [
        "",
        "## 各 donor",
        "",
        "| Held-out donor | Cells | Layer-12 F1 | Raw-expression F1 | L1-FFN F1 | Selected-64 F1 | Random-64 repeat0 F1 | Marker F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_metrics.itertuples(index=False):
        lines.append(
            f"| {row.held_out_donor} | {row.cells} | {row.macro_f1_layer12:.4f} | "
            f"{row.macro_f1_raw_linear:.4f} | {row.macro_f1_sparse_ffn:.4f} | "
            f"{row.macro_f1_sparse64:.4f} | {row.macro_f1_random64_repeat0:.4f} | {row.macro_f1_marker:.4f} |"
        )
    lines += [
        "",
        "## 主要混淆对",
        "",
        "| True | Predicted | Errors | Sparse candidates | Candidate fraction |",
        "|---|---|---:|---:|---:|",
    ]
    for row in top_pairs.itertuples(index=False):
        lines.append(
            f"| {row.true_cell_type} | {row.predicted_cell_type} | {row.errors} | "
            f"{row.sparse_candidates} | {row.sparse_candidate_fraction:.1%} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "本报告把 marker、早层正确、邻域支持和稀疏 FFN 救回作为诊断证据。"
        "固定 64 神经元已完成等预算随机 readout 对照，但它仍是读取既有表征，不是对 scGPT 计算路径的干预。"
        "它不证明神经元对应稳定生物机制，也不证明加入稀疏模块后总体准确率会提高。"
        "下一步应在开发 OOF 结果上学习不依赖未知真值的拒绝机制，再到新的外部数据或新 donor 上与随机等预算残差比较。",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text())
    seed_all(int(config["seed"]))
    torch.set_num_threads(int(config["cpu_threads"]))
    device = choose_device(config["device"])
    print(f"device={device}", flush=True)

    model, vocab, checkpoint_args, checkpoint_audit = load_dense(config)
    cohort = KangDevelopmentCohort(config, vocab)
    dataset_hash = sha256(Path(config["dataset"]))
    checkpoint_hash = sha256(Path(config["checkpoint_dir"]) / "best_model.pt")
    expected_feature_manifest = feature_manifest(
        config, cohort, dataset_hash, checkpoint_hash
    )
    model.to(device)
    feature_path = root / "cache" / "features.npz"
    extract_features(
        config, cohort, model, device, feature_path, expected_feature_manifest
    )
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    summary = evaluate(config, cohort, feature_path, root)
    atomic_json(
        root / "PROVENANCE.json",
        {
            "config_sha256": sha256(config_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "dataset_sha256": dataset_hash,
            "checkpoint_sha256": checkpoint_hash,
            "feature_cache_sha256": sha256(feature_path),
            "feature_manifest": expected_feature_manifest,
            "checkpoint_args": checkpoint_args,
            "checkpoint_audit": checkpoint_audit,
            "device": str(device),
            "atlas_excluded_donors_loaded_with_source_file": True,
            "atlas_excluded_donors_tokenized_fitted_or_evaluated": False,
            "atlas_excluded_donors_were_historically_evaluated_elsewhere": True,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
        },
    )
    print(json.dumps(summary["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
