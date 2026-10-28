"""Sparse, module-aware scGPT annotation training on donor-disjoint Kang splits."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
from scipy import sparse
from sklearn.metrics import f1_score
import torch
from torch import nn
import torch.nn.functional as F
from scgpt.model import TransformerModel


METHODS = ("embedding", "random_module_vicreg", "stable_module_vicreg",
           "stable_module_sparse")


def decode(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dense(config):
    directory = Path(config["checkpoint_dir"])
    args = json.loads((directory / "args.json").read_text())
    vocab = json.loads((directory / "vocab.json").read_text())
    model = TransformerModel(
        len(vocab), args["embsize"], args["nheads"], args["d_hid"], args["nlayers"],
        vocab=vocab, dropout=args["dropout"], pad_value=-2, do_mvc=True,
        input_emb_style=args["input_emb_style"], n_input_bins=args["n_bins"],
        cell_emb_style="avg-pool", use_fast_transformer=False)
    del model.cls_decoder
    model.flag_encoder = nn.Embedding(2, args["embsize"])
    state = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True, mmap=True)
    converted = {k.replace(".Wqkv.weight", ".in_proj_weight").replace(
        ".Wqkv.bias", ".in_proj_bias"): value for k, value in state.items()}
    model.load_state_dict(converted, strict=True)
    return model, vocab, args, {"checkpoint_tensors": len(state),
        "checkpoint_parameters": sum(v.numel() for v in state.values()), "strict_load": True}


class KangCohort:
    """Raw-count source, train-only gene filter, fixed donor-disjoint cohort."""
    def __init__(self, config, vocab):
        self.config, self.vocab = config, vocab
        with h5py.File(config["dataset"], "r") as handle:
            obs, var = handle["obs"][:], handle["var"][:]
            self.meta = {name: decode(handle["uns"][name + "_categories"][:])[obs[name]]
                         for name in ["replicate", "label", "cell_type"]}
            self.cell_ids = decode(obs["index"])
            self.names = decode(var["name"])
            self.x = sparse.csc_matrix((handle["X/data"][:], handle["X/indices"][:],
                                        handle["X/indptr"][:]),
                                       shape=tuple(handle["X"].attrs["h5sparse_shape"])).tocsr()
        self.x.eliminate_zeros()
        if len(set(self.names)) != len(self.names):
            raise ValueError("Duplicate gene symbols require an aggregation rule")
        if (self.x.data < 0).any() or not np.isfinite(self.x.data).all() or not np.allclose(
                self.x.data, np.rint(self.x.data), atol=1e-6, rtol=0):
            raise ValueError("Expected finite nonnegative integer-like raw counts")
        train_mask = np.isin(self.meta["replicate"], config["split"]["train"])
        detected = np.asarray((self.x[train_mask] > 0).sum(0)).ravel()
        keep = np.array([name in vocab for name in self.names]) & (
            detected >= config["min_train_cells_per_gene"])
        self.vocab_id_by_source_gene = np.full(len(self.names), -1, dtype=np.int64)
        self.vocab_id_by_source_gene[keep] = [vocab[name] for name in self.names[keep]]
        self.pad_id = int(vocab["<pad>"])
        self.rows = self._select_rows()
        self.classes = sorted({self.meta["cell_type"][i] for i in self.rows["train"]})
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
        for split, rows in self.rows.items():
            unknown = {self.meta["cell_type"][i] for i in rows} - set(self.classes)
            if unknown:
                raise ValueError(f"{split} has labels absent from training: {unknown}")
        counts = np.bincount([self.label(i) for i in self.rows["train"]], minlength=len(self.classes))
        self.class_weights = counts.sum() / (len(counts) * counts)

    def _select_rows(self):
        rng = np.random.default_rng(self.config["cohort_seed"])
        selected = {}
        excluded = set(self.config["excluded_cell_types"])
        for split, donors in self.config["split"].items():
            pools = []
            for donor in donors:
                for cell_type in sorted(set(self.meta["cell_type"])):
                    if cell_type in excluded:
                        continue
                    eligible = ((self.meta["replicate"] == donor) &
                                (self.meta["label"] == self.config["condition"]) &
                                (self.meta["cell_type"] == cell_type))
                    rows = np.flatnonzero(eligible)
                    usable = []
                    for row in rng.permutation(rows):
                        source_cols = self.x[int(row)].indices
                        if np.sum(self.vocab_id_by_source_gene[source_cols] >= 0) >= self.config["min_expressed_genes"]:
                            usable.append(int(row))
                    if usable:
                        pools.append(usable)
            limit = int(self.config["split_cell_limits"][split])
            if sum(map(len, pools)) < limit:
                raise ValueError(f"Insufficient eligible cells for {split}")
            chosen = []
            while len(chosen) < limit:
                progressed = False
                for pool in pools:
                    if pool and len(chosen) < limit:
                        chosen.append(pool.pop())
                        progressed = True
                if not progressed:
                    raise RuntimeError("Cohort allocator stalled")
            selected[split] = np.array(sorted(chosen), dtype=np.int64)
        all_rows = np.concatenate(list(selected.values()))
        if len(all_rows) != len(set(map(int, all_rows))):
            raise ValueError("Cohort splits overlap")
        return selected

    def label(self, source_row):
        return self.class_to_id[str(self.meta["cell_type"][source_row])]

    def record(self, source_row):
        return {"source_row": int(source_row), "cell_id": str(self.cell_ids[source_row]),
                "donor": str(self.meta["replicate"][source_row]),
                "cell_type": str(self.meta["cell_type"][source_row]),
                "condition": str(self.meta["label"][source_row])}

    def tokenize(self, source_rows, rate, draw_seed):
        seq_len = int(self.config["seq_len"])
        genes = np.full((len(source_rows), seq_len), self.pad_id, dtype=np.int64)
        values = np.full((len(source_rows), seq_len), -2, dtype=np.float32)
        labels = np.empty(len(source_rows), dtype=np.int64)
        for batch_index, source_row in enumerate(map(int, source_rows)):
            raw = self.x[source_row]
            source_cols = raw.indices.copy()
            counts = raw.data.astype(np.int64, copy=True)
            if rate < 1:
                seed = np.random.SeedSequence([int(draw_seed), source_row, int(round(rate * 1000))])
                counts = np.random.default_rng(seed).binomial(counts, rate)
            nonzero = counts > 0
            source_cols, counts = source_cols[nonzero], counts[nonzero]
            total = int(counts.sum())
            retained = self.vocab_id_by_source_gene[source_cols] >= 0
            source_cols, kept_counts = source_cols[retained], counts[retained]
            minimum_view_genes = self.config["min_expressed_genes"] if rate == 1 else 1
            if total <= 0 or len(source_cols) < minimum_view_genes:
                raise ValueError(f"View became ineligible at row {source_row}, rate {rate}")
            normalized = np.log1p(kept_counts.astype(np.float64) / total * 1e4)
            edges = np.quantile(normalized, np.linspace(0, 1, self.config["n_bins"] - 1))
            left = np.digitize(normalized, edges)
            right = np.digitize(normalized, edges, right=True)
            tie = np.random.default_rng(np.random.SeedSequence(
                [self.config["cohort_seed"], source_row, 11])).random(len(self.names))
            bins = np.ceil(tie[source_cols] * (right - left) + left).astype(np.float32)
            priority = np.random.default_rng(np.random.SeedSequence(
                [self.config["cohort_seed"], source_row, 12])).random(len(self.names))
            pick = np.arange(len(source_cols))
            if len(pick) > seq_len:
                pick = np.sort(np.argsort(priority[source_cols], kind="stable")[:seq_len])
            n = len(pick)
            genes[batch_index, :n] = self.vocab_id_by_source_gene[source_cols[pick]]
            values[batch_index, :n] = bins[pick]
            labels[batch_index] = self.label(source_row)
        return {"genes": torch.from_numpy(genes), "values": torch.from_numpy(values),
                "padding": torch.from_numpy(genes == self.pad_id), "labels": torch.from_numpy(labels)}

    def audit(self):
        return {"source_shape": list(self.x.shape), "condition": self.config["condition"],
                "classes": self.classes, "class_weights": self.class_weights.tolist(),
                "splits": {split: {"cells": len(rows),
                    "donors": sorted(set(self.meta["replicate"][rows])),
                    "cell_type_counts": {name: int(np.sum(self.meta["cell_type"][rows] == name))
                                         for name in self.classes}}
                    for split, rows in self.rows.items()},
                "gene_filter": "vocabulary membership and >= min_train_cells_per_gene in training donors only"}


class AnnotationModel(nn.Module):
    def __init__(self, base, classes, layer_index, method, module_spec):
        super().__init__()
        if layer_index != len(base.transformer_encoder.layers) - 1:
            raise ValueError("This MVP intentionally supports last-layer fine-tuning only")
        self.base, self.method, self.layer_index = base, method, layer_index
        self.base.requires_grad_(False)
        self.layer = self.base.transformer_encoder.layers[layer_index]
        self.layer.requires_grad_(True)
        self.head = nn.Linear(self.layer.linear2.out_features, classes)
        modules = None
        if method in {"stable_module_vicreg", "stable_module_sparse"}:
            modules = module_spec["stable_modules"]
        elif method == "random_module_vicreg":
            modules = module_spec["random_modules"]
        self.module_indices = ([torch.tensor(group, dtype=torch.long) for group in modules]
                               if modules is not None else None)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20261011)
        projection = torch.randn(512, 8, generator=generator)
        self.register_buffer("module_projection", projection, persistent=False)
        self._captured = None
        self._handle = self.layer.linear2.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self._captured = inputs[0]

    def optimization_mode(self):
        # Dropout is disabled so the paired difference represents measurement thinning.
        self.eval()
        self.head.train()

    def forward(self, batch):
        genes, values, padding = batch["genes"], batch["values"], batch["padding"]
        with torch.no_grad():
            hidden = self.base.encoder(genes) + self.base.value_encoder(values)
            for block in self.base.transformer_encoder.layers[:self.layer_index]:
                hidden = block(hidden, src_key_padding_mask=padding)
        self._captured = None
        hidden = self.layer(hidden, src_key_padding_mask=padding)
        if self._captured is None:
            raise RuntimeError("FFN activation hook did not run")
        valid = ~padding
        pooled = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        module_vectors, module_energy = None, None
        if self.module_indices is not None:
            activation = self._captured.abs()
            neuron_response = ((activation * valid.unsqueeze(-1)).sum(1) /
                               valid.sum(1, keepdim=True))
            vectors, energies = [], []
            for index in self.module_indices:
                index = index.to(activation.device)
                response = neuron_response.index_select(-1, index)
                standardized = F.layer_norm(torch.log1p(response), (len(index),))
                projection = self.module_projection.index_select(0, index)
                vectors.append(standardized @ projection / math.sqrt(len(index)))
                energies.append(torch.log1p(response.mean(-1)))
            module_vectors = torch.stack(vectors, dim=1)
            module_energy = torch.stack(energies, dim=-1)
        return self.head(pooled), pooled, module_vectors, module_energy


def normalized_consistency(first, second):
    first = F.normalize(first, dim=-1, eps=1e-8)
    second = F.normalize(second, dim=-1, eps=1e-8)
    return 0.5 * (first - second).square().sum(-1).mean()


def off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg(first, second, variance_target=1.0, variance_weight=1.0,
           covariance_weight=0.04):
    first = first.flatten(1)
    second = second.flatten(1)
    invariance = F.mse_loss(first, second)
    joined = torch.cat([first, second], dim=0)
    centered = joined - joined.mean(0, keepdim=True)
    std = torch.sqrt(centered.var(0, unbiased=True) + 1e-4)
    variance = F.relu(variance_target - std).mean()
    covariance = centered.T @ centered / max(len(joined) - 1, 1)
    covariance = off_diagonal(covariance).square().sum() / covariance.shape[0]
    total = invariance + variance_weight * variance + covariance_weight * covariance
    return total, {"invariance": invariance, "variance": variance,
                   "covariance": covariance}


def entmax15(logits, dim=-1):
    """Differentiable alpha=1.5 entmax with exact zeros."""
    values = logits / 2
    values = values - values.max(dim=dim, keepdim=True).values
    sorted_values = values.sort(dim=dim, descending=True).values
    count = values.shape[dim]
    shape = [1] * values.ndim
    shape[dim] = count
    rho = torch.arange(1, count + 1, dtype=values.dtype,
                       device=values.device).view(shape)
    mean = sorted_values.cumsum(dim) / rho
    mean_square = sorted_values.square().cumsum(dim) / rho
    delta = (1 - rho * (mean_square - mean.square())) / rho
    taus = mean - torch.sqrt(torch.clamp(delta, min=0))
    support = taus <= sorted_values
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = taus.gather(dim, support_size - 1)
    probabilities = torch.clamp(values - tau, min=0).square()
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def module_gate(energy, temperature):
    standardized = ((energy - energy.mean(-1, keepdim=True)) /
                    energy.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4))
    return entmax15(standardized / temperature)


def sparse_gate_loss(first_energy, second_energy, temperature, diversity_weight):
    first = module_gate(first_energy, temperature)
    second = module_gate(second_energy, temperature)
    consistency = F.mse_loss(first, second)
    mean_gate = 0.5 * (first + second)
    entropy = -(mean_gate * mean_gate.clamp_min(1e-8).log()).sum(-1).mean()
    population = mean_gate.mean(0)
    population_entropy = -(population * population.clamp_min(1e-8).log()).sum()
    objective = consistency + entropy - diversity_weight * population_entropy
    diagnostics = {"gate_consistency": consistency, "gate_entropy": entropy,
                   "population_entropy": population_entropy,
                   "active_modules": (mean_gate > 1e-6).float().sum(-1).mean(),
                   "max_gate": mean_gate.max(-1).values.mean()}
    return objective, diagnostics


def active_auxiliary_names(method):
    names = ["embedding"]
    if method != "embedding":
        names.append("module_vicreg")
    if method == "stable_module_sparse":
        names.append("sparse_gate")
    return names


def compute_loss(model, full, thin, class_weights, weights, config):
    logits_a, embedding_a, modules_a, energy_a = model(full)
    logits_b, embedding_b, modules_b, energy_b = model(thin)
    classification = 0.5 * (F.cross_entropy(logits_a, full["labels"], weight=class_weights) +
                            F.cross_entropy(logits_b, thin["labels"], weight=class_weights))
    terms = {"embedding": normalized_consistency(embedding_a, embedding_b)}
    diagnostics = {}
    if modules_a is not None:
        terms["module_vicreg"], pieces = vicreg(
            modules_a, modules_b,
            variance_target=float(config["vicreg_variance_target"]),
            variance_weight=float(config["vicreg_variance_weight"]),
            covariance_weight=float(config["vicreg_covariance_weight"]))
        diagnostics.update({f"vicreg_{key}": value for key, value in pieces.items()})
    if model.method == "stable_module_sparse":
        terms["sparse_gate"], pieces = sparse_gate_loss(
            energy_a, energy_b, float(config["gate_temperature"]),
            float(config["gate_diversity_weight"]))
        diagnostics.update(pieces)
    total = classification + sum(weights.get(name, 0.0) * value
                                 for name, value in terms.items())
    return total, classification, terms, diagnostics


def layer_gradient_norm(model):
    terms = [parameter.grad.detach().float().square().sum()
             for parameter in model.layer.parameters() if parameter.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def calibrate_auxiliary_weights(model, cohort, config, class_weights, seed, device):
    # Match the actual optimization forward pass: paired views differ only by
    # count thinning, while Transformer dropout is disabled.
    model.optimization_mode()
    batch_size = int(config["batch_size"])
    n_batches = int(config["gradient_calibration_batches"])
    total = batch_size * n_batches
    rng = np.random.default_rng(seed)
    pools = []
    for class_id in range(len(cohort.classes)):
        eligible = np.array([row for row in cohort.rows["train"]
                             if cohort.label(row) == class_id], dtype=np.int64)
        pools.append(list(map(int, rng.permutation(eligible))))
    rows = []
    while len(rows) < total:
        progressed = False
        for pool in pools:
            if pool and len(rows) < total:
                rows.append(pool.pop())
                progressed = True
        if not progressed:
            raise RuntimeError("Calibration cohort allocator stalled")
    rows = np.array(rows, dtype=np.int64)

    model.zero_grad(set_to_none=True)
    for batch_index in range(n_batches):
        take = rows[batch_index * batch_size:(batch_index + 1) * batch_size]
        full = to_device(cohort.tokenize(take, 1.0, seed), device)
        thin = to_device(cohort.tokenize(take, config["thin_rate_train"], seed), device)
        _, classification, _, _ = compute_loss(model, full, thin, class_weights, {}, config)
        (classification / n_batches).backward()
    classification_grad = layer_gradient_norm(model)

    if not (np.isfinite(classification_grad) and classification_grad > 0):
        raise FloatingPointError("Invalid classification gradient during calibration")
    weights, auxiliaries = {}, {}
    for name in active_auxiliary_names(model.method):
        model.zero_grad(set_to_none=True)
        for batch_index in range(n_batches):
            take = rows[batch_index * batch_size:(batch_index + 1) * batch_size]
            full = to_device(cohort.tokenize(take, 1.0, seed), device)
            thin = to_device(cohort.tokenize(take, config["thin_rate_train"], seed), device)
            _, _, terms, _ = compute_loss(model, full, thin, class_weights, {}, config)
            (terms[name] / n_batches).backward()
        auxiliary_grad = layer_gradient_norm(model)
        if not (np.isfinite(auxiliary_grad) and auxiliary_grad > 0):
            raise FloatingPointError(f"Invalid {name} gradient during calibration")
        target = float(config["aux_target_gradient_ratios"][name])
        raw_weight = target * classification_grad / auxiliary_grad
        cap = float(config["aux_weight_maxes"][name])
        weight = min(float(raw_weight), cap)
        weights[name] = weight
        auxiliaries[name] = {"auxiliary_grad_norm": auxiliary_grad,
            "target_ratio": target, "uncapped_weight": float(raw_weight),
            "effective_weight": weight, "capped": bool(weight < raw_weight),
            "realized_gradient_ratio": weight * auxiliary_grad / classification_grad}
    model.zero_grad(set_to_none=True)
    return weights, {"classification_grad_norm": classification_grad,
        "auxiliaries": auxiliaries, "cells": len(rows),
        "cell_type_counts": {name: int(sum(cohort.label(row) == class_id for row in rows))
                             for class_id, name in enumerate(cohort.classes)},
        "calibration_mode": "optimization_mode_no_dropout",
        "selection_data": "seeded class-balanced sample of training cohort only"}


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, cohort, split, rates, thinning_seeds, batch_size, device):
    model.optimization_mode()
    rows = cohort.rows[split]
    records = []
    for rate in rates:
        seeds = [0] if rate == 1 else thinning_seeds
        for thinning_seed in seeds:
            truth, prediction, donors, gates, module_vectors = [], [], [], [], []
            for start in range(0, len(rows), batch_size):
                take = rows[start:start + batch_size]
                batch = to_device(cohort.tokenize(take, rate, thinning_seed), device)
                logits, _, vectors, energy = model(batch)
                truth.extend(batch["labels"].cpu().tolist())
                prediction.extend(logits.argmax(-1).cpu().tolist())
                donors.extend(map(str, cohort.meta["replicate"][take]))
                if energy is not None:
                    gates.append(module_gate(energy, float(cohort.config["gate_temperature"])).cpu())
                    module_vectors.append(vectors.cpu())
            row = {"split": split, "rate": rate, "thinning_seed": None if rate == 1 else thinning_seed,
                   "macro_f1_pooled": float(f1_score(truth, prediction, average="macro")),
                   "per_class_f1": {name: float(value) for name, value in zip(cohort.classes,
                       f1_score(truth, prediction, labels=range(len(cohort.classes)),
                                average=None, zero_division=0))}, "donor_macro_f1": {}}
            truth, prediction, donors = np.array(truth), np.array(prediction), np.array(donors)
            for donor in sorted(set(donors)):
                mask = donors == donor
                row["donor_macro_f1"][donor] = float(f1_score(
                    truth[mask], prediction[mask], average="macro", zero_division=0))
            if gates:
                gate = torch.cat(gates).numpy()
                vectors = torch.cat(module_vectors).flatten(1).numpy()
                entropy = -(gate * np.log(np.clip(gate, 1e-8, None))).sum(1)
                row["module_diagnostics"] = {
                    "mean_active_modules": float(np.mean(np.sum(gate > 1e-6, axis=1))),
                    "mean_modules_over_1pct": float(np.mean(np.sum(gate > 0.01, axis=1))),
                    "mean_modules_over_5pct": float(np.mean(np.sum(gate > 0.05, axis=1))),
                    "mean_gate_entropy": float(np.mean(entropy)),
                    "mean_effective_modules": float(np.mean(np.exp(entropy))),
                    "mean_max_gate": float(np.mean(np.max(gate, axis=1))),
                    "mean_gate": np.mean(gate, axis=0).tolist(),
                    "feature_std_mean": float(np.std(vectors, axis=0, ddof=1).mean()),
                    "per_class_mean_gate": {
                        name: np.mean(gate[truth == class_id], axis=0).tolist()
                        for class_id, name in enumerate(cohort.classes)}}
            records.append(row)
    return records


def make_model(config, method, module_spec, classes, device):
    base, vocab, checkpoint_args, audit = load_dense(config)
    if int(checkpoint_args["d_hid"]) != 512:
        raise ValueError("Module indices assume 512 FFN neurons")
    model = AnnotationModel(base, classes, config["train_layer_index"], method, module_spec).to(device)
    model.optimization_mode()
    return model, vocab, audit


def optimizer_for(model, config):
    return torch.optim.AdamW([
        {"params": model.layer.parameters(), "lr": config["learning_rate"]},
        {"params": model.head.parameters(), "lr": config["head_learning_rate"]}],
        weight_decay=config["weight_decay"])


def smoke(config, module_spec, method, output, seed):
    device = torch.device(config["device"])
    base, vocab, _, checkpoint_audit = load_dense(config)
    cohort = KangCohort(config, vocab)
    model = AnnotationModel(base, len(cohort.classes), config["train_layer_index"], method,
                            module_spec).to(device)
    model.optimization_mode()
    optimizer = optimizer_for(model, config)
    take = cohort.rows["train"][:config["batch_size"]]
    full = to_device(cohort.tokenize(take, 1.0, seed), device)
    thin = to_device(cohort.tokenize(take, config["thin_rate_train"], seed), device)
    class_weights = torch.tensor(cohort.class_weights, dtype=torch.float32, device=device)
    effective_weights, calibration = calibrate_auxiliary_weights(
        model, cohort, config, class_weights, seed, device)
    frozen_parameter = next(model.base.encoder.parameters())
    frozen_before = frozen_parameter.detach().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    loss, classification, auxiliary, diagnostics = compute_loss(
        model, full, thin, class_weights, effective_weights, config)
    if not all(torch.isfinite(item) for item in
               [loss, classification, *auxiliary.values(), *diagnostics.values()]):
        raise FloatingPointError("Non-finite smoke loss")
    loss.backward()
    layer_grad = float(model.layer.linear1.weight.grad.norm())
    head_grad = float(model.head.weight.grad.norm())
    frozen_grad_absent = frozen_parameter.grad is None
    total_grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                config["grad_clip"], error_if_nonfinite=True)
    layer_before = model.layer.linear1.weight.detach().clone()
    optimizer.step()
    result = {"status": "pass", "method": method, "device": str(device),
        "batch_size": len(take), "classes": cohort.classes,
        "full_shape": list(full["genes"].shape), "thin_shape": list(thin["genes"].shape),
        "loss": float(loss), "classification_loss": float(classification),
        "auxiliary_losses": {name: float(value) for name, value in auxiliary.items()},
        "diagnostics": {name: float(value) for name, value in diagnostics.items()},
        "effective_auxiliary_weights": effective_weights,
        "layer_linear1_grad_norm": layer_grad, "gradient_calibration": calibration,
        "head_grad_norm": head_grad, "grad_norm_before_clip": float(total_grad),
        "frozen_gradient_absent": frozen_grad_absent,
        "frozen_parameter_unchanged": bool(torch.equal(frozen_before, frozen_parameter.detach().cpu())),
        "trainable_layer_updated": bool(not torch.equal(layer_before, model.layer.linear1.weight.detach())),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_audit": checkpoint_audit, "data_audit": cohort.audit()}
    if not (layer_grad > 0 and head_grad > 0 and frozen_grad_absent and
            result["frozen_parameter_unchanged"] and result["trainable_layer_updated"]):
        raise AssertionError("Gradient/update smoke checks failed")
    write_json(output / "smoke.json", result)
    print(json.dumps(result))


def train(config, module_spec, method, output, seed):
    device = torch.device(config["device"])
    base, vocab, _, checkpoint_audit = load_dense(config)
    cohort = KangCohort(config, vocab)
    model = AnnotationModel(base, len(cohort.classes), config["train_layer_index"], method,
                            module_spec).to(device)
    optimizer = optimizer_for(model, config)
    class_weights = torch.tensor(cohort.class_weights, dtype=torch.float32, device=device)
    effective_weights, calibration = calibrate_auxiliary_weights(
        model, cohort, config, class_weights, seed, device)
    train_rows = cohort.rows["train"]
    total = min(config["max_steps"], math.ceil(len(train_rows) / config["batch_size"]) * config["epochs"])
    rng = np.random.default_rng(seed)
    best_score, best_step, stale, step = -float("inf"), 0, 0, 0
    started = time.monotonic()
    metrics_path = output / "metrics.jsonl"
    metrics = metrics_path.open("x", buffering=1)
    write_json(output / "data_audit.json", cohort.audit())
    write_json(output / "cohort.json", {split: [cohort.record(i) for i in rows]
                                        for split, rows in cohort.rows.items()})
    write_json(output / "checkpoint_audit.json", checkpoint_audit)
    write_json(output / "gradient_calibration.json", calibration)

    def save(name):
        payload = {"format_version": 2, "method": method, "seed": seed, "step": step,
                   "trainable_state": {
                       "layer": {k: v.detach().cpu() for k, v in model.layer.state_dict().items()},
                       "head": {k: v.detach().cpu() for k, v in model.head.state_dict().items()}},
                   "optimizer": optimizer.state_dict(),
                   "best_validation_score": best_score, "classes": cohort.classes,
                   "config_sha256": sha256(output / "config.json"),
                   "module_spec_sha256": sha256(output / "module_spec.json")}
        temporary = output / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(output / name)

    try:
        for epoch in range(config["epochs"]):
            order = rng.permutation(train_rows)
            for start in range(0, len(order), config["batch_size"]):
                if step >= total:
                    break
                step += 1
                take = order[start:start + config["batch_size"]]
                full = to_device(cohort.tokenize(take, 1.0, seed + epoch), device)
                thin = to_device(cohort.tokenize(take, config["thin_rate_train"], seed + epoch), device)
                model.optimization_mode()
                optimizer.zero_grad(set_to_none=True)
                loss, classification, auxiliary, diagnostics = compute_loss(
                    model, full, thin, class_weights, effective_weights, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config["grad_clip"],
                    error_if_nonfinite=True)
                optimizer.step()
                row = {"step": step, "epoch": epoch + 1, "loss": float(loss),
                       "classification_loss": float(classification),
                       "auxiliary_losses": {name: float(value) for name, value in auxiliary.items()},
                       "diagnostics": {name: float(value) for name, value in diagnostics.items()},
                       "grad_norm": float(grad_norm),
                       "effective_auxiliary_weights": effective_weights,
                       "elapsed_seconds": time.monotonic() - started}
                metrics.write(json.dumps(row) + "\n")
                if step % config["validation_every"] == 0 or step == total:
                    validation = evaluate(model, cohort, "validation", [1.0, 0.5], [42],
                                          config["batch_size"], device)
                    score = float(np.mean([item["macro_f1_pooled"] for item in validation]))
                    metrics.write(json.dumps({"step": step, "validation": validation,
                                              "selection_score": score}) + "\n")
                    if score > best_score:
                        best_score, best_step, stale = score, step, 0
                        save("best.pt")
                    else:
                        stale += 1
                    save("last.pt")
                    if stale >= config["early_stop_evaluations"]:
                        break
            if step >= total or stale >= config["early_stop_evaluations"]:
                break
        write_json(output / "summary.json", {"state": "completed", "method": method,
            "seed": seed, "steps": step, "planned_steps": total, "best_step": best_step,
            "best_validation_score": best_score, "elapsed_seconds": time.monotonic() - started,
            "test_evaluated": False, "claim": "training completed; no test claim until evaluate mode"})
    finally:
        metrics.close()


def evaluate_checkpoint(config, module_spec, method, checkpoint, output, seed, split):
    device = torch.device(config["device"])
    base, vocab, _, _ = load_dense(config)
    cohort = KangCohort(config, vocab)
    model = AnnotationModel(base, len(cohort.classes), config["train_layer_index"], method,
                            module_spec).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload["method"] != method or payload["classes"] != cohort.classes:
        raise ValueError("Checkpoint method/classes mismatch")
    model.layer.load_state_dict(payload["trainable_state"]["layer"], strict=True)
    model.head.load_state_dict(payload["trainable_state"]["head"], strict=True)
    if split == "test":
        rates, thinning_seeds = config["evaluation_rates"], config["evaluation_thinning_seeds"]
    else:
        rates, thinning_seeds = [1.0, 0.5], [42]
    records = evaluate(model, cohort, split, rates, thinning_seeds,
                       config["batch_size"], device)
    full_f1 = next(row["macro_f1_pooled"] for row in records if row["rate"] == 1.0)
    for row in records:
        row["macro_f1_drop_from_full_pp"] = 100 * (full_f1 - row["macro_f1_pooled"])
    result = {"method": method, "seed": seed, "split": split,
              "checkpoint": str(Path(checkpoint).resolve()),
              "checkpoint_sha256": sha256(checkpoint),
              "test_used_for_selection": False,
              "evaluation_post_selection": split == "validation",
              "records": records}
    write_json(output / f"{split}_results.json", result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "train", "evaluate", "validate"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--module-spec", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    module_spec = json.loads(args.module_spec.read_text())
    if module_spec["layer_one_based"] != config["train_layer_index"] + 1:
        raise ValueError("Module layer and trainable layer differ")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "module_spec.json").write_text(json.dumps(module_spec, indent=2))
    write_json(args.output / "provenance.json", {"code_sha256": sha256(Path(__file__)),
        "input_config_sha256": sha256(args.config), "input_module_spec_sha256": sha256(args.module_spec),
        "seed": args.seed, "method": args.method, "torch": torch.__version__})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(config["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; run on the macOS host")
    if args.mode == "smoke":
        smoke(config, module_spec, args.method, args.output, args.seed)
    elif args.mode == "train":
        train(config, module_spec, args.method, args.output, args.seed)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for evaluate")
        split = "test" if args.mode == "evaluate" else "validation"
        evaluate_checkpoint(config, module_spec, args.method, args.checkpoint,
                            args.output, args.seed, split)


if __name__ == "__main__":
    main()
