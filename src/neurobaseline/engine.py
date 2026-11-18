from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModel

from .config import Config
from .data import BagCollator, encode_dataset, frame_digest, load_frame, require_disjoint, split_path, stratum, tokenizer_digest
from .metrics import deployment, fit_thresholds, function_pair_rank, summary
from .model import BaselineModel, autocast_context, choose_device, move_batch
from .objectives import Objective, prevalence_weights
from .cfa_heads import CFAObjective, ENV_OF_STRATUM, reference_log_lengths
from .cfa_model import (CFAModel, CFACollator, PairAdjacentSampler, PairBatchSampler, WindowLabeledDataset,
                        cfa_group_loss, complete_pairs, function_pair_loss, pair_keys, window_counts, window_label_map)

log = logging.getLogger(__name__)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_path(cfg: Config) -> Path:
    suffix = f"_smoke{cfg.smoke_limit}" if cfg.smoke_limit is not None else ""
    return Path(cfg.output_dir) / cfg.variant / f"seed{cfg.seed}{suffix}"


def get_tokenizer(cfg: Config):
    return AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.model, use_fast=True,
                                         local_files_only=cfg.local_files_only)


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_cfa_pairs(cfg, tokenizer=None):
    path = Path(cfg.cfa_pairs)
    if not path.is_file():
        raise ValueError(f"CFA pair table is missing: {path}")
    pairs = pd.read_parquet(path)
    required = {"split", "vuln_idx", "fixed_idx", "n_windows", "n_fixed_windows",
                "positive_windows", "hard_negative_windows"}
    if not required.issubset(pairs):
        raise ValueError(f"CFA pair table is missing {required - set(pairs)}")
    for key in ("vuln_idx", "fixed_idx"):
        if pairs[key].isna().any():
            raise ValueError(f"CFA pair table has null {key}")
        pairs[key] = pairs[key].astype(str)
    digest = file_sha256(path)
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("outputs", {}).get(path.name) != digest:
            raise ValueError("CFA pair table differs from its manifest")
        if tokenizer is not None and manifest.get("tokenizer_digest") != tokenizer_digest(tokenizer):
            raise ValueError("CFA pair table uses a different tokenizer")
        settings = manifest.get("settings", {})
        for field, key in (("window_tokens", "width"), ("stride_tokens", "stride"),
                           ("max_input_tokens", "max_input")):
            if settings.get(key) != getattr(cfg, field):
                raise ValueError(f"CFA pair table uses different {field}")
    return pairs, digest


def validate_cfa_train_pairs(pairs, train_frame):
    """Check train labels and window ordinals before attaching auxiliary targets."""
    part = pairs.loc[pairs.split == "train"]
    if part.empty:
        raise ValueError("CFA pair table has no train pairs")
    if part.vuln_idx.duplicated().any():
        raise ValueError("CFA train pairs repeat a vulnerable idx")
    lookup = train_frame.set_index(train_frame.idx.astype(str))
    counts = {}
    for row in part.itertuples(index=False):
        for idx, label, n in ((str(row.vuln_idx), 1, row.n_windows),
                              (str(row.fixed_idx), 0, row.n_fixed_windows)):
            if idx not in lookup.index or int(lookup.loc[idx, "target"]) != label:
                raise ValueError(f"CFA train pair idx={idx} is absent or has the wrong label")
            if int(n) < 1 or (idx in counts and counts[idx] != int(n)):
                raise ValueError(f"CFA pair window count is inconsistent for idx={idx}")
            counts[idx] = int(n)
        if "n_tokens" in lookup and hasattr(row, "n_tokens"):
            if int(lookup.loc[str(row.vuln_idx), "n_tokens"]) != row.n_tokens:
                raise ValueError("CFA vulnerable function length differs from the pair table")
            if int(lookup.loc[str(row.fixed_idx), "n_tokens"]) != row.fixed_n_tokens:
                raise ValueError("CFA fixed function length differs from the pair table")
        for values, n in ((row.positive_windows, row.n_windows),
                          (row.hard_negative_windows, row.n_fixed_windows)):
            if any(int(k) != k or not 0 <= int(k) < int(n) for k in values):
                raise ValueError("CFA pair has an out-of-range window label")
    return part


def optimization_groups(loader, accumulation_steps):
    """Yield bounded lists of CPU microbatches, including the final partial group."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    pending = []
    for batch in loader:
        pending.append(batch)
        if len(pending) == accumulation_steps:
            yield pending
            pending = []
    if pending:
        yield pending


def training_plan(cfg, n_functions, epochs=None):
    """Schedule in optimization groups, never in individual encoder forwards."""
    if n_functions < 1:
        raise ValueError("Training needs at least one function")
    epochs = (1 if cfg.smoke_limit is not None else cfg.epochs) if epochs is None else epochs
    microbatches = math.ceil(n_functions / cfg.batch_size)
    groups = math.ceil(microbatches / cfg.gradient_accumulation_steps)
    validations = sorted({math.ceil(groups * j / cfg.evals_per_epoch)
                          for j in range(1, cfg.evals_per_epoch + 1)})
    return {"physical_batch_size": cfg.batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "effective_batch_size": cfg.effective_batch_size,
            "gradient_cache": cfg.uses_gradient_cache,
            "epochs": epochs, "microbatches_per_epoch": microbatches,
            "optimization_groups_per_epoch": groups,
            "planned_optimizer_updates": epochs * groups,
            "final_group_functions": n_functions - (groups - 1) * cfg.effective_batch_size,
            "validation_group_boundaries": validations,
            "early_stopping_patience": cfg.patience,
            "warmup_updates": int(cfg.warmup_fraction * epochs * groups)}


def optimizer_update(model, optimizer, scheduler, scaler, grad_clip):
    """Advance LR only when GradScaler actually performed the parameter update."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= previous_scale
    if updated:
        scheduler.step()
    return updated


def make_optimizer(model, cfg):
    if cfg.variant == "cfa" and (cfg.head_lr is not None or cfg.intercept_lr is not None):
        buckets = {key: [] for key in ("encoder_decay", "encoder_no_decay", "head_decay",
                                        "head_no_decay", "intercept")}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "invariant.b":
                bucket = "intercept"
            elif name.startswith(("pool.", "head.", "window_scorer.", "invariant.v.")):
                bucket = "head_no_decay" if parameter.ndim < 2 or "LayerNorm" in name else "head_decay"
            elif name.startswith("encoder."):
                bucket = "encoder_no_decay" if parameter.ndim < 2 or "LayerNorm" in name else "encoder_decay"
            else:
                raise ValueError(f"Unclassified CFA parameter: {name}")
            buckets[bucket].append(parameter)
        head_lr = cfg.lr if cfg.head_lr is None else cfg.head_lr
        intercept_lr = cfg.lr if cfg.intercept_lr is None else cfg.intercept_lr
        groups = [{"params": buckets["encoder_decay"], "weight_decay": cfg.weight_decay, "lr": cfg.lr},
                  {"params": buckets["encoder_no_decay"], "weight_decay": 0.0, "lr": cfg.lr},
                  {"params": buckets["head_decay"], "weight_decay": cfg.weight_decay, "lr": head_lr},
                  {"params": buckets["head_no_decay"], "weight_decay": 0.0, "lr": head_lr},
                  {"params": buckets["intercept"], "weight_decay": 0.0, "lr": intercept_lr}]
        return torch.optim.AdamW(groups)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or "LayerNorm" in name else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)


def cfa_train_prior(frame, stratum_weights) -> dict:
    """CFA intercept priors from the selected TRAIN functions only."""
    y = np.asarray(frame.target, dtype=np.float64)
    stratum = np.asarray(frame.stratum, dtype=np.int64)
    weights = np.asarray(stratum_weights, dtype=np.float64)
    if len(y) == 0 or y.shape != stratum.shape or weights.shape != (5,):
        raise ValueError("CFA prior needs TRAIN targets, strata, and five stratum weights")
    if not np.isin(stratum, np.arange(1, 6)).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("CFA prior has invalid TRAIN labels or strata")
    env = np.asarray(ENV_OF_STRATUM, dtype=np.int64)[stratum - 1]
    counts = np.bincount(env, minlength=4)
    positives = np.bincount(env, weights=y, minlength=4)
    if np.any(counts == 0):
        raise ValueError("CFA prior needs all four TRAIN environments")
    weighted = weights[stratum - 1]
    if not np.isfinite(weighted).all() or np.any(weighted <= 0):
        raise ValueError("CFA prior needs finite positive weights")
    return {"env_prior": (positives / counts).tolist(),
            "prior": float(np.dot(weighted, y) / weighted.sum())}


def _rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["device"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["device"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["device"])


@contextmanager
def replay_rng(state, device):
    """Replay encoder dropout without consuming the post-loss RNG stream twice."""
    current = _rng_state(device)
    _set_rng_state(state, device)
    try:
        yield
    finally:
        _set_rng_state(current, device)


def merge_group_metadata(batches, device):
    """Concatenate function/window metadata; encoder inputs stay in CPU microbatches."""
    result = {key: torch.cat([batch[key] for batch in batches]).to(device)
              for key in ("target", "stratum", "log_length")}
    owners, offset = [], 0
    for batch in batches:
        owners.append(batch["owners"] + offset)
        offset += len(batch["target"])
    result["owners"] = torch.cat(owners).to(device)
    result["idx"] = [idx for batch in batches for idx in batch["idx"]]
    if any("window_labels" in batch for batch in batches):
        if not all("window_labels" in batch for batch in batches):
            raise ValueError("Every microbatch must supply window_labels when using auxiliary labels")
        result["window_labels"] = torch.cat([batch["window_labels"] for batch in batches]).to(device)
    if all("pair_key" in batch for batch in batches):
        result["pair_key"] = torch.cat([batch["pair_key"] for batch in batches]).to(device)
    if any("replay" in batch for batch in batches):
        result["replay"] = torch.cat([batch.get("replay", torch.zeros(len(batch["target"]), dtype=torch.bool))
                                      for batch in batches]).to(device)
    return result


def gradient_cache_backward(model, batches, loss_fn, device, *, amp=False, scaler=None):
    """Differentiate ONE whole-group loss with a two-pass encoder VJP.

    loss_fn(features, group_metadata) may use attention, external CFA heads,
    window supervision, IRM, or Group DRO. Heads/pooling receive gradients once.
    Cached feature gradients are already AMP-scaled; replay must not scale again.
    The caller zeroes/unscales/clips/updates the optimizer once per group.
    """
    if not model.training or not batches:
        raise ValueError("Gradient caching needs a nonempty group and model.train()")
    states, cached = [], []
    for cpu_batch in batches:
        states.append(_rng_state(device))
        batch = move_batch(cpu_batch, device)
        with torch.no_grad(), autocast_context(device, amp):
            cached.append(model.encode_windows(batch).detach())
        del batch
    sizes = [len(features) for features in cached]
    leaf = torch.cat(cached).detach().requires_grad_(True)
    del cached
    metadata = merge_group_metadata(batches, device)
    with autocast_context(device, amp):
        # encode_windows returns a nonleaf concatenation. Match that boundary so
        # multiple head/aux branches combine gradients before AccumulateGrad;
        # direct leaf accumulation can change FP32 addition order and FP16 VJPs.
        features = model.forward_from_windows(leaf.clone(), metadata, return_features=True)
        loss = loss_fn(features, metadata)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Whole-group loss must be a finite scalar")
    (scaler.scale(loss) if scaler is not None else loss).backward()
    if leaf.grad is None:
        raise ValueError("Whole-group loss does not differentiate the cached window CLS")
    value, offset = loss.detach(), 0
    # Release the pooling/head graph before replaying encoder microbatches.
    del features, loss, metadata
    for cpu_batch, state, size in zip(batches, states, sizes, strict=True):
        batch = move_batch(cpu_batch, device)
        with replay_rng(state, device):
            with autocast_context(device, amp):
                encoded = model.encode_windows(batch)
            tolerance = 1e-3 if amp and device.type == "cuda" else 1e-5
            if not torch.allclose(encoded.detach(), leaf[offset:offset + size].detach(), rtol=tolerance, atol=tolerance):
                raise ValueError("Cached encoder replay differs; check stochastic/stateful encoder behavior")
            # Autocast applies to forward only, as in the ordinary training path.
            torch.autograd.backward(encoded, leaf.grad[offset:offset + size])
        offset += size
        del batch, encoded
    return value


def bernoulli_symmetric_kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean over functions of (KL(p||q) + KL(q||p)) / 2 for p = sigmoid(a), q = sigmoid(b)."""
    a, b = a.float(), b.float()
    p, q = torch.sigmoid(a), torch.sigmoid(b)
    # KL(p||q) = p (log p - log q) + (1 - p)(log(1-p) - log(1-q)); log(1 - sigmoid(x)) = logsigmoid(-x)
    kl_pq = p * (F.logsigmoid(a) - F.logsigmoid(b)) + (1 - p) * (F.logsigmoid(-a) - F.logsigmoid(-b))
    kl_qp = q * (F.logsigmoid(b) - F.logsigmoid(a)) + (1 - q) * (F.logsigmoid(-b) - F.logsigmoid(-a))
    return (0.5 * (kl_pq + kl_qp)).mean()


def svuld_contrast_loss(z1, z2, target, keys) -> torch.Tensor:
    """SVulD cosine contrast: a vulnerable function is pushed away from its fixed version
    (max(0, cos)), a non-vulnerable one pulled towards its second dropout pass (1 - cos).
    Vulnerable functions whose fixed version is not in the microbatch are skipped."""
    z1, z2 = z1.float(), z2.float()
    terms = []
    where = {k: i for i, k in enumerate(keys) if k != 0} if keys is not None else {}
    for i, y in enumerate(target.tolist()):
        if y < 0.5:
            terms.append(1 - F.cosine_similarity(z1[i], z2[i], dim=0))
        elif keys is not None and keys[i] > 0 and -keys[i] in where:
            terms.append(torch.clamp(F.cosine_similarity(z1[i], z1[where[-keys[i]]], dim=0), min=0))
    return torch.stack(terms).mean() if terms else z1.sum() * 0


def svuld_loss(model, objective, batch, cfg, keys=None) -> torch.Tensor:
    """SVulD training loss on one microbatch (two stochastic forward passes)."""
    first = model(batch, return_features=True)
    second = model(batch, return_features=True)
    kl = bernoulli_symmetric_kl(first["logits"], second["logits"])
    if cfg.svuld == "rdrop":
        bce = 0.5 * (objective(first["logits"], batch["target"], batch["stratum"])
                     + objective(second["logits"], batch["target"], batch["stratum"]))
        return bce + cfg.svuld_kl * kl
    bce = objective(first["logits"], batch["target"], batch["stratum"])
    contrast = svuld_contrast_loss(first["content"], second["content"], batch["target"], keys)
    return bce + cfg.svuld_kl * kl + cfg.svuld_ct * contrast


def backward_optimization_group(model, objective, batches, cfg, device, scaler, pair_key_map=None):
    """``pair_key_map`` (idx -> +k/-k) adds lambda_pair times the mean pair loss over the group's complete pairs."""
    if cfg.uses_gradient_cache:
        def group_loss(features, metadata):
            loss = objective(features["logits"], metadata["target"], metadata["stratum"])
            if pair_key_map is not None and cfg.lambda_pair > 0:
                keys = [int(pair_key_map.get(str(i), 0)) for i in metadata["idx"]]
                loss = loss + cfg.lambda_pair * function_pair_loss(features["logits"], keys)
            return loss
        return gradient_cache_backward(model, batches, group_loss, device, amp=cfg.amp, scaler=scaler)
    # Additive function losses: normalize by actual function count, including tails.
    count = sum(len(batch["target"]) for batch in batches)
    # The pair term is a mean over the group's complete pairs, as in the gradient-cache path
    # (PairBatchSampler keeps both versions of a pair in one microbatch).
    use_pairs = pair_key_map is not None and cfg.lambda_pair > 0 and cfg.svuld == "none"
    pair_keys_by_batch = ([[int(pair_key_map.get(str(i), 0)) for i in batch["idx"]] for batch in batches]
                          if use_pairs else [None] * len(batches))
    group_pairs = sum(len(complete_pairs(keys)) for keys in pair_keys_by_batch) if use_pairs else 0
    total = torch.zeros((), device=device)
    for cpu_batch, keys in zip(batches, pair_keys_by_batch):
        batch = move_batch(cpu_batch, device)
        weight = len(batch["target"]) / count
        with autocast_context(device, cfg.amp):
            if cfg.svuld != "none":
                svuld_keys = ([int(pair_key_map.get(str(i), 0)) for i in batch["idx"]]
                              if pair_key_map is not None else None)
                loss = svuld_loss(model, objective, batch, cfg, svuld_keys) * weight
            else:
                logits = model(batch)
                loss = objective(logits, batch["target"], batch["stratum"]) * weight
                pairs = len(complete_pairs(keys)) if use_pairs else 0
                if pairs:
                    loss = loss + cfg.lambda_pair * function_pair_loss(logits, keys) * pairs / group_pairs
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        scaler.scale(loss).backward()
        total += loss.detach()
        del batch, loss
    return total


def smoke_frame(frame, cfg, tokenizer):
    if cfg.smoke_limit is None or len(frame) <= cfg.smoke_limit:
        return frame
    frame = frame.copy()
    if "n_tokens" not in frame:
        frame["n_tokens"] = [len(tokenizer(x, add_special_tokens=False)["input_ids"]) for x in frame.func]
    cells = pd.DataFrame({"k": stratum(frame.n_tokens), "y": frame.target})
    rng = np.random.default_rng(cfg.seed)
    mandatory = [int(rng.choice(cell.index)) for _, cell in cells.groupby(["k", "y"])]
    remaining = np.setdiff1d(np.arange(len(frame)), mandatory)
    extra = rng.choice(remaining, cfg.smoke_limit - len(mandatory), replace=False)
    return frame.iloc[np.sort(np.r_[mandatory, extra])].reset_index(drop=True)


def prepare(cfg: Config, splits=("train", "valid", "test")) -> dict:
    cfg.validate()
    tokenizer = get_tokenizer(cfg)
    report = {}
    frames = {}
    for split in splits:
        frames[split] = load_frame(split_path(cfg.data_dir, split))
        dataset = encode_dataset(frames[split], tokenizer, cfg, split)
        report[split] = dataset.metadata
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            require_disjoint(frames[a], frames[b], (a, b))
    return report


@torch.no_grad()
def predict(model, dataset, indices, tokenizer, cfg, device):
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=cfg.eval_batch_size,
                        shuffle=False, collate_fn=BagCollator(tokenizer.pad_token_id))
    previous = model.training
    model.eval()
    scores, ids = [], []
    try:
        for batch in loader:
            ids.extend(batch["idx"])
            batch = move_batch(batch, device)
            with autocast_context(device, cfg.amp):
                logits = model(batch)
            scores.extend(logits.float().cpu().tolist())
    finally:
        model.train(previous)
    expected = dataset.frame.iloc[list(indices)].idx.tolist()
    if ids != expected or len(scores) != len(expected) or not np.isfinite(scores).all():
        raise ValueError("Prediction failed to cover/order the requested rows")
    return np.asarray(scores, dtype=float)


@torch.no_grad()
def predict_cfa(model, dataset, indices, tokenizer, cfg, device):
    indices = list(indices)
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.eval_batch_size,
                        shuffle=False, collate_fn=CFACollator(tokenizer.pad_token_id))
    previous = model.training
    model.eval()
    values, ids = {"std": [], "raw": []}, []
    try:
        for batch in loader:
            ids.extend(batch["idx"])
            with autocast_context(device, cfg.amp):
                scores = model.score_batch(move_batch(batch, device))
            for key in values:
                values[key].extend(scores[key].float().cpu().tolist())
    finally:
        model.train(previous)
    expected = dataset.frame.iloc[indices].idx.tolist()
    if ids != expected or any(len(v) != len(expected) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("CFA prediction failed to cover/order the requested rows")
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def save_checkpoint(model, cfg, tokenizer, folder, extra):
    folder.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(folder / "encoder")
    tokenizer.save_pretrained(folder / "tokenizer")
    temporary = folder / "model.pt.tmp"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
    temporary.replace(folder / "model.pt")
    write_json(folder / "metadata.json", {"config": cfg.to_dict(), **extra})


def load_checkpoint(folder, device):
    metadata = json.loads((folder / "metadata.json").read_text())
    cfg = Config(**metadata["config"])
    cfg.validate()
    if cfg.variant == "cfa":
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(folder / "encoder", local_files_only=True),
                                        add_pooling_layer=False)
        model = CFAModel(encoder, cfg, reference=np.zeros(cfg.reference_k, dtype=np.float32))
    else:
        model = BaselineModel.from_encoder_config(folder / "encoder", cfg)
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device), cfg, metadata


def train_one(cfg, train_data, valid_data, tokenizer, root, train_indices, valid_indices, private_stratum=None):
    seed_all(cfg.seed)
    device = choose_device(cfg.device)
    tr = train_data.frame.iloc[list(train_indices)]
    va = valid_data.frame.iloc[list(valid_indices)]
    if tr.target.nunique() != 2 or va.target.nunique() != 2:
        raise ValueError(f"Training and validation both need two classes (stratum={private_stratum})")
    metric = "overall_auc" if private_stratum is not None else cfg.selection_metric
    # Train statistics only. No validation/test length statistics enter the head or loss.
    weights = prevalence_weights(tr.target, tr.stratum, cfg.rho) if cfg.variant in ("stratum_weighted", "cfa") else None
    if cfg.variant == "cfa":
        reference = reference_log_lengths(tr.log_length, cfg.reference_k, cfg.seed,
                                          cfg.reference_mode, tr.stratum)
        prior_init = cfa_train_prior(tr, weights) if cfg.cfa_prior_init else None
        model = CFAModel.pretrained(cfg, float(tr.log_length.mean()),
                                    float(tr.log_length.std(ddof=0)), reference=reference,
                                    prior_init=prior_init).to(device)
    else:
        model = BaselineModel.pretrained(cfg, float(tr.log_length.mean()), float(tr.log_length.std(ddof=0))).to(device)
    capacity = getattr(model.encoder.config, "max_position_embeddings", None)
    if capacity is not None and getattr(model.encoder.config, "model_type", "") == "roberta":
        required = cfg.max_input_tokens + model.encoder.config.pad_token_id + 1
        if required > capacity:
            raise ValueError(f"max_input_tokens={cfg.max_input_tokens} exceeds RoBERTa position capacity")
    objective = (CFAObjective(weights, lam_inv=cfg.lambda_inv, lam_aux=cfg.lambda_aux, lam_rank=cfg.lambda_rank, lam_pair=cfg.lambda_pair, lam_snip=cfg.lambda_snip, rank_max_stratum=cfg.rank_max_stratum or None,
                              irm_lambda=cfg.lambda_irm, irm_anneal_steps=cfg.irm_anneal_steps).to(device)
                 if cfg.variant == "cfa" else Objective(cfg.variant, weights, cfg.dro_eta).to(device))
    if cfg.variant == "cfa":
        objective.snip_start = cfg.snip_start_updates
    generator = torch.Generator().manual_seed(cfg.seed)
    collate = (CFACollator if cfg.variant == "cfa" else BagCollator)(tokenizer.pad_token_id)
    if cfg.variant == "cfa" and (cfg.lambda_pair > 0 or cfg.lambda_snip > 0):
        keys = getattr(train_data, "pair_keys", None)
        if not keys:
            raise ValueError("lambda_pair/lambda_snip > 0 needs train pair keys")
        subset_keys = [int(keys.get(str(i), 0)) for i in train_data.frame.idx.iloc[list(train_indices)]]
        loader = DataLoader(Subset(train_data, list(train_indices)), batch_size=cfg.batch_size,
                            sampler=PairAdjacentSampler(subset_keys, generator), collate_fn=collate)
    elif cfg.uses_pair_keys:
        keys = getattr(train_data, "pair_keys", None)
        if not keys:
            raise ValueError("lambda_pair > 0 or svuld=simct needs train pair keys")
        subset_keys = [int(keys.get(str(i), 0)) for i in train_data.frame.idx.iloc[list(train_indices)]]
        if cfg.uses_gradient_cache:
            # Whole-group loss: pairs only need to share an optimizer group, as in CFA.
            loader = DataLoader(Subset(train_data, list(train_indices)), batch_size=cfg.batch_size,
                                sampler=PairAdjacentSampler(subset_keys, generator), collate_fn=collate)
        else:
            loader = DataLoader(Subset(train_data, list(train_indices)), collate_fn=collate,
                                batch_sampler=PairBatchSampler(subset_keys, cfg.batch_size, generator))
    else:
        loader = DataLoader(Subset(train_data, list(train_indices)), batch_size=cfg.batch_size,
                            shuffle=True, generator=generator, collate_fn=collate)
    replay_pool, replay_generator = [], None
    if cfg.variant == "cfa" and cfg.replay_short > 0:
        frame = train_data.frame
        replay_pool = [i for i in train_indices
                       if frame.target.iloc[i] == 1 and frame.stratum.iloc[i] <= cfg.rank_max_stratum]
        if not replay_pool:
            raise ValueError("replay_short > 0 but the training split has no short positives")
        replay_generator = torch.Generator().manual_seed(cfg.seed + 1)

    def replay_batches():
        """Replayed short positives: ranking term only (no BCE, pair, or window labels)."""
        picks = torch.randint(len(replay_pool), (cfg.replay_short,), generator=replay_generator).tolist()
        out = []
        for p in picks:
            batch = collate([train_data[replay_pool[p]]])
            batch["replay"] = torch.ones(len(batch["target"]), dtype=torch.bool)
            if "pair_key" in batch:
                batch["pair_key"] = torch.zeros_like(batch["pair_key"])
            if "window_labels" in batch:
                batch["window_labels"] = torch.full_like(batch["window_labels"], -1)
            out.append(batch)
        return out

    optimizer = make_optimizer(model, cfg)
    plan = training_plan(cfg, len(tr))
    scheduler = get_linear_schedule_with_warmup(optimizer, plan["warmup_updates"],
                                                plan["planned_optimizer_updates"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    boundaries = set(plan["validation_group_boundaries"])
    best, stale, step, history = -math.inf, 0, 0, []
    best_cfa_scores = None
    groups_seen = microbatches_seen = functions_seen = 0
    adam_reset_at = None
    train_seconds = validation_seconds = 0.0
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log.info("%s stratum=%s training_plan=%s", cfg.variant, private_stratum, plan)
    for epoch in range(plan["epochs"]):
        model.train()
        epoch_loss_sum, epoch_count = 0.0, 0
        for group_no, batches in enumerate(optimization_groups(loader, cfg.gradient_accumulation_steps), 1):
            group_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            previous_q = objective.log_q.clone() if cfg.variant == "group_dro" else None
            if cfg.variant == "cfa" and cfg.lambda_irm > 1 and step == cfg.irm_anneal_steps and adam_reset_at is None:
                optimizer.state.clear()
                adam_reset_at = groups_seen + 1
            cfa_terms = {}
            if replay_pool:
                batches = batches + replay_batches()
            loss = (gradient_cache_backward(model, batches, cfa_group_loss(model, objective, step, cfa_terms),
                                            device, amp=cfg.amp, scaler=scaler)
                    if cfg.variant == "cfa" else backward_optimization_group(
                        model, objective, batches, cfg, device, scaler,
                        getattr(train_data, "pair_keys", None) if cfg.uses_pair_keys else None))
            updated = optimizer_update(model, optimizer, scheduler, scaler, cfg.grad_clip)
            if updated:
                step += 1
            elif previous_q is not None:
                objective.log_q.copy_(previous_q)
            count = sum(len(batch["target"]) for batch in batches if "replay" not in batch)
            groups_seen += 1
            microbatches_seen += sum(1 for batch in batches if "replay" not in batch)
            functions_seen += count
            epoch_count += count
            epoch_loss_sum += float(loss) * count
            train_seconds += time.perf_counter() - group_start
            if groups_seen <= 2 or groups_seen % cfg.log_every_updates == 0:
                log.info("%s epoch=%d group=%d/%d updates=%d functions=%d loss=%.5f lr=%.3g train_seconds=%.1f",
                         cfg.variant, epoch + 1, group_no, plan["optimization_groups_per_epoch"], step,
                         functions_seen, epoch_loss_sum / epoch_count, optimizer.param_groups[0]["lr"], train_seconds)
            if group_no in boundaries:
                validation_start = time.perf_counter()
                scores = (predict_cfa(model, valid_data, valid_indices, tokenizer, cfg, device)
                          if cfg.variant == "cfa" else predict(model, valid_data, valid_indices, tokenizer, cfg, device))
                metrics, _ = summary(va, scores["std"] if cfg.variant == "cfa" else scores, cfg.matched_delta)
                validation_seconds += time.perf_counter() - validation_start
                value = metrics[metric]
                if not math.isfinite(value):
                    raise ValueError(f"Undefined validation {metric}; inspect class/matching coverage per stratum")
                record = {"step": step, "optimization_groups": groups_seen,
                          "microbatches": microbatches_seen, "functions_seen": functions_seen,
                          "epoch": epoch + group_no / plan["optimization_groups_per_epoch"],
                          "train_loss": epoch_loss_sum / epoch_count, "validation": metrics,
                          "selection": metric, "train_seconds": train_seconds,
                          "validation_seconds": validation_seconds, "seconds": time.perf_counter() - start}
                if cfg.variant == "cfa":
                    record["cfa_terms"] = cfa_terms
                    record["raw_validation"], _ = summary(va, scores["raw"], cfg.matched_delta)
                history.append(record)
                log.info("%s stratum=%s step=%d valid_%s=%.5f", cfg.variant, private_stratum, step, metric, value)
                if value > best:
                    best, stale = value, 0
                    if cfg.variant == "cfa":
                        best_cfa_scores = {name: values.copy() for name, values in scores.items()}
                    save_checkpoint(model, cfg, tokenizer, root / "checkpoint",
                                    {"selection_metric": metric, "best_value": best, "best_step": step,
                                     "private_stratum": private_stratum, "training_plan": plan,
                                     "best_optimization_group": groups_seen,
                                     "objective_state": {"log_q": objective.log_q.tolist()} if cfg.variant != "cfa" else None,
                                     "reference_log_lengths": model.reference.detach().cpu().tolist() if cfg.variant == "cfa" else None,
                                     "score_definition": "s_std_probability" if cfg.variant == "cfa" else "logit"})
                else:
                    stale += 1
                write_json(root / "history.json", history)
                if cfg.patience and stale >= cfg.patience:
                    break
        if cfg.patience and stale >= cfg.patience:
            break
    dro_q = torch.softmax(objective.log_q, dim=0).cpu().tolist() if cfg.variant == "group_dro" else None
    del optimizer, scheduler, scaler, objective, loader, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, _, _ = load_checkpoint(root / "checkpoint", device)
    scores = (predict_cfa(model, valid_data, valid_indices, tokenizer, cfg, device)
              if cfg.variant == "cfa" else predict(model, valid_data, valid_indices, tokenizer, cfg, device))
    if cfg.variant == "cfa" and any(not np.array_equal(scores[name], best_cfa_scores[name])
                                    for name in ("std", "raw")):
        raise ValueError("Reloaded CFA checkpoint changed validation s_std or s_raw")
    write_json(root / "training.json", {"n_train": len(tr), "n_valid": len(va), "steps": step,
                                        "optimization_groups": groups_seen, "microbatches": microbatches_seen,
                                        "functions_seen": functions_seen,
                                        "skipped_optimizer_updates": groups_seen - step,
                                        "training_plan": plan, "train_seconds": train_seconds,
                                        "validation_seconds": validation_seconds,
                                        "best_value": best, "selection_metric": metric,
                                        "stratum_weights": weights,
                                        "adam_reset_at_group": adam_reset_at,
                                        "cfa_objective": {"lambda_inv": cfg.lambda_inv, "lambda_aux": cfg.lambda_aux,
                                                          "lambda_irm": cfg.lambda_irm,
                                                          "irm_anneal_steps": cfg.irm_anneal_steps}
                                        if cfg.variant == "cfa" else None,
                                        "dro_q": dro_q,
                                        "seconds": time.perf_counter() - start})
    return scores


def score_frame(frame, scores):
    result = frame[["idx", "project", "target", "n_tokens", "stratum", "log_length"]].copy()
    if isinstance(scores, dict):
        return result.assign(score=scores["std"], score_raw=scores["raw"])
    return result.assign(score=scores)


def train(cfg: Config) -> Path:
    cfg.validate()
    root = run_path(cfg)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Run directory already exists: {root}; use a new output_dir/seed")
    tokenizer = get_tokenizer(cfg)
    train_frame = load_frame(split_path(cfg.data_dir, "train"))
    valid_frame = load_frame(split_path(cfg.data_dir, "valid"))
    require_disjoint(train_frame, valid_frame)
    train_digest, valid_digest = frame_digest(train_frame), frame_digest(valid_frame)
    tr = smoke_frame(train_frame, cfg, tokenizer)
    va = smoke_frame(valid_frame, cfg, tokenizer)
    train_data = encode_dataset(tr, tokenizer, cfg, "train")
    valid_data = encode_dataset(va, tokenizer, cfg, "valid")
    cfa_pair_digest = None
    if cfg.variant == "cfa":
        pairs, cfa_pair_digest = read_cfa_pairs(cfg, tokenizer)
        validate_cfa_train_pairs(pairs, train_data.frame if cfg.smoke_limit is None else train_frame)
        labels = window_label_map(pairs, split="train")
        counts = window_counts(pairs, split="train")
        if cfg.smoke_limit is not None:
            present = set(train_data.frame.idx.astype(str))
            labels = {idx: value for idx, value in labels.items() if idx in present}
            counts = {idx: value for idx, value in counts.items() if idx in present}
        train_cache_metadata = train_data.metadata
        keys = None
        if cfg.lambda_pair > 0 or cfg.lambda_snip > 0:
            present = set(train_data.frame.idx.astype(str))
            keys = {idx: k for idx, k in pair_keys(pairs, split="train").items() if idx in present}
        train_data = WindowLabeledDataset(train_data, labels, counts, pair_keys=keys)
    else:
        train_cache_metadata = train_data.metadata
        if cfg.uses_pair_keys:
            # Only the vulnerable/fixed idx columns are used here, not the tokenizer-specific window labels.
            pairs, cfa_pair_digest = read_cfa_pairs(cfg, None)
            present = set(train_data.frame.idx.astype(str))
            train_data.pair_keys = {idx: k for idx, k in pair_keys(pairs, split="train").items() if idx in present}
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", cfg.to_dict())
    write_json(root / "protocol.json", {"train_digest": train_digest, "valid_digest": valid_digest,
                                        "train_cache": train_cache_metadata, "valid_cache": valid_data.metadata,
                                        "is_smoke": cfg.smoke_limit is not None,
                                        "score_scale": "probability_s_std" if cfg.variant == "cfa" else "logit",
                                        "score_raw_scale": "probability_s_raw" if cfg.variant == "cfa" else None,
                                        "cfa_pairs_sha256": cfa_pair_digest,
                                        "cfa_auxiliary_windows": {
                                            "positive": sum(v == 1 for rows in labels.values() for v in rows.values()),
                                            "hard_negative_unique": sum(v == 0 for rows in labels.values() for v in rows.values())}
                                        if cfg.variant == "cfa" else None,
                                        "cfa_window_labels": "train_pairs_positive_and_same_site_hard_negative_only"
                                        if cfg.variant == "cfa" else None,
                                        "matched_auc_weighting": "positive-negative pairs",
                                        "pauc_normalization": "raw_area_divided_by_max_fpr",
                                        "loss_unit": "function", "selection": cfg.selection_metric,
                                        "optimization": training_plan(cfg, len(train_data))
                                        if cfg.variant != "stratum_private" else "per-stratum checkpoint metadata",
                                        "validation_schedule": "epoch-local completed optimization groups",
                                        "best_checkpoint_ties": "earliest",
                                        "scheduler_steps": "successful optimizer updates only"})
    np.savez_compressed(root / "split_ids.npz", train=train_frame.idx.to_numpy(dtype=str),
                        valid=valid_frame.idx.to_numpy(dtype=str))
    if cfg.variant == "stratum_private":
        scores = np.empty(len(valid_data))
        for k in range(1, 6):
            ti = np.flatnonzero(train_data.frame.stratum.to_numpy() == k)
            vi = np.flatnonzero(valid_data.frame.stratum.to_numpy() == k)
            scores[vi] = train_one(cfg, train_data, valid_data, tokenizer, root / f"stratum{k}", ti, vi, k)
        tokenizer.save_pretrained(root / "tokenizer")
    else:
        scores = train_one(cfg, train_data, valid_data, tokenizer, root,
                           range(len(train_data)), range(len(valid_data)))
    scored = score_frame(valid_data.frame, scores)
    scored.to_parquet(root / "scores_valid.parquet", index=False)
    selected_scores = scores["std"] if cfg.variant == "cfa" else scores
    thresholds = fit_thresholds(valid_data.frame, selected_scores)
    write_json(root / "thresholds.json", thresholds)
    metrics, table = summary(valid_data.frame, selected_scores, cfg.matched_delta)
    write_json(root / "metrics_valid.json", metrics)
    table.to_csv(root / "strata_valid.csv", index=False)
    deployment(valid_data.frame, selected_scores, thresholds).to_csv(root / "deployment_valid.csv", index=False)
    if cfg.variant == "cfa":
        raw_metrics, _ = summary(valid_data.frame, scores["raw"], cfg.matched_delta)
        write_json(root / "metrics_valid_raw.json", raw_metrics)
    write_json(root / "done.json", {"status": "trained", "is_smoke": cfg.smoke_limit is not None})
    return root


def score_run(root: str | Path, split="test", device_name="auto") -> Path:
    root = Path(root)
    cfg = Config.read(root / "config.json")
    if not (root / "done.json").exists():
        raise ValueError("Training is incomplete; no done.json")
    if split not in ("valid", "test"):
        raise ValueError("score accepts valid or test")
    if cfg.smoke_limit is not None:
        raise ValueError("Smoke runs cannot produce paper evaluation scores")
    frame = load_frame(split_path(cfg.data_dir, split))
    protocol = json.loads((root / "protocol.json").read_text())
    if split == "valid" and frame_digest(frame) != protocol["valid_digest"]:
        raise ValueError("Validation data changed since training; frozen thresholds are no longer valid")
    with np.load(root / "split_ids.npz", allow_pickle=False) as ids:
        if split == "test" and (set(frame.idx) & (set(ids["train"]) | set(ids["valid"]))):
            raise ValueError("Test ids overlap the train/valid splits")
    device = choose_device(device_name)
    tokenizer_folder = root / ("tokenizer" if cfg.variant == "stratum_private" else "checkpoint/tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder, use_fast=True, local_files_only=True)
    # The saved tokenizer and architecture are used, not mutable paths to the initial model.
    data = encode_dataset(frame, tokenizer, cfg, split)
    scores = np.empty(len(data))
    if cfg.variant == "stratum_private":
        for k in range(1, 6):
            indices = np.flatnonzero(data.frame.stratum.to_numpy() == k)
            if len(indices):
                model, _, _ = load_checkpoint(root / f"stratum{k}/checkpoint", device)
                scores[indices] = predict(model, data, indices, tokenizer, cfg, device)
                del model
    else:
        model, _, _ = load_checkpoint(root / "checkpoint", device)
        scores = (predict_cfa(model, data, range(len(data)), tokenizer, cfg, device)
                  if cfg.variant == "cfa" else predict(model, data, range(len(data)), tokenizer, cfg, device))
    path = root / f"scores_{split}.parquet"
    score_frame(data.frame, scores).to_parquet(path, index=False)
    write_json(root / f"coverage_{split}.json", {"expected": len(frame),
                                               "scored": len(scores["std"]) if cfg.variant == "cfa" else len(scores),
                                               "coverage": 1.0,
                                               "data_digest": frame_digest(frame), "cache": data.metadata})
    return path


def validate_scores(frame: pd.DataFrame):
    required = {"idx", "target", "n_tokens", "score"}
    if not required.issubset(frame):
        raise ValueError(f"Scores missing {required - set(frame.columns)}")
    frame = frame.copy()
    frame["idx"] = frame.idx.astype(str)
    if len(frame) == 0 or frame.idx.duplicated().any() or frame[list(required)].isna().any().any():
        raise ValueError("Scores contain nulls, duplicate ids or no rows")
    if not frame.target.isin([0, 1]).all() or not np.isfinite(frame.score).all():
        raise ValueError("Scores need binary labels and finite scores")
    n = frame.n_tokens.to_numpy(dtype=float)
    if not np.isfinite(n).all() or np.any(n < 1) or np.any(n % 1):
        raise ValueError("Invalid full-function lengths")
    frame["stratum"] = stratum(n)
    frame["log_length"] = np.log(n)
    return frame


def evaluate_scores(valid, test, output, delta=0.05, pairs=None, frozen_thresholds=None):
    valid, test = validate_scores(valid), validate_scores(test)
    require_disjoint(valid, test, ("valid", "test"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    thresholds = frozen_thresholds if frozen_thresholds is not None else fit_thresholds(valid, valid.score)
    metrics, table = summary(test, test.score, delta)
    if pairs is not None:
        metrics.update(function_pair_rank(test, test.score, pairs))
    write_json(output / "metrics_test.json", metrics)
    write_json(output / "thresholds.json", thresholds)
    table.to_csv(output / "strata_test.csv", index=False)
    deployment(test, test.score, thresholds).to_csv(output / "deployment_test.csv", index=False)
    return metrics


def evaluate_run(root, pairs_path=None):
    root = Path(root)
    cfg = Config.read(root / "config.json")
    if cfg.smoke_limit is not None:
        raise ValueError("Smoke runs cannot be evaluated as paper experiments")
    valid = pd.read_parquet(root / "scores_valid.parquet")
    test = pd.read_parquet(root / "scores_test.parquet")
    with np.load(root / "split_ids.npz", allow_pickle=False) as ids:
        if valid.idx.astype(str).tolist() != ids["valid"].tolist():
            raise ValueError("Validation score coverage/order differs from the training manifest")
        if set(test.idx.astype(str)) & set(ids["train"]):
            raise ValueError("Evaluation test contains training ids")
    coverage = json.loads((root / "coverage_test.json").read_text())
    if coverage["scored"] != len(test) or coverage["expected"] != len(test):
        raise ValueError("Test score coverage differs from the scoring manifest")
    pairs = None
    if pairs_path:
        pairs = pd.read_parquet(pairs_path)
        if "split" in pairs:
            pairs = pairs[pairs.split == "test"]
        for key in ("vul_split", "fixed_split"):
            if key in pairs and not (pairs[key] == "test").all():
                raise ValueError("Pair file has versions crossing test split")
    thresholds = json.loads((root / "thresholds.json").read_text())
    expected = fit_thresholds(validate_scores(valid), valid.score)
    if thresholds != expected:
        raise ValueError("Frozen thresholds disagree with the saved validation scores")
    return evaluate_scores(valid, test, root, cfg.matched_delta, pairs, thresholds)
