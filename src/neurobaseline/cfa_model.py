"""CFA model and window-labeled data on top of the sliding-window baseline (task M5).

CFAModel reuses BaselineModel's encoder, chunked/checkpointed ``encode_windows``
and attention pooling, so engine.gradient_cache_backward works unchanged; it
replaces the classification head with the CFA heads from cfa_heads.
``cfa_group_loss`` is the whole-group loss_fn for gradient caching.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .cfa_heads import AdditiveHead, CFAObjective, FullHead, InvariantHead, WindowScorer
from .data import BagCollator
from .model import AttentionPool, BaselineModel


class CFAModel(BaselineModel):
    def __init__(self, encoder, cfg, length_mean: float = 0, length_std: float = 1,
                 reference=None, additive: bool = False, prior_init: dict | None = None):
        super().__init__(encoder, cfg, length_mean, length_std)
        hidden = encoder.config.hidden_size
        std = getattr(encoder.config, "initializer_range", 0.02)
        dropout = getattr(encoder.config, "classifier_dropout", None)
        if dropout is None:
            dropout = getattr(encoder.config, "hidden_dropout_prob", 0.1)
        if self.pool is None:
            self.pool = AttentionPool(hidden, cfg.attention_hidden)
            for layer in self.pool.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
        self.length_embedding = None
        head = AdditiveHead if additive else FullHead
        self.head = head(hidden, cfg.length_hidden, dropout, length_mean, length_std, init_std=std)
        self.invariant = InvariantHead(hidden, init_std=std)
        self.window_scorer = WindowScorer(hidden, init_std=std)
        if prior_init is not None:
            # Optional train-only base rates for the output biases; None keeps the original init.
            if set(prior_init) != {"env_prior", "prior"}:
                raise ValueError("prior_init needs exactly 'env_prior' and 'prior'")
            self.head.init_prior(prior_init["prior"])
            self.invariant.init_env_prior(prior_init["env_prior"])
        ref = torch.zeros(0) if reference is None else torch.as_tensor(np.asarray(reference), dtype=torch.float32)
        self.register_buffer("reference", ref)

    def _scorer_pool(self, windows, window_logits, owners, n_bags):
        """Localization-guided pooling: attention is the softmax of the window scores h(u)."""
        pooled, weights = [], []
        for bag in range(n_bags):
            mask = owners == bag
            if not mask.any():
                raise ValueError("Empty window bag")
            alpha = torch.softmax(window_logits[mask].float(), dim=0).to(windows.dtype)
            pooled.append(torch.sum(alpha[:, None] * windows[mask], dim=0))
            weights.append(alpha)
        return torch.stack(pooled), torch.cat(weights)

    def forward_from_windows(self, windows, batch, return_features: bool = False):
        window_logits = self.window_scorer(windows)
        if self.cfg.attention_from_scorer:
            z, alpha = self._scorer_pool(windows, window_logits, batch["owners"], len(batch["target"]))
        else:
            z, alpha = self.pool(windows, batch["owners"], len(batch["target"]))
        logits = self.head(z, batch["log_length"])
        if not return_features:
            return logits
        return {"logits": logits, "content": z, "attention": alpha, "window_features": windows,
                "window_logits": window_logits}

    @torch.no_grad()
    def score_batch(self, batch) -> dict:
        """raw and standardized probabilities, window scores and attention for one batch."""
        if not len(self.reference):
            raise ValueError("Reference lengths are not set; save them with the checkpoint")
        f = self.forward_from_windows(self.encode_windows(batch), batch, return_features=True)
        scores = self.head.scores(f["content"], batch["log_length"], self.reference)
        return {"raw": scores["raw"], "std": scores["std"], "logits": f["logits"],
                "content": f["content"], "attention": f["attention"], "window_logits": f["window_logits"]}


def paired_content_loss(model: CFAModel, z: torch.Tensor, log_length: torch.Tensor,
                        pair_key: torch.Tensor, shared_length: bool = True) -> torch.Tensor:
    """Vulnerable vs fixed version of the same function, scored at one shared length.

    pair_key is +k for the vulnerable and -k for the fixed member of pair k (0 otherwise).
    Both members are scored by the full head at their mean log-length, so the ordering
    can only come from content. Returns the mean logistic loss over complete pairs in
    the group, or zero when the group has none.
    """
    keys = pair_key.tolist()
    where = {k: i for i, k in enumerate(keys) if k != 0}
    complete = [(where[k], where[-k]) for k in where if k > 0 and -k in where]
    if not complete:
        return z.sum() * 0
    v = torch.tensor([a for a, _ in complete], device=z.device)
    f = torch.tensor([b for _, b in complete], device=z.device)
    if shared_length:
        shared = (log_length[v] + log_length[f]) / 2
        lengths = torch.cat([shared, shared])
    else:  # ablation: each version keeps its own length
        lengths = torch.cat([log_length[v], log_length[f]])
    logits = model.head(torch.cat([z[v], z[f]]), lengths)
    return F.softplus(logits[len(complete):].float() - logits[:len(complete)].float()).mean()


SNIPPET_LOG_LENGTH = float(np.log(128.0))


def snippet_pair_loss(model: CFAModel, windows: torch.Tensor, owners: torch.Tensor, window_labels: torch.Tensor,
                      pair_key: torch.Tensor) -> torch.Tensor:
    """Short-scale paired loss on single fix windows.

    For every complete fix pair in the group, the first fix-overlapping window of the
    vulnerable version and the first same-site window of the fixed version are scored as
    one-window bags (a one-window attention bag is the window vector itself) at one shared
    short length (128 tokens), and the vulnerable snippet is asked to score higher.
    Pairs without both labeled windows are skipped; zero when no pair qualifies.
    """
    keys = pair_key.tolist()
    where = {k: i for i, k in enumerate(keys) if k != 0}
    vuln, fixed = [], []
    for k, i in where.items():
        if k <= 0 or -k not in where:
            continue
        pos = torch.nonzero((owners == i) & (window_labels == 1)).flatten()
        neg = torch.nonzero((owners == where[-k]) & (window_labels == 0)).flatten()
        if len(pos) and len(neg):
            vuln.append(int(pos[0]))
            fixed.append(int(neg[0]))
    if not vuln:
        return windows.sum() * 0
    z = torch.cat([windows[vuln], windows[fixed]])
    length = torch.full((len(z),), SNIPPET_LOG_LENGTH, device=windows.device)
    logits = model.head(z, length)
    n = len(vuln)
    return F.softplus(logits[n:].float() - logits[:n].float()).mean()


def cfa_group_loss(model: CFAModel, objective: CFAObjective, step: int, record: dict | None = None):
    """loss_fn(features, metadata) for engine.gradient_cache_backward."""
    def loss_fn(features, metadata):
        terms = objective(full_logits=features["logits"], target=metadata["target"],
                          stratum=metadata["stratum"], step=step, z=features["content"],
                          invariant=model.invariant if objective.irm_lambda > 0 or objective.lam_inv > 0 else None,
                          window_logits=features["window_logits"],
                          window_labels=metadata.get("window_labels"), replay=metadata.get("replay"))
        if objective.lam_pair > 0:
            if "pair_key" not in metadata:
                raise ValueError("lambda_pair > 0 needs pair_key metadata")
            terms["pair"] = paired_content_loss(model, features["content"], metadata["log_length"],
                                                metadata["pair_key"], model.cfg.pair_shared_length)
            terms["total"] = terms["total"] + objective.lam_pair * terms["pair"]
        if getattr(objective, "lam_snip", 0.0) > 0 and step >= getattr(objective, "snip_start", 0):
            if "pair_key" not in metadata or "window_labels" not in metadata:
                raise ValueError("lambda_snip > 0 needs pair_key and window_labels metadata")
            terms["snip"] = snippet_pair_loss(model, features["window_features"], metadata["owners"],
                                              metadata["window_labels"], metadata["pair_key"])
            terms["total"] = terms["total"] + objective.lam_snip * terms["snip"]
        if record is not None:
            record.update({k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in terms.items()})
        return terms["total"]
    return loss_fn


# ---------------------------------------------------------------- window labels

def window_label_map(pairs: pd.DataFrame, split: str = "train") -> dict[str, dict[int, int]]:
    """idx -> {window ordinal: label}; fix-overlapping windows 1, same-site fixed windows 0.

    Hard negatives are deduplicated across pairs that share a fixed version.
    """
    labels: dict[str, dict[int, int]] = {}
    for row in pairs[pairs.split == split].itertuples(index=False):
        for k in row.positive_windows:
            labels.setdefault(row.vuln_idx, {})[int(k)] = 1
        for k in row.hard_negative_windows:
            labels.setdefault(row.fixed_idx, {})[int(k)] = 0
    return labels


def pair_keys(pairs: pd.DataFrame, split: str = "train") -> dict[str, int]:
    """idx -> +k (vulnerable) / -k (fixed) for pair k; a fixed version shared by several
    vulnerable versions is kept only for its first pair."""
    keys: dict[str, int] = {}
    for row in pairs[pairs.split == split].itertuples(index=False):
        v, f = str(row.vuln_idx), str(row.fixed_idx)
        if v in keys or f in keys:
            continue
        k = len(keys) // 2 + 1
        keys[v], keys[f] = k, -k
    return keys


class PairAdjacentSampler(Sampler):
    """Random order in which the two versions of each fix pair are adjacent, so that
    consecutive microbatches of one optimizer group usually contain both."""
    def __init__(self, keys: list[int], generator: torch.Generator):
        self.n, self.generator = len(keys), generator
        position = {k: i for i, k in enumerate(keys) if k != 0}
        paired = [(position[k], position[-k]) for k in position if k > 0 and -k in position]
        used = {i for pair in paired for i in pair}
        self.units = [list(p) for p in paired] + [[i] for i in range(self.n) if i not in used]

    def __len__(self):
        return self.n

    def __iter__(self):
        order = torch.randperm(len(self.units), generator=self.generator).tolist()
        flips = torch.randint(0, 2, (len(self.units),), generator=self.generator).tolist()
        for u, flip in zip(order, flips):
            unit = self.units[u]
            yield from (unit[::-1] if flip and len(unit) == 2 else unit)


class PairBatchSampler(Sampler):
    """Batches of ``batch_size`` positions in random order, with both versions of each fix
    pair in the same batch; single functions fill the slots a pair would overflow."""
    def __init__(self, keys: list[int], batch_size: int, generator: torch.Generator):
        self.n, self.batch_size, self.generator = len(keys), batch_size, generator
        position = {k: i for i, k in enumerate(keys) if k != 0}
        self.pairs = [(position[k], position[-k]) for k in position if k > 0 and -k in position]
        used = {i for pair in self.pairs for i in pair}
        self.singles = [i for i in range(self.n) if i not in used]

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        units = [list(p) for p in self.pairs] + [[i] for i in self.singles]
        batch, carry = [], []
        for u in torch.randperm(len(units), generator=self.generator).tolist():
            unit = units[u]
            if len(batch) + len(unit) <= self.batch_size:
                batch += unit
            else:
                carry.append(unit)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                while carry and len(batch) + len(carry[0]) <= self.batch_size:
                    batch += carry.pop(0)
        rest = batch + [i for unit in carry for i in unit]
        for start in range(0, len(rest), self.batch_size):
            yield rest[start:start + self.batch_size]


def complete_pairs(keys: list[int]) -> list[tuple[int, int]]:
    """(vulnerable, fixed) positions of the fix pairs with both versions in ``keys``."""
    where = {k: i for i, k in enumerate(keys) if k != 0}
    return [(i, where[-k]) for k, i in where.items() if k > 0 and -k in where]


def function_pair_loss(logits: torch.Tensor, keys: list[int]) -> torch.Tensor:
    """Mean softplus(f(fixed) - f(vulnerable)) over fix pairs complete within the batch."""
    pairs = complete_pairs(keys)
    if not pairs:
        return logits.sum() * 0
    vuln, fixed = [v for v, _ in pairs], [f for _, f in pairs]
    return F.softplus(logits[fixed].float() - logits[vuln].float()).mean()


def window_counts(pairs: pd.DataFrame, split: str = "train") -> dict[str, int]:
    part = pairs[pairs.split == split]
    return {**dict(zip(part.vuln_idx, part.n_windows)), **dict(zip(part.fixed_idx, part.n_fixed_windows))}


def _stable_rng(seed: int, idx: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{idx}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


class WindowLabeledDataset(Dataset):
    """Adds ``window_labels`` (1, 0, or -1 = unlabeled) to an EncodedDataset's samples.

    ``negative_windows`` > 0 also labels that many random windows of every other
    negative function 0 (the spec's variant); the draw is fixed per (seed, idx).
    """
    def __init__(self, base, labels: dict[str, dict[int, int]], expected_windows: dict[str, int],
                 negative_windows: int = 0, seed: int = 13, pair_keys: dict[str, int] | None = None):
        self.base, self.labels, self.negative_windows, self.seed = base, labels, negative_windows, seed
        self.pair_keys = pair_keys
        self.frame = base.frame
        present = dict(zip(base.frame.idx.astype(str), range(len(base.frame))))
        missing = [i for i in labels if i not in present]
        if missing:
            raise ValueError(f"{len(missing)} labeled functions are not in the dataset, e.g. {missing[:3]}")
        for idx, n in expected_windows.items():
            got = int(base.bag_ptr[present[idx] + 1] - base.bag_ptr[present[idx]])
            if got != n:
                raise ValueError(f"idx={idx}: {got} windows but the pair table has {n}; window settings differ")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample = self.base[i]
        labels = np.full(len(sample["windows"]), -1, dtype=np.int64)
        known = self.labels.get(str(sample["idx"]))
        if known:
            for k, v in known.items():
                labels[k] = v
        elif self.negative_windows and sample["target"] == 0:
            pick = _stable_rng(self.seed, str(sample["idx"])).choice(
                len(labels), size=min(self.negative_windows, len(labels)), replace=False)
            labels[pick] = 0
        out = {**sample, "window_labels": labels}
        if self.pair_keys is not None:
            out["pair_key"] = int(self.pair_keys.get(str(sample["idx"]), 0))
        return out


class CFACollator(BagCollator):
    def __call__(self, samples):
        batch = super().__call__(samples)
        if all("window_labels" in s for s in samples):
            batch["window_labels"] = torch.as_tensor(np.concatenate([s["window_labels"] for s in samples]))
        if all("pair_key" in s for s in samples):
            batch["pair_key"] = torch.tensor([s["pair_key"] for s in samples], dtype=torch.long)
        return batch
