"""CFA heads and loss terms on top of pooled content z (task M2).

Tensor-only: nothing here touches the encoder or the training loop. Inputs are
the content representation ``z`` (B x d), window CLS vectors, standardized or
raw log lengths, the length stratum (1..5) and labels. See docs/cfa-method-spec.md.

- FullHead / AdditiveHead: f(x) = MLP([z; phi(L)]) or g(z) + h(L) + b, with raw
  and counterfactually standardized scores (component C).
- InvariantHead: v^T z + b_e over four length environments {1+2, 3, 4, 5} with
  the IRMv1 penalty on the shared slope (component B-b).
- WindowScorer / aux_loss: window-level localization BCE (component A).
- CFAObjective: weighted function BCE (B-a) + invariant-head terms + aux loss.

IRM and group-weighted losses are nonlinear in the group of functions, so they
must be computed over the whole optimizer group (see the spec's gradient-cache
note), not summed over batch-size-1 micro-batches.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ENV_OF_STRATUM = (0, 0, 1, 2, 3)  # strata 1..5 -> environments {1+2, 3, 4, 5}
N_ENVS = 4


def environments(stratum: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(ENV_OF_STRATUM, device=stratum.device)[stratum - 1]


def prior_logit(p, eps: float = 1e-4) -> torch.Tensor:
    """logit of a (vector of) train positive rate(s), clamped away from 0 and 1."""
    p = torch.as_tensor(p, dtype=torch.float32)
    if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Prior rates must lie in [0, 1]")
    return torch.logit(p.clamp(eps, 1 - eps))


def _init_linear(module: nn.Module, std: float):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class LengthEmbedding(nn.Module):
    """phi(L): a small MLP of the train-standardized log length."""
    def __init__(self, hidden: int, length_mean: float, length_std: float):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(length_mean)))
        self.register_buffer("std", torch.tensor(max(float(length_std), 1e-6)))
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh())

    def forward(self, log_length: torch.Tensor) -> torch.Tensor:
        return self.net(((log_length - self.mean) / self.std).unsqueeze(-1))


class _LengthConditionedHead(nn.Module):
    """Shared raw / standardized scoring for heads f(z, L)."""

    def logits(self, z, log_length):  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, z, log_length):
        return self.logits(z, log_length)

    def standardized(self, z: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """s_std(x) = mean_k sigmoid(f(z(x), l_k)); z is computed once, only the head repeats."""
        b, k = len(z), len(reference)
        zz = z.unsqueeze(1).expand(b, k, z.shape[-1]).reshape(b * k, -1)
        ll = reference.to(z.device, z.dtype if z.is_floating_point() else torch.float32).repeat(b)
        probs = torch.sigmoid(self.logits(zz, ll).float()).view(b, k)
        return probs.mean(dim=1)

    def scores(self, z, log_length, reference) -> dict:
        return {"raw": torch.sigmoid(self.logits(z, log_length).float()),
                "std": self.standardized(z, reference)}


class FullHead(_LengthConditionedHead):
    """f(x) = MLP([z; phi(L)]), the RoBERTa-style dense-tanh-out head used by the baselines."""
    def __init__(self, width: int, length_hidden: int, dropout: float, length_mean: float,
                 length_std: float, init_std: float = 0.02):
        super().__init__()
        self.phi = LengthEmbedding(length_hidden, length_mean, length_std)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(width + length_hidden, width)
        self.out = nn.Linear(width, 1)
        _init_linear(self, init_std)

    @torch.no_grad()
    def init_prior(self, prior: float):
        """Start the output bias at logit(prior) instead of 0 (train-only weighted positive rate)."""
        self.out.bias.fill_(float(prior_logit(prior)))

    def logits(self, z, log_length):
        x = torch.cat((z, self.phi(log_length.to(z.dtype))), dim=-1)
        x = torch.tanh(self.dense(self.dropout(x)))
        return self.out(self.dropout(x)).squeeze(-1)


class AdditiveHead(_LengthConditionedHead):
    """p_add(x) = sigmoid(g(z) + h(L) + b): the additivity check of gate 4."""
    def __init__(self, width: int, length_hidden: int, dropout: float, length_mean: float,
                 length_std: float, init_std: float = 0.02):
        super().__init__()
        self.phi = LengthEmbedding(length_hidden, length_mean, length_std)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(width, width)
        self.g = nn.Linear(width, 1)
        self.h = nn.Linear(length_hidden, 1, bias=False)
        _init_linear(self, init_std)

    @torch.no_grad()
    def init_prior(self, prior: float):
        self.g.bias.fill_(float(prior_logit(prior)))

    def logits(self, z, log_length):
        content = self.g(self.dropout(torch.tanh(self.dense(self.dropout(z))))).squeeze(-1)
        return content + self.h(self.phi(log_length.to(z.dtype))).squeeze(-1)


class InvariantHead(nn.Module):
    """s_{phi,e}(x) = v^T z(x) + b_e: shared slope, one intercept per length environment."""
    def __init__(self, width: int, init_std: float = 0.02):
        super().__init__()
        self.v = nn.Linear(width, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(N_ENVS))
        _init_linear(self, init_std)

    def forward(self, z, env):
        return self.v(z).squeeze(-1) + self.b[env]

    @torch.no_grad()
    def init_env_prior(self, env_prior):
        """Start b_e at logit of each environment's train positive rate, so the intercepts,
        not the shared slope, carry the base-rate differences from the first update."""
        logits = prior_logit(env_prior)
        if logits.shape != self.b.shape:
            raise ValueError(f"Expected {N_ENVS} environment priors")
        self.b.copy_(logits.to(self.b))

    @torch.no_grad()
    def diagnostics(self, z, env, y) -> dict:
        """Intercepts, slope norm and the distribution of a = v^T z, with per-environment counts."""
        a = self.v(z).squeeze(-1).float()
        q = torch.quantile(a, torch.tensor([0.05, 0.5, 0.95], device=a.device)) if len(a) else torch.full((3,), float("nan"))
        counts = torch.bincount(env, minlength=N_ENVS)
        positives = torch.bincount(env, weights=y.float(), minlength=N_ENVS)
        return {"b_e": self.b.detach().float().tolist(), "v_norm": float(self.v.weight.norm()),
                "a_mean": float(a.mean()) if len(a) else float("nan"),
                "a_std": float(a.std()) if len(a) > 1 else float("nan"),
                "a_q05": float(q[0]), "a_q50": float(q[1]), "a_q95": float(q[2]),
                "n_per_env": counts.tolist(), "pos_per_env": [int(x) for x in positives.tolist()]}

    def terms(self, z, y, env, unbiased: bool = False) -> dict:
        """Unweighted per-environment BCE (ERM term) and the IRMv1 slope penalty.

        For loss_e(w) = mean BCE(sigmoid(w * a + b_e), y) with a = v^T z, the gradient at
        w = 1 is mean((sigmoid(a + b_e) - y) * a), so the penalty sum_e grad_e^2 has a
        closed form; no double backward is needed. ``unbiased`` uses the product of the
        two half-sample gradients (Arjovsky et al., Section 3.2) instead of the square.
        Environments absent from the group contribute nothing.
        """
        a = self.v(z).squeeze(-1).float()
        logit = a + self.b[env].float()
        y = y.float()
        risks, grads = [], []
        for e in torch.unique(env).tolist():
            m = env == e
            risks.append(F.binary_cross_entropy_with_logits(logit[m], y[m]))
            g = (torch.sigmoid(logit[m]) - y[m]) * a[m]
            if unbiased:
                if len(g) < 2:
                    continue
                half = len(g) // 2
                grads.append(g[:half].mean() * g[half:2 * half].mean())
            else:
                grads.append(g.mean() ** 2)
        zero = a.sum() * 0
        return {"erm": torch.stack(risks).mean() if risks else zero,
                "penalty": torch.stack(grads).sum() if grads else zero,
                "environments": len(risks)}


class WindowScorer(nn.Module):
    """h(u) = w^T f_theta(u) + b on window CLS vectors."""
    def __init__(self, width: int, init_std: float = 0.02):
        super().__init__()
        self.out = nn.Linear(width, 1)
        _init_linear(self, init_std)

    def forward(self, windows):
        return self.out(windows).squeeze(-1)


def aux_loss(window_logits: torch.Tensor, window_labels: torch.Tensor) -> torch.Tensor:
    """Mean window BCE over labeled windows; label -1 marks unlabeled windows."""
    labeled = window_labels >= 0
    if not labeled.any():
        return window_logits.sum() * 0
    return F.binary_cross_entropy_with_logits(window_logits[labeled].float(), window_labels[labeled].float())


def stratum_rank_loss(logits: torch.Tensor, target: torch.Tensor, stratum: torch.Tensor,
                      max_stratum: int | None = None, replay: torch.Tensor | None = None) -> torch.Tensor:
    """Pairwise logistic ranking loss between positives and negatives of the same length stratum.

    Averaged over all within-stratum (positive, negative) pairs of the group; zero when the
    group has no such pair. It trains the within-stratum ordering that R_k evaluates.
    ``max_stratum`` restricts it to strata 1..max_stratum; ``replay`` marks replayed positives,
    which are never used as negatives.
    """
    logits, y = logits.float(), target.float()
    real = torch.ones_like(y, dtype=torch.bool) if replay is None else ~replay.bool()
    losses = []
    for k in torch.unique(stratum).tolist():
        if max_stratum is not None and k > max_stratum:
            continue
        m = stratum == k
        pos, neg = logits[m & (y == 1)], logits[m & (y == 0) & real]
        if len(pos) and len(neg):
            losses.append(F.softplus(neg[None, :] - pos[:, None]).reshape(-1))
    if not losses:
        return logits.sum() * 0
    return torch.cat(losses).mean()


def reference_log_lengths(train_log_lengths, k: int, seed: int, mode: str = "empirical",
                          train_strata=None) -> np.ndarray:
    """Fixed reference vector for s_std, drawn once from TRAIN and saved with the checkpoint.

    empirical: k draws from the train log-length distribution.
    stratum_uniform: k draws with equal probability per stratum (sensitivity analysis).
    """
    rng = np.random.default_rng(seed)
    lengths = np.asarray(train_log_lengths, dtype=float)
    if mode == "empirical":
        return np.sort(rng.choice(lengths, size=k, replace=True))
    if mode == "stratum_uniform":
        strata = np.asarray(train_strata)
        picks = rng.integers(1, 6, size=k)
        return np.sort([rng.choice(lengths[strata == s]) for s in picks])
    raise ValueError(f"Unknown reference mode {mode}")


def irm_weight(step: int, anneal_steps: int, lam: float) -> float:
    """IRMv1 schedule: weight 1 during warm-up, lam afterwards (Arjovsky et al. reference code).

    lam = 0 switches the penalty off entirely, including the warm-up (the lambda_IRM = 0 ablation).
    """
    if lam == 0:
        return 0.0
    return 1.0 if step < anneal_steps else float(lam)


class CFAObjective(nn.Module):
    """L = BCE^w(f) + lam_inv * ERM_inv + lam_irm * P_IRM + lam_aux * L_aux over one optimizer group.

    ``stratum_weights`` are the five B-a weights from objectives.prevalence_weights
    (train only, mean 1 over train). The invariant-head ERM term is not in the draft's
    total loss; without it the slope v receives no signal except the penalty, which
    it can zero by collapsing v (see the spec). lam_inv = 0 reproduces the draft.
    When the IRM weight exceeds 1 the total is divided by it, as in the IRMv1 code.
    """
    def __init__(self, stratum_weights=None, lam_inv: float = 1.0, lam_aux: float = 1.0,
                 irm_lambda: float = 1000.0, irm_anneal_steps: int = 500, rescale: bool = True,
                 unbiased_irm: bool = False, lam_rank: float = 0.0, lam_pair: float = 0.0,
                 lam_snip: float = 0.0, rank_max_stratum: int | None = None):
        super().__init__()
        self.lam_rank, self.lam_pair, self.lam_snip = lam_rank, lam_pair, lam_snip
        self.rank_max_stratum = rank_max_stratum
        w = np.ones(5) if stratum_weights is None else np.asarray(stratum_weights, dtype=float)
        self.register_buffer("stratum_weights", torch.tensor(w, dtype=torch.float32))
        self.lam_inv, self.lam_aux = lam_inv, lam_aux
        self.irm_lambda, self.irm_anneal_steps = irm_lambda, irm_anneal_steps
        self.rescale, self.unbiased_irm = rescale, unbiased_irm

    def forward(self, *, full_logits, target, stratum, step: int, z=None, invariant=None,
                window_logits=None, window_labels=None, replay=None) -> dict:
        """``replay`` marks replayed positives: they enter only the ranking term."""
        rank_logits, rank_target, rank_stratum = full_logits, target, stratum
        if replay is not None and bool(replay.any()):
            keep = ~replay.bool()
            full_logits, target, stratum = full_logits[keep], target[keep], stratum[keep]
            z = z[keep] if z is not None else None
        y = target.float()
        bce = F.binary_cross_entropy_with_logits(full_logits.float(), y, reduction="none")
        terms = {"bce_w": (bce * self.stratum_weights[stratum - 1]).mean()}
        zero = terms["bce_w"] * 0
        lam_irm = irm_weight(step, self.irm_anneal_steps, self.irm_lambda) if invariant is not None else 0.0
        if invariant is not None:
            inv = invariant.terms(z, y, environments(stratum), self.unbiased_irm)
            terms["erm_inv"], terms["irm_penalty"] = inv["erm"], inv["penalty"]
        else:
            terms["erm_inv"] = terms["irm_penalty"] = zero
        terms["aux"] = (aux_loss(window_logits, window_labels)
                        if window_logits is not None and self.lam_aux > 0 else zero)
        terms["rank"] = (stratum_rank_loss(rank_logits, rank_target, rank_stratum, self.rank_max_stratum, replay)
                         if self.lam_rank > 0 else zero)
        total = (terms["bce_w"] + self.lam_inv * terms["erm_inv"] + lam_irm * terms["irm_penalty"]
                 + self.lam_aux * terms["aux"] + self.lam_rank * terms["rank"])
        if self.rescale and lam_irm > 1:
            total = total / lam_irm
        terms["total"] = total
        terms["irm_weight"] = lam_irm
        return terms
