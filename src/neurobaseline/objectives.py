from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def prevalence_weights(y, groups, rho: float) -> np.ndarray:
    """Inverse-positive-rate stratum weighting, normalized by train sample counts.

    Every example in a stratum has the same weight. This is the RQ1/CFA
    weighting control, not an implementation of continuous length matching.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    rates = np.array([y[groups == k].mean() if np.any(groups == k) else np.nan for k in range(1, 6)])
    if not np.isfinite(rates).all() or np.any(rates <= 0) or np.any(rates >= 1):
        raise ValueError("Stratum weighting needs both classes in all five TRAIN strata")
    weights = (rates.mean() / rates) ** rho
    weights /= weights[groups - 1].mean()
    return weights


class Objective(nn.Module):
    def __init__(self, variant: str, weights=None, dro_eta: float = 0.01):
        super().__init__()
        self.variant = variant
        self.dro_eta = dro_eta
        self.register_buffer("weights", torch.tensor(np.ones(5) if weights is None else weights, dtype=torch.float32))
        self.register_buffer("log_q", torch.zeros(5))

    def forward(self, logits, y, groups):
        losses = F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="none")
        if self.variant == "stratum_weighted":
            return (losses * self.weights[groups - 1]).mean()
        if self.variant != "group_dro":
            return losses.mean()
        # Five original length strata. Missing minibatch groups neither update nor
        # contribute a fabricated zero risk; normalize q over observed groups.
        active = torch.unique(groups).sort().values
        risks = torch.stack([losses[groups == k].mean() for k in active])
        if self.training:
            with torch.no_grad():
                self.log_q[active - 1] += self.dro_eta * risks.detach()
                self.log_q -= torch.logsumexp(self.log_q, dim=0)
        q = torch.softmax(self.log_q[active - 1], dim=0)
        return torch.sum(q * risks)

