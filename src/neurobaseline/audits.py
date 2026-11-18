"""Window PairRank and gate-1 length-leakage audits (task M4).

Score-level only: callers supply window scores h(u), content vectors z,
function scores and lengths from a frozen checkpoint. Perturbation helpers
return transformed source code or window bags; re-scoring them is the caller's job.

Gate 1 pass criteria (CFA design 6.5): probe R^2 < 0.05, within-stratum
|Corr(s_std, log N)| < 0.05, and |Delta R_k| < 0.01 under each perturbation.
"""
from __future__ import annotations

import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import length_matched_auc

PROBE_R2_MAX = 0.05
CORR_MAX = 0.05
DELTA_R_MAX = 0.01


# ---------------------------------------------------------------- window PairRank

def window_pair_rank(window_scores: dict, pairs: pd.DataFrame, max_tokens: int | None = None) -> dict:
    """Share of (fix-overlapping window, same-site fixed window) pairs ranked correctly.

    ``window_scores`` maps function idx -> array of h(u) in window order (the
    ordering of data.code_windows). Within each pair, compare every positive window
    of the vulnerable version with every hard-negative window of the fixed version
    (ties 1/2); average within the pair, then equally over pairs.
    """
    if max_tokens is not None:
        pairs = pairs[pairs.n_tokens <= max_tokens]
    per_pair, skipped = [], 0
    for row in pairs.itertuples(index=False):
        pos, neg = list(row.positive_windows), list(row.hard_negative_windows)
        if not pos or not neg or row.vuln_idx not in window_scores or row.fixed_idx not in window_scores:
            skipped += 1
            continue
        sv, sf = np.asarray(window_scores[row.vuln_idx]), np.asarray(window_scores[row.fixed_idx])
        if len(sv) != row.n_windows or len(sf) != row.n_fixed_windows:
            raise ValueError(f"pair {row.vuln_idx}/{row.fixed_idx}: window count differs from the pair table")
        a, b = sv[pos][:, None], sf[neg][None, :]
        per_pair.append(float(((a > b) + 0.5 * (a == b)).mean()))
    return {"window_PairRank": float(np.mean(per_pair)) if per_pair else float("nan"),
            "pairs": len(per_pair), "pairs_skipped": skipped}


def attention_on_fix(attention: dict, pairs: pd.DataFrame) -> dict:
    """Attention mass on fix-overlapping windows vs the uniform share (gate 2 diagnostic)."""
    mass, uniform = [], []
    for row in pairs.itertuples(index=False):
        if row.vuln_idx not in attention or not len(row.positive_windows):
            continue
        alpha = np.asarray(attention[row.vuln_idx])
        mass.append(float(alpha[list(row.positive_windows)].sum()))
        uniform.append(len(row.positive_windows) / len(alpha))
    mass, uniform = np.asarray(mass), np.asarray(uniform)
    return {"mean_mass_on_fix": float(mass.mean()) if len(mass) else float("nan"),
            "mean_uniform_share": float(uniform.mean()) if len(uniform) else float("nan"),
            "pairs": int(len(mass))}


# ---------------------------------------------------------------- gate 1 (a)(b)

def length_probe(z_fit, log_length_fit, z_eval, log_length_eval) -> dict:
    """Linear probe z -> log N, fitted on one split (valid) and scored on another (test)."""
    probe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    probe.fit(np.asarray(z_fit), np.asarray(log_length_fit))
    r2 = r2_score(np.asarray(log_length_eval), probe.predict(np.asarray(z_eval)))
    return {"probe_R2": float(r2), "alpha": float(probe[-1].alpha_), "pass": bool(r2 < PROBE_R2_MAX)}


def within_stratum_correlation(score, log_length, stratum) -> pd.DataFrame:
    score, log_length, stratum = map(np.asarray, (score, log_length, stratum))
    rows = []
    for k in range(1, 6):
        m = stratum == k
        if m.sum() < 3:
            rows.append({"stratum": k, "n": int(m.sum()), "pearson": np.nan, "spearman": np.nan, "pass": False})
            continue
        p = pearsonr(score[m], log_length[m]).statistic
        s = spearmanr(score[m], log_length[m]).statistic
        rows.append({"stratum": k, "n": int(m.sum()), "pearson": float(p), "spearman": float(s),
                     "pass": bool(abs(p) < CORR_MAX)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- gate 1 (c) perturbations

DEAD_CODE = "if (0) { int cfa_pad_{k} = {k}; (void)cfa_pad_{k}; }"


def insert_dead_code(code: str, copies: int) -> str | None:
    """Insert ``copies`` never-executed statements right after the first '{'.

    Returns None when the function has no body brace. The inserted names are unique
    per copy so the result stays valid C/C++.
    """
    brace = code.find("{")
    if brace < 0:
        return None
    line_end = code.find("\n", brace)
    cut = len(code) if line_end < 0 else line_end + 1
    indent = re.match(r"[ \t]*", code[cut:]).group(0) if cut < len(code) else "    "
    block = "".join(indent + DEAD_CODE.replace("{k}", str(k)) + "\n" for k in range(copies))
    prefix = code[:cut] if line_end >= 0 else code + "\n"
    return prefix + block + code[cut:]


def reformat(code: str, style: str = "LLVM", binary: str | None = None, timeout: float = 20) -> str | None:
    """clang-format with a fixed style; None if the tool is missing or fails."""
    exe = binary or shutil.which("clang-format")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, f"--style={style}", "--assume-filename=f.c"], input=code, text=True,
                             capture_output=True, timeout=timeout, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return out if out.strip() else None


def duplicate_low_attention_window(windows: list, attention, copies: int = 1) -> list:
    """Append copies of the lowest-attention window to the bag (a bag-level, code-free perturbation)."""
    k = int(np.argmin(np.asarray(attention)))
    return list(windows) + [windows[k]] * copies


def perturbation_delta(frame: pd.DataFrame, score_before, score_after, applied=None) -> pd.DataFrame:
    """Per-stratum matched AUROC before/after, on the rows where the perturbation applied.

    Strata and matching use the ORIGINAL length, so only the scores change.
    """
    y, z, g = frame.target.to_numpy(), frame.log_length.to_numpy(), frame.stratum.to_numpy()
    before, after = np.asarray(score_before, float), np.asarray(score_after, float)
    keep = np.ones(len(y), bool) if applied is None else np.asarray(applied, bool)
    rows = []
    for k in range(1, 6):
        m = keep & (g == k)
        rb = length_matched_auc(y[keep], before[keep], z[keep], m[keep])["auc"]
        ra = length_matched_auc(y[keep], after[keep], z[keep], m[keep])["auc"]
        diff = np.abs(after[m] - before[m])
        rows.append({"stratum": k, "n": int(m.sum()), "R_before": rb, "R_after": ra, "delta_R": ra - rb,
                     "median_abs_score_change": float(np.median(diff)) if len(diff) else np.nan,
                     "pass": bool(np.isfinite(ra - rb) and abs(ra - rb) < DELTA_R_MAX)})
    return pd.DataFrame(rows)
