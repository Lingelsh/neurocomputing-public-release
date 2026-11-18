from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auc(y, score) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def length_matched_auc(y, score, log_length, positive_mask=None, delta: float = 0.05) -> dict:
    """Pair-weighted AUROC; each selected positive compares to ALL length-close negatives.

    Matching is |log(N_pos)-log(N_neg)| <= delta, including negatives just over
    stratum boundaries. Ties receive 1/2. Positives without a match are disclosed.
    No dense positive-by-negative matrix is materialized.
    """
    y, score, z = np.asarray(y), np.asarray(score), np.asarray(log_length)
    if delta <= 0 or not (len(y) == len(score) == len(z)):
        raise ValueError("Invalid matched-AUROC inputs")
    mask = np.ones(len(y), dtype=bool) if positive_mask is None else np.asarray(positive_mask, dtype=bool)
    if len(mask) != len(y) or not np.isfinite(score).all() or not np.isfinite(z).all():
        raise ValueError("Invalid scores/lengths/mask")
    pos = np.flatnonzero((y == 1) & mask)
    neg = np.flatnonzero(y == 0)
    order = neg[np.argsort(z[neg], kind="stable")]
    neg_z, neg_score = z[order], score[order]
    wins, comparisons, matched_positives = 0.0, 0, 0
    for i in pos:
        # Search a one-ULP envelope, then apply the stated absolute-difference
        # rule exactly; subtraction at the matching boundary can round differently.
        lo = np.searchsorted(neg_z, np.nextafter(z[i] - delta, -np.inf), side="left")
        hi = np.searchsorted(neg_z, np.nextafter(z[i] + delta, np.inf), side="right")
        close = np.abs(neg_z[lo:hi] - z[i]) <= delta
        candidates = neg_score[lo:hi][close]
        if len(candidates):
            wins += np.count_nonzero(score[i] > candidates) + 0.5 * np.count_nonzero(score[i] == candidates)
            comparisons += len(candidates)
            matched_positives += 1
    return {"auc": wins / comparisons if comparisons else float("nan"),
            "pairs": int(comparisons), "positives": int(len(pos)),
            "matched_positives": int(matched_positives),
            "unmatched_positives": int(len(pos) - matched_positives)}


def partial_auc(y, score, max_fpr=0.05) -> float:
    """Raw area in [0,max_fpr] divided by max_fpr, NOT sklearn's standardized pAUC."""
    if len(np.unique(y)) != 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    end = np.searchsorted(fpr, max_fpr, side="right")
    x = np.r_[fpr[:end], max_fpr]
    v = np.r_[tpr[:end], np.interp(max_fpr, fpr, tpr)]
    return float(np.sum(np.diff(x) * (v[:-1] + v[1:]) * 0.5) / max_fpr)


def summary(frame: pd.DataFrame, scores, delta=0.05) -> tuple[dict, pd.DataFrame]:
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and cover every row")
    y, z, groups = frame.target.to_numpy(), frame.log_length.to_numpy(), frame.stratum.to_numpy()
    rows = []
    for k in range(1, 6):
        m = groups == k
        matched = length_matched_auc(y, scores, z, m, delta)
        rows.append({"stratum": k, "n": int(m.sum()), "positives": int(y[m].sum()),
                     "matched_auc": matched["auc"], "bucket_auc": auc(y[m], scores[m]),
                     "matched_pairs": matched["pairs"], "unmatched_positives": matched["unmatched_positives"]})
    table = pd.DataFrame(rows)
    all_r = table.matched_auc.to_numpy()
    r12 = length_matched_auc(y, scores, z, groups <= 2, delta)
    overall = {"overall_auc": auc(y, scores),
               "auprc": float(average_precision_score(y, scores)) if np.any(y == 1) else float("nan"),
               "macro_matched_auc": float(all_r.mean()) if np.isfinite(all_r).all() else float("nan"),
               "mean_R3_R5": float(all_r[2:].mean()) if np.isfinite(all_r[2:]).all() else float("nan"),
               "R_1+2": r12["auc"], "R_1+2_pairs": r12["pairs"],
               "pAUC_5_raw_normalized": partial_auc(y[groups == 5], scores[groups == 5])}
    return overall, table


def fit_thresholds(valid: pd.DataFrame, scores, alphas=(0.01, 0.05)) -> dict:
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(valid) or not np.isfinite(scores).all():
        raise ValueError("Invalid validation scores")
    y, groups = valid.target.to_numpy(), valid.stratum.to_numpy()
    if not np.any(y == 0):
        raise ValueError("Validation set has no negatives for threshold fitting")
    result = {}
    for alpha in alphas:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0,1)")
        per = {}
        for k in range(1, 6):
            neg = scores[(y == 0) & (groups == k)]
            per[str(k)] = float(np.quantile(neg, 1 - alpha)) if len(neg) else None
        result[str(alpha)] = {"uniform": float(np.quantile(scores[y == 0], 1 - alpha)), "stratum": per}
    return result


def deployment(frame: pd.DataFrame, scores, thresholds: dict) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    y, groups = frame.target.to_numpy(), frame.stratum.to_numpy()
    rows = []
    for alpha, ts in thresholds.items():
        for mode in ("uniform", "stratum"):
            for k in range(1, 6):
                threshold = ts[mode] if mode == "uniform" else ts[mode][str(k)]
                m = groups == k
                pos, neg = m & (y == 1), m & (y == 0)
                alarms = scores > threshold if threshold is not None else np.zeros(len(y), dtype=bool)
                rows.append({"alpha_valid": float(alpha), "threshold_mode": mode, "stratum": k,
                             "threshold": threshold,
                             "TPR": float(alarms[pos].mean()) if pos.any() and threshold is not None else float("nan"),
                             "FPR": float(alarms[neg].mean()) if neg.any() and threshold is not None else float("nan"),
                             "alarms": int((alarms & m).sum()) if threshold is not None else None})
    return pd.DataFrame(rows)


def function_pair_rank(frame: pd.DataFrame, scores, pairs: pd.DataFrame) -> dict:
    """FUNCTION version score ordering; never label this as CFA's window PairRank."""
    pairs = pairs.copy()
    # M1's canonical table names versus the original external-score interface.
    for key, source in (("vul_id", "vuln_idx"), ("fixed_id", "fixed_idx")):
        if source in pairs:
            if key in pairs and not pairs[key].astype(str).equals(pairs[source].astype(str)):
                raise ValueError(f"Conflicting pair ID columns: {key} and {source}")
            pairs[key] = pairs[source]
    required = {"vul_id", "fixed_id"}
    if not required.issubset(pairs.columns):
        raise ValueError("Pair file requires vul_id and fixed_id")
    for key in required:
        pairs[key] = pairs[key].astype(str)
    indexed = frame.assign(score=np.asarray(scores)).set_index("idx")
    present = pairs.vul_id.isin(indexed.index) & pairs.fixed_id.isin(indexed.index)
    kept = pairs[present]
    if len(kept) and (not (indexed.loc[kept.vul_id, "target"].to_numpy() == 1).all()
                      or not (indexed.loc[kept.fixed_id, "target"].to_numpy() == 0).all()):
        raise ValueError("Pair labels must be vulnerable=1 and fixed=0")
    results = {"pairs_missing_versions": int((~present).sum())}
    for name, p in (("all", kept), ("le160", kept[kept.vul_id.map(indexed.n_tokens) <= 160])):
        sv = indexed.loc[p.vul_id, "score"].to_numpy()
        sf = indexed.loc[p.fixed_id, "score"].to_numpy()
        results[f"function_PairRank_{name}"] = float(((sv > sf) + 0.5 * (sv == sf)).mean()) if len(p) else float("nan")
        results[f"function_pairs_{name}"] = len(p)
    return results
