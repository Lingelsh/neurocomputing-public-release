"""Recompute paper metrics from the frozen scores without training a model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurobaseline.metrics import summary

DATA = ROOT / "reproduction"
DATASETS = ("primevul", "bigvul", "reveal", "megavul")
METADATA = {"idx", "target", "n_tokens", "stratum", "log_length", "has_cpg"}


def wins(left, right):
    return (left > right) + 0.5 * (left == right)


def classification(valid, test):
    precision, recall, thresholds = precision_recall_curve(valid.target, valid.score)
    valid_f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    threshold = thresholds[np.argmax(valid_f1[:-1])]
    predicted = test.score >= threshold
    correct = (predicted & test.target.eq(1)).sum()
    precision = correct / max(predicted.sum(), 1)
    recall = correct / test.target.sum()
    return {
        "validation_f1_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def evaluate_dataset(dataset, models):
    valid, test = [
        pd.read_parquet(DATA / "scores" / f"{dataset}-{split}.parquet").set_index("idx")
        for split in ("valid", "test")
    ]
    pairs = pd.read_parquet(DATA / "datasets" / dataset / f"{dataset}_pairs.parquet")
    pairs = pairs.loc[pairs.split.eq("test")]
    rows = []
    if not models:
        models = [col for col in test if col not in METADATA]
    for model in models:
        if model not in test or model in METADATA:
            raise ValueError(f"Unknown score for {dataset}: {model}")

        def frame(source):
            columns = ["target", "n_tokens", "stratum", "log_length", model]
            return source[columns].dropna(subset=[model]).rename(columns={model: "score"})

        valid_model, test_model = frame(valid), frame(test)
        match = pairs.loc[pairs.vuln_idx.isin(test_model.index) & pairs.fixed_idx.isin(test_model.index)]
        vulnerable = test_model.score.loc[match.vuln_idx].to_numpy()
        fixed = test_model.score.loc[match.fixed_idx].to_numpy()
        pair_wins = wins(vulnerable, fixed)
        length_change = np.log(match.fixed_n_tokens.to_numpy() / match.n_tokens.to_numpy())
        longer = length_change > np.log(1.15)
        negatives = test_model.loc[test_model.target.eq(0)]
        metrics, _ = summary(test_model, test_model.score)
        rho_delta = 1.0 if model == "length" else spearmanr(length_change, fixed - vulnerable).statistic
        rows.append({
            "dataset": dataset,
            "model": model,
            "test_functions": len(test_model),
            "test_pairs": len(match),
            "pairrank": pair_wins.mean(),
            "test_pairs_length_gain_gt15": int(longer.sum()),
            "pairrank_length_gain_gt15": pair_wins[longer].mean(),
            "rho_delta": rho_delta,
            "rho_l": spearmanr(negatives.log_length, negatives.score).statistic,
            "length_matched_auroc": metrics["macro_matched_auc"],
            "auroc": metrics["overall_auc"],
            **classification(valid_model, test_model),
        })
    return rows


def evaluate_transformations():
    folder = DATA / "transformations"
    reported = json.loads((folder / "reported-results.json").read_text())
    flags = pd.read_parquet(folder / "selection-flags.parquet")
    exact_visible = (flags.exact & flags.visible).groupby([flags.pair, flags.model]).all()
    passes = exact_visible.unstack(fill_value=False).all(axis=1)
    cohort = pd.read_csv(folder / "cohort.csv", dtype={"vuln_idx": str}).sort_values("vuln_idx")
    if passes[passes].index.tolist() != cohort.vuln_idx.tolist():
        raise ValueError("The selected pairs differ from the recorded cohort")
    idx = np.random.default_rng(reported["bootstrap_seed"]).integers(
        len(cohort), size=(reported["bootstrap_resamples"], len(cohort))
    )

    def estimate(values):
        lower, upper = np.percentile(values[idx].mean(axis=1), (2.5, 97.5))
        return {"PairRank": round(float(values.mean()), 4), "ci": [round(lower, 3), round(upper, 3)]}

    def aligned(frame):
        return frame.set_index("pair").loc[cohort.vuln_idx]

    results = {}
    scores = pd.read_parquet(folder / "scores.parquet")
    for model, rows in scores.groupby("model"):
        original = aligned(rows.loc[rows.cond.eq("orig")])
        baseline = estimate(wins(original.s_v.to_numpy(), original.s_f.to_numpy()))
        fillers = {}
        for filler, part in rows.loc[rows.filler.notna()].groupby("filler"):
            padded = part.loc[part.cond.eq("pad")]
            equal = aligned(padded.loc[padded.k.eq(1)])
            control = aligned(part.loc[part.cond.eq("placebo")])
            dose = [baseline]
            for amount in reported["dose_multipliers"][1:]:
                longer = aligned(padded.loc[padded.k.eq(amount)])
                dose.append(estimate(wins(longer.s_v.to_numpy(), longer.s_f.to_numpy())))
            fillers[filler] = {
                "dose": dose,
                "both_padded": estimate(wins(control.s_v.to_numpy(), control.s_f.to_numpy())),
                "length_only": estimate(wins(original.s_v.to_numpy(), equal.s_v.to_numpy())),
                "content_only": estimate(wins(equal.s_v.to_numpy(), equal.s_f.to_numpy())),
            }
        results[model] = {"original": baseline, "fillers": fillers}
    if results != reported["models"]:
        raise ValueError("Recomputed transformation results differ from the archived results")
    return {**reported, "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", action="append", help="Frozen score column, repeatable")
    parser.add_argument("--transformations", action="store_true", help="Recompute RQ1/RQ2 padding contrasts")
    args = parser.parse_args()
    if args.transformations:
        if args.dataset or args.model:
            parser.error("--transformations takes no dataset or model argument")
        print(json.dumps(evaluate_transformations(), indent=2))
        return
    if args.model and not args.dataset:
        parser.error("--model requires --dataset")
    datasets = [args.dataset] if args.dataset else DATASETS
    rows = [row for dataset in datasets for row in evaluate_dataset(dataset, args.model)]
    pd.DataFrame(rows).to_csv(sys.stdout, index=False, float_format="%.8g")


if __name__ == "__main__":
    main()
