from __future__ import annotations

import argparse
from dataclasses import fields
import json
import logging

from .config import Config, VARIANTS


def experiment_arguments(parser):
    parser.add_argument("--config", help="JSON experiment configuration")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"))
    for name in ("data_dir", "model", "tokenizer", "output_dir", "cache_dir", "device", "cfa_pairs", "reference_mode"):
        parser.add_argument("--" + name.replace("_", "-"))
    for name in ("seed", "max_input_tokens", "window_tokens", "stride_tokens", "encoder_chunk_size",
                 "attention_hidden", "length_hidden", "batch_size", "eval_batch_size", "epochs",
                 "evals_per_epoch", "patience", "gradient_accumulation_steps", "log_every_updates",
                 "irm_anneal_steps", "reference_k"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("lr", "weight_decay", "warmup_fraction", "grad_clip", "rho", "dro_eta", "matched_delta",
                 "lambda_inv", "lambda_aux", "lambda_irm"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--selection", choices=("auto", "overall_auc", "macro_matched_auc"))
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--encoder-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", type=int, dest="smoke_limit", help="One epoch on limited train/valid; separate artifacts")


def configurations(args):
    config = Config.read(args.config).to_dict() if args.config else Config().to_dict()
    valid = {f.name for f in fields(Config)}
    for name, value in vars(args).items():
        if name in valid and value is not None and value != "all":
            config[name] = value
    variants = VARIANTS if args.variant == "all" else (config["variant"],)
    for variant in variants:
        cfg = Config(**{**config, "variant": variant})
        cfg.validate()
        yield cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description="CodeBERT baselines: prepare → train → score → evaluate")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List implemented variants")
    prep = sub.add_parser("prepare", help="Tokenize and audit dataset splits")
    experiment_arguments(prep)
    prep.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid", "test"])
    trainer = sub.add_parser("train", help="Train on train; select/check thresholds on valid")
    experiment_arguments(trainer)
    scorer = sub.add_parser("score", help="Score a frozen checkpoint (never train on test)")
    scorer.add_argument("--run", required=True)
    scorer.add_argument("--split", choices=("valid", "test"), default="test")
    scorer.add_argument("--device", default="auto")
    evaluator = sub.add_parser("evaluate", help="Test metrics using frozen validation thresholds")
    evaluator.add_argument("--run", required=True)
    evaluator.add_argument("--pairs", help="Optional official pair parquet; function PairRank only")
    external = sub.add_parser("evaluate-scores", help="Evaluate another algorithm's complete score tables")
    external.add_argument("--valid", required=True)
    external.add_argument("--test", required=True)
    external.add_argument("--output", required=True)
    external.add_argument("--matched-delta", type=float, default=0.05)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "list":
        print("\n".join(VARIANTS))
        return
    # Import ML dependencies only after parsing; --help/list works without torch.
    from . import engine
    if args.command in ("prepare", "train"):
        for cfg in configurations(args):
            if args.command == "prepare":
                result = engine.prepare(cfg, tuple(args.splits))
            else:
                result = str(engine.train(cfg))
            print(json.dumps(engine.clean_json(result), ensure_ascii=False, indent=2))
    elif args.command == "score":
        print(engine.score_run(args.run, args.split, args.device))
    elif args.command == "evaluate":
        print(json.dumps(engine.clean_json(engine.evaluate_run(args.run, args.pairs)), indent=2))
    else:
        import pandas as pd
        result = engine.evaluate_scores(pd.read_parquet(args.valid), pd.read_parquet(args.test),
                                         args.output, args.matched_delta)
        print(json.dumps(engine.clean_json(result), indent=2))
