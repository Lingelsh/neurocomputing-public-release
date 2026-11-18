from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

VARIANTS = (
    "global", "global_macro", "global_length", "stratum_private",
    "stratum_weighted", "group_dro", "sliding_attention", "cfa",
)
CUTS = (79, 160, 300, 660)


@dataclass
class Config:
    variant: str = "global"
    data_dir: str = "data/primevul"
    model: str = "microsoft/codebert-base"
    tokenizer: str | None = None
    output_dir: str = "artifacts/baselines"
    cache_dir: str = "artifacts/baselines/cache"
    seed: int = 13
    device: str = "auto"
    local_files_only: bool = False
    max_input_tokens: int = 512
    window_tokens: int = 128
    stride_tokens: int = 64
    encoder_chunk_size: int = 64
    attention_hidden: int = 128
    length_hidden: int = 32
    lr: float = 2e-5
    head_lr: float | None = None
    intercept_lr: float | None = None
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_cache: bool = False
    eval_batch_size: int = 32
    epochs: int = 3
    evals_per_epoch: int = 4
    patience: int = 4
    log_every_updates: int = 50
    warmup_fraction: float = 0.06
    grad_clip: float = 1.0
    rho: float = 0.5
    dro_eta: float = 0.01
    cfa_pairs: str = "artifacts/fix_pairs/primevul_pairs.parquet"
    lambda_inv: float = 1.0
    lambda_aux: float = 1.0
    lambda_rank: float = 0.0
    lambda_pair: float = 0.0
    pair_shared_length: bool = True
    lambda_snip: float = 0.0
    snip_start_updates: int = 0
    rank_max_stratum: int = 0
    replay_short: int = 0
    # SVulD reproduction (Ni et al., FSE'23 code): "rdrop" = two dropout passes, mean of their BCE
    # plus symmetric KL; "simct" = BCE + KL + cosine contrast, a vulnerable function against its
    # fixed version (pushed apart) and a non-vulnerable function against its second pass (pulled).
    # Both use mean pooling and a linear head. svuld_kl is per function: the original sums KL over a
    # batch of 32 with weight 0.1, i.e. 3.2 per function.
    svuld: str = "none"
    svuld_kl: float = 3.2
    svuld_ct: float = 0.2
    # "auto" = mean pooling + linear head for SVulD, CLS + RoBERTa head otherwise; "mean" puts the
    # SVulD pooling and head on a plain BCE detector (E25 control), "cls" forces the CLS head.
    pooling: str = "auto"
    length_tokenizer: str | None = None
    attention_from_scorer: bool = False
    lambda_irm: float = 1000.0
    cfa_prior_init: bool = False
    irm_anneal_steps: int = 500
    reference_k: int = 32
    reference_mode: str = "empirical"
    amp: bool = True
    gradient_checkpointing: bool = False
    encoder_checkpointing: bool = False
    selection: str = "auto"
    matched_delta: float = 0.05
    smoke_limit: int | None = None

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; choose from {VARIANTS}")
        if self.selection not in ("auto", "overall_auc", "macro_matched_auc"):
            raise ValueError("selection must be auto, overall_auc or macro_matched_auc")
        for key in ("max_input_tokens", "window_tokens", "stride_tokens", "encoder_chunk_size",
                    "batch_size", "gradient_accumulation_steps", "eval_batch_size", "epochs", "evals_per_epoch", "log_every_updates",
                    "attention_hidden", "length_hidden", "reference_k"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.patience < 0:
            raise ValueError("patience must be nonnegative; 0 disables early stopping")
        if not 0 <= self.rank_max_stratum <= 5 or self.replay_short < 0:
            raise ValueError("rank_max_stratum must be in 0..5 (0 = all) and replay_short nonnegative")
        if self.replay_short > 0 and (self.lambda_rank <= 0 or self.rank_max_stratum == 0):
            raise ValueError("replay_short needs lambda_rank > 0 and rank_max_stratum > 0")
        if self.svuld not in ("none", "rdrop", "simct"):
            raise ValueError("svuld must be none, rdrop or simct")
        if self.svuld != "none" and (self.variant != "global" or self.uses_gradient_cache):
            raise ValueError("svuld reproduces a truncated single-input detector: variant global, no gradient cache")
        if self.pooling not in ("auto", "cls", "mean"):
            raise ValueError("pooling must be auto, cls or mean")
        if self.pooling != "auto" and self.uses_windows:
            raise ValueError("pooling applies to single-input detectors; window variants use attention pooling")
        if min(self.svuld_kl, self.svuld_ct) < 0:
            raise ValueError("svuld_kl and svuld_ct must be nonnegative")
        if self.snip_start_updates < 0:
            raise ValueError("snip_start_updates must be nonnegative")
        if self.irm_anneal_steps < 0 or min(self.lambda_inv, self.lambda_aux, self.lambda_irm, self.lambda_rank, self.lambda_pair, self.lambda_snip) < 0:
            raise ValueError("CFA loss weights and IRM anneal steps must be nonnegative")
        if self.reference_mode not in ("empirical", "stratum_uniform"):
            raise ValueError("Unknown reference_mode")
        if self.max_input_tokens < 4:
            raise ValueError("max_input_tokens must be at least 4")
        if self.stride_tokens > self.window_tokens:
            raise ValueError("stride_tokens must not exceed window_tokens (coverage would have gaps)")
        if self.uses_windows and self.window_tokens > self.max_input_tokens - 2:
            raise ValueError("window_tokens must leave space for special tokens")
        if not 0 <= self.rho <= 1 or self.dro_eta <= 0 or self.matched_delta <= 0:
            raise ValueError("Require rho in [0,1], dro_eta > 0, matched_delta > 0")
        if self.lr <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer parameters")
        if any(value is not None and (not math.isfinite(value) or value <= 0)
               for value in (self.head_lr, self.intercept_lr)):
            raise ValueError("head_lr and intercept_lr must be positive when set")
        if not isinstance(self.cfa_prior_init, bool):
            raise ValueError("cfa_prior_init must be a boolean")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0,1)")
        if self.smoke_limit is not None and self.smoke_limit < 10:
            raise ValueError("smoke_limit must be at least 10 to retain stratum/class cells")

    @property
    def selection_metric(self) -> str:
        if self.selection != "auto":
            return self.selection
        return "macro_matched_auc" if self.variant in ("global_macro", "cfa") else "overall_auc"

    @property
    def uses_windows(self) -> bool:
        return self.variant in ("sliding_attention", "cfa")

    @property
    def uses_mean_pool(self) -> bool:
        return self.pooling == "mean" or (self.pooling == "auto" and self.svuld != "none")

    @property
    def uses_pair_keys(self) -> bool:
        return self.lambda_pair > 0 or self.lambda_snip > 0 or self.svuld == "simct"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def uses_gradient_cache(self) -> bool:
        # Group risks and q must be computed once over the whole optimization group.
        return self.variant == "cfa" or self.gradient_cache or (self.variant == "group_dro" and self.gradient_accumulation_steps > 1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def read(cls, path: str | Path) -> "Config":
        cfg = cls(**json.loads(Path(path).read_text()))
        cfg.validate()
        return cfg
