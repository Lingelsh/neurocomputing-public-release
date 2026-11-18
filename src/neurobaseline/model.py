from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel

from .config import Config


class ClassificationHead(nn.Module):
    """RoBERTa-style CLS head, shared in architecture across mechanism controls."""
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(width, width)
        self.out = nn.Linear(width, 1)

    def forward(self, z):
        z = self.dropout(z)
        z = torch.tanh(self.dense(z))
        return self.out(self.dropout(z)).squeeze(-1)


class LinearHead(nn.Module):
    """Single linear layer on the pooled vector, as in the SVulD classifier."""
    def __init__(self, width: int):
        super().__init__()
        self.out = nn.Linear(width, 1)

    def forward(self, z):
        return self.out(z).squeeze(-1)


class AttentionPool(nn.Module):
    """The single tanh attention formula in the CFA draft (not gated attention)."""
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))

    def forward(self, windows, owners, n_bags: int):
        energies = self.network(windows).squeeze(-1)
        pooled, weights = [], []
        for bag in range(n_bags):
            mask = owners == bag
            if not mask.any():
                raise ValueError("Empty window bag")
            alpha = torch.softmax(energies[mask].float(), dim=0).to(windows.dtype)
            pooled.append(torch.sum(alpha[:, None] * windows[mask], dim=0))
            weights.append(alpha)
        return torch.stack(pooled), torch.cat(weights)


class BaselineModel(nn.Module):
    def __init__(self, encoder, cfg: Config, length_mean: float = 0, length_std: float = 1):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.register_buffer("length_mean", torch.tensor(float(length_mean)))
        self.register_buffer("length_std", torch.tensor(max(float(length_std), 1e-6)))
        hidden = encoder.config.hidden_size
        dropout = getattr(encoder.config, "classifier_dropout", None)
        if dropout is None:
            dropout = getattr(encoder.config, "hidden_dropout_prob", 0.1)
        self.pool = AttentionPool(hidden, cfg.attention_hidden) if cfg.uses_windows else None
        self.length_embedding = (nn.Sequential(nn.Linear(1, cfg.length_hidden), nn.Tanh())
                                 if cfg.variant == "global_length" else None)
        head_width = hidden + (cfg.length_hidden if self.length_embedding is not None else 0)
        self.mean_pool = cfg.uses_mean_pool
        self.head = LinearHead(head_width) if self.mean_pool else ClassificationHead(head_width, dropout)
        # Apply the same initialization rule to newly introduced layers, never to the encoder.
        std = getattr(encoder.config, "initializer_range", 0.02)
        for module in (self.pool, self.length_embedding, self.head):
            if module is not None:
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=0.0, std=std)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
        if cfg.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

    @classmethod
    def pretrained(cls, cfg: Config, length_mean=0, length_std=1, **kwargs):
        encoder = AutoModel.from_pretrained(cfg.model, add_pooling_layer=False,
                                           local_files_only=cfg.local_files_only)
        return cls(encoder, cfg, length_mean, length_std, **kwargs)

    @classmethod
    def from_encoder_config(cls, path, cfg: Config):
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(path, local_files_only=True),
                                        add_pooling_layer=False)
        return cls(encoder, cfg)

    def _encode_cls(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if not self.mean_pool:
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)

    @contextmanager
    def _encoder_checkpoint_mode(self):
        # A no-grad cache pass needs dropout, but no activation checkpoints.
        # Preserve the exact layer settings/checkpoint functions for replay.
        saved = [] if torch.is_grad_enabled() else [
            (module, module.gradient_checkpointing) for module in self.encoder.modules()
            if getattr(module, "gradient_checkpointing", False)]
        try:
            for module, _ in saved:
                module.gradient_checkpointing = False
            yield
        finally:
            for module, value in saved:
                module.gradient_checkpointing = value

    def encode_windows(self, batch):
        """Encode CLS with autograd, or without it for an explicitly replayed cache pass."""
        representations = []
        with self._encoder_checkpoint_mode():
            for start in range(0, len(batch["input_ids"]), self.cfg.encoder_chunk_size):
                stop = start + self.cfg.encoder_chunk_size
                ids = batch["input_ids"][start:stop]
                mask = batch["attention_mask"][start:stop]
                if self.cfg.encoder_checkpointing and self.training and torch.is_grad_enabled():
                    # Recompute entire chunks; preserve dropout RNG for backward.
                    cls = checkpoint(self._encode_cls, ids, mask, use_reentrant=False, preserve_rng_state=True)
                else:
                    cls = self._encode_cls(ids, mask)
                representations.append(cls)
        return torch.cat(representations)

    def forward_from_windows(self, windows, batch, return_features: bool = False):
        """Pool and classify supplied CLS vectors; supports whole-group cached losses."""
        n_bags = len(batch["target"])
        if self.pool is not None:
            z, alpha = self.pool(windows, batch["owners"], n_bags)
        else:
            if len(windows) != n_bags or not torch.equal(batch["owners"], torch.arange(n_bags, device=windows.device)):
                raise ValueError("Truncated global model requires exactly one input per function")
            z, alpha = windows, None
        head_input = z
        if self.length_embedding is not None:
            length = ((batch["log_length"] - self.length_mean) / self.length_std).unsqueeze(-1)
            head_input = torch.cat((z, self.length_embedding(length.to(z.dtype))), dim=-1)
        logits = self.head(head_input)
        return {"logits": logits, "content": z, "attention": alpha,
                "window_features": windows} if return_features else logits

    def forward(self, batch, return_features: bool = False):
        return self.forward_from_windows(self.encode_windows(batch), batch, return_features)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return device


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
