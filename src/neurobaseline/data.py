from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CUTS, Config


def stratum(n_tokens) -> np.ndarray:
    return np.searchsorted(CUTS, np.asarray(n_tokens), side="left") + 1


def split_path(data_dir: str | Path, split: str) -> Path:
    if split not in ("train", "valid", "test"):
        raise ValueError("split must be train, valid or test")
    root = Path(data_dir)
    candidates = [root / f"{split}.{ext}" for ext in ("parquet", "jsonl", "csv")]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected one {split} file in {root}, found {found}")
    return found[0]


def jsonl_records(path: Path):
    """Use Python JSON to preserve IDs and oversized numeric metadata exactly."""
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON record") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield record


def load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        # Metadata can exceed the numeric limits of pandas' ujson
        # parser. Parsing one record at a time also avoids one giant JSON array.
        df = pd.DataFrame.from_records(jsonl_records(path))
    elif path.suffix == ".csv":
        df = pd.read_csv(path, dtype={"idx": str})
    else:
        raise ValueError(f"Unsupported dataset: {path}")
    required = {"idx", "func", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(df.columns))}")
    if len(df) == 0 or df[list(required)].isna().any().any():
        raise ValueError(f"{path}: empty dataset or null idx/func/target")
    if not df.func.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{path}: func must contain nonempty code strings")
    if not df.target.isin([0, 1]).all():
        raise ValueError(f"{path}: target must be 0 or 1")
    df = df.copy()
    df["idx"] = df.idx.astype(str)
    if df.idx.duplicated().any():
        raise ValueError(f"{path}: duplicate idx values")
    df["target"] = df.target.astype(np.int8)
    if "project" not in df:
        df["project"] = "unknown"
    if "n_tokens" in df:
        n = pd.to_numeric(df.n_tokens, errors="raise")
        if n.isna().any() or (n < 1).any() or (n % 1 != 0).any():
            raise ValueError(f"{path}: n_tokens must be positive integers")
        df["n_tokens"] = n.astype(np.int64)
    return df.reset_index(drop=True)


def require_disjoint(a: pd.DataFrame, b: pd.DataFrame, names=("train", "valid")) -> None:
    overlap = set(a.idx) & set(b.idx)
    if overlap:
        raise ValueError(f"Overlapping idx between {names[0]} and {names[1]}: {len(overlap)}")


@dataclass(frozen=True)
class Window:
    ids: tuple[int, ...]  # content only; signature followed by body
    lo: int             # half-open positions in the body token sequence
    hi: int
    hard_split: bool


def code_windows(code: str, tokenizer, width: int, stride: int, max_input: int) -> tuple[list[Window], dict]:
    """Line-aligned windows, with token fallback for long lines; no body token is dropped.

    The first '{' is a signature heuristic, not a C/C++ parser. Keep the brace and
    same-line body statements. Long signatures are truncated only in the repeated
    signature prefix; the full function length used for evaluation is unchanged.
    """
    if not 1 <= stride <= width:
        raise ValueError("Require 1 <= stride <= width")
    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    all_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if not all_ids:
        raise ValueError("Function has no content tokens")
    capacity = max_input - tokenizer.num_special_tokens_to_add(pair=False)
    if width > capacity or capacity < 1:
        raise ValueError("Window exceeds encoder content capacity")
    brace = code.find("{")
    signature = tokenizer(code[:brace], add_special_tokens=False, verbose=False)["input_ids"] if brace >= 0 else []
    body_start = next((i for i, (_, end) in enumerate(offsets) if end > brace), 0) if brace >= 0 else 0
    body = all_ids[body_start:]
    sig_budget = capacity - width
    signature_truncated = len(signature) > sig_budget
    signature = signature[:sig_budget]
    boundaries = {0, len(body)}
    token_starts = [start for start, _ in offsets]
    char_start = 0
    for line in code.splitlines(keepends=True):
        if char_start > brace:
            boundaries.add(max(0, min(len(body), bisect_left(token_starts, char_start) - body_start)))
        char_start += len(line)
    boundaries = sorted(boundaries)
    windows, start = [], 0
    while start < len(body):
        limit = min(start + width, len(body))
        aligned_end = boundaries[bisect_right(boundaries, limit) - 1]
        end = aligned_end if aligned_end > start else limit
        windows.append(Window(tuple(signature + body[start:end]), start, end,
                              start not in boundaries or end not in boundaries))
        if end == len(body):
            break
        desired = min(start + stride, end)
        aligned_start = boundaries[bisect_right(boundaries, desired) - 1]
        start = aligned_start if aligned_start > start else desired
    if not windows:
        raise ValueError("Function produced an empty window bag")
    return windows, {"n_tokens": len(all_ids), "body_tokens": len(body),
                     "signature_truncated": signature_truncated,
                     "hard_split_windows": sum(w.hard_split for w in windows)}


def frame_digest(frame: pd.DataFrame) -> str:
    cols = ["idx", "func", "target"] + (["n_tokens"] if "n_tokens" in frame else [])
    values = pd.util.hash_pandas_object(frame[cols], index=False).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def tokenizer_digest(tokenizer) -> str:
    if not tokenizer.is_fast:
        raise ValueError("Use a fast CodeBERT tokenizer (offset mapping is required)")
    payload = tokenizer.backend_tokenizer.to_str() + repr(tokenizer.special_tokens_map)
    return hashlib.sha256(payload.encode()).hexdigest()


class EncodedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, arrays: dict, metadata: dict):
        self.frame = frame.copy()
        self.values = arrays["values"]
        self.token_ptr = arrays["token_ptr"]
        self.bag_ptr = arrays["bag_ptr"]
        self.body_spans = arrays["body_spans"]
        self.frame["n_tokens"] = arrays["n_tokens"]
        self.frame["stratum"] = stratum(self.frame.n_tokens).astype(np.int8)
        self.frame["log_length"] = np.log(self.frame.n_tokens.to_numpy(dtype=float))
        self.metadata = metadata

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        windows = [self.values[self.token_ptr[j]:self.token_ptr[j + 1]]
                   for j in range(self.bag_ptr[i], self.bag_ptr[i + 1])]
        row = self.frame.iloc[i]
        return {"windows": windows, "target": float(row.target), "stratum": int(row.stratum),
                "log_length": float(row.log_length), "idx": row.idx}


def _length_counter(cfg: Config):
    """Tokenizer that defines N(x) and the strata; None means the model tokenizer.

    Lets a backbone with a different vocabulary keep the CodeBERT length axis, so strata
    and length-matched metrics stay comparable across backbones.
    """
    if cfg.length_tokenizer is None:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.length_tokenizer, use_fast=True, local_files_only=True)


def encode_dataset(frame: pd.DataFrame, tokenizer, cfg: Config, split: str) -> EncodedDataset:
    counter = _length_counter(cfg)
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = cfg.max_input_tokens - special_count
    if capacity < 1:
        raise ValueError("No capacity left for content tokens")
    identity = {"format": 1, "data": frame_digest(frame), "tokenizer": tokenizer_digest(tokenizer),
                "windows": cfg.uses_windows, "max_input_tokens": cfg.max_input_tokens,
                "window_tokens": cfg.window_tokens, "stride_tokens": cfg.stride_tokens}
    if counter is not None:
        identity["length_tokenizer"] = tokenizer_digest(counter)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(cfg.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{split}-{digest}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in ("values", "token_ptr", "bag_ptr", "body_spans", "n_tokens")}
            metadata = json.loads(str(cache["metadata"]))
        if metadata["identity"] != identity:
            raise ValueError(f"Cache identity mismatch: {path}")
        return EncodedDataset(frame, arrays, metadata)

    pieces, token_ptr, bag_ptr, spans, lengths = [], [0], [0], [], []
    signature_truncations, hard_splits = 0, 0
    for row in frame.itertuples(index=False):
        if cfg.uses_windows:
            windows, info = code_windows(row.func, tokenizer, cfg.window_tokens,
                                         cfg.stride_tokens, cfg.max_input_tokens)
            signature_truncations += int(info["signature_truncated"])
            hard_splits += info["hard_split_windows"]
            n_tokens = info["n_tokens"]
        else:
            ids = tokenizer(row.func, add_special_tokens=False, verbose=False)["input_ids"]
            n_tokens = len(ids)
            if not n_tokens:
                raise ValueError(f"{split} idx={row.idx}: no content tokens")
            windows = [Window(tuple(ids[:capacity]), 0, min(n_tokens, capacity), False)]
        if counter is not None:
            n_tokens = len(counter(row.func, add_special_tokens=False, verbose=False)["input_ids"])
            if not n_tokens:
                raise ValueError(f"{split} idx={row.idx}: no content tokens under the length tokenizer")
        supplied = getattr(row, "n_tokens", n_tokens)
        if supplied != n_tokens:
            raise ValueError(f"{split} idx={row.idx}: stored n_tokens={supplied}, tokenizer gives {n_tokens}")
        lengths.append(n_tokens)
        for w in windows:
            seq = np.asarray(tokenizer.build_inputs_with_special_tokens(list(w.ids)), dtype=np.int32)
            if len(seq) != len(w.ids) + special_count:
                raise ValueError("Tokenizer special-token builder disagrees with its declared special-token count")
            if len(seq) > cfg.max_input_tokens:
                raise ValueError("Internal error: input exceeds max_input_tokens")
            pieces.append(seq)
            token_ptr.append(token_ptr[-1] + len(seq))
            spans.append((w.lo, w.hi))
        bag_ptr.append(len(pieces))
    arrays = {"values": np.concatenate(pieces), "token_ptr": np.asarray(token_ptr, dtype=np.int64),
              "bag_ptr": np.asarray(bag_ptr, dtype=np.int64), "body_spans": np.asarray(spans, dtype=np.int64),
              "n_tokens": np.asarray(lengths, dtype=np.int64)}
    metadata = {"identity": identity, "functions": len(frame), "windows": len(pieces),
                "signature_truncations": signature_truncations, "hard_split_windows": hard_splits}
    # Replace only this cache key after a complete write; partial caches are never reused.
    # Different GPU workers may prepare the same deterministic cache concurrently.
    with tempfile.NamedTemporaryFile(dir=root, prefix=path.stem + "-", suffix=".tmp.npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays, metadata=json.dumps(metadata, sort_keys=True))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return EncodedDataset(frame, arrays, metadata)


class BagCollator:
    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token")
        self.pad = pad_token_id

    def __call__(self, samples: list[dict]) -> dict:
        windows = [w for s in samples for w in s["windows"]]
        width = max(map(len, windows))
        ids = torch.full((len(windows), width), self.pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        owners = []
        offset = 0
        for i, sample in enumerate(samples):
            for w in sample["windows"]:
                ids[offset, :len(w)] = torch.as_tensor(np.asarray(w, dtype=np.int64))
                mask[offset, :len(w)] = 1
                owners.append(i)
                offset += 1
        return {"input_ids": ids, "attention_mask": mask,
                "owners": torch.tensor(owners, dtype=torch.long),
                "target": torch.tensor([s["target"] for s in samples], dtype=torch.float32),
                "stratum": torch.tensor([s["stratum"] for s in samples], dtype=torch.long),
                "log_length": torch.tensor([s["log_length"] for s in samples], dtype=torch.float32),
                "idx": [s["idx"] for s in samples]}
