"""Fix-pair table and window localization labels for CFA component A.

A PrimeVul paired file stores (vulnerable, fixed) versions as adjacent rows.
Fix blocks come from a line diff after whitespace normalization, so the
BigVul-sourced whitespace artifacts do not create spurious blocks. Windows come
from ``data.code_windows``; this module only maps diff lines to body tokens.

Window labels (see docs/cfa-method-spec.md):
- positive: a window of the vulnerable version whose body span overlaps a fix block;
- hard negative: a window of the fixed version overlapping the same block's
  fixed-side lines;
- all other windows of paired functions are left unlabeled.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import difflib
from pathlib import Path

import numpy as np

from .data import code_windows, jsonl_records, stratum


def normalize_line(line: str) -> str:
    return " ".join(line.split())


@dataclass(frozen=True)
class FixBlock:
    kind: str                    # replace, delete or insert (difflib opcodes)
    vuln_lines: tuple[int, int]  # half-open line range located in the vulnerable version
    fixed_lines: tuple[int, int]  # half-open line range at the same site in the fixed version


def fix_blocks(vuln: str, fixed: str, r_ins: int = 1) -> tuple[list[FixBlock], int]:
    """Diff normalized lines; return located blocks and the count of whitespace-only blocks.

    Pure insertions locate r_ins lines on each side of the insertion point in the
    vulnerable version; pure deletions do the same in the fixed version.
    """
    a = [normalize_line(x) for x in vuln.splitlines()]
    b = [normalize_line(x) for x in fixed.splitlines()]
    blocks, blank_only = [], 0
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if not any(a[i1:i2]) and not any(b[j1:j2]):
            blank_only += 1
            continue
        vl = (i1, i2) if i2 > i1 else (max(0, i1 - r_ins), min(len(a), i1 + r_ins))
        fl = (j1, j2) if j2 > j1 else (max(0, j1 - r_ins), min(len(b), j1 + r_ins))
        blocks.append(FixBlock(tag, vl, fl))
    return blocks, blank_only


def line_token_ranges(code: str, offsets) -> list[tuple[int, int]]:
    """Half-open token index range covering the non-whitespace characters of each line."""
    starts = [s for s, _ in offsets]
    ends = [e for _, e in offsets]
    ranges, pos = [], 0
    for line in code.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            lo = pos + (len(line) - len(line.lstrip()))
            hi = lo + len(stripped)
            # tokens whose [start, end) intersects [lo, hi)
            first = bisect_right(ends, lo)
            last = bisect_left(starts, hi)
            ranges.append((first, max(first, last)))
        else:
            ranges.append((0, 0))
        pos += len(line)
    return ranges


@dataclass
class FunctionView:
    windows: list            # data.Window objects
    n_tokens: int
    body_start: int
    line_tokens: list[tuple[int, int]]
    signature_truncated: bool


def function_view(code: str, tokenizer, width: int, stride: int, max_input: int) -> FunctionView:
    windows, info = code_windows(code, tokenizer, width, stride, max_input)
    enc = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    if len(enc["input_ids"]) != info["n_tokens"]:
        raise ValueError("Tokenization differs from code_windows")
    return FunctionView(windows, info["n_tokens"], info["n_tokens"] - info["body_tokens"],
                        line_token_ranges(code, enc["offset_mapping"]), info["signature_truncated"])


def block_body_span(view: FunctionView, lines: tuple[int, int]) -> tuple[int, int] | None:
    """Body-token span of a line range, or None if the lines hold no body tokens."""
    spans = [view.line_tokens[i] for i in range(*lines) if i < len(view.line_tokens)]
    spans = [(lo, hi) for lo, hi in spans if hi > lo]
    if not spans:
        return None
    lo = max(min(s for s, _ in spans), view.body_start) - view.body_start
    hi = max(e for _, e in spans) - view.body_start
    return (lo, hi) if hi > lo else None


def overlapping_windows(view: FunctionView, span: tuple[int, int]) -> list[int]:
    return [k for k, w in enumerate(view.windows) if w.lo < span[1] and span[0] < w.hi]


def covering_windows(view: FunctionView, span: tuple[int, int]) -> list[int]:
    return [k for k, w in enumerate(view.windows) if w.lo <= span[0] and span[1] <= w.hi]


def read_pairs(path: str | Path) -> list[tuple[dict, dict]]:
    rows = list(jsonl_records(Path(path)))
    if len(rows) % 2:
        raise ValueError(f"{path}: odd number of paired rows")
    pairs = []
    for k in range(0, len(rows), 2):
        v, f = rows[k], rows[k + 1]
        if (v["target"], f["target"]) != (1, 0):
            raise ValueError(f"{path}: pair {k // 2} is not ordered vulnerable/fixed")
        if (v["project"], v["commit_id"]) != (f["project"], f["commit_id"]):
            raise ValueError(f"{path}: pair {k // 2} crosses commits")
        pairs.append((v, f))
    return pairs


def label_pair(vuln: dict, fixed: dict, tokenizer, width: int, stride: int, max_input: int,
               r_ins: int = 1) -> dict:
    """One row of the pair table, with positive and hard-negative window ordinals."""
    blocks, blank_only = fix_blocks(vuln["func"], fixed["func"], r_ins)
    vv = function_view(vuln["func"], tokenizer, width, stride, max_input)
    fv = function_view(fixed["func"], tokenizer, width, stride, max_input)
    positive, hard, spans, covered = set(), set(), [], 0
    signature_only = 0
    for block in blocks:
        vs = block_body_span(vv, block.vuln_lines)
        if vs is None:
            signature_only += 1
            continue
        spans.append(vs[1] - vs[0])
        positive.update(overlapping_windows(vv, vs))
        covered += bool(covering_windows(vv, vs))
        fs = block_body_span(fv, block.fixed_lines)
        if fs is not None:
            hard.update(overlapping_windows(fv, fs))
    return {
        "vuln_idx": str(vuln["idx"]), "fixed_idx": str(fixed["idx"]),
        "project": vuln["project"], "commit_id": vuln["commit_id"],
        "n_tokens": vv.n_tokens, "stratum": int(stratum([vv.n_tokens])[0]),
        "fixed_n_tokens": fv.n_tokens,
        "identical_after_normalization": [normalize_line(x) for x in vuln["func"].splitlines() if x.strip()]
                                         == [normalize_line(x) for x in fixed["func"].splitlines() if x.strip()],
        "n_blocks": len(blocks), "blank_only_blocks": blank_only,
        "signature_only_blocks": signature_only, "located_blocks": len(spans),
        "covered_blocks": covered,
        "block_spans": spans, "max_block_span": max(spans) if spans else 0,
        "n_windows": len(vv.windows), "n_fixed_windows": len(fv.windows),
        "positive_windows": sorted(positive), "hard_negative_windows": sorted(hard),
        "signature_truncated": vv.signature_truncated,
    }


def check_against_split(pairs: list[tuple[dict, dict]], main_path: str | Path) -> dict:
    """Every paired version must exist in the main split with identical code and label."""
    need = {str(r["idx"]) for pair in pairs for r in pair}
    main = {str(r["idx"]): (r["func"], r["target"]) for r in jsonl_records(Path(main_path))
            if str(r["idx"]) in need}
    missing = sorted(need - set(main))
    mismatched = sorted(str(r["idx"]) for pair in pairs for r in pair
                        if str(r["idx"]) in main and main[str(r["idx"])] != (r["func"], r["target"]))
    return {"paired_ids": len(need), "missing_in_main": len(missing), "mismatched": len(mismatched),
            "examples": (missing + mismatched)[:5]}


def coverage_by_stratum(table, positives_by_stratum: dict[int, int], width: int) -> list[dict]:
    """Per-stratum localization coverage of the split's positive functions."""
    rows = []
    for k in range(1, 6):
        part = table[table.stratum == k]
        n_pos = positives_by_stratum.get(k, 0)
        located = part[part.positive_windows.map(len) > 0]
        rows.append({
            "stratum": k, "positives": n_pos, "paired": len(part),
            "paired_rate": len(part) / n_pos if n_pos else np.nan,
            "located": len(located), "located_rate": len(located) / n_pos if n_pos else np.nan,
            "positive_windows": int(part.positive_windows.map(len).sum()),
            "hard_negative_windows": int(part.hard_negative_windows.map(len).sum()),
            "median_windows": float(part.n_windows.median()) if len(part) else np.nan,
            "median_positive_windows": float(located.positive_windows.map(len).median()) if len(located) else np.nan,
            "positive_window_share": (float(part.positive_windows.map(len).sum() / part.n_windows.sum())
                                      if len(part) else np.nan),
            "median_max_block_span": float(located.max_block_span.median()) if len(located) else np.nan,
            "max_span_le_width": (float((located.max_block_span <= width).mean()) if len(located) else np.nan),
        })
    return rows
