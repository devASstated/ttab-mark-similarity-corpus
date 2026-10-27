#!/usr/bin/env python3
"""
feline.py — FELINE deterministic orthographic scorer.

FELINE
======
Fixed Exact-character Local alIgNment mEtric

A deterministic LASA-oriented character similarity measure:
1. exact-character Smith-Waterman local alignment
2. affine gap penalties
3. normalization by the longer name
4. bounded exact-prefix bonus over at most the first 4 characters

There is NO training step and NO model artifact.

Default parameters:
    gap_open     = 0.60
    gap_extend   = 0.15
    prefix_scale = 0.175
    prefix_max   = 4

Usage:
    from feline import score
    s = score("dopamine", "dobutamine")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict
import numpy as np

_KEEP = re.compile(r"[^a-z0-9\- ]")


@dataclass(frozen=True)
class FelineConfig:
    gap_open: float = 0.60
    gap_extend: float = 0.15
    prefix_scale: float = 0.175
    prefix_max: int = 4


DEFAULT_CONFIG = FelineConfig()


def normalize_name(name: str) -> str:
    s = _KEEP.sub("", str(name).lower())
    return re.sub(r"\s+", " ", s).strip()


def _local_alignment_score(a: str, b: str, gap_open: float, gap_extend: float) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0

    H = np.zeros((n + 1, m + 1), dtype=float)
    E = np.zeros((n + 1, m + 1), dtype=float)
    F = np.zeros((n + 1, m + 1), dtype=float)

    best = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            E[i, j] = max(
                H[i, j - 1] - gap_open,
                E[i, j - 1] - gap_extend,
            )
            F[i, j] = max(
                H[i - 1, j] - gap_open,
                F[i - 1, j] - gap_extend,
            )

            match_reward = 1.0 if ai == b[j - 1] else 0.0
            diag = H[i - 1, j - 1] + match_reward

            v = max(0.0, diag, E[i, j], F[i, j])
            H[i, j] = v
            if v > best:
                best = v

    return float(best / max(n, m))


def _prefix_length(a: str, b: str, prefix_max: int) -> int:
    k = 0
    for ca, cb in zip(a[:prefix_max], b[:prefix_max]):
        if ca != cb:
            break
        k += 1
    return k


def score(name_a: str, name_b: str, *, config: FelineConfig = DEFAULT_CONFIG) -> float:
    """
    FELINE similarity in [0,1]. Higher means more orthographically similar.
    """
    a = normalize_name(name_a)
    b = normalize_name(name_b)

    if not a or not b:
        return 0.0

    raw = _local_alignment_score(
        a, b,
        gap_open=config.gap_open,
        gap_extend=config.gap_extend,
    )
    prefix_len = _prefix_length(a, b, config.prefix_max)
    final = raw + prefix_len * config.prefix_scale * (1.0 - raw)
    return float(np.clip(final, 0.0, 1.0))


def score_components(
    name_a: str,
    name_b: str,
    *,
    config: FelineConfig = DEFAULT_CONFIG,
) -> Dict[str, float | int | str]:
    """
    Diagnostic version of score(); useful for audit logs and experiments.
    """
    a = normalize_name(name_a)
    b = normalize_name(name_b)

    if not a or not b:
        return {
            "score": 0.0,
            "local_alignment": 0.0,
            "prefix_len": 0,
            "normalized_a": a,
            "normalized_b": b,
            "gap_open": config.gap_open,
            "gap_extend": config.gap_extend,
            "prefix_scale": config.prefix_scale,
            "prefix_max": config.prefix_max,
        }

    raw = _local_alignment_score(a, b, config.gap_open, config.gap_extend)
    prefix_len = _prefix_length(a, b, config.prefix_max)
    final = float(np.clip(
        raw + prefix_len * config.prefix_scale * (1.0 - raw),
        0.0,
        1.0,
    ))

    return {
        "score": final,
        "local_alignment": raw,
        "prefix_len": prefix_len,
        "normalized_a": a,
        "normalized_b": b,
        "gap_open": config.gap_open,
        "gap_extend": config.gap_extend,
        "prefix_scale": config.prefix_scale,
        "prefix_max": config.prefix_max,
    }


def model_info() -> dict:
    c = DEFAULT_CONFIG
    return {
        "name": "FELINE",
        "type": "deterministic",
        "trained": False,
        "gap_open": c.gap_open,
        "gap_extend": c.gap_extend,
        "prefix_scale": c.prefix_scale,
        "prefix_max": c.prefix_max,
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) >= 3:
        x = score_components(sys.argv[1], sys.argv[2])
        print(f"{x['score']:.4f}")
        print(json.dumps(x, indent=2))
    else:
        print(__doc__)
        print("config:", model_info())
