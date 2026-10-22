#!/usr/bin/env python3
"""Paired uncertainty for the public end-to-end track.

Paired bootstrap over the shared record ids, plus McNemar for binary
correctness.  Nothing here rounds a non-significant difference into a claim:
the caller receives the interval and the p-value and reports both.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def paired_bootstrap(
    treatment: Sequence[float],
    control: Sequence[float],
    iterations: int = 10000,
    seed: int = 20260805,
    alpha: float = 0.05,
) -> dict:
    a = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs aligned samples")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n == 0:
        return {
            "n": 0,
            "treatment_mean": float("nan"),
            "control_mean": float("nan"),
            "difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "significant_at_95": False,
        }
    difference = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iterations, n))
    draws = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(n),
        "treatment_mean": float(a.mean()),
        "control_mean": float(b.mean()),
        "difference": difference,
        "ci_low": float(low),
        "ci_high": float(high),
        "significant_at_95": bool(low > 0.0 or high < 0.0),
        "iterations": iterations,
        "seed": seed,
    }


def mcnemar(
    treatment: Sequence[float], control: Sequence[float], correct_at: float = 0.5
) -> dict:
    """Exact binomial McNemar on discordant pairs."""
    a = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask] >= correct_at, b[mask] >= correct_at
    only_treatment = int(np.sum(a & ~b))
    only_control = int(np.sum(~a & b))
    discordant = only_treatment + only_control
    if discordant == 0:
        return {
            "treatment_only_correct": 0,
            "control_only_correct": 0,
            "discordant": 0,
            "p_value": 1.0,
            "test": "exact_binomial",
            "note": "no discordant pairs; the two systems agree on every record",
        }
    smaller = min(only_treatment, only_control)
    tail = sum(
        math.comb(discordant, k) for k in range(0, smaller + 1)
    ) * (0.5 ** discordant)
    p_value = min(1.0, 2.0 * tail)
    return {
        "treatment_only_correct": only_treatment,
        "control_only_correct": only_control,
        "discordant": discordant,
        "p_value": float(p_value),
        "test": "exact_binomial",
    }


def align(rows_a: dict, rows_b: dict, metric: str) -> tuple[list[float], list[float]]:
    """Align two uid -> row maps on their shared uids, in a stable order."""
    shared = sorted(set(rows_a) & set(rows_b))
    left, right = [], []
    for uid in shared:
        va = rows_a[uid].get(metric)
        vb = rows_b[uid].get(metric)
        left.append(float("nan") if va is None else float(va))
        right.append(float("nan") if vb is None else float(vb))
    return left, right
