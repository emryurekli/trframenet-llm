from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score


def safe_kappa(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    yt = np.asarray(list(y_true), dtype=object)
    yp = np.asarray(list(y_pred), dtype=object)
    if len(yt) < 2:
        return float("nan")
    try:
        return float(cohen_kappa_score(yt, yp))
    except Exception:
        return float("nan")


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 5000, seed: int = 42,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(list(values), dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff(a_correct: Iterable[bool], b_correct: Iterable[bool],
                          n_boot: int = 5000, seed: int = 42,
                          alpha: float = 0.05) -> dict[str, float]:
    a = np.asarray(list(a_correct), dtype=float)
    b = np.asarray(list(b_correct), dtype=float)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Paired arrays must be non-empty and have equal length")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {
        "difference": float(diff.mean()),
        "ci_low": float(np.quantile(boots, alpha / 2)),
        "ci_high": float(np.quantile(boots, 1 - alpha / 2)),
        "bootstrap_p": min(1.0, p),
    }


def exact_mcnemar(a_correct: Iterable[bool], b_correct: Iterable[bool]) -> dict[str, float | int]:
    a = np.asarray(list(a_correct), dtype=bool)
    b = np.asarray(list(b_correct), dtype=bool)
    if len(a) != len(b):
        raise ValueError("Paired arrays must have equal length")
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_value": p}
