"""Statistical protocol (ROADMAP.md §5.3, referee check R-6).

Binding rules, implemented here so no analysis script can quietly skip them:

* **Omnibus** Friedman test across methods over instances; if significant, a **Nemenyi**
  post-hoc with a critical-difference diagram.  This is the Demsar protocol and referees at
  both target venues expect it.
* **Pairwise** Wilcoxon signed-rank with **Holm--Bonferroni** correction.
* **Effect size** Vargha--Delaney ``A12`` reported alongside every significant p-value.
  A p-value without an effect size is a delivery-blocking defect.
* Never a mean without a dispersion measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class FriedmanResult:
    statistic: float
    p_value: float
    n_methods: int
    n_instances: int
    mean_ranks: dict[str, float]
    critical_difference: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    method_a: str
    method_b: str
    p_raw: float
    p_holm: float
    a12: float
    median_diff: float

    @property
    def significant(self) -> bool:
        return self.p_holm < 0.05

    @property
    def effect_label(self) -> str:
        """Vargha--Delaney magnitude labels (0.56 small, 0.64 medium, 0.71 large)."""
        d = abs(self.a12 - 0.5)
        if d < 0.06:
            return "negligible"
        if d < 0.14:
            return "small"
        if d < 0.21:
            return "medium"
        return "large"


def friedman(scores: Mapping[str, Sequence[float]]) -> FriedmanResult:
    """Friedman omnibus test with mean ranks and the Nemenyi critical difference.

    ``scores`` maps a method name to its per-instance objective values (lower is better).
    """
    methods = list(scores)
    matrix = np.array([list(scores[m]) for m in methods], dtype=float)
    n_methods, n_instances = matrix.shape
    if n_methods < 3:
        raise ValueError("the Friedman test needs at least three methods")
    statistic, p_value = stats.friedmanchisquare(*matrix)
    ranks = np.apply_along_axis(stats.rankdata, 0, matrix)  # rank within each instance
    mean_ranks = {m: float(ranks[i].mean()) for i, m in enumerate(methods)}
    q_alpha = _nemenyi_q(n_methods)
    cd = q_alpha * float(np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_instances)))
    return FriedmanResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_methods=n_methods,
        n_instances=n_instances,
        mean_ranks=mean_ranks,
        critical_difference=cd,
    )


def _nemenyi_q(k: int) -> float:
    """Studentised range critical value at alpha = 0.05, divided by sqrt(2)."""
    table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
        9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    }
    return table.get(k, 3.391)


def vargha_delaney_a12(a: Sequence[float], b: Sequence[float]) -> float:
    """``A12``: probability that a random draw from ``a`` exceeds one from ``b``.

    0.5 means no effect.  Reported with every significant p-value.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    greater = sum(1.0 for i in x for j in y if i > j)
    equal = sum(1.0 for i in x for j in y if i == j)
    return float((greater + 0.5 * equal) / (len(x) * len(y)))


def pairwise_wilcoxon_holm(
    scores: Mapping[str, Sequence[float]],
    reference: str | None = None,
) -> list[PairwiseResult]:
    """Wilcoxon signed-rank on every pair (or against a reference), Holm-corrected."""
    methods = list(scores)
    pairs: list[tuple[str, str]] = []
    if reference is not None:
        pairs = [(reference, m) for m in methods if m != reference]
    else:
        pairs = [
            (methods[i], methods[j])
            for i in range(len(methods))
            for j in range(i + 1, len(methods))
        ]
    raw: list[tuple[str, str, float, float, float]] = []
    for a, b in pairs:
        xa = np.asarray(scores[a], dtype=float)
        xb = np.asarray(scores[b], dtype=float)
        if np.allclose(xa, xb):
            p = 1.0
        else:
            try:
                p = float(stats.wilcoxon(xa, xb, zero_method="zsplit").pvalue)
            except ValueError:  # pragma: no cover - degenerate input
                p = 1.0
        raw.append(
            (a, b, p, vargha_delaney_a12(xa.tolist(), xb.tolist()), float(np.median(xa - xb)))
        )

    order = sorted(range(len(raw)), key=lambda i: raw[i][2])
    m = len(raw)
    holm = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * raw[idx][2])
        running = max(running, adjusted)  # Holm's step-down is monotone
        holm[idx] = running
    return [
        PairwiseResult(
            method_a=a,
            method_b=b,
            p_raw=p,
            p_holm=holm[i],
            a12=a12,
            median_diff=diff,
        )
        for i, (a, b, p, a12, diff) in enumerate(raw)
    ]


def describe(values: Sequence[float]) -> dict[str, float]:
    """Mean **with** dispersion -- never one without the other."""
    arr = np.asarray(values, dtype=float)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
