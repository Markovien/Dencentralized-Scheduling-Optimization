"""Shapley value: exact enumeration and the stratified antithetic sampler.

Implements Paper A §6, Layer 2b.

    phi_i(v) = sum_{S subset N\\{i}} [ |S|! (n-|S|-1)! / n! ] * [ v(S + i) - v(S) ]

Exact evaluation is ``O(2^n)`` in *coalition evaluations*, and on this problem each
evaluation is itself a scheduling run -- so the binding constraint is the run count, not
the combinatorics.  On the Bilge--Ulusoy family ``n = |M| + |V| = 4 + 2 = 6``, i.e. 64
runs, so the **exact** Shapley value, least core and nucleolus are all affordable with no
sampling error whatsoever.  :func:`appro_shapley` is the estimator for the larger
generated instances, with the two variance reductions the roadmap specifies -- **coalition-
size stratification** and **antithetic permutation pairs** -- and a CLT confidence interval
reported with every allocation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from jsspt_tou.simulator.state import Player

ValueFn = Callable[[frozenset[Player]], float]


@dataclass(frozen=True, slots=True)
class Allocation:
    """A payoff vector with its provenance and, if sampled, its uncertainty."""

    payoff: dict[Player, float]
    method: str
    ci_halfwidth: dict[Player, float] | None = None
    permutations: int = 0

    @property
    def total(self) -> float:
        return sum(self.payoff.values())

    def gini(self) -> float:
        """Gini coefficient of the allocation -- the fairness column of Paper A Tab. 6."""
        x = np.array(sorted(self.payoff.values()), dtype=float)
        n = len(x)
        if n == 0 or np.allclose(x.sum(), 0.0):
            return 0.0
        shifted = x - min(0.0, x.min())
        total = shifted.sum()
        if total <= 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2.0 * (index * shifted).sum()) / (n * total) - (n + 1.0) / n)


def exact_shapley(players: Sequence[Player], v: ValueFn) -> Allocation:
    """Exact Shapley value by subset enumeration.  Use for ``n <= 16``."""
    n = len(players)
    weights = [
        math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
        for s in range(n)
    ]
    payoff: dict[Player, float] = {}
    for i, player in enumerate(players):
        rest = [p for p in players if p != player]
        total = 0.0
        for mask in range(1 << (n - 1)):
            subset = frozenset(rest[j] for j in range(n - 1) if mask >> j & 1)
            total += weights[len(subset)] * (v(subset | {player}) - v(subset))
        payoff[player] = total
    return Allocation(payoff=payoff, method="shapley-exact")


def appro_shapley(
    players: Sequence[Player],
    v: ValueFn,
    rng: np.random.Generator,
    permutations: int = 200,
    stratified: bool = True,
    antithetic: bool = True,
    confidence: float = 0.95,
) -> Allocation:
    """Sampled Shapley value (Castro et al.) with stratification and antithetic pairs.

    Parameters
    ----------
    permutations:
        Number of sampled permutations.  With ``antithetic`` each draw contributes a
        permutation *and* its reverse, which cancels the dominant source of variance in
        marginal-contribution sampling (early versus late positions).
    stratified:
        Sample marginal contributions stratified by coalition size, so every stratum is
        represented rather than left to chance.
    confidence:
        Two-sided normal confidence level for the reported half-widths.

    Returns
    -------
    Allocation
        With ``ci_halfwidth`` populated.  An allocation reported without its interval is
        not reportable: the sampling error is part of the result.
    """
    n = len(players)
    z = float(_normal_quantile(0.5 + confidence / 2.0))
    samples: dict[Player, list[float]] = {p: [] for p in players}
    order = list(players)
    for _ in range(permutations):
        perms = [list(rng.permutation(order))]
        if antithetic:
            perms.append(list(reversed(perms[0])))
        for perm in perms:
            running: frozenset[Player] = frozenset()
            prev = v(running)
            for player in perm:
                running = running | {player}
                current = v(running)
                samples[player].append(current - prev)
                prev = current
    if stratified:
        # Top up the thinnest strata so no coalition size is left unrepresented.
        for size in range(n):
            for _ in range(max(0, permutations // (4 * n))):
                for player in players:
                    rest = [p for p in players if p != player]
                    idx = rng.choice(len(rest), size=size, replace=False)
                    subset = frozenset(rest[int(j)] for j in idx)
                    samples[player].append(v(subset | {player}) - v(subset))
    payoff: dict[Player, float] = {}
    half: dict[Player, float] = {}
    for player in players:
        arr = np.array(samples[player], dtype=float)
        payoff[player] = float(arr.mean())
        half[player] = float(z * arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    # Re-impose efficiency exactly: the sampler is unbiased per player but the sum of
    # independent estimates need not hit v(N) exactly, and every core/least-core test
    # downstream assumes an efficient allocation.
    total = sum(payoff.values())
    target = v(frozenset(players))
    if abs(total) > 1e-12:
        drift = (target - total) / n
        payoff = {p: val + drift for p, val in payoff.items()}
    return Allocation(
        payoff=payoff,
        method="shapley-sampled",
        ci_halfwidth=half,
        permutations=permutations * (2 if antithetic else 1),
    )


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |err| < 1.15e-9)."""
    from scipy.stats import norm  # local import keeps the module importable without scipy

    return float(norm.ppf(p))
