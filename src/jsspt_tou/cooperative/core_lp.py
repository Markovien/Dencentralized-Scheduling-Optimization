"""Core, least core and nucleolus of the savings game.

Implements Paper A §6, result **T5** (certified stability) and the numerical certification
of ROADMAP.md §3.4.

The direction of the LP is the thing to get right
-------------------------------------------------
``v`` is a **savings (profit) game**, so an allocation ``x`` is blocked by a coalition that
could secure *more* on its own than it is being given.  The least-core programme is

    maximise    eps
    subject to  sum_{i in S} x_i  >=  v(S) + eps      for all S, empty != S != N
                sum_{i in N} x_i  =   v(N)

``eps* >= 0`` certifies a **non-empty core**; ``eps* < 0`` is the **least-core radius** --
how far from stable the most stable allocation is.

    The cost-game form ``sum_{i in S} x_i <= v(S) + eps``, with "``eps <= 0`` certifies
    membership", is the **wrong direction** for this game and would certify the wrong
    condition with an inverted sign test.  ``tests/test_cooperative.py`` pins the
    direction against a hand-computed three-player example, and
    ``tests/test_theory.py`` pins it against the N5 congestion instance, whose core must
    come out **empty**.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linprog

from jsspt_tou.simulator.state import Player

ValueFn = Callable[[frozenset[Player]], float]


@dataclass(frozen=True, slots=True)
class CoreResult:
    """Outcome of the least-core programme."""

    epsilon: float
    allocation: dict[Player, float]
    core_nonempty: bool
    binding: tuple[frozenset[Player], ...]
    status: str

    def excess(self, v: ValueFn, coalition: frozenset[Player]) -> float:
        """``sum_{i in S} x_i - v(S)`` -- non-negative means the coalition cannot block."""
        return sum(self.allocation[p] for p in coalition) - v(coalition)


def least_core(
    players: Sequence[Player],
    v: ValueFn,
    coalitions: Sequence[frozenset[Player]] | None = None,
) -> CoreResult:
    """Solve the least-core LP in the savings-game direction.

    Parameters
    ----------
    coalitions:
        Proper non-empty sub-coalitions to constrain.  Defaults to *all* of them, which is
        exact and affordable at ``n = 6`` (62 constraints).  Pass a subset to run
        constraint generation on larger games via :func:`least_core_generated`.
    """
    n = len(players)
    grand = frozenset(players)
    subsets = (
        list(coalitions)
        if coalitions is not None
        else [
            frozenset(c)
            for size in range(1, n)
            for c in itertools.combinations(players, size)
        ]
    )
    index = {p: i for i, p in enumerate(players)}

    # variables z = (x_1, ..., x_n, eps); maximise eps  <=>  minimise -eps
    obj = np.zeros(n + 1)
    obj[-1] = -1.0

    a_ub = np.zeros((len(subsets), n + 1))
    b_ub = np.zeros(len(subsets))
    for row, subset in enumerate(subsets):
        for p in subset:
            a_ub[row, index[p]] = -1.0
        a_ub[row, -1] = 1.0
        b_ub[row] = -v(subset)

    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    b_eq = np.array([v(grand)])

    res = linprog(
        obj,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * (n + 1),
        method="highs",
    )
    if not res.success:
        return CoreResult(
            epsilon=float("nan"),
            allocation={p: float("nan") for p in players},
            core_nonempty=False,
            binding=(),
            status=str(res.message),
        )
    x = {p: float(res.x[index[p]]) for p in players}
    eps = float(res.x[-1])
    binding = tuple(
        s
        for s in subsets
        if abs(sum(x[p] for p in s) - v(s) - eps) < 1e-7
    )
    return CoreResult(
        epsilon=eps,
        allocation=x,
        core_nonempty=eps >= -1e-9,
        binding=binding,
        status="optimal",
    )


def least_core_generated(
    players: Sequence[Player],
    v: ValueFn,
    max_rounds: int = 30,
    seed_coalitions: Sequence[frozenset[Player]] | None = None,
) -> CoreResult:
    """Least core by **constraint generation**, for games too large to enumerate.

    Starts from the singletons, solves, then adds the most-violated coalition found by
    exhaustive separation over the coalitions that have already been evaluated.  Exact when
    it terminates without finding a violated constraint.
    """
    active: list[frozenset[Player]] = list(
        seed_coalitions if seed_coalitions is not None else [frozenset({p}) for p in players]
    )
    n = len(players)
    result = least_core(players, v, active)
    for _ in range(max_rounds):
        worst: tuple[float, frozenset[Player]] | None = None
        for size in range(1, n):
            for combo in itertools.combinations(players, size):
                s = frozenset(combo)
                if s in active:
                    continue
                slack = sum(result.allocation[p] for p in s) - v(s) - result.epsilon
                if worst is None or slack < worst[0]:
                    worst = (slack, s)
        if worst is None or worst[0] >= -1e-9:
            break
        active.append(worst[1])
        result = least_core(players, v, active)
    return result


def nucleolus(
    players: Sequence[Player],
    v: ValueFn,
    max_levels: int = 12,
) -> CoreResult:
    """Nucleolus by the standard sequence of LPs (exact; intended for ``n <= 12``).

    Solve the least-core LP; freeze the coalitions whose excess is tight at the optimum as
    equalities; re-solve on the remainder; repeat until no free coalition remains.  The
    result lexicographically maximises the sorted vector of excesses, which is the
    nucleolus of a savings game.
    """
    n = len(players)
    index = {p: i for i, p in enumerate(players)}
    grand = frozenset(players)
    every = [
        frozenset(c)
        for size in range(1, n)
        for c in itertools.combinations(players, size)
    ]
    fixed: list[tuple[frozenset[Player], float]] = []
    free = list(every)
    x: dict[Player, float] = {p: 0.0 for p in players}
    eps = float("nan")
    first_eps = float("nan")

    for _ in range(max_levels):
        if not free:
            break
        obj = np.zeros(n + 1)
        obj[-1] = -1.0
        a_ub = np.zeros((len(free), n + 1))
        b_ub = np.zeros(len(free))
        for row, subset in enumerate(free):
            for p in subset:
                a_ub[row, index[p]] = -1.0
            a_ub[row, -1] = 1.0
            b_ub[row] = -v(subset)
        rows_eq = [np.concatenate([np.ones(n), [0.0]])]
        vals_eq = [v(grand)]
        for subset, level in fixed:
            row = np.zeros(n + 1)
            for p in subset:
                row[index[p]] = 1.0
            rows_eq.append(row)
            vals_eq.append(v(subset) + level)
        res = linprog(
            obj,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=np.array(rows_eq),
            b_eq=np.array(vals_eq),
            bounds=[(None, None)] * (n + 1),
            method="highs",
        )
        if not res.success:
            break
        x = {p: float(res.x[index[p]]) for p in players}
        eps = float(res.x[-1])
        if np.isnan(first_eps):
            # The first level *is* the least-core programme; later levels maximise the
            # excess of the coalitions still free and therefore return larger values.
            # Reporting a later level as "the" epsilon would overstate stability.
            first_eps = eps
        tight = [s for s in free if abs(sum(x[p] for p in s) - v(s) - eps) < 1e-7]
        if not tight:
            break
        for s in tight:
            fixed.append((s, eps))
            free.remove(s)
    return CoreResult(
        epsilon=first_eps,
        allocation=x,
        core_nonempty=first_eps >= -1e-9,
        binding=tuple(s for s, _ in fixed),
        status="nucleolus",
    )


def is_in_core(
    allocation: dict[Player, float],
    players: Sequence[Player],
    v: ValueFn,
    tol: float = 1e-7,
) -> tuple[bool, float]:
    """Is ``allocation`` core-stable?  Returns the verdict and the worst excess deficit.

    Used to report *the fraction of instances on which the Shapley value is core-stable* --
    the headline number of T5 alongside the least-core radius.
    """
    n = len(players)
    worst = float("inf")
    for size in range(1, n):
        for combo in itertools.combinations(players, size):
            s = frozenset(combo)
            worst = min(worst, sum(allocation[p] for p in s) - v(s))
    total_ok = abs(sum(allocation.values()) - v(frozenset(players))) < 1e-6
    return (worst >= -tol and total_ok), worst
