"""``M2`` Layer 2a — the characteristic cost function ``c(S)`` and the savings game ``v(S)``.

Implements Paper A §6 (ROADMAP.md §3.4).

    c(S) = Phi of the schedule produced when the members of S coordinate and every agent
           in N \\ S follows the fixed reference policy pi_0,
    c(0) = Phi(pi_0 everywhere),
    v(S) = c(0) - c(S),        v(0) = 0.

Two properties of this construction must be stated in the paper, because referees look for
exactly these admissions: it *fixes the externalities* from the complement (which is what
makes a partition-function-free TU game legitimate here), and **``c(S)`` is a whole-system
cost for every ``S``, including singletons**.

    Do **not** define the savings game as ``sum_{i in S} c({i}) - c(S)``.  That form
    presupposes ``c({i})`` is a *standalone* cost.  Under the construction above it is a
    whole-system cost, so the sum double-counts roughly ``(|S| - 1) c(0)``: it gives
    ``v(0) = -c(0) != 0``, forces ``v({i}) = 0`` for every ``i`` by construction, and makes
    ``v(N)`` measure something that is not the total savings -- which the Shapley
    efficiency condition would then allocate.

The deadline makes ``c(S)`` finite, not ``+inf``
------------------------------------------------
A small coalition coordinating against a ``pi_0`` complement may have no deadline-feasible
completion.  A characteristic function taking the value ``+inf`` has no Shapley value, no
core LP and no least-core radius -- the entire Layer-2 apparatus would stop existing on
exactly the instances where coordination matters most.  :func:`jsspt_tou.domain.objective.phi`
therefore carries the **penalised** form ``Phi = w Chat + (1-w) Ehat + M max(0, Chat - 1)``
with finite ``M``: zero on deadline-feasible outcomes, finite beyond them.  Experiment E11
ablates ``M in {2, 5, 10, 50}`` -- an allocation that moves with ``M`` is an artefact.

Cost discipline
---------------
Every ``c(S)`` evaluation is itself a scheduling run, so the binding constraint is the
*number of evaluations*, not the Shapley combinatorics.  Three measures, all declared in
the paper: ``c(S)`` is evaluated by a **single sequential best-response pass** (not a full
optimisation); results are **memoised** across permutations, which is the dominant saving;
and the call count and wall-clock are reported.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from jsspt_tou.baselines.dispatching import DispatchingPolicy
from jsspt_tou.simulator.engine import Engine
from jsspt_tou.simulator.state import Action, Player, State


def all_players(engine: Engine) -> tuple[Player, ...]:
    """``N = M union V`` in the documented total order."""
    inst = engine.inst
    return tuple(
        [Player("machine", m) for m in range(1, inst.n_machines + 1)]
        + [Player("robot", r) for r in range(inst.n_robots)]
    )


@dataclass(slots=True)
class CoalitionPolicy:
    """Members of ``S`` coordinate; everyone else follows ``pi_0``.

    Members run a sequential best-response sweep on their marginal-contribution utilities
    while non-members' actions are held at their ``pi_0`` choices.  Because the stage game
    is an exact potential game with potential ``W`` (T1), best response on the individual
    MCU *is* coordinate descent on the coalition's contribution to the global cost -- the
    coalition needs no separate objective, which is a structural convenience worth stating.

    ``i_max = 1`` is the declared surrogate of ROADMAP.md §3.4: one sweep, not a full
    optimisation.  Its gap to ``EX-CP`` is measured on the small instances rather than
    assumed.
    """

    members: frozenset[Player]
    i_max: int = 1
    reference: DispatchingPolicy | None = None

    def __call__(
        self,
        engine: Engine,
        state: State,
        players: Sequence[Player],
        rng: np.random.Generator,
    ) -> dict[Player, Action]:
        alpha: dict[Player, Action] = {
            p: engine.reference_action(state, p, state.t) for p in sorted(players)
        }
        coord = [p for p in sorted(players) if p in self.members]
        if not coord:
            return alpha
        for _ in range(self.i_max):
            improved = False
            for p in coord:
                others = {q: a for q, a in alpha.items() if q != p}
                cand = engine.actions_or_fallback(state, p)
                if not cand:
                    continue
                us = engine.utilities(state, p, cand, others)
                best = cand[int(np.argmax(us))]
                if best != alpha[p]:
                    alpha[p] = best
                    improved = True
            if not improved:
                break
        return alpha


@dataclass(slots=True)
class PartitionPolicy:
    """Block-local coordination: each block of a coalition structure sweeps on its own.

    Used to *realise* a coalition structure produced by
    :mod:`jsspt_tou.cooperative.coalition_formation`.  It is not the same object as the
    grand coalition: a player best-responds only within its own block, while every other
    block's current choice is treated as given.  Blocks are processed in the documented
    total order, so the policy is a deterministic function of the state.

    This distinction matters for experiment E4.  ``sum_k v(S_k)`` is an *accounting* figure
    -- it adds savings that were each measured against a ``pi_0`` complement -- whereas
    running this policy measures what the partition actually delivers when all blocks act
    at once.  Reporting only the former would overstate the value of partitioning.
    """

    blocks: tuple[frozenset[Player], ...]
    i_max: int = 1

    def __call__(
        self,
        engine: Engine,
        state: State,
        players: Sequence[Player],
        rng: np.random.Generator,
    ) -> dict[Player, Action]:
        alpha: dict[Player, Action] = {
            p: engine.reference_action(state, p, state.t) for p in sorted(players)
        }
        for block in self.blocks:
            coord = [p for p in sorted(players) if p in block]
            if not coord:
                continue
            for _ in range(self.i_max):
                improved = False
                for p in coord:
                    others = {q: a for q, a in alpha.items() if q != p}
                    cand = engine.actions_or_fallback(state, p)
                    if not cand:
                        continue
                    us = engine.utilities(state, p, cand, others)
                    best = cand[int(np.argmax(us))]
                    if best != alpha[p]:
                        alpha[p] = best
                        improved = True
                if not improved:
                    break
        return alpha


@dataclass(slots=True)
class CharacteristicFunction:
    """Memoised ``c(S)`` and ``v(S)`` for one instance at one ``omega``.

    Attributes
    ----------
    calls:
        Number of *evaluated* (non-cached) coalition runs -- reported in the paper, so
        that a truncated certification budget cannot read as "we certified everything".
    """

    engine: Engine
    i_max: int = 1
    _cache: dict[frozenset[Player], float] = field(default_factory=dict)
    calls: int = 0
    seconds: float = 0.0

    @property
    def players(self) -> tuple[Player, ...]:
        return all_players(self.engine)

    def c(self, members: frozenset[Player]) -> float:
        """Coalition cost.  Deterministic, so repeated calls are free after the first."""
        if members in self._cache:
            return self._cache[members]
        t0 = time.perf_counter()
        policy = CoalitionPolicy(members=members, i_max=self.i_max)
        outcome = self.engine.run(policy, np.random.default_rng(0))
        value = self.engine.phi(outcome)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self._cache[members] = value
        return value

    def c_empty(self) -> float:
        return self.c(frozenset())

    def v(self, members: frozenset[Player]) -> float:
        """Savings ``v(S) = c(0) - c(S)``.  ``v(0) = 0`` by construction."""
        return self.c_empty() - self.c(members)

    def enumerate_all(self) -> dict[frozenset[Player], float]:
        """Evaluate every coalition.

        Tractable exactly where it matters: on the Bilge--Ulusoy family
        ``n = |M| + |V| = 4 + 2 = 6``, so ``2^6 = 64`` runs give the *exact* Shapley
        value, the *exact* least core and the *exact* nucleolus with no sampling error at
        all.  :mod:`jsspt_tou.cooperative.shapley` keeps the sampling estimator for the
        larger generated instances.
        """
        players = self.players
        for size in range(len(players) + 1):
            for combo in itertools.combinations(players, size):
                self.c(frozenset(combo))
        return dict(self._cache)

    def values_vector(self) -> dict[frozenset[Player], float]:
        """All savings values, assuming :meth:`enumerate_all` has been run."""
        base = self.c_empty()
        return {s: base - val for s, val in self._cache.items()}


def realised_partition_value(
    cf: "CharacteristicFunction", blocks: Sequence[frozenset[Player]]
) -> float:
    """Savings actually delivered by a coalition structure, ``c(0) - Phi(CS schedule)``."""
    policy = PartitionPolicy(blocks=tuple(blocks), i_max=cf.i_max)
    outcome = cf.engine.run(policy, np.random.default_rng(0))
    return cf.c_empty() - cf.engine.phi(outcome)


def infeasible_coalition_rate(
    cf: CharacteristicFunction, deadline_penalty_threshold: float | None = None
) -> float:
    """Fraction of evaluated coalitions whose schedule misses the deadline.

    Reported per instance (ROADMAP.md §3.4, obligation 3).  If it is high, the honest
    reading is that the coalition structure is organised around the *deadline* rather than
    around the *tariff*, and Paper A says so.
    """
    engine = cf.engine
    threshold = (
        deadline_penalty_threshold
        if deadline_penalty_threshold is not None
        else 1.0 + max(engine.omega, 1.0 - engine.omega)
    )
    if not cf._cache:
        return 0.0
    n_bad = sum(1 for value in cf._cache.values() if value > threshold)
    return n_bad / len(cf._cache)


def submodularity_report(
    cf: CharacteristicFunction,
    rng: np.random.Generator,
    n_triples: int = 400,
) -> dict[str, float]:
    """Sampling test for submodularity of the **cost** ``c``.

    Draws triples ``(A subset B, x not in B)`` and checks the diminishing-returns
    inequality in the *cost* direction

        c(A + x) - c(A)  >=  c(B + x) - c(B).

    Returns the violation rate and the worst violation magnitude.  Under charger
    congestion this test is *expected to fail* -- that failure is result **N5**, not a bug,
    and the correlation of the violation magnitude with charger utilisation is the evidence
    for the congestion mechanism (a figure in Paper A).
    """
    players = cf.players
    n = len(players)
    violations = 0
    worst = 0.0
    total = 0
    for _ in range(n_triples):
        mask_b = rng.random(n) < 0.5
        idx_out = [i for i in range(n) if not mask_b[i]]
        if not idx_out:
            continue
        x = players[int(rng.choice(idx_out))]
        b = frozenset(players[i] for i in range(n) if mask_b[i])
        keep = rng.random(len(b)) < 0.5
        a = frozenset(p for p, k in zip(sorted(b), keep) if k)
        delta_a = cf.c(a | {x}) - cf.c(a)
        delta_b = cf.c(b | {x}) - cf.c(b)
        total += 1
        gap = delta_b - delta_a  # > 0 means submodularity is violated
        if gap > 1e-9:
            violations += 1
            worst = max(worst, gap)
    return {
        "triples": float(total),
        "violation_rate": violations / total if total else 0.0,
        "worst_violation": worst,
    }


def monotonicity_report(
    cf: CharacteristicFunction, rng: np.random.Generator, n_pairs: int = 400
) -> dict[str, float]:
    """Check ``v(S) <= v(S')`` for ``S subset S'`` (T6, unconditional part).

    Monotonicity holds *by construction* for the exact characteristic function -- a larger
    coalition can replicate the smaller one's policy.  It need not hold for the
    single-sweep **surrogate** used here, since a greedy sweep with more players is not
    guaranteed to dominate one with fewer.  The violation rate of the surrogate is
    therefore measured and reported rather than asserted away.
    """
    players = cf.players
    n = len(players)
    violations = 0
    worst = 0.0
    total = 0
    for _ in range(n_pairs):
        mask = rng.random(n) < 0.5
        big = frozenset(players[i] for i in range(n) if mask[i])
        if not big:
            continue
        keep = rng.random(len(big)) < 0.6
        small = frozenset(p for p, k in zip(sorted(big), keep) if k)
        total += 1
        gap = cf.v(small) - cf.v(big)
        if gap > 1e-9:
            violations += 1
            worst = max(worst, gap)
    return {
        "pairs": float(total),
        "violation_rate": violations / total if total else 0.0,
        "worst_violation": worst,
    }
