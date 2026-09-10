"""``M2`` Layer 3 — the trade-off weight ``omega``, derived rather than assumed.

Implements Paper A §6, Layer 3 and contribution **C2** (ROADMAP.md §3.4).

``omega`` is the standard weak point of every scalarised bi-objective scheduling paper: it
is set by hand, and the sensitivity plot that follows is a caveat rather than a result.
Here it is *derived*.  Partition the agents into two interest blocs -- **production** (the
machines, whose payoff decreases in ``C_max``) and **logistics/energy** (the robots, whose
payoff decreases in ``E_cost``) -- take the feasible payoff set to be the Pareto front the
system can actually reach, and take the **disagreement point** to be the payoff pair
realised at the *non-cooperative equilibrium of M1*: what happens if the two blocs do not
coordinate is exactly the threat point.

    Nash bargaining:   argmax over the front of  (u_M - d_M)(u_V - d_V)
    Kalai-Smorodinsky: the front point where the two normalised gains are equal

Both are reported, and any disagreement between them is a discussion point rather than
something to hide.  NBS satisfies Pareto optimality, symmetry, scale invariance and
independence of irrelevant alternatives; KS replaces IIA with individual monotonicity.

Why this layer is immune to the commensurability problem
--------------------------------------------------------
Both solutions are **invariant under independent positive affine rescaling of each
player's utility**.  Whatever units or normalisation are chosen -- minutes or seconds,
euros or cents, raw or normalised -- Layer 3 returns the same operating point.  That is a
genuine argument for deriving ``omega`` rather than setting it, it is provable in two
lines, and it is why this layer, alone among the scalarised machinery, needed no repair
for finding F10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from jsspt_tou.domain.anchors import Anchors
from jsspt_tou.domain.objective import Outcome


@dataclass(frozen=True, slots=True)
class FrontPoint:
    """One attainable operating point, tagged with the weight that produced it."""

    omega: float
    c_max: float
    e_cost: float
    norm_cmax: float
    norm_ecost: float


@dataclass(frozen=True, slots=True)
class BargainingResult:
    """The derived operating point and the weight it induces."""

    solution: str
    point: FrontPoint
    omega_induced: float
    omega_interval: tuple[float, float]
    disagreement: tuple[float, float]
    product: float


def build_front(
    outcomes: Sequence[tuple[float, Outcome]], anchors: Anchors
) -> list[FrontPoint]:
    """Non-dominated ``(C_max, E_cost)`` points from an ``omega`` sweep."""
    points = [
        FrontPoint(
            omega=w,
            c_max=o.c_max,
            e_cost=o.e_cost,
            norm_cmax=anchors.norm_cmax(o.c_max),
            norm_ecost=anchors.norm_ecost(o.e_cost),
        )
        for w, o in outcomes
        if o.deadline_met
    ]
    front: list[FrontPoint] = []
    for p in points:
        dominated = any(
            q.c_max <= p.c_max
            and q.e_cost <= p.e_cost
            and (q.c_max < p.c_max or q.e_cost < p.e_cost)
            for q in points
        )
        if not dominated and not any(
            abs(q.c_max - p.c_max) < 1e-9 and abs(q.e_cost - p.e_cost) < 1e-9
            for q in front
        ):
            front.append(p)
    front.sort(key=lambda p: p.c_max)
    return front


def nadir_disagreement(front: Sequence[FrontPoint]) -> Outcome:
    """The threat point: what each bloc suffers when the other dictates the schedule.

    The roadmap names the M1 equilibrium as the disagreement point.  Taken literally at a
    single ``omega`` that point is *itself on the front*, so the bargaining set is empty and
    neither solution is defined.  The equilibrium payoffs that actually bound the
    negotiation are the two extremes of the equilibrium front: production's worst outcome
    is the makespan it suffers when logistics dictates (``omega = 0``), and logistics'
    worst is the energy bill it pays when production dictates (``omega = 1``).  That pair
    is the **nadir of the equilibrium front**, and it is the disagreement point used here.
    """
    return Outcome(
        c_max=max(p.c_max for p in front),
        e_cost=max(p.e_cost for p in front),
        energy_kwh=0.0,
        peak_kwh=0.0,
        charge_blocks=0,
        deadline_met=True,
    )


def nash_bargaining(
    front: Sequence[FrontPoint], disagreement: Outcome, anchors: Anchors
) -> BargainingResult | None:
    """Nash bargaining solution over the discrete front.

    Payoffs are *losses avoided* relative to the disagreement outcome:
    ``u_M = C_max^d - C_max`` for production and ``u_V = E_cost^d - E_cost`` for
    logistics.  The maximiser of the Nash product over the discrete front is reported; for
    a non-convex front this is the discrete NBS, and the convexified relaxation would only
    ever lie weakly above it.
    """
    d_c, d_e = disagreement.c_max, disagreement.e_cost
    best: tuple[float, FrontPoint] | None = None
    for p in front:
        u_m = d_c - p.c_max
        u_v = d_e - p.e_cost
        if u_m < 0 or u_v < 0:
            continue
        product = u_m * u_v
        if best is None or product > best[0]:
            best = (product, p)
    if best is None:
        return None
    lo, hi = supporting_weights(front, best[1])
    return BargainingResult(
        solution="NBS",
        point=best[1],
        omega_induced=0.5 * (lo + hi),
        omega_interval=(lo, hi),
        disagreement=(d_c, d_e),
        product=best[0],
    )


def kalai_smorodinsky(
    front: Sequence[FrontPoint], disagreement: Outcome, anchors: Anchors
) -> BargainingResult | None:
    """Kalai--Smorodinsky solution: equal normalised gains for the two blocs."""
    if not front:
        return None
    d_c, d_e = disagreement.c_max, disagreement.e_cost
    ideal_m = d_c - min(p.c_max for p in front)
    ideal_v = d_e - min(p.e_cost for p in front)
    if ideal_m <= 0 or ideal_v <= 0:
        return None
    best: tuple[float, FrontPoint] | None = None
    for p in front:
        r_m = (d_c - p.c_max) / ideal_m
        r_v = (d_e - p.e_cost) / ideal_v
        if r_m < 0 or r_v < 0:
            continue
        gap = abs(r_m - r_v)
        if best is None or gap < best[0]:
            best = (gap, p)
    if best is None:
        return None
    lo, hi = supporting_weights(front, best[1])
    return BargainingResult(
        solution="KS",
        point=best[1],
        omega_induced=0.5 * (lo + hi),
        omega_interval=(lo, hi),
        disagreement=(d_c, d_e),
        product=(d_c - best[1].c_max) * (d_e - best[1].e_cost),
    )


def supporting_weights(
    front: Sequence[FrontPoint], point: FrontPoint, grid: int = 2001
) -> tuple[float, float]:
    """The interval of ``omega`` for which ``point`` minimises the normalised scalarisation.

    This is the supporting hyperplane of the front at the bargaining solution, read off in
    weight space: it is what converts an operating point into the weight a plant should
    actually use.  An empty interval (returned as a degenerate one at the closest weight)
    means the point is *unsupported* -- it lies in a non-convex pocket of the front and no
    weighted sum can reach it, which is itself worth reporting.
    """
    weights = [i / (grid - 1) for i in range(grid)]
    hits = []
    for w in weights:
        best = min(front, key=lambda p: w * p.norm_cmax + (1 - w) * p.norm_ecost)
        if (
            abs(best.norm_cmax - point.norm_cmax) < 1e-12
            and abs(best.norm_ecost - point.norm_ecost) < 1e-12
        ):
            hits.append(w)
    if not hits:
        closest = min(
            weights,
            key=lambda w: (w * point.norm_cmax + (1 - w) * point.norm_ecost)
            - min(w * p.norm_cmax + (1 - w) * p.norm_ecost for p in front),
        )
        return (closest, closest)
    return (min(hits), max(hits))


def breakeven_time_cost(front: Sequence[FrontPoint]) -> float | None:
    """The value of shop time ``c_time`` [EUR/min] at which the cheapest schedule flips.

    The managerial variant of ROADMAP.md §3.7.4: with ``Phi_EUR = c_time * C_max + E_cost``
    there is no weight at all, and the break-even ``c_time`` -- below which the cost-optimal
    schedule is off-peak-seeking and above which it is deadline-seeking -- is a number a
    plant actually knows how to argue about.  It is the headline managerial figure of
    Paper A §9.
    """
    if len(front) < 2:
        return None
    ordered = sorted(front, key=lambda p: p.c_max)
    cheapest_energy = min(ordered, key=lambda p: p.e_cost)
    fastest = ordered[0]
    if abs(cheapest_energy.c_max - fastest.c_max) < 1e-9:
        return None
    return (cheapest_energy.e_cost - fastest.e_cost) / (
        fastest.c_max - cheapest_energy.c_max
    )
