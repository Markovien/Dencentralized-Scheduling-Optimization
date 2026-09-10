"""THE objective.  Single source of truth for ``Phi``.

Every method in this repository -- ``EX-CP``, the dispatching baselines, the
non-cooperative game ``M1``, the cooperative game ``M2``, the RL environment -- computes
its objective by calling this module.  Nothing re-implements it anywhere.  That rule is
what makes the cross-method comparison tables meaningful, and it is the guard against
finding **F6** recurring (in the draft, the centralised model priced *robot charging only*
while the game and the RL reward priced *machines plus charging*, so no "MILP vs game"
table could have compared the same quantity).

    Phi(w) = w * Chat_max + (1 - w) * Ehat_cost + M * max(0, Chat_max - 1)

The penalty term is the deadline (ROADMAP.md §3.0, §3.4).  ``Chat_max = 1`` is *exactly*
the deadline, so the penalty is zero on every deadline-feasible outcome and finite -- not
``+inf`` -- beyond it.  Finiteness matters downstream: a characteristic function that can
take the value ``+inf`` has no Shapley value, no core LP and no least-core radius, which
would remove the whole cooperative apparatus on exactly the instances where coordination
matters most.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jsspt_tou.domain.anchors import Anchors

DEADLINE_PENALTY_DEFAULT: Final[float] = 10.0
"""``M`` in the penalised objective.

Calibrated from the instance, not by taste: ``M >= 1`` in normalised terms already makes
any deadline-feasible outcome preferable to any infeasible one, because the entire
normalised energy range spans 1.  ``M = 10`` is the default and experiment ``E11``
ablates ``M in {2, 5, 10, 50}``; an allocation that moves with ``M`` is an artefact, not a
result.
"""


@dataclass(frozen=True, slots=True)
class Outcome:
    """The measured outcome of one complete schedule.

    Attributes
    ----------
    c_max:
        Makespan [min], to the last **return to L/U** (the Bilge--Ulusoy convention).
    e_cost:
        Total ToU energy cost [EUR]: machine processing **plus** robot charging.
    energy_kwh:
        Total energy [kWh] -- machine processing plus charging.
    peak_kwh:
        Energy drawn during the dearest tariff band, for the peak-share metric.
    charge_blocks:
        Number of charging blocks executed by the fleet.
    deadline_met:
        Whether ``c_max <= H``.
    used_fallback:
        Whether the deadline-feasibility safeguard had to fire during the episode.
        Reported as the *deadlock rate* -- a method that meets the deadline only by
        falling back most of the time is not a scheduler (ROADMAP.md §3.0(2)).
    """

    c_max: float
    e_cost: float
    energy_kwh: float
    peak_kwh: float
    charge_blocks: int
    deadline_met: bool
    used_fallback: bool = False

    def norm_cmax(self, anchors: Anchors) -> float:
        return anchors.norm_cmax(self.c_max)

    def norm_ecost(self, anchors: Anchors) -> float:
        return anchors.norm_ecost(self.e_cost)


def phi(
    outcome: Outcome,
    anchors: Anchors,
    omega: float,
    penalty: float = DEADLINE_PENALTY_DEFAULT,
) -> float:
    """The global cost ``Phi``, on normalised quantities, with the deadline penalty.

    Parameters
    ----------
    outcome:
        Measured schedule outcome.
    anchors:
        Instance-fixed anchors.  **Must** be the ones computed at instance load; passing
        anchors recomputed mid-episode silently invalidates T1 and T7 (see
        :mod:`jsspt_tou.domain.anchors`).
    omega:
        Trade-off weight ``w in [0, 1]``.  ``w -> 1`` prioritises makespan.
    penalty:
        ``M`` in the deadline penalty.
    """
    c_hat = anchors.norm_cmax(outcome.c_max)
    e_hat = anchors.norm_ecost(outcome.e_cost)
    return omega * c_hat + (1.0 - omega) * e_hat + penalty * max(0.0, c_hat - 1.0)


def phi_monetised(outcome: Outcome, cost_of_time_eur_per_min: float) -> float:
    """The managerial variant ``Phi_EUR = c_time * C_max + E_cost`` (ROADMAP.md §3.7.4).

    One unit throughout, no ``omega``, and ``c_time`` is a quantity a plant actually knows.
    Used only in Paper A §9, where the break-even ``c_time`` at which the cost-optimal
    schedule flips from off-peak-seeking to deadline-seeking is the headline managerial
    number.
    """
    return cost_of_time_eur_per_min * outcome.c_max + outcome.e_cost


def welfare(
    outcome: Outcome,
    idle_outcome: Outcome,
    anchors: Anchors,
    omega: float,
    penalty: float = DEADLINE_PENALTY_DEFAULT,
) -> float:
    """Welfare ``W(alpha) = Phi(alpha^0) - Phi(alpha) >= 0``.

    The **single sign convention** of the whole project (ROADMAP.md §3.3, repairing
    finding **F4**, errata A11--A12): welfare is a *cost reduction* relative to the
    all-idle profile, agent utility is a marginal contribution to welfare, and **every
    agent maximises**.  The draft defined ``u_p`` as a cost *increase* in one section and
    a cost *reduction* in another, then minimised one and maximised the other.
    """
    return phi(idle_outcome, anchors, omega, penalty) - phi(
        outcome, anchors, omega, penalty
    )
