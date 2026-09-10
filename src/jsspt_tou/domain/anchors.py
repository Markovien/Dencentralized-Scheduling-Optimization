"""Normalisation anchors — closed-form, instance-fixed, computed once.

Implements ROADMAP.md §3.7 (Upper--Lower Bound / Nadir--Ideal scaling), which repairs
finding **F10** / erratum **A29**: the draft's ``Phi = w*C_max + (1-w)*E_cost`` adds
*minutes to euros*.  On the draft's own EX11 the terms differ by a factor of 110--159, so
at ``w = 0.5`` energy contributes under 1 % of the objective and the bi-objective study
silently collapses into a makespan study.  Every scalarised object in this project --
the game utilities, the coalition cost, the RL reward -- is computed on **normalised**
quantities instead:

    Chat_max  = (C_max  - C_max^LB) / (H - C_max^LB)
    Ehat_cost = (E_cost - E^LB)     / (E^UB - E^LB)
    Phi       = w * Chat_max + (1 - w) * Ehat_cost

Invariance condition (guards T1 and T7)
---------------------------------------
The anchors must be computed **once per instance, before any game or episode begins, and
never updated**.  Recomputing them from running best-known values -- the tempting
"adaptive normalisation" -- makes ``Phi`` a function of history rather than of the joint
action profile: the exact-potential identity of **T1** fails, the finite-improvement
convergence argument fails with it, and the telescoping of **T7** no longer sums to
``-Phi``.  :class:`Anchors` is frozen and ``tests/test_scalarisation.py`` asserts the
anchors are identical at episode start and end.

None of the four anchors needs a solver call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jsspt_tou.domain.instance import Instance


@dataclass(frozen=True, slots=True)
class Anchors:
    """The four instance-fixed normalisation anchors, plus the certified feasibility margin."""

    cmax_lb: float
    cmax_ub: float  # = the deadline H
    ecost_lb: float
    ecost_ub: float
    kwh_lb: float
    kwh_ub: float
    charge_blocks_lb: int
    charge_blocks_ub: int

    @property
    def feasibility_margin(self) -> float:
        """``H / C_max^LB``.

        ROADMAP.md §3.0(1): every instance must carry this margin, and WP4 rejects or
        re-scales any instance whose margin falls outside ``[1.05, 2.0]``.  A margin below
        1 means the deadline admits *no* feasible schedule -- a generation defect, not a
        result.
        """
        return self.cmax_ub / self.cmax_lb if self.cmax_lb > 0 else math.inf

    def norm_cmax(self, c_max: float) -> float:
        """Normalised makespan.  ``0`` at the lower bound, ``1`` exactly at the deadline."""
        span = self.cmax_ub - self.cmax_lb
        if span <= 0:
            return 0.0
        return (c_max - self.cmax_lb) / span

    def norm_ecost(self, e_cost: float) -> float:
        """Normalised energy cost.  Measures *where in the tariff profile the load sits*."""
        span = self.ecost_ub - self.ecost_lb
        if span <= 0:
            return 0.0
        return (e_cost - self.ecost_lb) / span


def compute_anchors(inst: Instance) -> Anchors:
    """Closed-form anchors for one instance.  Call once, at load time.

    ``C_max^LB`` is the maximum of three valid lower bounds (ROADMAP.md §3.7.3):

    1. the **job critical path**, including every transport leg and the final return;
    2. the **one-machine head/tail bound**: for each machine, the earliest any of its
       operations can start, plus all of its processing, plus the shortest remaining work
       after its last operation.  Strictly tighter than the plain machine-load bound and
       still valid, because the operations on a machine must be sequenced;
    3. the **fleet work bound**, total transport plus unavoidable charging time spread
       over the robots -- valid because some robot carries at least the average.

    ``E^LB``/``E^UB`` price the same reference kWh at the cheapest and dearest tariff.
    Machine processing energy is schedule-invariant (idle and standby power are excluded
    by assumption), so the only schedule-dependent part of the reference kWh is the number
    of charging blocks, bounded below by the fleet's unavoidable discharge and above by the
    fastest possible discharge (the loaded rate sustained over the whole horizon).
    """
    bat = inst.battery

    # (1) job critical paths, and the head/tail of every operation ----------------------------
    critical = 0.0
    head: dict[tuple[int, int], float] = {}
    tail: dict[tuple[int, int], float] = {}
    for i, ops in enumerate(inst.jobs):
        legs = inst.transports_of_job(i)  # legs[k] precedes ops[k]; legs[-1] is the return
        path = sum(t.duration for t in legs) + sum(op.duration for op in ops)
        critical = max(critical, path)
        acc = 0.0
        for k, op in enumerate(ops):
            acc += legs[k].duration
            head[(i, k)] = acc  # earliest possible start of op (i, k)
            acc += op.duration
        # tail(i,k) = sum_{k' > k} (leg_{k'} + p_{k'}) + return leg
        suffix = legs[-1].duration
        for k in range(len(ops) - 1, -1, -1):
            tail[(i, k)] = suffix
            suffix += ops[k].duration + legs[k].duration

    # (2) one-machine head/tail bound ---------------------------------------------------------
    machine_load = 0.0
    for m in range(1, inst.n_machines + 1):
        keys = [
            (op.job, op.index) for op in inst.operations if op.machine == m
        ]
        if not keys:
            continue
        load = sum(
            op.duration for op in inst.operations if op.machine == m
        )
        bound = min(head[k] for k in keys) + load + min(tail[k] for k in keys)
        machine_load = max(machine_load, bound)

    # (3) fleet work bound -------------------------------------------------------------------
    loaded_discharge = sum(t.duration for t in inst.transports) * bat.loaded_mah_min
    fleet_reserve = inst.n_robots * (bat.start_mah - bat.floor_mah)
    blocks_lb = max(
        0, int(math.ceil((loaded_discharge - fleet_reserve) / bat.block_mah - 1e-9))
    )
    fleet_work = (
        inst.total_transport + blocks_lb * bat.charge_block_min
    ) / inst.n_robots

    cmax_lb = max(critical, machine_load, fleet_work)

    # energy anchors --------------------------------------------------------------------------
    proc_kwh = inst.machine_power_kw * inst.total_processing / 60.0
    # A robot cannot discharge faster than the loaded rate, so over the horizon it can
    # absorb at most this many blocks; this is what keeps Ehat_cost inside [0, 1].
    per_robot_blocks_ub = int(
        math.ceil(bat.loaded_mah_min * inst.horizon / bat.block_mah)
    )
    blocks_ub = inst.n_robots * per_robot_blocks_ub
    kwh_lb = proc_kwh + blocks_lb * bat.block_kwh
    kwh_ub = proc_kwh + blocks_ub * bat.block_kwh

    return Anchors(
        cmax_lb=cmax_lb,
        cmax_ub=inst.deadline,
        ecost_lb=kwh_lb * inst.tariff.min_price,
        ecost_ub=kwh_ub * inst.tariff.max_price,
        kwh_lb=kwh_lb,
        kwh_ub=kwh_ub,
        charge_blocks_lb=blocks_lb,
        charge_blocks_ub=blocks_ub,
    )
