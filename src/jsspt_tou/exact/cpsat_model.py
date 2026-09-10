"""``EX-CP`` — the exact CP-SAT reference, as an **adapter** over the author's solver.

Implements Paper A §4 (ROADMAP.md §3.2).  This module *extends* the classical JSSPT model
in ``external/jsspt_cpsat_original/solve_jsspt_cp.py``; it does not reimplement it.  The
skeleton is the original's -- interval variables per operation, ``NoOverlap`` per machine,
a transport task before every operation, a final return leg to the L/U station, exactly one
robot per transport, sequence-dependent **empty travel** between consecutive transports --
and on top of it ``EX-CP`` adds the five things the draft's models lack:

======================================  ==========  ==============================================
Addition                                Finding     Why the original cannot carry it unchanged
======================================  ==========  ==============================================
ToU-indexed **processing** cost         F6 / A8     the original has no cost term at all
Charging as optional intervals          §3.2.1      no battery in the original
``Cumulative`` charger capacity K_CH    F7 / A9     no charger in the original
**Exact prefix SoC** along the route    F7 / A10    the draft's aggregate balance ignores ordering
Hard deadline ``C_max <= H``            F11 / A30   the original minimises C_max unconstrained
======================================  ==========  ==============================================

One constraint block is **ported rather than reused**: the original sequences transports on
a robot with pairwise big-M disjunctions (``solve_jsspt_cp.py`` lines 317--364).  That form
cannot express a *prefix* state such as state of charge, because it fixes only relative
order, never adjacency, and it cannot interleave charging visits with transports.  ``EX-CP``
therefore replaces it with a per-robot ``AddCircuit`` over {depot} + transports + that
robot's charging slots.  Circuit arcs give adjacency directly, so empty travel, idle waiting
and the SoC recursion are all exact along the route rather than relaxed.  Recorded in
``DECISIONS.md``; the flat-tariff regression in ``tests/test_exact_regression.py`` proves
the replacement is faithful by reproducing the original's makespan exactly.

Integer arithmetic.  Times are integer minutes, charge is integer mAh, and money is integer
micro-euros, so CP-SAT never leaves integer arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model

from jsspt_tou.domain.anchors import Anchors, compute_anchors
from jsspt_tou.domain.instance import Instance
from jsspt_tou.domain.objective import Outcome

MICRO: int = 1_000_000
"""Money scale: one unit is 1e-6 EUR."""


@dataclass(frozen=True, slots=True)
class ExactResult:
    """What one ``EX-CP`` solve produced."""

    status: str
    objective: float | None
    best_bound: float | None
    outcome: Outcome | None
    wall_clock: float
    truncated: bool
    n_variables: int
    n_constraints: int

    @property
    def gap(self) -> float | None:
        if self.objective is None or self.best_bound is None:
            return None
        if abs(self.objective) < 1e-12:
            return 0.0
        return abs(self.objective - self.best_bound) / max(abs(self.objective), 1e-12)


def solve_exact(
    inst: Instance,
    omega: float = 0.5,
    time_limit_s: float = 60.0,
    num_workers: int = 8,
    enforce_deadline: bool = True,
    enforce_battery: bool = True,
    max_charges_per_robot: int | None = None,
    anchors: Anchors | None = None,
    flat_tariff: bool = False,
    hint: dict[str, dict[Any, float]] | None = None,

) -> ExactResult:
    """Solve one instance exactly (or to a bound) under the shared objective.

    Parameters
    ----------
    omega:
        Trade-off weight.  The objective minimised is the **same normalised** ``Phi`` the
        simulator scores, so an ``EX-CP`` row and a game row in one table are comparable.
    enforce_deadline, enforce_battery:
        Switches used by the regression tests: turning both off and flattening the tariff
        must reproduce the vendored solver's makespan exactly.
    flat_tariff:
        Price every period at the cheapest rate.  The energy term is then
        schedule-invariant, which is what makes the regression test meaningful.
    hint:
        Optional warm start from :func:`hint_from_schedule` -- typically the ``M1``
        equilibrium schedule.  Hinting the operation and transport start times gives
        CP-SAT an incumbent it would otherwise spend most of its budget looking for; on
        the battery-constrained model that is the difference between a usable bound and a
        useless one.  A hint can never change the optimum, only the time to reach it.
    """
    anchors = anchors if anchors is not None else compute_anchors(inst)
    bat = inst.battery
    horizon = int(math.ceil(inst.deadline)) if enforce_deadline else _loose_horizon(inst)

    model = cp_model.CpModel()

    # ---- operations: intervals + machine NoOverlap  (as in the original) ------------------
    op_start: dict[tuple[int, int], Any] = {}
    op_end: dict[tuple[int, int], Any] = {}
    machine_intervals: dict[int, list[Any]] = {
        m: [] for m in range(1, inst.n_machines + 1)
    }
    for i, ops in enumerate(inst.jobs):
        for k, op in enumerate(ops):
            dur = int(op.duration)
            s = model.new_int_var(0, horizon, f"S_{i}_{k}")
            e = model.new_int_var(0, horizon, f"C_{i}_{k}")
            iv = model.new_interval_var(s, dur, e, f"I_{i}_{k}")
            op_start[(i, k)] = s
            op_end[(i, k)] = e
            machine_intervals[op.machine].append(iv)
    for m in range(1, inst.n_machines + 1):
        model.add_no_overlap(machine_intervals[m])

    # ---- transport tasks (as in the original) ----------------------------------------------
    tasks = list(inst.transports)
    n_t = len(tasks)
    t_start = [model.new_int_var(0, horizon, f"TS_{t}") for t in range(n_t)]
    t_end = [model.new_int_var(0, horizon, f"TC_{t}") for t in range(n_t)]
    for t, task in enumerate(tasks):
        model.add(t_end[t] == t_start[t] + int(task.duration))

    index_of = {(task.job, task.index): t for t, task in enumerate(tasks)}

    # ---- precedence: transport before processing, processing before next transport ----------
    for i, ops in enumerate(inst.jobs):
        for k in range(len(ops)):
            t = index_of[(i, k)]
            model.add(op_start[(i, k)] >= t_end[t])
            if k > 0:
                model.add(t_start[t] >= op_end[(i, k - 1)])
        ret = index_of[(i, len(ops))]
        model.add(t_start[ret] >= op_end[(i, len(ops) - 1)])

    # ---- charging slots: optional intervals, one Cumulative charger ------------------------
    n_charge = (
        max_charges_per_robot
        if max_charges_per_robot is not None
        else _charge_slot_bound(inst, horizon)
    )
    if not enforce_battery:
        n_charge = 0
    block = int(bat.charge_block_min)
    ch_start: list[list[Any]] = []
    ch_end: list[list[Any]] = []
    ch_used: list[list[Any]] = []
    charge_intervals: list[Any] = []
    for r in range(inst.n_robots):
        starts, ends, used = [], [], []
        for c in range(n_charge):
            lit = model.new_bool_var(f"chuse_{r}_{c}")
            s = model.new_int_var(0, horizon, f"chS_{r}_{c}")
            e = model.new_int_var(0, horizon, f"chE_{r}_{c}")
            model.add(e == s + block)
            iv = model.new_optional_interval_var(s, block, e, lit, f"chI_{r}_{c}")
            charge_intervals.append(iv)
            starts.append(s)
            ends.append(e)
            used.append(lit)
            if c > 0:
                # symmetry break: slot c only after slot c-1, and in time order
                model.add_implication(lit, used[c - 1])
                model.add(s >= ends[c - 1]).only_enforce_if(lit)
        ch_start.append(starts)
        ch_end.append(ends)
        ch_used.append(used)
    if charge_intervals:
        # The charger constraint absent from every model in the draft (F7 / A9).
        model.add_cumulative(
            charge_intervals,
            [1] * len(charge_intervals),
            inst.n_chargers,
        )

    # ---- routing: one AddCircuit per robot over depot + transports + its charge slots --------
    # Ported (not reused) from the original's pairwise big-M block; see the module docstring.
    n_nodes = 1 + n_t + n_charge
    visit: list[list[Any]] = []  # visit[r][node] -- node served by robot r
    soc: list[list[Any]] = []
    for r in range(inst.n_robots):
        arcs: list[tuple[int, int, Any]] = []
        node_visit = [model.new_bool_var(f"vis_{r}_{n}") for n in range(n_nodes)]
        node_soc = [
            model.new_int_var(int(bat.floor_mah), int(bat.ceiling_mah), f"soc_{r}_{n}")
            for n in range(n_nodes)
        ]
        model.add(node_soc[0] == int(bat.start_mah))
        # A robot that serves nothing takes the empty circuit: AddCircuit accepts it only
        # when *every* node, the depot included, carries a true self-loop literal.
        arcs.append((0, 0, node_visit[0].negated()))
        for n in range(1, n_nodes):
            model.add_implication(node_visit[n], node_visit[0])
            arcs.append((n, n, node_visit[n].negated()))  # self-loop == not visited
            if n > n_t:  # charge node
                model.add(node_visit[n] == ch_used[r][n - n_t - 1])
        for a in range(n_nodes):
            for b in range(n_nodes):
                if a == b:
                    continue
                lit = model.new_bool_var(f"arc_{r}_{a}_{b}")
                arcs.append((a, b, lit))
                model.add_implication(lit, node_visit[a])
                model.add_implication(lit, node_visit[b])
                _link_arc(
                    model,
                    inst,
                    lit,
                    a,
                    b,
                    r,
                    n_t,
                    tasks,
                    t_start,
                    t_end,
                    ch_start,
                    ch_end,
                    node_soc,
                    horizon,
                    enforce_battery,
                )
        model.add_circuit(arcs)
        visit.append(node_visit)
        soc.append(node_soc)

    # every transport served by exactly one robot (the original's ExactlyOne)
    for t in range(n_t):
        model.add_exactly_one([visit[r][1 + t] for r in range(inst.n_robots)])

    # ---- makespan: last return to L/U (Bilge--Ulusoy convention) -----------------------------
    c_max = model.new_int_var(0, horizon, "Cmax")
    model.add_max_equality(
        c_max, [t_end[index_of[(i, len(ops))]] for i, ops in enumerate(inst.jobs)]
    )
    if enforce_deadline:
        model.add(c_max <= int(inst.deadline))  # A30: the deadline, finally a constraint

    # ---- ToU energy cost: processing + charging ---------------------------------------------
    periods = _period_table(inst, horizon, flat_tariff)
    cost_terms: list[Any] = []
    for i, ops in enumerate(inst.jobs):
        for k, op in enumerate(ops):
            cost_terms += _tou_cost_terms(
                model,
                op_start[(i, k)],
                op_end[(i, k)],
                int(op.duration),
                inst.machine_power_kw,
                periods,
                f"op_{i}_{k}",
                horizon,
            )
    for r in range(inst.n_robots):
        for c in range(n_charge):
            cost_terms += _tou_cost_terms(
                model,
                ch_start[r][c],
                ch_end[r][c],
                block,
                bat.charger_power_kw,
                periods,
                f"ch_{r}_{c}",
                horizon,
                active=ch_used[r][c],
            )
    e_cost = model.new_int_var(0, 10 * MICRO * 1000, "Ecost")
    model.add(e_cost == sum(cost_terms) if cost_terms else e_cost == 0)

    # ---- objective: the SAME normalised Phi the simulator scores ----------------------------
    span_c = max(anchors.cmax_ub - anchors.cmax_lb, 1e-9)
    span_e = max(anchors.ecost_ub - anchors.ecost_lb, 1e-9)
    # Phi = w_c_f * C_max + w_e_f * E_micro - const, an affine function of the two integer
    # quantities, so CP-SAT minimises it exactly once the coefficients are integers.  The
    # common scale K is chosen from the *smaller* coefficient: fixing it in advance rounds
    # the energy weight to zero on these instances (span_e is O(1) EUR against E in
    # micro-euros), which silently makes energy free and the "bi-objective" model a pure
    # makespan model -- the very collapse finding F10 is about.
    w_c_f = omega / span_c
    w_e_f = (1.0 - omega) / (span_e * MICRO)
    smallest = min(x for x in (w_c_f, w_e_f) if x > 0.0) if max(w_c_f, w_e_f) > 0 else 1.0
    scale = 1.0e5 / smallest
    w_c = int(round(scale * w_c_f))
    w_e = int(round(scale * w_e_f))
    offset = omega * anchors.cmax_lb / span_c + (1.0 - omega) * anchors.ecost_lb / span_e
    model.minimize(w_c * c_max + w_e * e_cost)

    if hint:
        for (i, k), value in hint.get("ops", {}).items():
            if (i, k) in op_start:
                model.add_hint(op_start[(i, k)], int(value))
        for key, value in hint.get("transports", {}).items():
            if key in index_of:
                model.add_hint(t_start[index_of[key]], int(value))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(num_workers)
    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactResult(
            status=status_name,
            objective=None,
            best_bound=None,
            outcome=None,
            wall_clock=solver.wall_time,
            truncated=status_name != "INFEASIBLE",
            n_variables=0,
            n_constraints=0,
        )

    cmax_val = float(solver.value(c_max))
    ecost_val = solver.value(e_cost) / MICRO
    kwh = inst.machine_power_kw * inst.total_processing / 60.0 + sum(
        bat.charger_power_kw * block / 60.0
        for r in range(inst.n_robots)
        for c in range(n_charge)
        if solver.value(ch_used[r][c])
    )
    blocks = sum(
        int(solver.value(ch_used[r][c]))
        for r in range(inst.n_robots)
        for c in range(n_charge)
    )
    outcome = Outcome(
        c_max=cmax_val,
        e_cost=ecost_val,
        energy_kwh=kwh,
        peak_kwh=0.0,
        charge_blocks=blocks,
        deadline_met=cmax_val <= inst.deadline + 1e-9,
    )
    return ExactResult(
        status=status_name,
        objective=(
            omega * anchors.norm_cmax(cmax_val)
            + (1.0 - omega) * anchors.norm_ecost(ecost_val)
        ),
        best_bound=solver.best_objective_bound / scale - offset,
        outcome=outcome,
        wall_clock=solver.wall_time,
        truncated=status_name != "OPTIMAL",
        n_variables=len(model.proto.variables),
        n_constraints=len(model.proto.constraints),
    )


def hint_from_schedule(
    inst: Instance, log: list[tuple[str, int, int, float, float]]
) -> dict[str, dict[Any, float]]:
    """Turn a simulator event log into a CP-SAT warm start.

    Operations and transport legs of a job appear in the log in execution order, which is
    exactly their index order, so the mapping needs no extra bookkeeping.
    """
    ops: dict[Any, float] = {}
    transports: dict[Any, float] = {}
    seen_ops: dict[int, int] = {}
    seen_legs: dict[int, int] = {}
    for kind, _, job, start, _end in sorted(log, key=lambda e: (e[3], e[0])):
        if kind == "process":
            k = seen_ops.get(job, 0)
            ops[(job, k)] = start
            seen_ops[job] = k + 1
        elif kind == "transport":
            k = seen_legs.get(job, 0)
            transports[(job, k)] = start
            seen_legs[job] = k + 1
    return {"ops": ops, "transports": transports}


# --------------------------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------------------------
def _loose_horizon(inst: Instance) -> int:
    """A horizon large enough not to constrain anything (regression mode)."""
    max_sigma = max(max(row) for row in inst.sigma)
    return int(inst.total_processing + len(inst.transports) * max_sigma + 10)


def _charge_slot_bound(inst: Instance, horizon: int) -> int:
    """Valid upper bound on charging visits per robot.

    The naive bound "loaded rate sustained over the whole horizon" is valid but far too
    loose -- on EX11 it allows six blocks per robot where two suffice, and the resulting
    model does not close in three minutes.  A robot's discharge over an episode is bounded
    by

        loaded_rate * (all loaded travel) + empty_rate * (all loaded travel) + idle_rate * H

    -- loaded travel bounds the loaded time, the same figure bounds the repositioning that
    can precede it, and everything else is idling.  Subtracting the robot's usable reserve
    gives the charge it can possibly need.  Still valid, several times tighter.
    """
    bat = inst.battery
    transport = inst.total_transport
    drain = (
        bat.loaded_mah_min * transport
        + bat.empty_mah_min * transport
        + bat.idle_mah_min * horizon
    )
    need = max(0.0, drain - (bat.start_mah - bat.floor_mah))
    return max(1, int(math.ceil(need / bat.block_mah)))


def _link_arc(
    model: cp_model.CpModel,
    inst: Instance,
    lit: Any,
    a: int,
    b: int,
    robot: int,
    n_t: int,
    tasks: list[Any],
    t_start: list[Any],
    t_end: list[Any],
    ch_start: list[list[Any]],
    ch_end: list[list[Any]],
    node_soc: list[Any],
    horizon: int,
    enforce_battery: bool,
) -> None:
    """Time and SoC propagation along one circuit arc ``a -> b``.

    Because the arc gives **adjacency**, the empty leg, the idle wait and the state of
    charge are all exact -- which is precisely what the draft's aggregate energy balance
    could not express (F7): a robot can satisfy a total-energy inequality and still cross
    the SoC floor midway through its sequence.
    """
    bat = inst.battery
    drop_a = _drop_location(inst, a, n_t, tasks)
    pick_b = _pick_location(inst, b, n_t, tasks)
    empty = int(inst.travel(drop_a, pick_b))

    if b == 0:
        return  # returning to the depot constrains nothing further
    end_a = _node_end(a, n_t, t_end, ch_end, robot)
    start_b = _node_start(b, n_t, t_start, ch_start, robot)
    if end_a is not None:
        model.add(start_b >= end_a + empty).only_enforce_if(lit)
    else:  # leaving the depot at time zero
        model.add(start_b >= empty).only_enforce_if(lit)

    if not enforce_battery:
        return

    # SoC recursion: empty travel, then idle waiting, then the node's own activity.
    wait = model.new_int_var(0, horizon, f"wait_{robot}_{a}_{b}")
    if end_a is not None:
        model.add(wait == start_b - end_a - empty).only_enforce_if(lit)
    else:
        model.add(wait == start_b - empty).only_enforce_if(lit)

    drain_transit = int(bat.empty_mah_min) * empty
    if b <= n_t:  # transport node: loaded travel, no gain
        loaded = int(bat.loaded_mah_min * tasks[b - 1].duration)
        model.add(
            node_soc[b]
            == node_soc[a] - drain_transit - int(bat.idle_mah_min) * wait - loaded
        ).only_enforce_if(lit)
    else:  # charging node: idle-drain to the charger, then one block of gain, capped
        raw = model.new_int_var(
            -10 * int(bat.capacity_mah), int(bat.capacity_mah), f"raw_{robot}_{a}_{b}"
        )
        model.add(
            raw
            == node_soc[a]
            - drain_transit
            - int(bat.idle_mah_min) * wait
            + int(bat.block_mah)
        ).only_enforce_if(lit)
        model.add_min_equality(node_soc[b], [raw, int(bat.ceiling_mah)]).only_enforce_if(lit)


def _drop_location(inst: Instance, node: int, n_t: int, tasks: list[Any]) -> int:
    if node == 0:
        return inst.charger_location
    if node <= n_t:
        return int(tasks[node - 1].dest)
    return inst.charger_location


def _pick_location(inst: Instance, node: int, n_t: int, tasks: list[Any]) -> int:
    if node == 0:
        return inst.charger_location
    if node <= n_t:
        return int(tasks[node - 1].origin)
    return inst.charger_location


def _node_start(
    node: int, n_t: int, t_start: list[Any], ch_start: list[list[Any]], robot: int
) -> Any:
    if node <= n_t:
        return t_start[node - 1]
    return ch_start[robot][node - n_t - 1]


def _node_end(
    node: int, n_t: int, t_end: list[Any], ch_end: list[list[Any]], robot: int
) -> Any | None:
    if node == 0:
        return None
    if node <= n_t:
        return t_end[node - 1]
    return ch_end[robot][node - n_t - 1]


def _period_table(
    inst: Instance, horizon: int, flat: bool
) -> list[tuple[int, int, float]]:
    """Integer-bounded tariff periods, extended to cover the whole solver horizon."""
    price = inst.tariff.min_price
    rows: list[tuple[int, int, float]] = []
    for p in inst.tariff.periods:
        rows.append(
            (int(p.start), int(math.ceil(p.end)), price if flat else p.price)
        )
    last_end = rows[-1][1] if rows else 0
    if last_end < horizon:
        rows.append((last_end, horizon, price if flat else inst.tariff.periods[-1].price))
    return rows


def _tou_cost_terms(
    model: cp_model.CpModel,
    start: Any,
    end: Any,
    duration: int,
    power_kw: float,
    periods: list[tuple[int, int, float]],
    tag: str,
    horizon: int,
    active: Any | None = None,
) -> list[Any]:
    """Exact ToU cost of one activity as a linear expression over per-period overlaps.

    ``overlap_h = max(0, min(end, T_h) - max(start, T_{h-1}))``, encoded with min/max
    equalities.  This is the piecewise-constant integral of ROADMAP.md §3.0 and it is what
    puts **machine processing energy** into the exact model at all (F6 / A8): the draft's
    centralised model priced robot charging only, so no "MILP versus game" table could have
    compared the same quantity.
    """
    terms: list[Any] = []
    if duration <= 0 or power_kw <= 0.0:
        return terms
    for h, (lo_b, hi_b, price) in enumerate(periods):
        # Domains must admit the *unclipped* values: max(start, T_{h-1}) can run all the
        # way to the horizon and min(end, T_h) all the way down to zero.  Tightening them
        # to the period bounds silently makes the whole model infeasible.
        lo = model.new_int_var(lo_b, horizon, f"lo_{tag}_{h}")
        hi = model.new_int_var(0, hi_b, f"hi_{tag}_{h}")
        model.add_max_equality(lo, [start, lo_b])
        model.add_min_equality(hi, [end, hi_b])
        diff = model.new_int_var(-horizon, duration, f"df_{tag}_{h}")
        model.add(diff == hi - lo)
        ov = model.new_int_var(0, duration, f"ov_{tag}_{h}")
        model.add_max_equality(ov, [diff, 0])
        coeff = int(round(price * power_kw / 60.0 * MICRO))
        if coeff:
            terms.append(coeff * ov)
    if active is None or not terms:
        return terms
    # An *optional* activity (a charging slot that may go unused) keeps its start and end
    # variables whether or not it happens, so its per-period overlaps are still real: an
    # interval of positive length always overlaps some period of a profile that tiles the
    # horizon.  Zeroing those overlaps when the slot is unused is therefore unsatisfiable,
    # and forces every slot to be used -- which showed up as the exact model returning a
    # proven "optimum" worse than a feasible heuristic schedule.  Gate the *cost* instead.
    upper = sum(
        int(round(price * power_kw / 60.0 * MICRO)) * duration for _, _, price in periods
    )
    gated = model.new_int_var(0, max(upper, 1), f"cost_{tag}")
    model.add(gated == sum(terms)).only_enforce_if(active)
    model.add(gated == 0).only_enforce_if(active.negated())
    return [gated]
