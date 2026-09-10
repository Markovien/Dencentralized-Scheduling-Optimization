"""Commensurability and normalisation tests (finding F10 / erratum A29).

The draft's ``Phi = w*C_max + (1-w)*E_cost`` adds **minutes to euros**.  On its own EX11
the two terms differ by a factor of 110--159, so at ``w = 0.5`` energy contributes under
1 % of the objective, the omega sweep is flat over almost all of [0, 1], and the marginal
utilities and the RL reward are dominated by makespan.  These tests pin the repair:
everything scalarised is computed on **normalised** quantities with **instance-fixed**
anchors.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jsspt_tou.baselines.dispatching import REFERENCE_POLICY, DispatchingPolicy
from jsspt_tou.benchmark.bilge_ulusoy import all_instances, build_instance
from jsspt_tou.domain.anchors import compute_anchors
from jsspt_tou.domain.objective import Outcome, phi
from jsspt_tou.simulator.engine import Engine


def test_raw_weighted_sum_collapses_to_a_makespan_objective() -> None:
    """The arithmetic of finding F10, reproduced on the draft's own instance.

    This test does not check the repair -- it checks that the *problem the repair fixes is
    real*.  If it ever stopped failing to be lopsided, the motivation for §3.7 would be
    gone and the paper should say so.
    """
    inst = build_instance(1, 1)
    engine = Engine(inst, omega=0.5)
    outcome = engine.run(REFERENCE_POLICY, np.random.default_rng(0))
    ratio = outcome.c_max / outcome.e_cost
    assert ratio > 50.0, "the raw terms are supposed to be orders of magnitude apart"
    energy_share = 0.5 * outcome.e_cost / (0.5 * outcome.c_max + 0.5 * outcome.e_cost)
    assert energy_share < 0.05, "raw scalarisation should be dominated by makespan"


def test_normalised_terms_are_commensurable() -> None:
    """After normalisation both terms live on [0, 1] and neither dominates by construction."""
    checked = 0
    for inst in all_instances()[:10]:
        engine = Engine(inst, omega=0.5)
        outcome = engine.run(REFERENCE_POLICY, np.random.default_rng(0))
        if not outcome.deadline_met:
            # The claim is "[0, 1] on *feasible* schedules".  A ToU-greedy rule that
            # defers into the off-peak band can overrun the deadline; that is the
            # deferral risk R13, measured in test_deadline.py, not a normalisation defect.
            continue
        checked += 1
        c_hat = engine.anchors.norm_cmax(outcome.c_max)
        e_hat = engine.anchors.norm_ecost(outcome.e_cost)
        assert -1e-9 <= c_hat <= 1.0 + 1e-9, inst.instance_id
        assert -1e-9 <= e_hat <= 1.0 + 1e-9, inst.instance_id
    assert checked > 0


def test_normalised_bounds_hold_across_the_rule_family() -> None:
    """[0, 1] must hold for every deadline-feasible schedule, not just the good ones."""
    inst = build_instance(7, 4)
    engine = Engine(inst, omega=0.5)
    rules = [
        DispatchingPolicy(machine_rule=mr, vehicle_rule=vr, charge_rule=cr, delay_rule=dr)
        for mr in ("SPT", "LPT", "MWKR")
        for vr in ("NT", "EFT")
        for cr in ("reactive", "tou")
        for dr in ("asap", "tou")
    ]
    for rule in rules:
        outcome = engine.run(rule, np.random.default_rng(0))
        if not outcome.deadline_met:
            continue
        assert -1e-9 <= engine.anchors.norm_cmax(outcome.c_max) <= 1.0 + 1e-9
        assert -1e-9 <= engine.anchors.norm_ecost(outcome.e_cost) <= 1.0 + 1e-9


def test_dimensional_consistency_under_unit_rescaling() -> None:
    """Rescale minutes to seconds and euros to cents: nothing normalised may move.

    A model that fails this test is measuring its own units rather than the schedule.
    """
    inst = build_instance(1, 1)
    anchors = compute_anchors(inst)
    outcomes = [
        Outcome(c_max=c, e_cost=e, energy_kwh=0.0, peak_kwh=0.0, charge_blocks=0,
                deadline_met=True)
        for c, e in ((120.0, 1.10), (150.0, 0.80), (190.0, 0.60))
    ]
    base = [phi(o, anchors, 0.5) for o in outcomes]

    scaled_anchors = dataclasses.replace(
        anchors,
        cmax_lb=anchors.cmax_lb * 60.0,
        cmax_ub=anchors.cmax_ub * 60.0,
        ecost_lb=anchors.ecost_lb * 100.0,
        ecost_ub=anchors.ecost_ub * 100.0,
    )
    scaled = [
        phi(
            Outcome(c_max=o.c_max * 60.0, e_cost=o.e_cost * 100.0, energy_kwh=0.0,
                    peak_kwh=0.0, charge_blocks=0, deadline_met=True),
            scaled_anchors,
            0.5,
        )
        for o in outcomes
    ]
    for a, b in zip(base, scaled):
        assert a == pytest.approx(b, abs=1e-12)
    # and the induced ranking is identical
    assert np.argsort(base).tolist() == np.argsort(scaled).tolist()


def test_anchors_are_instance_fixed_and_never_updated() -> None:
    """The invariance condition that T1 and T7 depend on.

    Recomputing the anchors from running best-known values -- "adaptive normalisation" --
    would make ``Phi`` a function of history rather than of the joint action profile: the
    exact-potential identity of T1 fails and the telescoping of T7 no longer sums to
    ``-Phi``.  It is a silent, plausible-looking bug, so it is asserted directly.
    """
    engine = Engine(build_instance(1, 1), omega=0.5)
    before = dataclasses.astuple(engine.anchors)
    engine.run(REFERENCE_POLICY, np.random.default_rng(0))
    after = dataclasses.astuple(engine.anchors)
    assert before == after
    with pytest.raises(dataclasses.FrozenInstanceError):
        engine.anchors.cmax_lb = 1.0  # type: ignore[misc]


def test_deadline_penalty_is_finite_and_inactive_when_feasible() -> None:
    """The penalised objective must stay real-valued -- see §3.4's obligation."""
    anchors = compute_anchors(build_instance(1, 1))
    feasible = Outcome(c_max=anchors.cmax_ub, e_cost=anchors.ecost_lb, energy_kwh=0.0,
                       peak_kwh=0.0, charge_blocks=0, deadline_met=True)
    infeasible = Outcome(c_max=anchors.cmax_ub * 2, e_cost=anchors.ecost_lb,
                         energy_kwh=0.0, peak_kwh=0.0, charge_blocks=0, deadline_met=False)
    assert phi(feasible, anchors, 0.5) == pytest.approx(0.5)
    value = phi(infeasible, anchors, 0.5)
    assert np.isfinite(value)
    assert value > phi(feasible, anchors, 0.5) + 1.0
