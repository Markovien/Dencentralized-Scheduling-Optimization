"""One objective, computed one way (guards finding F6 from recurring).

The draft's centralised model priced **robot charging only** while the game and the RL
reward priced **machines plus charging**, so no "MILP versus game" table could have compared
the same quantity.  ``domain/objective.py`` is now the only place ``Phi`` is computed and
these tests hold every producer to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from jsspt_tou.baselines.dispatching import REFERENCE_POLICY
from jsspt_tou.benchmark.bilge_ulusoy import all_instances, build_instance
from jsspt_tou.domain.objective import phi
from jsspt_tou.exact.cpsat_model import solve_exact
from jsspt_tou.simulator.engine import Engine


def test_engine_phi_delegates_to_the_single_implementation() -> None:
    for inst in all_instances()[:8]:
        engine = Engine(inst, omega=0.5)
        outcome = engine.run(REFERENCE_POLICY, np.random.default_rng(0))
        assert engine.phi(outcome) == pytest.approx(
            phi(outcome, engine.anchors, 0.5), abs=1e-12
        )


def test_energy_cost_covers_machines_and_charging() -> None:
    """F6: both terms must be present, or the models are not comparable."""
    inst = build_instance(1, 1)
    engine = Engine(inst, omega=0.5)
    state = engine.reset()
    engine.rollout(state)
    processing = sum(
        inst.tariff.cost(inst.machine_power_kw, s, e)
        for kind, _, _, s, e in state.log
        if kind == "process"
    )
    charging = sum(
        inst.tariff.cost(inst.battery.charger_power_kw, s, e)
        for kind, _, _, s, e in state.log
        if kind == "charge"
    )
    assert processing > 0.0
    assert state.e_cost == pytest.approx(processing + charging, abs=1e-9)


@pytest.mark.slow
def test_exact_model_scores_its_own_schedule_with_the_shared_objective() -> None:
    """The EX-CP objective value must agree with the shared normalised ``Phi``."""
    inst = build_instance(5, 1)
    engine = Engine(inst, omega=0.5)
    result = solve_exact(inst, omega=0.5, time_limit_s=60.0, anchors=engine.anchors)
    if result.outcome is None:
        pytest.skip("EX-CP found no feasible solution inside the test budget")
    recomputed = (
        0.5 * engine.anchors.norm_cmax(result.outcome.c_max)
        + 0.5 * engine.anchors.norm_ecost(result.outcome.e_cost)
    )
    assert result.objective == pytest.approx(recomputed, abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("job_set,layout", [(1, 2), (5, 3)])
def test_proven_optimum_is_never_worse_than_a_feasible_heuristic(
    job_set: int, layout: int
) -> None:
    """The invariant that catches cross-model drift.

    ``EX-CP`` and the simulator must optimise the same objective over the same feasible
    set.  Whenever ``EX-CP`` *proves* optimality, its value must therefore be at most the
    value of any schedule the simulator produces -- ``M1a``'s included.  Two real defects
    were found only by this comparison, and neither was visible from inside either model:
    the simulator charged idle battery drain only while waiting at a pickup rather than for
    the whole idle gap, and ``EX-CP`` forced every optional charging slot to be used.
    A ``FEASIBLE`` (not closed) run is exempt -- an incumbent is under no obligation to
    beat a heuristic.
    """
    import numpy as np

    from jsspt_tou.game.best_response import BestResponsePolicy

    inst = build_instance(job_set, layout)
    engine = Engine(inst, omega=0.5)
    heuristic = engine.run(BestResponsePolicy(i_max=6), np.random.default_rng(0))
    exact = solve_exact(inst, omega=0.5, time_limit_s=90.0, anchors=engine.anchors)
    if exact.status != "OPTIMAL" or exact.objective is None:
        pytest.skip("EX-CP did not close this instance inside the test budget")
    assert exact.objective <= engine.phi(heuristic) + 1e-9, (
        f"{inst.instance_id}: proven optimum {exact.objective:.4f} exceeds the heuristic "
        f"{engine.phi(heuristic):.4f} -- the two models disagree about the feasible set"
    )
