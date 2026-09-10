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
