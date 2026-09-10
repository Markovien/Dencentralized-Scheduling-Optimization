"""``M1`` tests: convergence, the eps-Nash certificate, and the potential trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from jsspt_tou.baselines.dispatching import rule_grid
from jsspt_tou.benchmark.bilge_ulusoy import build_instance
from jsspt_tou.game.best_response import (
    BestResponsePolicy,
    GameTrace,
    LogLinearPolicy,
)
from jsspt_tou.simulator.engine import Engine


def test_best_response_terminates_and_certifies_equilibrium() -> None:
    trace = GameTrace()
    engine = Engine(build_instance(1, 1), omega=0.5)
    engine.run(BestResponsePolicy(i_max=8, trace=trace), np.random.default_rng(0))
    assert trace.stages > 0
    assert trace.eps_ne <= 1e-9, "committed profiles must be exact stage equilibria"
    assert trace.imax_binding == 0, "the iteration cap should not bind (FIP)"
    assert trace.mean_iterations >= 1.0


def test_log_linear_learning_runs_and_anneals_to_near_equilibrium() -> None:
    """M1b is an *equilibrium-selection* device, not an equilibrium *solver*.

    Its final sweep is a single sequential pass, so a player updated early may no longer be
    best-responding once later players move: the committed profile is an eps-equilibrium
    with small but generally non-zero eps.  Asserting eps = 0 here would be asserting
    something the method does not claim.  What must hold is that eps stays small relative
    to the objective scale.
    """
    trace = GameTrace()
    engine = Engine(build_instance(1, 1), omega=0.5)
    outcome = engine.run(LogLinearPolicy(i_max=4, trace=trace), np.random.default_rng(1))
    assert outcome.c_max > 0
    assert np.isfinite(trace.eps_ne)
    assert trace.eps_ne < 0.25


def test_game_beats_the_reference_policy_everywhere() -> None:
    """M1 must improve on ``pi_0``, the policy its own utilities are measured against."""
    for job_set in (1, 5, 7):
        inst = build_instance(job_set, 1)
        engine = Engine(inst, omega=0.5)
        reference = engine.reset()
        engine.rollout(reference)
        game = engine.run(BestResponsePolicy(i_max=4), np.random.default_rng(0))
        assert engine.phi(game) <= engine.phi(engine.outcome(reference)) + 1e-9, (
            inst.instance_id
        )


def test_game_wins_against_the_rule_family_on_most_instances() -> None:
    """Measured, not asserted instance by instance.

    The comparator here is an *oracle*: the best of 120 composite rules selected with
    hindsight per instance.  M1 does not dominate it everywhere, and the paper reports the
    win rate rather than claiming dominance.  What the test pins is that the win rate stays
    a majority -- if it collapsed, the mechanism would have stopped working.
    """
    wins = 0
    total = 0
    for job_set in (1, 2, 5, 7, 10):
        for layout in (1, 2):
            inst = build_instance(job_set, layout)
            engine = Engine(inst, omega=0.5)
            best_rule = min(
                (engine.run(p, np.random.default_rng(0)) for p in rule_grid()),
                key=engine.phi,
            )
            game = engine.run(BestResponsePolicy(i_max=4), np.random.default_rng(0))
            total += 1
            wins += int(engine.phi(game) <= engine.phi(best_rule) + 1e-9)
    assert total == 10
    assert wins >= 6, f"M1 beat the per-instance oracle rule on only {wins}/{total}"


def test_omega_actually_moves_the_trade_off() -> None:
    """The corrected normalisation must make the weight bite (finding F10)."""
    inst = build_instance(1, 1)
    results = {}
    for w in (0.0, 0.5, 1.0):
        engine = Engine(inst, omega=w)
        results[w] = engine.run(BestResponsePolicy(i_max=4), np.random.default_rng(0))
    assert results[0.0].e_cost < results[1.0].e_cost - 1e-9
    assert results[1.0].c_max <= results[0.0].c_max + 1e-9
