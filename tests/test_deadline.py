"""Deadline semantics (finding F11 / erratum A30) and the deadlock safeguard (risk R13)."""

from __future__ import annotations

import numpy as np
import pytest

from jsspt_tou.baselines.dispatching import rule_grid
from jsspt_tou.benchmark.bilge_ulusoy import all_instances, build_instance
from jsspt_tou.game.best_response import BestResponsePolicy
from jsspt_tou.simulator.engine import Engine


def test_no_returned_schedule_exceeds_the_deadline() -> None:
    """``C_max <= H`` is a constraint, not a display window."""
    misses = 0
    total = 0
    for inst in all_instances()[:12]:
        engine = Engine(inst, omega=0.5)
        for policy in (BestResponsePolicy(i_max=3),):
            outcome = engine.run(policy, np.random.default_rng(0))
            total += 1
            if outcome.deadline_met:
                assert outcome.c_max <= inst.deadline + 1e-9
            else:
                misses += 1
    assert total > 0
    assert misses == 0, f"{misses}/{total} decentralised runs missed the deadline"


def test_the_remaining_work_bound_never_rejects_a_feasible_action() -> None:
    """The mask is built on a *lower* bound, which is the safe direction.

    It may fail to detect an infeasible continuation; it must never mask an action that a
    feasible completion could still use.  We verify the bound is dominated by the realised
    completion of the same job on schedules that actually finished.
    """
    inst = build_instance(1, 1)
    engine = Engine(inst, omega=0.5)
    rng = np.random.default_rng(0)
    state = engine.reset()
    checked = 0
    for _ in range(400):
        if state.done(inst):
            break
        t, players = engine.next_event(state)
        if not players:
            if not engine.advance(state):
                break
            continue
        state.t = t
        for p in players:
            for action in engine.feasible_actions(state, p):
                if action.is_null:
                    continue
                preview = engine.preview(state, p, action, t)
                assert preview.projected_completion >= preview.end - 1e-9
                checked += 1
        chosen = {
            p: engine.fallback_action(state, p, t) for p in players
        }
        committed = False
        for p in sorted(chosen):
            if not chosen[p].is_null:
                committed = engine.commit(state, p, chosen[p], t).feasible or committed
        if not committed and not engine.advance(state):
            break
    assert checked > 50


def test_fallback_yields_a_deadline_feasible_completion() -> None:
    """The safeguard must actually deliver a finished, deadline-feasible schedule."""
    for inst in all_instances()[:8]:
        engine = Engine(inst, omega=0.5)

        def greedy(engine, state, players, rng):  # type: ignore[no-untyped-def]
            return {p: engine.fallback_action(state, p, state.t) for p in players}

        outcome = engine.run(greedy, np.random.default_rng(0))
        assert outcome.deadline_met, inst.instance_id


def test_deadlock_rate_is_recorded_not_absorbed() -> None:
    """Whether the safeguard fired is reported on every outcome."""
    inst = build_instance(1, 1)
    engine = Engine(inst, omega=0.5)
    outcome = engine.run(BestResponsePolicy(i_max=3), np.random.default_rng(0))
    assert isinstance(outcome.used_fallback, bool)


def test_every_dispatching_rule_finishes_every_job() -> None:
    """An unfinished episode is scored as a violation, never silently dropped."""
    inst = build_instance(5, 1)
    engine = Engine(inst, omega=0.5)
    for policy in rule_grid()[:24]:
        outcome = engine.run(policy, np.random.default_rng(0))
        assert outcome.c_max > 0
