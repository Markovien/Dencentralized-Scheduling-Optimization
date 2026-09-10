"""Simulator invariants: determinism, conservation, and constraint satisfaction."""

from __future__ import annotations

import numpy as np
import pytest

from jsspt_tou.benchmark.bilge_ulusoy import all_instances, build_instance
from jsspt_tou.game.best_response import BestResponsePolicy
from jsspt_tou.simulator.engine import Engine


def test_determinism_under_a_fixed_seed() -> None:
    inst = build_instance(1, 1)
    engine = Engine(inst, omega=0.5)
    a = engine.run(BestResponsePolicy(i_max=3), np.random.default_rng(7))
    b = engine.run(BestResponsePolicy(i_max=3), np.random.default_rng(7))
    assert a == b


def test_no_global_seed_is_read() -> None:
    """Different generators must be able to give different runs; the same one, the same run."""
    engine = Engine(build_instance(2, 1), omega=0.5)
    np.random.seed(1234)  # deliberately poisoning the global state
    a = engine.run(BestResponsePolicy(i_max=3), np.random.default_rng(0))
    np.random.seed(4321)
    b = engine.run(BestResponsePolicy(i_max=3), np.random.default_rng(0))
    assert a == b


def test_all_jobs_complete_and_makespan_is_the_last_return() -> None:
    for inst in all_instances()[:10]:
        engine = Engine(inst, omega=0.5)
        state = engine.reset()
        engine.rollout(state)
        assert state.done(inst), inst.instance_id
        returns = [
            e for e in state.log if e[0] == "transport"
        ]
        assert state.c_max() == pytest.approx(max(c for c in state.job_completion))
        assert state.c_max() <= max(e[4] for e in returns) + 1e-9


def test_machine_and_robot_capacity_are_never_violated() -> None:
    """Unary machines, unary robots, and a charger of capacity ``K_CH``."""
    for inst in all_instances()[:8]:
        engine = Engine(inst, omega=0.5)
        state = engine.reset()
        engine.rollout(state)
        _assert_unary(state.log, "process")
        _assert_unary(state.log, "transport")
        charges = sorted(
            [(s, e) for kind, _, _, s, e in state.log if kind == "charge"]
        )
        for i, (s, e) in enumerate(charges):
            overlapping = sum(1 for (s2, e2) in charges if s2 < e - 1e-9 and s < e2 - 1e-9)
            assert overlapping <= inst.n_chargers, inst.instance_id


def _assert_unary(log, kind: str) -> None:
    by_owner: dict[int, list[tuple[float, float]]] = {}
    for k, owner, _, s, e in log:
        if k == kind:
            by_owner.setdefault(owner, []).append((s, e))
    for owner, spans in by_owner.items():
        spans.sort()
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert s2 >= e1 - 1e-9, f"{kind} overlap on {owner}: {(s1,e1)} {(s2,e2)}"


def test_soc_never_crosses_the_floor_while_travelling() -> None:
    for inst in all_instances()[:8]:
        engine = Engine(inst, omega=0.5)
        state = engine.reset()
        engine.rollout(state)
        for soc in state.robot_soc:
            assert soc >= -1e-6


def test_energy_accounting_matches_the_tariff_integral() -> None:
    """Total cost recomputed from the log must equal the accumulated cost."""
    for inst in all_instances()[:6]:
        engine = Engine(inst, omega=0.5)
        state = engine.reset()
        engine.rollout(state)
        total = 0.0
        for kind, _, _, s, e in state.log:
            if kind == "process":
                total += inst.tariff.cost(inst.machine_power_kw, s, e)
            elif kind == "charge":
                total += inst.tariff.cost(inst.battery.charger_power_kw, s, e)
        assert total == pytest.approx(state.e_cost, abs=1e-9), inst.instance_id


def test_snapshot_is_isolated_from_the_live_state() -> None:
    engine = Engine(build_instance(1, 1), omega=0.5)
    state = engine.reset()
    snap = state.snapshot()
    engine.rollout(state)
    assert snap.t == 0.0
    assert all(c < 0 for c in snap.job_completion)
