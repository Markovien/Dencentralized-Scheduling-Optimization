"""``M2`` tests: characteristic function, allocations, coalition formation, bargaining."""

from __future__ import annotations

import numpy as np
import pytest

from jsspt_tou.benchmark.bilge_ulusoy import build_instance
from jsspt_tou.cooperative.bargaining import (
    build_front,
    kalai_smorodinsky,
    nadir_disagreement,
    nash_bargaining,
)
from jsspt_tou.cooperative.characteristic import (
    CharacteristicFunction,
    all_players,
    realised_partition_value,
)
from jsspt_tou.cooperative.coalition_formation import is_dhp_stable, merge_split
from jsspt_tou.cooperative.core_lp import least_core, least_core_generated, nucleolus
from jsspt_tou.cooperative.shapley import exact_shapley
from jsspt_tou.game.best_response import BestResponsePolicy
from jsspt_tou.simulator.engine import Engine


def _cf(job_set: int = 1, layout: int = 1, omega: float = 0.5) -> CharacteristicFunction:
    engine = Engine(build_instance(job_set, layout), omega=omega)
    cf = CharacteristicFunction(engine=engine, i_max=1)
    cf.enumerate_all()
    return cf


def test_savings_game_is_grounded_and_deterministic() -> None:
    """``v(0) = 0`` -- the property the wrong savings definition would destroy."""
    cf = _cf()
    assert cf.v(frozenset()) == pytest.approx(0.0)
    players = all_players(cf.engine)
    first = cf.c(frozenset(players))
    calls = cf.calls
    assert cf.c(frozenset(players)) == first
    assert cf.calls == calls, "c(S) must be memoised, not recomputed"


def test_grand_coalition_beats_the_reference_policy() -> None:
    cf = _cf()
    players = all_players(cf.engine)
    assert cf.v(frozenset(players)) > 0.0
    assert cf.c(frozenset(players)) < cf.c_empty()


def test_shapley_is_efficient_and_least_core_is_consistent() -> None:
    cf = _cf()
    players = all_players(cf.engine)
    shapley = exact_shapley(players, cf.v)
    assert shapley.total == pytest.approx(cf.v(frozenset(players)), abs=1e-9)
    lc = least_core(players, cf.v)
    assert sum(lc.allocation.values()) == pytest.approx(
        cf.v(frozenset(players)), abs=1e-6
    )
    nu = nucleolus(players, cf.v)
    assert nu.epsilon == pytest.approx(lc.epsilon, abs=1e-6), (
        "the nucleolus's first level *is* the least-core programme"
    )


def test_constraint_generation_matches_full_enumeration() -> None:
    cf = _cf(5, 1)
    players = all_players(cf.engine)
    full = least_core(players, cf.v)
    generated = least_core_generated(players, cf.v)
    assert generated.epsilon == pytest.approx(full.epsilon, abs=1e-6)


def test_merge_split_terminates_at_a_dhp_stable_partition() -> None:
    cf = _cf()
    players = all_players(cf.engine)
    structure = merge_split(players, cf.v)
    assert set().union(*structure.blocks) == set(players)
    assert sum(len(b) for b in structure.blocks) == len(players)
    assert is_dhp_stable(structure, cf.v)
    assert structure.messages > 0


def test_partition_value_is_measured_not_summed() -> None:
    """``sum_k v(S_k)`` is accounting; the realised value is a simulator run."""
    cf = _cf()
    players = all_players(cf.engine)
    structure = merge_split(players, cf.v)
    realised = realised_partition_value(cf, structure.blocks)
    assert realised <= cf.v(frozenset(players)) + 1e-9


def test_bargaining_returns_a_supported_weight() -> None:
    inst = build_instance(1, 1)
    anchors = Engine(inst, omega=0.5).anchors
    outcomes = [
        (float(w), Engine(inst, omega=float(w)).run(
            BestResponsePolicy(i_max=3), np.random.default_rng(0)))
        for w in np.linspace(0.0, 1.0, 11)
    ]
    front = build_front(outcomes, anchors)
    assert len(front) >= 2
    disagreement = nadir_disagreement(front)
    for solution in (
        nash_bargaining(front, disagreement, anchors),
        kalai_smorodinsky(front, disagreement, anchors),
    ):
        assert solution is not None
        lo, hi = solution.omega_interval
        assert 0.0 <= lo <= solution.omega_induced <= hi <= 1.0


def test_bargaining_is_invariant_under_affine_rescaling() -> None:
    """Scale invariance of NBS/KS -- the two-line argument of §3.7.5, checked."""
    inst = build_instance(1, 1)
    anchors = Engine(inst, omega=0.5).anchors
    outcomes = [
        (float(w), Engine(inst, omega=float(w)).run(
            BestResponsePolicy(i_max=3), np.random.default_rng(0)))
        for w in np.linspace(0.0, 1.0, 11)
    ]
    front = build_front(outcomes, anchors)
    base = nash_bargaining(front, nadir_disagreement(front), anchors)
    import dataclasses

    scaled_front = [
        dataclasses.replace(p, c_max=p.c_max * 60.0, e_cost=p.e_cost * 100.0)
        for p in front
    ]
    scaled = nash_bargaining(scaled_front, nadir_disagreement(scaled_front), anchors)
    assert base is not None and scaled is not None
    assert base.point.omega == scaled.point.omega
