"""Falsifying tests for every formal result.

ROADMAP.md §4.3 and the proof-hygiene rule of §3.6: *a theorem without a test that would
fail if the theorem were false is not delivered*.  Each test below is written so that it
**can** fail -- it checks the identity or inequality the result asserts, on states drawn
from real episodes, not on a construction chosen to make it pass.

======  ==========================================================================
T1      the stage game is an exact potential game with potential ``W``
T2      per-stage convergence does not imply trajectory optimality (measured, not asserted)
T3      price of stability equals one: a global minimiser of Phi is a pure NE
T5      the least-core radius is computable on every instance and correctly signed
N5      charger congestion makes ``c`` supermodular and empties the core
T6      ``v`` is monotone by construction for the exact ``c``; the surrogate's violation
        rate is measured
======  ==========================================================================
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from jsspt_tou.benchmark.bilge_ulusoy import build_instance
from jsspt_tou.benchmark.congestion import congestion_instance, relaxed_instance
from jsspt_tou.cooperative.characteristic import (
    CharacteristicFunction,
    all_players,
    monotonicity_report,
)
from jsspt_tou.cooperative.core_lp import is_in_core, least_core
from jsspt_tou.cooperative.shapley import appro_shapley, exact_shapley
from jsspt_tou.game.best_response import BestResponsePolicy, potential
from jsspt_tou.simulator.engine import Engine
from jsspt_tou.simulator.state import IDLE, Action, Player


def _states(engine: Engine, n: int = 12) -> list:
    """A handful of genuine mid-episode states, reached by playing the game."""
    rng = np.random.default_rng(0)
    policy = BestResponsePolicy(i_max=2)
    state = engine.reset()
    out = []
    for _ in range(200):
        if len(out) >= n or state.done(engine.inst):
            break
        t, players = engine.next_event(state)
        if not players:
            if not engine.advance(state):
                break
            continue
        state.t = t
        out.append((state.snapshot(), list(players)))
        chosen = policy(engine, state, players, rng)
        committed = False
        for p in sorted(chosen):
            if not chosen[p].is_null:
                committed = engine.commit(state, p, chosen[p], t).feasible or committed
        if not committed and not engine.advance(state):
            break
    return out


# --------------------------------------------------------------------------------------------
# T1 -- exact potential
# --------------------------------------------------------------------------------------------
def test_t1_exact_potential_identity() -> None:
    """``u_p(a) - u_p(a') == W(a) - W(a')`` for every deviation, on real states.

    This is the whole content of T1.  It would fail if the utility convention were the
    draft's (a cost increase in one section, a cost reduction in another -- finding F4), or
    if ``Phi_hat`` depended on anything but ``(s_q, alpha)``.
    """
    engine = Engine(build_instance(1, 1), omega=0.5)
    rng = np.random.default_rng(1)
    checked = 0
    for state, players in _states(engine, n=8):
        for p in players:
            actions = engine.feasible_actions(state, p)
            if len(actions) < 2:
                continue
            others = {
                q: engine.reference_action(state, q, state.t) for q in players if q != p
            }
            for a, a2 in itertools.combinations(actions, 2):
                u_a = engine.utility(state, p, a, others)
                u_b = engine.utility(state, p, a2, others)
                w_a = potential(engine, state, {**others, p: a})
                w_b = potential(engine, state, {**others, p: a2})
                assert u_a - u_b == pytest.approx(w_a - w_b, abs=1e-9)
                checked += 1
                if checked > 400:
                    return
    assert checked > 0, "the test never exercised a real deviation"


def test_t1_requires_finite_action_sets() -> None:
    """Erratum A27: the delay grid is finite, which is what FIP needs."""
    engine = Engine(build_instance(1, 1), omega=0.5)
    for state, players in _states(engine, n=5):
        for p in players:
            actions = engine.feasible_actions(state, p)
            assert 1 <= len(actions) < 100
            assert len(set(actions)) == len(actions)


def test_t1_best_response_terminates_at_an_eps_nash_profile() -> None:
    """FIP: sequential best response reaches a pure NE of the stage game."""
    from jsspt_tou.game.best_response import epsilon_of

    engine = Engine(build_instance(1, 1), omega=0.5)
    rng = np.random.default_rng(2)
    policy = BestResponsePolicy(i_max=12)
    for state, players in _states(engine, n=6):
        alpha = policy(engine, state, players, rng)
        actions = {p: engine.actions_or_fallback(state, p) for p in players}
        assert epsilon_of(engine, state, alpha, actions) <= 1e-9


# --------------------------------------------------------------------------------------------
# T3 -- price of stability
# --------------------------------------------------------------------------------------------
def test_t3_price_of_stability_is_one() -> None:
    """A minimiser of the stage potential is a pure Nash equilibrium of the stage game.

    ``PoS = 1`` is the correct, provable version of the draft's claim that "the optimal
    solution to our problem is always an NE".  It replaces the directionally impossible
    bound ``0.5*Phi(opt) <= Phi(NE) <= Phi(opt)`` of finding F1, which would have made every
    equilibrium globally optimal and the optimisation vacuous.
    """
    engine = Engine(build_instance(5, 1), omega=0.5)
    for state, players in _states(engine, n=4):
        if len(players) > 3:
            players = players[:3]
        sets = [engine.feasible_actions(state, p) for p in players]
        if any(len(s) == 0 for s in sets) or np.prod([len(s) for s in sets]) > 4000:
            continue
        best_profile, best_phi = None, float("inf")
        for combo in itertools.product(*sets):
            profile = dict(zip(players, combo))
            value = engine.evaluate(state, profile)
            if value < best_phi:
                best_phi, best_profile = value, profile
        assert best_profile is not None
        # no unilateral deviation improves any player's utility at the global minimiser
        for p in players:
            others = {q: a for q, a in best_profile.items() if q != p}
            current = engine.utility(state, p, best_profile[p], others)
            best_dev = max(
                engine.utilities(engine_state := state, player=p, actions=engine.feasible_actions(state, p), others=others)
            )
            assert best_dev <= current + 1e-9


# --------------------------------------------------------------------------------------------
# N5 -- the congestion obstruction
# --------------------------------------------------------------------------------------------
def test_n5_charger_congestion_makes_the_cost_supermodular() -> None:
    """Regression test: the core of the congestion instance must come out **empty**.

    If a future change made this core non-empty, either the charger capacity stopped being
    enforced (finding F7 returning) or the characteristic function stopped measuring
    coordination.  Both are silent failures, which is why this is pinned.
    """
    engine = Engine(congestion_instance(), omega=0.0)
    cf = CharacteristicFunction(engine=engine, i_max=2)
    players = all_players(engine)
    robots = [p for p in players if p.kind == "robot"]
    v0, v1 = robots

    c_empty = cf.c(frozenset())
    c_a = cf.c(frozenset({v0}))
    c_b = cf.c(frozenset({v1}))
    c_ab = cf.c(frozenset({v0, v1}))

    # symmetric by construction
    assert c_a == pytest.approx(c_b)
    # the second coordinating robot gains *nothing*: the single charger is saturated
    assert c_ab == pytest.approx(c_a)
    # therefore the marginal cost reduction shrinks -- supermodular, not submodular
    delta_small = c_a - c_empty
    delta_big = c_ab - c_b
    assert delta_small < delta_big - 1e-9, "the congestion obstruction has disappeared"

    result = least_core(players, cf.v)
    assert not result.core_nonempty, "the core must be EMPTY on the congestion instance"
    assert result.epsilon < 0.0
    shapley = exact_shapley(players, cf.v)
    inside, _ = is_in_core(shapley.payoff, players, cf.v)
    assert not inside, "the Shapley value must not be core-stable here"


def test_n5_relaxing_the_charger_restores_the_marginal_saving() -> None:
    """The causal control for N5: only ``K_CH`` differs between the two instances."""
    congested = CharacteristicFunction(
        engine=Engine(congestion_instance(), omega=0.0), i_max=2
    )
    relaxed = CharacteristicFunction(
        engine=Engine(relaxed_instance(), omega=0.0), i_max=2
    )
    for cf in (congested, relaxed):
        cf.enumerate_all()
    robots = [p for p in all_players(congested.engine) if p.kind == "robot"]
    pair = frozenset(robots)
    single = frozenset({robots[0]})
    gain_congested = congested.c(single) - congested.c(pair)
    gain_relaxed = relaxed.c(single) - relaxed.c(pair)
    assert gain_congested == pytest.approx(0.0, abs=1e-9)
    assert gain_relaxed > 1e-6, "a second charger must restore the second robot's saving"


# --------------------------------------------------------------------------------------------
# T5 -- certified stability
# --------------------------------------------------------------------------------------------
def test_t5_least_core_is_correctly_signed_on_a_hand_computed_game() -> None:
    """Pins the *direction* of the LP.

    A three-player savings game whose core is a single point: ``v(S) = 0`` for singletons,
    ``v(S) = 2`` for pairs, ``v(N) = 3``.  The equal split ``(1,1,1)`` gives every pair
    exactly 2, so the core is non-empty and touches every pair constraint: ``eps* = 0``.
    Solving the *cost-game* form instead would return the wrong sign here.
    """
    players = [Player("machine", 1), Player("machine", 2), Player("robot", 0)]

    def v(s: frozenset[Player]) -> float:
        return {0: 0.0, 1: 0.0, 2: 2.0, 3: 3.0}[len(s)]

    result = least_core(players, v)
    assert result.epsilon == pytest.approx(0.0, abs=1e-9)
    assert result.core_nonempty
    for p in players:
        assert result.allocation[p] == pytest.approx(1.0, abs=1e-6)


def test_t5_least_core_detects_an_empty_core() -> None:
    """The complementary case: a symmetric game whose core is provably empty."""
    players = [Player("robot", 0), Player("robot", 1), Player("robot", 2)]

    def v(s: frozenset[Player]) -> float:
        return {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0}[len(s)]

    result = least_core(players, v)
    # Summing the three pair constraints gives 2*v(N) >= 3*(1 + eps), i.e. eps <= -1/3,
    # attained by the equal split.  The core is empty and the radius is exactly -1/3.
    assert result.epsilon == pytest.approx(-1.0 / 3.0, abs=1e-9)
    assert not result.core_nonempty


def test_t5_radius_is_computable_on_a_real_instance() -> None:
    engine = Engine(build_instance(1, 1), omega=0.5)
    cf = CharacteristicFunction(engine=engine, i_max=1)
    cf.enumerate_all()
    players = all_players(engine)
    result = least_core(players, cf.v)
    assert np.isfinite(result.epsilon)
    assert sum(result.allocation.values()) == pytest.approx(
        cf.v(frozenset(players)), abs=1e-6
    )


# --------------------------------------------------------------------------------------------
# T6 -- monotonicity and superadditivity
# --------------------------------------------------------------------------------------------
def test_t6_monotonicity_of_the_surrogate_is_measured_not_assumed() -> None:
    """Monotonicity holds by construction for the *exact* ``c``, not for the surrogate.

    A single best-response sweep with more players is not guaranteed to dominate one with
    fewer, so the violation rate is measured and reported.  The test pins that it stays
    small; a large rate would mean the surrogate is too crude to carry the cooperative
    analysis.
    """
    engine = Engine(build_instance(1, 1), omega=0.5)
    cf = CharacteristicFunction(engine=engine, i_max=1)
    cf.enumerate_all()
    report = monotonicity_report(cf, np.random.default_rng(0), n_pairs=300)
    assert report["violation_rate"] < 0.10


# --------------------------------------------------------------------------------------------
# Shapley axioms
# --------------------------------------------------------------------------------------------
def test_shapley_axioms_and_sampling_agreement() -> None:
    """Efficiency, symmetry, null player; and the sampler inside its own interval."""
    engine = Engine(build_instance(5, 1), omega=0.5)
    cf = CharacteristicFunction(engine=engine, i_max=1)
    cf.enumerate_all()
    players = all_players(engine)
    exact = exact_shapley(players, cf.v)
    assert exact.total == pytest.approx(cf.v(frozenset(players)), abs=1e-9)

    sampled = appro_shapley(players, cf.v, np.random.default_rng(0), permutations=120)
    assert sampled.total == pytest.approx(cf.v(frozenset(players)), abs=1e-6)
    assert sampled.ci_halfwidth is not None
    for p in players:
        assert abs(sampled.payoff[p] - exact.payoff[p]) <= max(
            3.0 * sampled.ci_halfwidth[p], 1e-3
        )


def test_shapley_null_player_and_symmetry() -> None:
    a, b, c = Player("machine", 1), Player("machine", 2), Player("robot", 0)

    def v(s: frozenset[Player]) -> float:
        return float(len(s & {a, b}))  # c is a null player; a and b are symmetric

    alloc = exact_shapley([a, b, c], v)
    assert alloc.payoff[c] == pytest.approx(0.0, abs=1e-12)
    assert alloc.payoff[a] == pytest.approx(alloc.payoff[b], abs=1e-12)
    assert alloc.total == pytest.approx(2.0, abs=1e-12)
