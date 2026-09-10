"""``M1`` — the non-cooperative potential game.

Implements Paper A §5: the corrected stage game, sequential best-response dynamics
(**M1a**) and log-linear learning (**M1b**), with the eps-Nash certificate and the
potential trajectory both instrumented.

The single sign convention (repairing F4)
-----------------------------------------
    W(alpha)  = Phi_hat(alpha^0) - Phi_hat(alpha)        welfare, a cost *reduction*
    u_p(alpha)= W(alpha) - W(a_p^0, alpha_{-p})          marginal contribution
              = Phi_hat(a_p^0, alpha_{-p}) - Phi_hat(alpha)
**every player maximises ``u_p``.**

T1 (exact stage potential).  For any ``a_p``, ``a_p'`` and any ``alpha_{-p}``,

    u_p(a_p, alpha_{-p}) - u_p(a_p', alpha_{-p})
        = [Phi_hat(a_p^0, alpha_{-p}) - Phi_hat(a_p, alpha_{-p})]
        - [Phi_hat(a_p^0, alpha_{-p}) - Phi_hat(a_p', alpha_{-p})]
        = W(a_p, alpha_{-p}) - W(a_p', alpha_{-p}),

because the ``Phi_hat(a_p^0, alpha_{-p})`` term does not depend on ``p``'s own action and
cancels.  The stage game is therefore an **exact potential game** with potential ``W``.
It has finite action sets (the ToU-aligned delay grid, erratum A27) and the deviations
below are **sequential**, so the finite-improvement property applies and best-response
dynamics reach a pure Nash equilibrium of the stage game in finitely many steps
(Monderer & Shapley, 1996).

T2 (dynamic scope, stated honestly).  The multi-stage game is a *sequence* of stage
potential games.  Per-stage convergence does **not** imply optimality of the resulting
trajectory; the draft claimed the potential property for the trajectory-dependent
objective, where it does not hold (finding F5).  The loss is quantified empirically
against ``EX-CP`` rather than asserted away.

**Updates must be sequential.**  Genuinely synchronous best response can cycle even in a
potential game; FIP guarantees termination only when one player deviates at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from jsspt_tou.simulator.engine import Engine
from jsspt_tou.simulator.state import IDLE, Action, Player, State

TOL: float = 1e-10


@dataclass(slots=True)
class GameTrace:
    """Instrumentation of the decentralised dynamics over one episode."""

    stages: int = 0
    total_iterations: int = 0
    total_improvements: int = 0
    imax_binding: int = 0
    eps_ne: float = 0.0
    potential_trajectory: list[float] = field(default_factory=list)
    utility_evaluations: int = 0

    @property
    def mean_iterations(self) -> float:
        return self.total_iterations / self.stages if self.stages else 0.0


@dataclass(slots=True)
class BestResponsePolicy:
    """**M1a** — round-robin sequential best response at every event.

    Parameters
    ----------
    i_max:
        Iteration cap.  Whether it binds is *reported*, not hidden: FIP guarantees
        termination, so a binding cap means the stage game is larger than expected.
    trace:
        Optional instrumentation object shared across the episode.
    """

    i_max: int = 8
    trace: GameTrace | None = None

    @property
    def name(self) -> str:
        return "M1a-BR"

    def __call__(
        self,
        engine: Engine,
        state: State,
        players: Sequence[Player],
        rng: np.random.Generator,
    ) -> dict[Player, Action]:
        actions = {p: engine.actions_or_fallback(state, p) for p in players}
        alpha: dict[Player, Action] = {p: IDLE for p in players}
        order = list(players)
        rng.shuffle(order)  # a random permutation per stage, as specified
        iterations = 0
        improvements = 0
        n_eval = 0
        for _ in range(self.i_max):
            iterations += 1
            improved = False
            for p in order:
                others = {q: a for q, a in alpha.items() if q != p}
                cand = actions[p]
                us = engine.utilities(state, p, cand, others)
                n_eval += len(cand)
                best_a, best_u = alpha[p], us[cand.index(alpha[p])] if alpha[p] in cand else -math.inf
                for a, u in zip(cand, us):
                    if u > best_u + TOL:
                        best_a, best_u = a, u
                if best_a != alpha[p]:
                    alpha[p] = best_a
                    improved = True
                    improvements += 1
            if not improved:
                break
        if self.trace is not None:
            self.trace.stages += 1
            self.trace.total_iterations += iterations
            self.trace.total_improvements += improvements
            self.trace.utility_evaluations += n_eval
            self.trace.imax_binding += int(iterations >= self.i_max)
            self.trace.eps_ne = max(
                self.trace.eps_ne, epsilon_of(engine, state, alpha, actions)
            )
            self.trace.potential_trajectory.append(
                potential(engine, state, alpha)
            )
        return alpha


@dataclass(slots=True)
class LogLinearPolicy:
    """**M1b** — log-linear learning (Gibbs best response with annealing).

    Replaces the arg-max with ``Pr(a_p) proportional to exp(u_p / theta)`` and anneals the
    temperature ``theta``.  As ``theta -> 0`` the stationary distribution of the induced
    Markov chain concentrates on **potential maximisers**, i.e. the stochastically stable
    states, so this is an *equilibrium selection* mechanism that targets the best Nash
    equilibrium rather than an arbitrary one -- which is what tightens the realised price
    of anarchy in experiment E6.

    The temperature is named ``theta`` and not ``T``: ``T`` is already the set of transport
    tasks (Appendix B, symbol-collision list).
    """

    i_max: int = 8
    theta0: float = 0.25
    anneal: float = 0.7
    trace: GameTrace | None = None

    @property
    def name(self) -> str:
        return "M1b-LLL"

    def __call__(
        self,
        engine: Engine,
        state: State,
        players: Sequence[Player],
        rng: np.random.Generator,
    ) -> dict[Player, Action]:
        actions = {p: engine.actions_or_fallback(state, p) for p in players}
        alpha: dict[Player, Action] = {p: IDLE for p in players}
        order = list(players)
        rng.shuffle(order)
        theta = self.theta0
        iterations = 0
        for _ in range(self.i_max):
            iterations += 1
            for p in order:
                others = {q: a for q, a in alpha.items() if q != p}
                utils = np.array(engine.utilities(state, p, actions[p], others), dtype=float)
                utils[~np.isfinite(utils)] = -1e12
                logits = (utils - utils.max()) / max(theta, 1e-6)
                weights = np.exp(logits)
                total = weights.sum()
                if not np.isfinite(total) or total <= 0.0:
                    continue
                idx = int(rng.choice(len(actions[p]), p=weights / total))
                alpha[p] = actions[p][idx]
            theta *= self.anneal
        # Final greedy sweep: anneal to zero exactly, so the committed profile is a
        # best response and the eps-NE certificate is meaningful.
        for p in order:
            others = {q: a for q, a in alpha.items() if q != p}
            us = engine.utilities(state, p, actions[p], others)
            alpha[p] = actions[p][int(np.argmax(us))]
        if self.trace is not None:
            self.trace.stages += 1
            self.trace.total_iterations += iterations
            self.trace.eps_ne = max(
                self.trace.eps_ne, epsilon_of(engine, state, alpha, actions)
            )
            self.trace.potential_trajectory.append(potential(engine, state, alpha))
        return alpha


def potential(
    engine: Engine, state: State, alpha: dict[Player, Action]
) -> float:
    """The exact potential ``W(alpha) = Phi_hat(alpha^0) - Phi_hat(alpha)``.

    Must go through :meth:`Engine.evaluate`, the *same* stage evaluation the utilities use.
    Mixing the two evaluators -- the rollout for ``u_p`` and the stage surrogate for ``W``
    -- silently breaks the exact-potential identity of T1 while leaving both quantities
    looking plausible; ``tests/test_theory.py`` catches it.
    """
    null = {p: IDLE for p in alpha}
    return engine.evaluate(state, null) - engine.evaluate(state, alpha)


def epsilon_of(
    engine: Engine,
    state: State,
    alpha: dict[Player, Action],
    actions: dict[Player, list[Action]],
) -> float:
    """The eps-Nash certificate: the largest unilateral improvement still available."""
    worst = 0.0
    for p, a_p in alpha.items():
        others = {q: a for q, a in alpha.items() if q != p}
        cand = list(actions.get(p, [a_p]))
        if a_p not in cand:
            cand.append(a_p)
        us = engine.utilities(state, p, cand, others)
        current = us[cand.index(a_p)]
        best = max(us)
        if math.isfinite(best) and math.isfinite(current):
            worst = max(worst, best - current)
    return worst
