"""Event-driven simulator: the one execution rule ``Gamma`` shared by every method.

Implements the decision architecture of Paper A §5 -- decisions are taken at *events*
(``RobotFree``, ``MachineFree``, ``ChargerFree``, part arrival), never on a time grid, so
the decision count scales with system activity rather than with horizon length.  The same
engine serves the dispatching baselines, the non-cooperative game ``M1``, the coalition
cost ``c(S)`` of ``M2``, and it is what makes the cross-method tables comparable: an
``EX-CP`` row and a game row score the same schedule object with the same objective.

What this module adds relative to the draft
-------------------------------------------
* **Charger capacity.**  The draft mentions a single charging station and then imposes no
  charger non-overlap constraint anywhere (finding F7, erratum A9).  Here charging
  occupies one of ``K_CH`` stations.  This is not a detail: it is the scarce shared
  resource whose congestion makes the coalition cost *supermodular* and can empty the core
  (result N5).
* **The deadline as a constraint.**  ``C_max <= H`` is enforced by masking, with a
  remaining-work lower bound, plus a fallback policy and a recorded deadlock rate
  (finding F11, erratum A30).
* **Machine processing energy.**  Priced against the tariff, not omitted (finding F6).
* **One objective.**  Outcomes are scored by :mod:`jsspt_tou.domain.objective` only.

Determinism.  Every tie is broken by the documented total order
``(time, player kind, player index, action)``; no stochastic component reads a global seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Final, Protocol, Sequence

import numpy as np

from jsspt_tou.domain.anchors import Anchors, compute_anchors
from jsspt_tou.domain.battery import b_needed
from jsspt_tou.domain.instance import Instance
from jsspt_tou.domain.objective import (
    DEADLINE_PENALTY_DEFAULT,
    Outcome,
    phi,
)
from jsspt_tou.simulator.state import IDLE, Action, Player, State

EPS: Final[float] = 1e-9
MAX_DELAY_OPTIONS: Final[int] = 3
"""Size of the finite ToU-aligned delay grid ``D`` beyond the zero delay.

Finiteness is required by T1: the finite-improvement property of a potential game
guarantees termination only for finite action sets (erratum A27).
"""


@dataclass(frozen=True, slots=True)
class Preview:
    """The consequence of one action, computed without mutating the state.

    ``ret`` is the draft's **return function** ``R_p^q(a_p)`` -- the projected completion
    time of the action -- reproduced here exactly as specified, including the
    empty-travel + idle-wait + loaded-travel decomposition for robots.
    """

    feasible: bool
    ret: float
    cost: float
    kwh: float
    peak_kwh: float
    start: float
    end: float
    blocks: int
    soc_after: float
    charger: int
    projected_completion: float
    deadline_safe: bool


class Policy(Protocol):
    """A decision rule: choose one action per active player at one event."""

    def __call__(
        self,
        engine: "Engine",
        state: State,
        players: Sequence[Player],
        rng: np.random.Generator,
    ) -> dict[Player, Action]:  # pragma: no cover - protocol
        ...


class Engine:
    """The discrete-event execution rule.

    Parameters
    ----------
    inst:
        The instance.  Immutable.
    omega:
        Trade-off weight used by the myopic stage objective and by the reported ``Phi``.
    penalty:
        ``M`` in the deadline penalty.
    anchors:
        Instance-fixed normalisation anchors.  Computed once here if not supplied, and
        never recomputed -- see :mod:`jsspt_tou.domain.anchors` for why that matters to
        T1 and T7.
    evaluator:
        ``"rollout"`` (default) or ``"stage"``.  See :meth:`evaluate`.
    """

    def __init__(
        self,
        inst: Instance,
        omega: float = 0.5,
        penalty: float = DEADLINE_PENALTY_DEFAULT,
        anchors: Anchors | None = None,
        evaluator: str = "rollout",
    ) -> None:
        self.inst = inst
        self.evaluator = evaluator
        self.omega = omega
        self.penalty = penalty
        self.anchors = anchors if anchors is not None else compute_anchors(inst)
        self._suffix = _suffix_tables(inst)
        self._boundaries = tuple(
            b for b in inst.tariff.boundaries() if b > 0.0
        )
        self._max_price = inst.tariff.max_price
        # Time scale for the null action when no future event exists (e.g. at t = 0, when
        # every clock reads zero).  The shortest positive task duration is the natural
        # quantum of the instance.
        durations = [op.duration for op in inst.operations] + [
            tk.duration for tk in inst.transports if tk.duration > 0
        ]
        self._defer_quantum = min(durations) if durations else 1.0

    # -- state ---------------------------------------------------------------------------
    def reset(self) -> State:
        return State.initial(self.inst)

    # -- events --------------------------------------------------------------------------
    def next_event(self, state: State) -> tuple[float, list[Player]]:
        """Earliest time at which some player has a non-null action, and who they are.

        Returns ``(inf, [])`` when no player can ever act again.
        """
        inst = self.inst
        best = math.inf
        # robots: free, and either a transport is claimable or they can charge
        claimable = self._claimable_transports(state)
        for r in range(inst.n_robots):
            can_charge = state.robot_soc[r] < inst.battery.ceiling_mah - EPS
            if not claimable and not can_charge:
                continue
            best = min(best, max(state.t, state.robot_free[r]))
        # machines: idle, with a delivered part waiting in the input buffer
        for m in range(1, inst.n_machines + 1):
            ready = [
                state.job_ready[i]
                for i in self._buffered_jobs(state, m)
            ]
            if ready:
                # Clamp to the current time: a machine that has been idle since before
                # ``state.t`` is available *now*, not retroactively.  Without the clamp the
                # event clock can run backwards and the episode livelocks.
                best = min(best, max(state.t, state.machine_free[m], min(ready)))
        if not math.isfinite(best):
            return math.inf, []
        players: list[Player] = []
        for r in range(inst.n_robots):
            if state.robot_free[r] <= best + EPS:
                if claimable or state.robot_soc[r] < inst.battery.ceiling_mah - EPS:
                    players.append(Player("robot", r))
        for m in range(1, inst.n_machines + 1):
            if state.machine_free[m] <= best + EPS and any(
                state.job_ready[i] <= best + EPS for i in self._buffered_jobs(state, m)
            ):
                players.append(Player("machine", m))
        players.sort()
        return best, players

    def advance(self, state: State) -> bool:
        """Jump to the next future event.  ``False`` when time cannot advance."""
        future = [
            x for x in state.future_event_times() if x > state.t + EPS
        ]
        if not future:
            return False
        state.t = min(future)
        return True

    # -- action sets ----------------------------------------------------------------------
    def feasible_actions(
        self, state: State, player: Player, t: float | None = None
    ) -> list[Action]:
        """Feasible, deadline-safe actions of ``player`` at time ``t``.

        Actions failing the *battery* predicate ``B_needed`` or the *deadline* lower bound
        are excluded; masking on a lower bound can never remove a genuinely feasible
        action, only fail to remove an infeasible one, which is the safe direction.

        **The null action is a baseline, not a strategy.**  ``IDLE`` is always the
        reference against which marginal contribution is measured, but it is removed from
        the *choice* set whenever the player has productive work available.  Two reasons,
        and both are modelling points rather than implementation convenience.  First it is
        redundant: deferral is already a first-class action through the ToU-aligned delay
        grid ``d in D``, so a machine that wants to wait for the off-peak band says so with
        a delay, not by refusing to act.  Second, leaving it in is *degenerate*:
        ``u_p(a_p^0) = 0`` identically, while a real action must beat a completion policy
        that will simply do the same work a moment later, so idling weakly dominates and
        best-response dynamics stall the shop at the first event where every clock has
        caught up.  A robot whose only options are charging keeps ``IDLE``, because
        charging really is optional and forcing it would buy energy nobody needs.
        """
        now = state.t if t is None else t
        actions: list[Action] = [IDLE]
        if player.kind == "robot":
            for job, stage in self._claimable_transports(state):
                act = Action(kind="transport", job=job, stage=stage)
                pv = self.preview(state, player, act, now)
                if pv.feasible and pv.deadline_safe:
                    actions.append(act)
            if state.robot_soc[player.index] < self.inst.battery.ceiling_mah - EPS:
                for delay in self._charge_delays(state, player, now):
                    act = Action(kind="charge", delay=delay)
                    pv = self.preview(state, player, act, now)
                    if pv.feasible and pv.deadline_safe:
                        actions.append(act)
        else:
            for job in self._buffered_jobs(state, player.index):
                if state.job_ready[job] > now + EPS:
                    continue
                stage = state.job_stage[job]
                for delay in self._process_delays(state, player, job, now):
                    act = Action(kind="process", job=job, stage=stage, delay=delay)
                    pv = self.preview(state, player, act, now)
                    if pv.feasible and pv.deadline_safe:
                        actions.append(act)
        return _drop_redundant_idle(actions)

    def actions_or_fallback(
        self, state: State, player: Player, t: float | None = None
    ) -> list[Action]:
        """Feasible actions, or the deadline fallback when the mask leaves nothing.

        Records the fallback on the state so the **deadlock rate** is reported rather than
        silently absorbed.
        """
        actions = self.feasible_actions(state, player, t)
        if any(not a.is_null for a in actions):
            return actions
        fallback = self.fallback_action(state, player, t)
        if fallback.is_null:
            return actions
        state.used_fallback = True
        return [IDLE, fallback]

    def fallback_action(
        self, state: State, player: Player, t: float | None = None
    ) -> Action:
        """Deadline-feasible fallback: the reference completion rule ``pi_0``.

        Fired when the deadline mask leaves a player with nothing to do.  It delegates to
        :meth:`reference_action` rather than inventing a second rule, for a reason worth
        stating: ``pi_0`` is *deadline-directed* -- it never defers into a cheaper tariff
        band and it charges only when no transport is battery-feasible -- so it is the
        policy for which a deadline-feasible completion is actually plausible.  A fallback
        built on "minimise this action's own completion time" would happily send an idle
        robot to top up its battery and lose the deadline while doing it.

        Every use is recorded (``State.used_fallback``) and reported as the **deadlock
        rate**: a method that meets the deadline only by falling back most of the time is
        not a scheduler (ROADMAP.md §3.0(2)).
        """
        now = state.t if t is None else t
        return self.reference_action(state, player, now)

    # -- preview / commit ------------------------------------------------------------------
    def preview(
        self, state: State, player: Player, action: Action, t: float | None = None
    ) -> Preview:
        """Consequences of ``action`` for ``player`` at ``t``.  Pure."""
        inst = self.inst
        now = state.t if t is None else t
        if action.is_null:
            return Preview(
                feasible=True,
                ret=0.0,
                cost=0.0,
                kwh=0.0,
                peak_kwh=0.0,
                start=now,
                end=now,
                blocks=0,
                soc_after=(
                    state.robot_soc[player.index] if player.kind == "robot" else 0.0
                ),
                charger=-1,
                projected_completion=now,
                deadline_safe=True,
            )
        if player.kind == "machine":
            return self._preview_process(state, player.index, action, now)
        if action.kind == "transport":
            return self._preview_transport(state, player.index, action, now)
        return self._preview_charge(state, player.index, action, now)

    def commit(
        self, state: State, player: Player, action: Action, t: float | None = None
    ) -> Preview:
        """Apply ``action``.  Mutates ``state``.  Returns the applied preview."""
        now = state.t if t is None else t
        pv = self.preview(state, player, action, now)
        if action.is_null or not pv.feasible:
            return pv
        state.n_decisions += 1
        state.e_cost += pv.cost
        state.energy_kwh += pv.kwh
        state.peak_kwh += pv.peak_kwh
        state.cmax_stage = max(state.cmax_stage, pv.ret)
        if player.kind == "machine":
            m = player.index
            state.machine_free[m] = pv.end
            state.job_ready[action.job] = pv.end
            state.job_stage[action.job] += 1
            state.log.append(("process", m, action.job, pv.start, pv.end))
        elif action.kind == "transport":
            r = player.index
            task = self._transport(action.job, action.stage // 2)
            state.robot_free[r] = pv.end
            state.robot_loc[r] = task.dest
            state.robot_soc[r] = pv.soc_after
            state.job_ready[action.job] = pv.end
            state.job_loc[action.job] = task.dest
            state.job_stage[action.job] += 1
            if task.is_return:
                state.job_completion[action.job] = pv.end
            state.log.append(("transport", r, action.job, pv.start, pv.end))
        else:
            r = player.index
            state.robot_free[r] = pv.end
            state.robot_loc[r] = self.inst.charger_location
            state.robot_soc[r] = pv.soc_after
            state.charger_free[pv.charger] = pv.end
            state.charge_blocks += pv.blocks
            state.charger_busy_min += pv.end - pv.start
            state.log.append(("charge", r, -1, pv.start, pv.end))
        return pv

    # -- running --------------------------------------------------------------------------
    def run(
        self,
        policy: Policy,
        rng: np.random.Generator | None = None,
        max_events: int = 100_000,
    ) -> Outcome:
        """Run one full episode under ``policy`` and score it with the one objective."""
        generator = rng if rng is not None else np.random.default_rng(0)
        state = self.reset()
        for _ in range(max_events):
            if state.done(self.inst):
                break
            t, players = self.next_event(state)
            if not players:
                if not self.advance(state):
                    break
                continue
            state.t = t
            chosen = policy(self, state, players, generator)
            committed = False
            for player in sorted(chosen):
                action = chosen[player]
                if action.is_null:
                    continue
                pv = self.commit(state, player, action, t)
                committed = committed or pv.feasible
            if committed:
                continue
            if self.advance(state):
                continue
            # Deadlock guard (risk R13): nothing was committed and the clock cannot move,
            # yet jobs remain.  Force every active player onto its deadline-feasible
            # fallback and record it -- the deadlock rate is a reported metric, never an
            # absorbed exception.
            state.used_fallback = True
            forced = False
            for player in players:
                action = self.fallback_action(state, player, t)
                if action.is_null:
                    continue
                forced = self.commit(state, player, action, t).feasible or forced
            if not forced:
                break
        return self.outcome(state)

    def outcome(self, state: State) -> Outcome:
        """Score a (usually terminal) state."""
        c_max = state.c_max()
        incomplete = any(c < 0.0 for c in state.job_completion)
        if incomplete:
            # An unfinished episode is scored as a deadline violation, never silently
            # dropped: the penalty makes it worse than any feasible schedule.
            c_max = max(c_max, self.inst.deadline * 2.0)
        return Outcome(
            c_max=c_max,
            e_cost=state.e_cost,
            energy_kwh=state.energy_kwh,
            peak_kwh=state.peak_kwh,
            charge_blocks=state.charge_blocks,
            deadline_met=(not incomplete) and c_max <= self.inst.deadline + EPS,
            used_fallback=state.used_fallback,
        )

    def phi(self, outcome: Outcome) -> float:
        """``Phi`` of an outcome under this engine's ``omega``."""
        return phi(outcome, self.anchors, self.omega, self.penalty)

    # -- the stage game (T1) ----------------------------------------------------------------
    def evaluate(
        self, state: State, joint: dict[Player, Action], t: float | None = None
    ) -> float:
        """The stage evaluation ``Phi_hat(s_q, alpha)`` the agents actually optimise.

        Two evaluators are available (``Engine.evaluator``), and **T1 holds for either**:
        the exact-potential identity follows from cancellation and needs only that
        ``Phi_hat`` be a *deterministic function of* ``(s_q, alpha)``.  What separates them
        is whether the resulting game is meaningful.

        ``"rollout"`` (default) commits ``alpha`` and completes the episode with the fixed
        deterministic reference policy ``pi_0``, then scores the finished schedule with the
        one objective.  This is the draft's ``Phi(Gamma(s_q, alpha))``, made well defined by
        pinning the completion rule.

        ``"stage"`` is the draft's stage-indexed surrogate
        ``C^q = max(C^{q-1}, max_p R_p(a_p))``, ``E^q = E^{q-1} + sum_p cost_e(a_p)``.
        It is retained because it is what the draft specifies and because experiment E13
        reports it -- but it is **degenerate as a decision rule**: it accumulates cost
        already incurred and never credits work done, so the null action weakly dominates
        every real action and best-response dynamics idle the shop to a standstill.  That
        is a finding about the draft's stage formulation, not an implementation choice.
        """
        if self.evaluator == "stage":
            return self.myopic_phi(state, joint, t)
        return self.rollout_phi(state, joint, t)

    def rollout_phi(
        self, state: State, joint: dict[Player, Action], t: float | None = None
    ) -> float:
        """``Phi`` of the schedule obtained by committing ``alpha`` then following ``pi_0``.

        The **null action is a one-event deferral, not a no-op.**  An idling player is made
        unavailable until the next event; only then does ``pi_0`` take over for it.  This
        matters and is easy to get wrong: if idling merely handed the player's turn to the
        completion policy, the null action would evaluate as "let ``pi_0`` decide for me",
        would weakly dominate every real action whenever ``pi_0`` is competent, and
        best-response dynamics would idle the shop to a standstill -- an equilibrium of a
        game nobody intended.  Deferral makes ``u_p`` measure what the draft says it
        measures: the value of *this* player acting *now*.
        """
        now = state.t if t is None else t
        snap = state.snapshot()
        idlers = [p for p, a in joint.items() if a.is_null]
        for player in sorted(joint):
            if not joint[player].is_null:
                self.commit(snap, player, joint[player], now)
        if idlers:
            future = [x for x in snap.future_event_times() if x > now + EPS]
            defer_to = min(future) if future else now + self._defer_quantum
            for player in idlers:
                if player.kind == "robot":
                    snap.robot_free[player.index] = max(
                        snap.robot_free[player.index], defer_to
                    )
                else:
                    snap.machine_free[player.index] = max(
                        snap.machine_free[player.index], defer_to
                    )
        self.rollout(snap)
        return self.phi(self.outcome(snap))

    def rollout(self, state: State, max_events: int = 100_000) -> State:
        """Complete an episode in place under the reference completion policy ``pi_0``."""
        for _ in range(max_events):
            if state.done(self.inst):
                break
            t, players = self.next_event(state)
            if not players:
                if not self.advance(state):
                    break
                continue
            state.t = t
            committed = False
            for player in players:
                action = self.reference_action(state, player, t)
                if action.is_null:
                    continue
                pv = self.commit(state, player, action, t)
                committed = committed or pv.feasible
            if not committed and not self.advance(state):
                break
        return state

    def reference_action(
        self, state: State, player: Player, t: float
    ) -> Action:
        """``pi_0``: the fast, deterministic completion rule used inside rollouts.

        Deliberately cheap -- it is on the innermost loop of every utility evaluation and
        every coalition-cost evaluation.  Machines take the shortest buffered operation and
        start it as early as possible; robots take the transport they can deliver earliest,
        and charge only when no transport is battery-feasible.  Ties break on the
        documented total order, so the rule is a *function*, which is what makes
        ``Phi_hat`` well defined.
        """
        inst = self.inst
        if player.kind == "machine":
            best: tuple[tuple[float, int], Action] | None = None
            for job in self._buffered_jobs(state, player.index):
                if state.job_ready[job] > t + EPS:
                    continue
                stage = state.job_stage[job]
                k = (stage - 1) // 2
                act = Action(kind="process", job=job, stage=stage, delay=0.0)
                key = (inst.jobs[job][k].duration, job)
                if best is None or key < best[0]:
                    best = (key, act)
            return best[1] if best is not None else IDLE
        best_r: tuple[tuple[float, int], Action] | None = None
        for job, stage in self._claimable_transports(state):
            act = Action(kind="transport", job=job, stage=stage)
            pv = self.preview(state, player, act, t)
            if not pv.feasible:
                continue
            key = (pv.end, job)
            if best_r is None or key < best_r[0]:
                best_r = (key, act)
        if best_r is not None:
            return best_r[1]
        charge = Action(kind="charge", delay=0.0)
        return charge if self.preview(state, player, charge, t).feasible else IDLE

    def myopic_phi(
        self, state: State, joint: dict[Player, Action], t: float | None = None
    ) -> float:
        """The stage objective ``Phi_hat(s_q, alpha)``.

        Implements the draft's stage-indexed makespan and energy

            C^q     = max( C^{q-1}, max_p R_p^q(a_p) )
            E^q     = E^{q-1} + sum_p cost_e(a_p)

        normalised and combined by :func:`jsspt_tou.domain.objective.phi`.

        This function depends **only** on ``(s_q, alpha)`` -- not on the downstream rollout.
        That is precisely the scope at which the exact-potential property T1 holds; the
        draft asserted it for the trajectory-dependent objective, where it does not survive
        (finding F5).  T2 states the honest dynamic-scope version.
        """
        now = state.t if t is None else t
        # Apply the joint action through the *execution rule* on a snapshot, in the
        # documented total order.  Conflicts -- two robots claiming the same transport --
        # are resolved exactly as they would be at commit time: the later player's action
        # is infeasible and degenerates to idling.  This keeps Phi_hat a well-defined
        # function of (s_q, alpha), which is all T1's cancellation argument needs.
        snap = state.snapshot()
        for player in sorted(joint):
            self.commit(snap, player, joint[player], now)
        outcome = Outcome(
            c_max=snap.cmax_stage,
            e_cost=snap.e_cost,
            energy_kwh=0.0,
            peak_kwh=0.0,
            charge_blocks=0,
            deadline_met=snap.cmax_stage <= self.inst.deadline + EPS,
        )
        return phi(outcome, self.anchors, self.omega, self.penalty)

    def utility(
        self,
        state: State,
        player: Player,
        action: Action,
        others: dict[Player, Action],
        t: float | None = None,
    ) -> float:
        """Marginal-contribution utility ``u_p``, in the project's single convention.

        ``u_p(a_p, alpha_{-p}) = Phi_hat(a_p^0, alpha_{-p}) - Phi_hat(a_p, alpha_{-p})``:
        the cost *reduction* this player's action produces relative to idling, everything
        else held fixed.  **All players maximise** (ROADMAP.md §3.3; repairs F4/A11/A12,
        where the draft defined a cost increase, said players minimise it, then defined a
        cost reduction and said they maximise it).

        Equivalently ``u_p = W - W(a_p^0, .)`` with welfare ``W = Phi_hat(alpha^0) -
        Phi_hat(alpha)``: the ``W(alpha^0)`` term is independent of ``p``'s action and
        cancels, which is exactly the one-line proof of T1.
        """
        with_action = dict(others)
        with_action[player] = action
        without = dict(others)
        without[player] = IDLE
        return self.evaluate(state, without, t) - self.evaluate(state, with_action, t)

    def utilities(
        self,
        state: State,
        player: Player,
        actions: Sequence[Action],
        others: dict[Player, Action],
        t: float | None = None,
    ) -> list[float]:
        """``u_p`` for a whole candidate set, sharing the null baseline.

        ``Phi_hat(a_p^0, alpha_{-p})`` does not depend on which action ``p`` is evaluating,
        so it is computed once instead of once per candidate.  That halves the rollout
        count of every best-response sweep, which is the dominant cost of both M1 and the
        coalition-cost evaluations of M2.
        """
        without = dict(others)
        without[player] = IDLE
        baseline = self.evaluate(state, without, t)
        out: list[float] = []
        for action in actions:
            profile = dict(others)
            profile[player] = action
            out.append(baseline - self.evaluate(state, profile, t))
        return out

    # -- internals ---------------------------------------------------------------------------
    def _transport(self, job: int, k: int):  # type: ignore[no-untyped-def]
        return self.inst.transports_of_job(job)[k]

    def _claimable_transports(self, state: State) -> list[tuple[int, int]]:
        """Transport tasks whose part is placed and whose stage nobody is executing."""
        out: list[tuple[int, int]] = []
        for i in range(self.inst.n_jobs):
            stage = state.job_stage[i]
            if stage % 2 == 0 and stage <= 2 * len(self.inst.jobs[i]):
                out.append((i, stage))  # stage 2k <-> transport leg k
        return out

    def _buffered_jobs(self, state: State, machine: int) -> list[int]:
        """Jobs sitting in ``machine``'s input buffer awaiting processing."""
        out: list[int] = []
        for i in range(self.inst.n_jobs):
            stage = state.job_stage[i]
            if stage % 2 == 1:
                k = (stage - 1) // 2
                # stage == 2*m_i + 1 marks a finished job; it has no k-th operation.
                if k < len(self.inst.jobs[i]) and self.inst.jobs[i][k].machine == machine:
                    out.append(i)
        return out

    def _delay_grid(self, earliest: float, latest: float) -> list[float]:
        """Finite ToU-aligned delay grid: zero, plus the next tariff-period starts."""
        delays = [0.0]
        for b in self._boundaries:
            if b > earliest + EPS and b <= latest + EPS:
                delays.append(b - earliest)
                if len(delays) > MAX_DELAY_OPTIONS:
                    break
        return delays

    def _process_delays(
        self, state: State, player: Player, job: int, now: float
    ) -> list[float]:
        stage = state.job_stage[job]
        k = (stage - 1) // 2
        op = self.inst.jobs[job][k]
        earliest = max(now, state.machine_free[player.index], state.job_ready[job])
        latest = self.inst.deadline - op.duration
        return self._delay_grid(earliest, latest)

    def _charge_delays(self, state: State, player: Player, now: float) -> list[float]:
        r = player.index
        arrive = max(now, state.robot_free[r]) + self.inst.travel(
            state.robot_loc[r], self.inst.charger_location
        )
        earliest = max(arrive, min(state.charger_free))
        return self._delay_grid(earliest, self.inst.deadline)

    def _preview_process(
        self, state: State, machine: int, action: Action, now: float
    ) -> Preview:
        inst = self.inst
        k = (action.stage - 1) // 2
        if action.stage != state.job_stage[action.job] or action.stage % 2 == 0:
            return _infeasible(now)
        op = inst.jobs[action.job][k]
        if op.machine != machine:
            return _infeasible(now)
        earliest = max(now, state.machine_free[machine], state.job_ready[action.job])
        start = earliest + action.delay
        end = start + op.duration
        cost = inst.tariff.cost(inst.machine_power_kw, start, end)
        kwh = inst.machine_power_kw * op.duration / 60.0
        peak = self._peak_kwh(inst.machine_power_kw, start, end)
        rest = self._suffix[action.job][action.stage + 1]
        proj = end + rest
        return Preview(
            feasible=True,
            ret=end,
            cost=cost,
            kwh=kwh,
            peak_kwh=peak,
            start=start,
            end=end,
            blocks=0,
            soc_after=0.0,
            charger=-1,
            projected_completion=proj,
            deadline_safe=proj <= inst.deadline + EPS,
        )

    def _preview_transport(
        self, state: State, robot: int, action: Action, now: float
    ) -> Preview:
        inst = self.inst
        bat = inst.battery
        if action.stage != state.job_stage[action.job] or action.stage % 2 == 1:
            return _infeasible(now)
        k = action.stage // 2
        legs = inst.transports_of_job(action.job)
        if k >= len(legs):
            return _infeasible(now)
        task = legs[k]
        depart = max(now, state.robot_free[robot])
        # Idle drain accrues from the moment the robot became free, not only while it waits
        # at a pickup.  Charging it only for the pickup wait silently understates depletion
        # and makes the simulator's feasible set *larger* than the exact model's, which
        # showed up as EX-CP returning a proven optimum worse than a heuristic schedule.
        idle_before = depart - state.robot_free[robot]
        to_pickup = inst.travel(state.robot_loc[robot], task.origin)
        arrive = depart + to_pickup
        pickup = max(arrive, state.job_ready[action.job])
        wait = pickup - arrive
        delivery = pickup + task.duration
        # Battery predicate B_needed (erratum A25: everything in mAh, floor = 20 000 mAh).
        need = b_needed(
            bat,
            travel_to_pickup=to_pickup,
            wait_at_pickup=idle_before + wait,
            loaded_travel=task.duration,
            travel_to_charger=inst.travel(task.dest, inst.charger_location),
        )
        soc = state.robot_soc[robot]
        if soc + EPS < need:
            return _infeasible(now)
        soc_after = (
            soc
            - bat.idle_mah_min * idle_before
            - bat.empty_mah_min * to_pickup
            - bat.idle_mah_min * wait
            - bat.loaded_mah_min * task.duration
        )
        rest = self._suffix[action.job][action.stage + 1]
        proj = delivery + rest
        # Robot traction energy is zero *in the cost*: it is paid for later, at charging
        # time, when the grid draw actually happens.  The depletion from the same travel is
        # fully modelled above.  Making this explicit answers finding F8.
        return Preview(
            feasible=True,
            ret=delivery,
            cost=0.0,
            kwh=0.0,
            peak_kwh=0.0,
            start=depart,
            end=delivery,
            blocks=0,
            soc_after=soc_after,
            charger=-1,
            projected_completion=proj,
            deadline_safe=proj <= inst.deadline + EPS,
        )

    def _preview_charge(
        self, state: State, robot: int, action: Action, now: float
    ) -> Preview:
        inst = self.inst
        bat = inst.battery
        depart = max(now, state.robot_free[robot])
        idle_before = depart - state.robot_free[robot]  # see _preview_transport
        travel = inst.travel(state.robot_loc[robot], inst.charger_location)
        arrive = depart + travel
        # Unary (or K_CH-ary) charger: pick the station free earliest.  This is the
        # constraint the draft omits entirely (F7) and the source of the N5 congestion.
        charger = min(range(inst.n_chargers), key=lambda c: state.charger_free[c])
        earliest = max(arrive, state.charger_free[charger])
        start = earliest + action.delay
        soc_at_start = (
            state.robot_soc[robot]
            - bat.idle_mah_min * idle_before
            - bat.empty_mah_min * travel
            - bat.idle_mah_min * (start - arrive)
        )
        # The SoC floor is an *operational reserve for travel*.  A robot queued at the
        # charging station is stationary and safe, so waiting there may draw the reserve
        # down; only a physically empty pack is infeasible.  In practice the idle rate
        # (60 mAh/min) makes this immaterial against a 20 000 mAh floor, but without the
        # relaxation charger congestion could deadlock a robot that B_needed had certified.
        if soc_at_start <= 0.0:
            return _infeasible(now)
        blocks = bat.blocks_to_full(soc_at_start)
        if blocks <= 0:
            return _infeasible(now)
        end = start + blocks * bat.charge_block_min
        cost = inst.tariff.cost(bat.charger_power_kw, start, end)
        kwh = bat.charger_power_kw * (end - start) / 60.0
        peak = self._peak_kwh(bat.charger_power_kw, start, end)
        soc_after = bat.charge(soc_at_start, blocks)
        return Preview(
            feasible=True,
            ret=end,
            cost=cost,
            kwh=kwh,
            peak_kwh=peak,
            start=start,
            end=end,
            blocks=blocks,
            soc_after=soc_after,
            charger=charger,
            projected_completion=end,
            deadline_safe=end <= inst.deadline + EPS,
        )

    def _peak_kwh(self, power_kw: float, start: float, end: float) -> float:
        """Energy drawn inside the dearest tariff band, for the peak-share metric."""
        total = 0.0
        for period in self.inst.tariff.periods:
            if period.price < self._max_price - 1e-12:
                continue
            lo, hi = max(start, period.start), min(end, period.end)
            if hi > lo:
                total += power_kw * (hi - lo) / 60.0
        return total


def _drop_redundant_idle(actions: list[Action]) -> list[Action]:
    """Remove the null action when productive work is available.  See
    :meth:`Engine.feasible_actions` for why."""
    real = [a for a in actions if not a.is_null]
    if real and any(a.kind != "charge" for a in real):
        return real
    return actions


def _infeasible(now: float) -> Preview:
    return Preview(
        feasible=False,
        ret=math.inf,
        cost=0.0,
        kwh=0.0,
        peak_kwh=0.0,
        start=now,
        end=math.inf,
        blocks=0,
        soc_after=0.0,
        charger=-1,
        projected_completion=math.inf,
        deadline_safe=False,
    )


def _suffix_tables(inst: Instance) -> list[list[float]]:
    """``suffix[i][s]`` = work that must still elapse for job ``i`` from stage ``s``.

    A valid *lower bound* on the remaining span of the job (it ignores every resource
    conflict), which is what makes the deadline mask safe: it can fail to detect an
    infeasible continuation, but it never rejects a feasible one.
    """
    tables: list[list[float]] = []
    for i, ops in enumerate(inst.jobs):
        legs = inst.transports_of_job(i)
        n_stages = 2 * len(ops) + 2
        suffix = [0.0] * n_stages
        acc = 0.0
        for s in range(n_stages - 2, -1, -1):
            if s % 2 == 0:
                acc += legs[s // 2].duration
            else:
                acc += ops[(s - 1) // 2].duration
            suffix[s] = acc
        # suffix[s] currently includes the work *of* stage s; the caller asks for the work
        # strictly after a stage, and passes ``stage + 1``.
        tables.append(suffix)
    return tables


PolicyFn = Callable[
    ["Engine", State, Sequence[Player], np.random.Generator], dict[Player, Action]
]
