"""Simulator state, players and actions.

Implements the state and action sets of Paper A §5 ("State and feasible actions").

Job stages
----------
Job ``i`` moves through ``2 m_i + 1`` stages.  Stage ``2k`` means "transport task ``k`` is
the next thing to happen"; stage ``2k+1`` means "operation ``k`` is next".  The job is
complete once transport ``m_i`` (the return leg to L/U) has been executed, i.e. at stage
``2 m_i + 1``.  ``in_progress[i]`` is set while an agent is executing the current stage,
which is what stops two robots from claiming the same transport task.

Action spaces (erratum A27)
---------------------------
The draft gives three mutually incompatible machine action spaces in three sections --
``(J_i, t_start)`` with a *continuous* start time in one, ``(i, d)`` with ``d`` in a finite
grid in another.  Only the finite one is usable: **T1's finite-improvement argument
requires finite action sets**.  This module fixes the finite ToU-aligned delay grid
everywhere: a machine may start an operation at the earliest feasible time, or defer it to
the start of any later tariff period still inside the deadline.  Robot charging carries the
same lever, which is what lets a robot shift its own load into an off-peak band.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

from jsspt_tou.domain.instance import Instance

PlayerKind = Literal["machine", "robot"]


@dataclass(frozen=True, slots=True, order=True)
class Player:
    """A decision maker: a machine or a robot.

    Ordering is total and documented, so every tie is broken deterministically
    (``(kind, index)``) rather than by dict order.
    """

    kind: PlayerKind
    index: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{'M' if self.kind == 'machine' else 'V'}{self.index}"


@dataclass(frozen=True, slots=True, order=True)
class Action:
    """One action of one player.

    Attributes
    ----------
    kind:
        ``"idle"`` -- the null action ``a_p^0`` against which marginal contribution is
        measured; ``"process"``, ``"transport"`` or ``"charge"``.
    job:
        Job index for ``process``/``transport``.
    stage:
        Job stage the action executes, so an action names a unique task.
    delay:
        ToU-aligned deferral [min] added to the earliest feasible start.
    """

    kind: Literal["idle", "process", "transport", "charge"]
    job: int = -1
    stage: int = -1
    delay: float = 0.0

    @property
    def is_null(self) -> bool:
        return self.kind == "idle"


IDLE = Action(kind="idle")


@dataclass(slots=True)
class State:
    """Mutable simulator state.  Only the engine mutates it.

    ``snapshot`` / ``restore`` give the cheap copy the counterfactual rollouts of the
    marginal-contribution utilities need (one evaluation per candidate action per player
    per stage, so this is on the hot path).
    """

    t: float
    machine_free: list[float]
    robot_free: list[float]
    robot_loc: list[int]
    robot_soc: list[float]
    charger_free: list[float]
    job_stage: list[int]
    job_ready: list[float]
    job_loc: list[int]
    job_in_progress: list[bool]
    job_completion: list[float]
    e_cost: float = 0.0
    energy_kwh: float = 0.0
    peak_kwh: float = 0.0
    charge_blocks: int = 0
    cmax_stage: float = 0.0
    used_fallback: bool = False
    n_decisions: int = 0
    charger_busy_min: float = 0.0
    log: list[tuple[str, int, int, float, float]] = field(default_factory=list)

    @staticmethod
    def initial(inst: Instance) -> "State":
        """State at ``t = 0``: every part at the L/U station, every robot at the charger."""
        bat = inst.battery
        return State(
            t=0.0,
            machine_free=[0.0] * (inst.n_machines + 1),
            robot_free=[0.0] * inst.n_robots,
            robot_loc=[inst.charger_location] * inst.n_robots,
            robot_soc=[bat.start_mah] * inst.n_robots,
            charger_free=[0.0] * inst.n_chargers,
            job_stage=[0] * inst.n_jobs,
            job_ready=[0.0] * inst.n_jobs,
            job_loc=[0] * inst.n_jobs,
            job_in_progress=[False] * inst.n_jobs,
            job_completion=[-1.0] * inst.n_jobs,
        )

    def snapshot(self, with_log: bool = False) -> "State":
        """A deep-enough copy for counterfactual evaluation.

        The event log is dropped by default: the marginal-contribution utilities snapshot
        the state once per candidate action per player per stage, and copying a growing
        history there would dominate the cost of the whole game.
        """
        return replace(
            self,
            machine_free=list(self.machine_free),
            robot_free=list(self.robot_free),
            robot_loc=list(self.robot_loc),
            robot_soc=list(self.robot_soc),
            charger_free=list(self.charger_free),
            job_stage=list(self.job_stage),
            job_ready=list(self.job_ready),
            job_loc=list(self.job_loc),
            job_in_progress=list(self.job_in_progress),
            job_completion=list(self.job_completion),
            log=list(self.log) if with_log else [],
        )

    # -- queries --------------------------------------------------------------------------
    def done(self, inst: Instance) -> bool:
        return all(
            self.job_stage[i] > 2 * len(inst.jobs[i]) for i in range(inst.n_jobs)
        )

    def c_max(self) -> float:
        """Realised makespan: the last **return to L/U** (Bilge--Ulusoy convention).

        Distinct from :attr:`cmax_stage`, which is the draft's stage-indexed *myopic*
        surrogate ``C^q = max(C^{q-1}, max_p R_p(a_p))`` and can include a charging block
        that finishes after the last delivery.  Reported results always use this one.
        """
        return max((c for c in self.job_completion if c >= 0.0), default=0.0)

    def pending_stage(self, inst: Instance, job: int) -> int:
        return self.job_stage[job]

    def future_event_times(self) -> Iterable[float]:
        """Every time strictly in the future at which the state can change."""
        yield from self.machine_free
        yield from self.robot_free
        yield from self.charger_free
        yield from self.job_ready
