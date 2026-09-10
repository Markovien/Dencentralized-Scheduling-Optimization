"""Dispatching-rule baselines: machine rule x vehicle rule x charge policy.

Implements the ``DR-*`` family of ROADMAP.md §3.1.  These are the cheap, industrially
realistic baselines every other method is measured against, and -- because the roadmap
defines the coalition characteristic function as "coordinate inside ``S``, everybody else
follows a fixed reference policy ``pi_0``" -- one of them *is* ``pi_0``.  They therefore
have to be good, not strawmen (referee check R-4).

All three rule families run through the shared :class:`~jsspt_tou.simulator.engine.Engine`
and are scored by the single objective, so a dispatching row and a game row in the same
table measure the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Sequence

import numpy as np

from jsspt_tou.simulator.engine import Engine
from jsspt_tou.simulator.state import IDLE, Action, Player, State

MachineRule = Literal["SPT", "LPT", "FIFO", "MWKR", "LWKR"]
VehicleRule = Literal["NT", "EFT", "FIFO", "LWKR"]
ChargeRule = Literal["reactive", "threshold", "tou"]
DelayRule = Literal["asap", "tou"]

MACHINE_RULES: Final[tuple[MachineRule, ...]] = ("SPT", "LPT", "FIFO", "MWKR", "LWKR")
VEHICLE_RULES: Final[tuple[VehicleRule, ...]] = ("NT", "EFT", "FIFO", "LWKR")
CHARGE_RULES: Final[tuple[ChargeRule, ...]] = ("reactive", "threshold", "tou")


@dataclass(frozen=True, slots=True)
class DispatchingPolicy:
    """A composite priority rule.

    Parameters
    ----------
    machine_rule:
        Which buffered job an idle machine takes.  ``SPT``/``LPT`` on processing time,
        ``FIFO`` on arrival in the buffer, ``MWKR``/``LWKR`` on the job's remaining work.
    vehicle_rule:
        Which transport task a free robot takes.  ``NT`` nearest task (least empty
        travel), ``EFT`` earliest delivery, ``FIFO`` longest-waiting part, ``LWKR`` the
        part closest to finishing.
    charge_rule:
        ``reactive`` charges only when no transport is battery-feasible; ``threshold``
        charges below ``soc_threshold`` of capacity; ``tou`` additionally waits for the
        cheapest tariff band when the battery still allows it -- the energy-aware variant.
    delay_rule:
        ``asap`` starts every operation as early as possible; ``tou`` picks the ToU-aligned
        delay that minimises the myopic stage objective, i.e. it uses the same deferral
        lever the game gives machines.
    soc_threshold:
        Fraction of *capacity* below which ``threshold``/``tou`` seek a charge.
    """

    machine_rule: MachineRule = "SPT"
    vehicle_rule: VehicleRule = "NT"
    charge_rule: ChargeRule = "reactive"
    delay_rule: DelayRule = "asap"
    soc_threshold: float = 0.35

    @property
    def name(self) -> str:
        return (
            f"DR-{self.machine_rule}-{self.vehicle_rule}-"
            f"{self.charge_rule}-{self.delay_rule}"
        )

    # -- the Policy protocol -------------------------------------------------------------
    def __call__(
        self,
        engine: Engine,
        state: State,
        players: Sequence[Player],
        rng: np.random.Generator,
    ) -> dict[Player, Action]:
        """Choose one action per active player.

        Players are served in the documented total order and each choice is made against
        the *already updated* claim set, so two robots never target the same task.
        """
        chosen: dict[Player, Action] = {}
        claimed: set[tuple[int, int]] = set()
        for player in sorted(players):
            actions = engine.actions_or_fallback(state, player)
            actions = [
                a
                for a in actions
                if not (a.kind == "transport" and (a.job, a.stage) in claimed)
            ]
            action = self._select(engine, state, player, actions)
            if action.kind == "transport":
                claimed.add((action.job, action.stage))
            chosen[player] = action
        return chosen

    # -- selection ------------------------------------------------------------------------
    def _select(
        self,
        engine: Engine,
        state: State,
        player: Player,
        actions: Sequence[Action],
    ) -> Action:
        real = [a for a in actions if not a.is_null]
        if not real:
            return IDLE
        if player.kind == "machine":
            return self._select_machine(engine, state, real)
        return self._select_robot(engine, state, player, real)

    def _select_machine(
        self, engine: Engine, state: State, actions: Sequence[Action]
    ) -> Action:
        inst = engine.inst
        best: tuple[tuple[float, ...], Action] | None = None
        for act in actions:
            k = (act.stage - 1) // 2
            op = inst.jobs[act.job][k]
            remaining = sum(o.duration for o in inst.jobs[act.job][k:])
            if self.machine_rule == "SPT":
                prio = op.duration
            elif self.machine_rule == "LPT":
                prio = -op.duration
            elif self.machine_rule == "FIFO":
                prio = state.job_ready[act.job]
            elif self.machine_rule == "MWKR":
                prio = -remaining
            else:  # LWKR
                prio = remaining
            key = (prio, self._delay_key(engine, state, Player("machine", op.machine), act))
            if best is None or key < best[0]:
                best = (key, act)
        assert best is not None
        return best[1]

    def _delay_key(
        self, engine: Engine, state: State, player: Player, act: Action
    ) -> float:
        """Tie-break across the ToU delay grid: 0 for ASAP, myopic cost for the ToU rule."""
        if self.delay_rule == "asap":
            return act.delay
        pv = engine.preview(state, player, act)
        return pv.cost

    def _select_robot(
        self,
        engine: Engine,
        state: State,
        player: Player,
        actions: Sequence[Action],
    ) -> Action:
        inst = engine.inst
        bat = inst.battery
        soc = state.robot_soc[player.index]
        transports = [a for a in actions if a.kind == "transport"]
        charges = [a for a in actions if a.kind == "charge"]

        wants_charge = False
        if charges:
            if self.charge_rule == "reactive":
                wants_charge = not transports
            else:
                low = soc < self.soc_threshold * bat.capacity_mah
                wants_charge = low or not transports
                if self.charge_rule == "tou" and low and transports:
                    # Energy-aware variant: only pre-empt work for a charge if the cheapest
                    # band is reachable, otherwise keep working and charge later.
                    cheapest = min(engine.preview(state, player, a).cost for a in charges)
                    now_cost = engine.preview(
                        state, player, Action(kind="charge", delay=0.0)
                    ).cost
                    wants_charge = cheapest < now_cost - 1e-12 or soc < (
                        0.5 * self.soc_threshold * bat.capacity_mah
                    )
        if wants_charge and charges:
            return min(charges, key=lambda a: (engine.preview(state, player, a).cost, a.delay))
        if not transports:
            return charges[0] if charges else IDLE

        best: tuple[tuple[float, ...], Action] | None = None
        for act in transports:
            pv = engine.preview(state, player, act)
            task = inst.transports_of_job(act.job)[act.stage // 2]
            empty = inst.travel(state.robot_loc[player.index], task.origin)
            k = act.stage // 2
            remaining = sum(o.duration for o in inst.jobs[act.job][k:])
            if self.vehicle_rule == "NT":
                prio = empty
            elif self.vehicle_rule == "EFT":
                prio = pv.end
            elif self.vehicle_rule == "FIFO":
                prio = state.job_ready[act.job]
            else:  # LWKR
                prio = remaining
            key = (prio, float(act.job), float(act.stage))
            if best is None or key < best[0]:
                best = (key, act)
        assert best is not None
        return best[1]


REFERENCE_POLICY: Final[DispatchingPolicy] = DispatchingPolicy(
    machine_rule="SPT",
    vehicle_rule="EFT",
    charge_rule="tou",
    delay_rule="tou",
)
"""The energy-aware composite rule used as the headline dispatching baseline.

**This is not** ``pi_0``.  The reference completion policy that agents outside a coalition
follow is :meth:`jsspt_tou.simulator.engine.Engine.reference_action` -- a deliberately fast,
deadline-safe rule, because it sits on the innermost loop of every utility and every
coalition-cost evaluation.  Keeping the two distinct matters: ``c(0)`` is defined against
``pi_0``, so quoting a different policy as the reference would misstate every cooperative
saving in the paper.  Experiment E1 reports both, so the gap between them is visible.

Note that this rule can *miss the deadline* on instances where deferring into the off-peak
band leaves too little slack -- risk R13, measured rather than hidden.
"""


def rule_grid() -> list[DispatchingPolicy]:
    """The full baseline family swept in experiment E1."""
    out: list[DispatchingPolicy] = []
    for mr in MACHINE_RULES:
        for vr in VEHICLE_RULES:
            for cr in CHARGE_RULES:
                for dr in ("asap", "tou"):
                    out.append(
                        DispatchingPolicy(
                            machine_rule=mr,
                            vehicle_rule=vr,
                            charge_rule=cr,
                            delay_rule=dr,  # type: ignore[arg-type]
                        )
                    )
    return out
