"""E2 -- where does the exact model break down?

ROADMAP.md §5.4, block E2.  Three nested models are solved on the same instances under the
same wall-clock budget:

``makespan``   the classical variant: flat tariff, no deadline, no battery.  This is the
               model the vendored solver already solves, and reproducing its optimum exactly
               is the provenance regression (referee check R-11).
``tou``        adds the ToU-indexed processing cost and the deadline.
``full``       adds charging, charger capacity and exact prefix state of charge.

The point of the block is not to produce optima -- it is to locate the wall.  Pricing energy
against the tariff destroys the makespan-optimal plateau the classical model relies on, and
the state-of-charge recursion couples the vehicle routes; the result is that a model which
closes a four-machine instance in seconds stops closing the same instance at all.  That is
the concrete form of the scalability argument that motivates decentralisation, and it is
reported as evidence rather than as a limitation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jsspt_tou.analysis.registry import Registry
from jsspt_tou.benchmark.bilge_ulusoy import build_instance
from jsspt_tou.domain.instance import Instance
from jsspt_tou.domain.objective import Outcome
from jsspt_tou.exact.cpsat_model import hint_from_schedule, solve_exact
from jsspt_tou.game.best_response import BestResponsePolicy
from jsspt_tou.simulator.engine import Engine


@dataclass(frozen=True, slots=True)
class Config:
    """One rung of the nested-model ladder.

    Held as a frozen record rather than a dict of keyword arguments: the runner used to
    ``pop`` the weight out of a module-level dict and put it back afterwards, which made the
    table mutable state shared across instances.
    """

    enforce_deadline: bool
    enforce_battery: bool
    flat_tariff: bool
    omega: float


CONFIGS: dict[str, Config] = {
    "makespan": Config(enforce_deadline=False, enforce_battery=False, flat_tariff=True, omega=1.0),
    "tou": Config(enforce_deadline=True, enforce_battery=False, flat_tariff=False, omega=0.5),
    "full": Config(enforce_deadline=True, enforce_battery=True, flat_tariff=False, omega=0.5),
}


def m1_schedule(
    inst: Instance, omega: float
) -> tuple[Engine, Outcome, list[tuple[str, int, int, float, float]]]:
    """Play M1a once and return (outcome, event log) for the warm start."""
    engine = Engine(inst, omega=omega)
    policy = BestResponsePolicy(i_max=6)
    rng = np.random.default_rng(0)
    state = engine.reset()
    for _ in range(5000):
        if state.done(inst):
            break
        t, players = engine.next_event(state)
        if not players:
            if not engine.advance(state):
                break
            continue
        state.t = t
        chosen = policy(engine, state, players, rng)
        committed = False
        for p in sorted(chosen):
            if not chosen[p].is_null:
                committed = engine.commit(state, p, chosen[p], t).feasible or committed
        if not committed and not engine.advance(state):
            break
    return engine, engine.outcome(state), state.log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--pairs", type=str, default="1:1,5:1,7:3,5:3,1:2,2:1")
    args = parser.parse_args(argv)

    pairs = [tuple(int(x) for x in p.split(":")) for p in args.pairs.split(",")]
    rows = []
    for job_set, layout in pairs:
        inst = build_instance(job_set, layout)
        engine, m1, log = m1_schedule(inst, 0.5)
        hint = hint_from_schedule(inst, log)
        for name, cfg in CONFIGS.items():
            omega = cfg.omega
            result = solve_exact(
                inst,
                omega=omega,
                time_limit_s=args.budget,
                anchors=engine.anchors,
                hint=hint if name != "makespan" else None,
                enforce_deadline=cfg.enforce_deadline,
                enforce_battery=cfg.enforce_battery,
                flat_tariff=cfg.flat_tariff,
            )
            rows.append(
                {
                    "instance": inst.instance_id,
                    "config": name,
                    "omega": omega,
                    "status": result.status,
                    "closed": result.status == "OPTIMAL",
                    "objective": result.objective,
                    "best_bound": result.best_bound,
                    "c_max": result.outcome.c_max if result.outcome else np.nan,
                    "seconds": result.wall_clock,
                    "n_variables": result.n_variables,
                    "n_constraints": result.n_constraints,
                    "m1_phi": engine.phi(m1),
                    "m1_cmax": m1.c_max,
                }
            )
            print(rows[-1], flush=True)

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "e2_exact.parquet")

    reg = Registry(args.out / "numbers_e2.json")
    reg.add("e2.budget.seconds", int(args.budget), unit="s", precision=0, source="E2")
    reg.add("e2.instances", int(frame["instance"].nunique()), precision=0, source="E2")
    for name in CONFIGS:
        sub = frame[frame["config"] == name]
        reg.add(f"e2.{name}.closed", int(sub["closed"].sum()), precision=0, source="E2")
        reg.add(f"e2.{name}.closed.rate", 100.0 * float(sub["closed"].mean()), unit="%", precision=1, source="E2")
        reg.add(f"e2.{name}.seconds.mean", float(sub["seconds"].mean()), unit="s", precision=1, source="E2")
        reg.add(f"e2.{name}.vars.mean", float(sub["n_variables"].mean()), precision=0, source="E2")
        reg.add(f"e2.{name}.cons.mean", float(sub["n_constraints"].mean()), precision=0, source="E2")
    closed_ms = frame[(frame["config"] == "makespan") & frame["closed"]]
    if not closed_ms.empty:
        reg.add("e2.makespan.seconds.max", float(closed_ms["seconds"].max()), unit="s", precision=2, source="E2")

    # Where the full model *closes*, the proven optimum is a genuine optimality reference
    # and M1a's distance from it is measurable.  This is the only place in the paper where
    # a gap to optimality is reported, and it is reported only on the closed subset.
    closed_full = frame[(frame["config"] == "full") & frame["closed"]].copy()
    reg.add("e2.full.closed.count", int(len(closed_full)), precision=0, source="E2")
    if not closed_full.empty:
        closed_full["gap"] = (
            100.0 * (closed_full["m1_phi"] - closed_full["objective"]) / closed_full["objective"]
        )
        reg.add("e2.m1a.gap.mean", float(closed_full["gap"].mean()), unit="%", precision=1, source="E2")
        reg.add("e2.m1a.gap.max", float(closed_full["gap"].max()), unit="%", precision=1, source="E2")
        reg.add("e2.m1a.gap.min", float(closed_full["gap"].min()), unit="%", precision=1, source="E2")
        reg.add(
            "e2.instances.list",
            ", ".join(sorted(closed_full["instance"])),
            source="E2",
        )
    reg.write()
    print(json.dumps({k: v.value for k, v in reg.entries.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
