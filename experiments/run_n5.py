"""N5 -- the congestion counterexample, and the units-contract binding check.

Produces the registry keys quoted in the proof of Proposition~\\ref{prop:n5} and in the
units contract, so that no numeral in either is typed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jsspt_tou.analysis.registry import Registry
from jsspt_tou.baselines.dispatching import DispatchingPolicy
from jsspt_tou.benchmark.bilge_ulusoy import all_instances
from jsspt_tou.benchmark.congestion import congestion_instance, relaxed_instance
from jsspt_tou.cooperative.characteristic import CharacteristicFunction, all_players
from jsspt_tou.cooperative.core_lp import is_in_core, least_core
from jsspt_tou.cooperative.shapley import exact_shapley
from jsspt_tou.simulator.engine import Engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    reg = Registry(args.out / "numbers_n5.json")

    # --- the congestion instance ---------------------------------------------------------
    engine = Engine(congestion_instance(), omega=0.0)
    cf = CharacteristicFunction(engine=engine, i_max=2)
    cf.enumerate_all()
    players = all_players(engine)
    robots = [p for p in players if p.kind == "robot"]
    pair = frozenset(robots)
    single = frozenset({robots[0]})
    other = frozenset({robots[1]})

    c0, c1, c2, c12 = cf.c(frozenset()), cf.c(single), cf.c(other), cf.c(pair)
    reg.add("n5.c.empty", c0, precision=4, source="N5")
    reg.add("n5.c.single", c1, precision=4, source="N5")
    reg.add("n5.c.pair", c12, precision=4, source="N5")
    reg.add("n5.delta.small", c1 - c0, precision=4, source="N5")
    reg.add("n5.delta.big", c12 - c2, precision=4, source="N5")
    reg.add("n5.v", cf.v(pair), precision=4, source="N5")
    reg.add("n5.v.half", cf.v(pair) / 2.0, precision=4, source="N5")

    lc = least_core(players, cf.v)
    shapley = exact_shapley(players, cf.v)
    inside, worst = is_in_core(shapley.payoff, players, cf.v)
    reg.add("n5.epsilon", lc.epsilon, precision=4, source="N5")
    reg.add("n5.core.nonempty", str(lc.core_nonempty), source="N5")
    reg.add("n5.shapley.incore", str(inside), source="N5")
    reg.add("n5.worst.excess", worst, precision=4, source="N5")

    # --- the two-charger control -----------------------------------------------------------
    engine2 = Engine(relaxed_instance(), omega=0.0)
    cf2 = CharacteristicFunction(engine=engine2, i_max=2)
    cf2.enumerate_all()
    players2 = all_players(engine2)
    robots2 = [p for p in players2 if p.kind == "robot"]
    gain_relaxed = cf2.c(frozenset({robots2[0]})) - cf2.c(frozenset(robots2))
    lc2 = least_core(players2, cf2.v)
    reg.add("n5.gain.congested", c1 - c12, precision=4, source="N5")
    reg.add("n5.gain.relaxed", gain_relaxed, precision=4, source="N5")
    reg.add("n5.epsilon.relaxed", lc2.epsilon, precision=4, source="N5")
    reg.add(
        "n5.epsilon.improvement",
        100.0 * (lc2.epsilon - lc.epsilon) / abs(lc.epsilon) if lc.epsilon else 0.0,
        unit="%", precision=1, source="N5",
    )

    # --- units contract: does the SoC floor bind? -------------------------------------------
    policy = DispatchingPolicy(
        machine_rule="SPT", vehicle_rule="NT", charge_rule="reactive", delay_rule="asap"
    )
    instances = all_instances()
    charging = sum(
        int(Engine(inst, omega=0.5).run(policy, np.random.default_rng(0)).charge_blocks > 0)
        for inst in instances
    )
    reg.add("units.floor.binding.count", charging, precision=0, source="units")
    reg.add(
        "units.floor.binding.rate",
        100.0 * charging / len(instances),
        unit="%", precision=1, source="units",
    )
    reg.add("units.minutes.to.empty", 60_000.0 / 420.0, unit="min", precision=0, source="units")

    reg.write()
    print(json.dumps({k: v.value for k, v in reg.entries.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
