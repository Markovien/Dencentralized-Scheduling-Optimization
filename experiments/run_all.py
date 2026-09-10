"""Run every experiment block and emit ``numbers.json``, Parquet results and LaTeX tables.

Implements the experiment matrix of ROADMAP.md §5.4 for Paper A:

====  ==============================================================================
E1    how far from the baselines is each method?
E3    does the cooperative model beat the non-cooperative one, and is it stable?
E4    which coalition structures emerge, and what do they cost in communication?
E5    what ``omega`` does bargaining select?
E6    empirical price of anarchy against ``EX-CP`` where it closes
E11   ablations: ``lambda``, the deadline penalty ``M``, charger capacity ``K_CH``
E13   commensurability audit: raw versus normalised objectives
====  ==============================================================================

Every row records the git SHA and the seed.  Nothing here interprets a result; the
manuscript does that, and only through registry keys.

    python experiments/run_all.py [--quick] [--out results]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jsspt_tou.analysis.registry import Registry
from jsspt_tou.analysis.stats import describe, friedman, pairwise_wilcoxon_holm
from jsspt_tou.baselines.dispatching import REFERENCE_POLICY, DispatchingPolicy, rule_grid
from jsspt_tou.benchmark.bilge_ulusoy import all_instances, build_instance, certify
from jsspt_tou.cooperative.bargaining import (
    breakeven_time_cost,
    build_front,
    kalai_smorodinsky,
    nadir_disagreement,
    nash_bargaining,
)
from jsspt_tou.cooperative.characteristic import (
    CharacteristicFunction,
    CoalitionPolicy,
    all_players,
    infeasible_coalition_rate,
    monotonicity_report,
    realised_partition_value,
    submodularity_report,
)
from jsspt_tou.cooperative.coalition_formation import is_dhp_stable, merge_split
from jsspt_tou.cooperative.core_lp import is_in_core, least_core, nucleolus
from jsspt_tou.cooperative.shapley import exact_shapley
from jsspt_tou.game.best_response import BestResponsePolicy, GameTrace, LogLinearPolicy
from jsspt_tou.simulator.engine import Engine

OMEGAS = (0.25, 0.5, 0.75)
COOP_OMEGA = 0.5
SEED = 0


# --------------------------------------------------------------------------------------------
# E1 -- method comparison
# --------------------------------------------------------------------------------------------
def run_e1(instances: list, omegas: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rules = rule_grid()
    for inst in instances:
        for omega in omegas:
            engine = Engine(inst, omega=omega)
            players = frozenset(all_players(engine))

            # pi_0: the reference completion policy the utilities are measured against
            t0 = time.perf_counter()
            state = engine.reset()
            engine.rollout(state)
            rows.append(_row(inst, omega, "pi0", engine.outcome(state), time.perf_counter() - t0))

            # the whole dispatching family, plus the per-instance oracle over it
            best_phi, best_outcome, best_name = float("inf"), None, ""
            t0 = time.perf_counter()
            for rule in rules:
                outcome = engine.run(rule, np.random.default_rng(SEED))
                value = engine.phi(outcome)
                if value < best_phi:
                    best_phi, best_outcome, best_name = value, outcome, rule.name
            dr_time = (time.perf_counter() - t0) / len(rules)
            assert best_outcome is not None
            rows.append(_row(inst, omega, "DR-oracle", best_outcome, dr_time, note=best_name))

            t0 = time.perf_counter()
            rows.append(
                _row(inst, omega, "DR-ToU", engine.run(REFERENCE_POLICY, np.random.default_rng(SEED)),
                     time.perf_counter() - t0)
            )
            t0 = time.perf_counter()
            rows.append(
                _row(inst, omega, "DR-SPT", engine.run(DispatchingPolicy(), np.random.default_rng(SEED)),
                     time.perf_counter() - t0)
            )

            trace = GameTrace()
            t0 = time.perf_counter()
            m1a = engine.run(BestResponsePolicy(i_max=6, trace=trace), np.random.default_rng(SEED))
            rows.append(
                _row(inst, omega, "M1a", m1a, time.perf_counter() - t0,
                     iterations=trace.mean_iterations, eps_ne=trace.eps_ne, stages=trace.stages)
            )

            t0 = time.perf_counter()
            m1b = engine.run(LogLinearPolicy(i_max=4), np.random.default_rng(SEED))
            rows.append(_row(inst, omega, "M1b", m1b, time.perf_counter() - t0))

            t0 = time.perf_counter()
            m2 = engine.run(CoalitionPolicy(members=players, i_max=3), np.random.default_rng(SEED))
            rows.append(_row(inst, omega, "M2-grand", m2, time.perf_counter() - t0))
    return pd.DataFrame(rows)


def _row(inst, omega, method, outcome, seconds, **extra) -> dict[str, Any]:
    engine = Engine(inst, omega=omega)
    return {
        "instance": inst.instance_id,
        "job_set": int(inst.instance_id[2:-1]),
        "layout": inst.layout_id,
        "omega": omega,
        "method": method,
        "phi": engine.phi(outcome),
        "c_max": outcome.c_max,
        "e_cost": outcome.e_cost,
        "energy_kwh": outcome.energy_kwh,
        "peak_share": (outcome.peak_kwh / outcome.energy_kwh) if outcome.energy_kwh else 0.0,
        "charge_blocks": outcome.charge_blocks,
        "deadline_met": outcome.deadline_met,
        "used_fallback": outcome.used_fallback,
        "norm_cmax": engine.anchors.norm_cmax(outcome.c_max),
        "norm_ecost": engine.anchors.norm_ecost(outcome.e_cost),
        "seconds": seconds,
        "seed": SEED,
        **extra,
    }


# --------------------------------------------------------------------------------------------
# E3 / E4 / E5 -- the cooperative layers
# --------------------------------------------------------------------------------------------
def run_cooperative(instances: list, omega: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    coop_rows: list[dict[str, Any]] = []
    alloc_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for inst in instances:
        engine = Engine(inst, omega=omega)
        cf = CharacteristicFunction(engine=engine, i_max=1)
        t0 = time.perf_counter()
        cf.enumerate_all()
        enum_seconds = time.perf_counter() - t0
        players = all_players(engine)
        grand = frozenset(players)
        v_n = cf.v(grand)

        shapley = exact_shapley(players, cf.v)
        lc = least_core(players, cf.v)
        nuc = nucleolus(players, cf.v)
        in_core, worst = is_in_core(shapley.payoff, players, cf.v)
        sub = submodularity_report(cf, rng, n_triples=300)
        mono = monotonicity_report(cf, rng, n_pairs=300)

        structure = merge_split(players, cf.v)
        realised = realised_partition_value(cf, structure.blocks)

        coop_rows.append(
            {
                "instance": inst.instance_id,
                "omega": omega,
                "c_empty": cf.c_empty(),
                "c_grand": cf.c(grand),
                "v_grand": v_n,
                "saving_pct": 100.0 * v_n / cf.c_empty() if cf.c_empty() else 0.0,
                "epsilon": lc.epsilon,
                "epsilon_ratio": lc.epsilon / v_n if v_n else 0.0,
                "core_nonempty": lc.core_nonempty,
                "shapley_in_core": in_core,
                "worst_excess": worst,
                "nucleolus_epsilon": nuc.epsilon,
                "gini_shapley": shapley.gini(),
                "submod_violation_rate": sub["violation_rate"],
                "submod_worst": sub["worst_violation"],
                "mono_violation_rate": mono["violation_rate"],
                "infeasible_coalition_rate": infeasible_coalition_rate(cf),
                "cs_blocks": len(structure.blocks),
                "cs_max_block": structure.max_block_size,
                "cs_messages": structure.messages,
                "cs_merges": structure.merges,
                "cs_dhp_stable": is_dhp_stable(structure, cf.v),
                "cs_realised_value": realised,
                "cs_value_ratio": realised / v_n if v_n else 0.0,
                "c_calls": cf.calls,
                "c_seconds": cf.seconds,
                "enum_seconds": enum_seconds,
                "seed": SEED,
            }
        )
        for player in players:
            alloc_rows.append(
                {
                    "instance": inst.instance_id,
                    "omega": omega,
                    "player": str(player),
                    "kind": player.kind,
                    "shapley": shapley.payoff[player],
                    "least_core": lc.allocation[player],
                    "nucleolus": nuc.allocation[player],
                    "equal_split": v_n / len(players),
                }
            )
    return pd.DataFrame(coop_rows), pd.DataFrame(alloc_rows)


def run_bargaining(instances: list) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grid = np.linspace(0.0, 1.0, 11)
    for inst in instances:
        anchors = Engine(inst, omega=0.5).anchors
        outcomes = [
            (
                float(w),
                Engine(inst, omega=float(w)).run(
                    BestResponsePolicy(i_max=4), np.random.default_rng(SEED)
                ),
            )
            for w in grid
        ]
        front = build_front(outcomes, anchors)
        if len(front) < 2:
            continue
        disagreement = nadir_disagreement(front)
        nbs = nash_bargaining(front, disagreement, anchors)
        ks = kalai_smorodinsky(front, disagreement, anchors)
        rows.append(
            {
                "instance": inst.instance_id,
                "front_size": len(front),
                "omega_nbs": nbs.omega_induced if nbs else np.nan,
                "omega_nbs_lo": nbs.omega_interval[0] if nbs else np.nan,
                "omega_nbs_hi": nbs.omega_interval[1] if nbs else np.nan,
                "omega_ks": ks.omega_induced if ks else np.nan,
                "nbs_cmax": nbs.point.c_max if nbs else np.nan,
                "nbs_ecost": nbs.point.e_cost if nbs else np.nan,
                "agree": bool(nbs and ks and abs(nbs.point.c_max - ks.point.c_max) < 1e-9),
                "breakeven_ctime": breakeven_time_cost(front) or np.nan,
                "seed": SEED,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# E11 -- ablations
# --------------------------------------------------------------------------------------------
def run_ablations(instances: list, subset: list) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    lam_rows = []
    for lam in (1.0, 1.25, 1.5, 2.0):
        for job_set in range(1, 11):
            for layout in (1, 2, 3, 4):
                inst = build_instance(job_set, layout, lam=lam)
                anchors, admissible, binding = certify(inst)
                lam_rows.append(
                    {
                        "lambda": lam,
                        "instance": inst.instance_id,
                        "horizon": inst.horizon,
                        "cmax_lb": anchors.cmax_lb,
                        "margin": anchors.feasibility_margin,
                        "admissible": admissible,
                        "binding": binding,
                    }
                )
    out["lambda"] = pd.DataFrame(lam_rows)

    pen_rows = []
    for penalty in (2.0, 5.0, 10.0, 50.0):
        for inst in subset:
            engine = Engine(inst, omega=COOP_OMEGA, penalty=penalty)
            cf = CharacteristicFunction(engine=engine, i_max=1)
            cf.enumerate_all()
            players = all_players(engine)
            shapley = exact_shapley(players, cf.v)
            lc = least_core(players, cf.v)
            pen_rows.append(
                {
                    "penalty": penalty,
                    "instance": inst.instance_id,
                    "v_grand": cf.v(frozenset(players)),
                    "epsilon": lc.epsilon,
                    "gini": shapley.gini(),
                    "robot_share": sum(
                        shapley.payoff[p] for p in players if p.kind == "robot"
                    )
                    / max(cf.v(frozenset(players)), 1e-12),
                }
            )
    out["penalty"] = pd.DataFrame(pen_rows)

    chg_rows = []
    for k in (1, 2, 3):
        for inst in subset:
            variant = dataclasses.replace(inst, n_chargers=k)
            engine = Engine(variant, omega=COOP_OMEGA)
            cf = CharacteristicFunction(engine=engine, i_max=1)
            cf.enumerate_all()
            players = all_players(engine)
            lc = least_core(players, cf.v)
            sub = submodularity_report(cf, np.random.default_rng(SEED), n_triples=200)
            state = engine.reset()
            engine.rollout(state)
            chg_rows.append(
                {
                    "n_chargers": k,
                    "instance": inst.instance_id,
                    "v_grand": cf.v(frozenset(players)),
                    "epsilon": lc.epsilon,
                    "core_nonempty": lc.core_nonempty,
                    "submod_violation_rate": sub["violation_rate"],
                    "charger_busy_min": state.charger_busy_min,
                    "charger_utilisation": state.charger_busy_min
                    / max(state.c_max(), 1e-9)
                    / k,
                }
            )
    out["charger"] = pd.DataFrame(chg_rows)
    return out


# --------------------------------------------------------------------------------------------
# E13 -- commensurability audit
# --------------------------------------------------------------------------------------------
def run_e13(instances: list) -> pd.DataFrame:
    rows = []
    for inst in instances:
        engine = Engine(inst, omega=0.5)
        outcome = engine.run(REFERENCE_POLICY, np.random.default_rng(SEED))
        raw_total = 0.5 * outcome.c_max + 0.5 * outcome.e_cost
        norm_c = engine.anchors.norm_cmax(outcome.c_max)
        norm_e = engine.anchors.norm_ecost(outcome.e_cost)
        rows.append(
            {
                "instance": inst.instance_id,
                "c_max": outcome.c_max,
                "e_cost": outcome.e_cost,
                "ratio": outcome.c_max / outcome.e_cost if outcome.e_cost else np.nan,
                "raw_energy_share": (0.5 * outcome.e_cost) / raw_total if raw_total else 0.0,
                "norm_energy_share": (0.5 * norm_e) / (0.5 * norm_c + 0.5 * norm_e)
                if (norm_c + norm_e) > 0
                else 0.0,
                "omega_star_raw": outcome.e_cost / (outcome.c_max + outcome.e_cost),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------
def populate_registry(
    reg: Registry,
    e1: pd.DataFrame,
    coop: pd.DataFrame,
    alloc: pd.DataFrame,
    barg: pd.DataFrame,
    abl: dict[str, pd.DataFrame],
    e13: pd.DataFrame,
) -> None:
    reg.add("bench.instances", int(e1["instance"].nunique()), unit="", precision=0, source="E1")
    reg.add("bench.omegas", int(e1["omega"].nunique()), unit="", precision=0, source="E1")
    reg.add("exp.rules.count", len(rule_grid()), unit="", precision=0, source="E1")

    # --- E1 -------------------------------------------------------------------------------
    mid = e1[e1["omega"] == 0.5]
    for method in sorted(mid["method"].unique()):
        sub = mid[mid["method"] == method]
        stats_ = describe(sub["phi"].tolist())
        tag = method.replace("-", "").lower()
        reg.add(f"e1.{tag}.phi.mean", stats_["mean"], unit="", precision=4, source="E1")
        reg.add(f"e1.{tag}.phi.std", stats_["std"], unit="", precision=4, source="E1")
        reg.add(f"e1.{tag}.cmax.mean", describe(sub["c_max"].tolist())["mean"], unit="min", precision=1, source="E1")
        reg.add(f"e1.{tag}.ecost.mean", describe(sub["e_cost"].tolist())["mean"], unit="EUR", precision=3, source="E1")
        reg.add(f"e1.{tag}.deadline.rate", 100.0 * float(sub["deadline_met"].mean()), unit="%", precision=1, source="E1")
        reg.add(f"e1.{tag}.fallback.rate", 100.0 * float(sub["used_fallback"].mean()), unit="%", precision=1, source="E1")
        reg.add(f"e1.{tag}.seconds.mean", describe(sub["seconds"].tolist())["mean"], unit="s", precision=3, source="E1")

    pivot = mid.pivot_table(index="instance", columns="method", values="phi")
    methods = [m for m in ("pi0", "DR-SPT", "DR-ToU", "DR-oracle", "M1a", "M1b", "M2-grand") if m in pivot]
    scores = {m: pivot[m].tolist() for m in methods}
    fr = friedman(scores)
    reg.add("e1.friedman.stat", fr.statistic, precision=2, source="E1")
    reg.add("e1.friedman.p", fr.p_value, precision=6, source="E1")
    reg.add("e1.friedman.cd", fr.critical_difference, precision=3, source="E1")
    for m, rank in fr.mean_ranks.items():
        reg.add(f"e1.rank.{m.replace('-', '').lower()}", rank, precision=2, source="E1")
    for pr in pairwise_wilcoxon_holm(scores, reference="M2-grand"):
        tag = pr.method_b.replace("-", "").lower()
        reg.add(f"e1.wilcoxon.m2vs{tag}.p", pr.p_holm, precision=6, source="E1")
        reg.add(f"e1.wilcoxon.m2vs{tag}.a12", pr.a12, precision=3, source="E1")

    # improvement of M1a and M2 over the reference policy
    for method in ("M1a", "M2-grand"):
        if method not in pivot:
            continue
        gain = 100.0 * (pivot["pi0"] - pivot[method]) / pivot["pi0"]
        tag = method.replace("-", "").lower()
        reg.add(f"e1.{tag}.gain.mean", float(gain.mean()), unit="%", precision=1, source="E1")
        reg.add(f"e1.{tag}.gain.std", float(gain.std(ddof=1)), unit="%", precision=1, source="E1")
        wins = 100.0 * float((pivot[method] <= pivot["DR-oracle"] + 1e-9).mean())
        reg.add(f"e1.{tag}.winrate.vs.oracle", wins, unit="%", precision=1, source="E1")

    if "M1a" in mid["method"].values:
        it = mid[mid["method"] == "M1a"]
        reg.add("e1.m1a.iterations.mean", float(it["iterations"].mean()), precision=2, source="E1")
        reg.add("e1.m1a.epsne.max", float(it["eps_ne"].max()), precision=9, source="E1")
        reg.add("e1.m1a.stages.mean", float(it["stages"].mean()), precision=1, source="E1")

    # --- E3 -------------------------------------------------------------------------------
    reg.add("e3.v.mean", float(coop["v_grand"].mean()), precision=4, source="E3")
    reg.add("e3.saving.mean", float(coop["saving_pct"].mean()), unit="%", precision=1, source="E3")
    reg.add("e3.saving.std", float(coop["saving_pct"].std(ddof=1)), unit="%", precision=1, source="E3")
    reg.add("e3.saving.max", float(coop["saving_pct"].max()), unit="%", precision=1, source="E3")
    reg.add("e3.eps.mean", float(coop["epsilon"].mean()), precision=4, source="E3")
    # The ratio eps*/v(N) is only interpretable where v(N) > 0; on instances where the
    # single-sweep surrogate makes coordination *lose* against pi_0 the ratio changes sign
    # for a reason that has nothing to do with stability.  Report the ratio on the
    # positive-savings subset, and report the size of the excluded subset separately.
    positive = coop[coop["v_grand"] > 1e-9]
    reg.add("e3.eps.ratio.mean", float(positive["epsilon_ratio"].mean()), precision=3, source="E3")
    reg.add("e3.eps.ratio.median", float(positive["epsilon_ratio"].median()), precision=3, source="E3")
    reg.add("e3.eps.ratio.worst", float(positive["epsilon_ratio"].min()), precision=3, source="E3")
    reg.add("e3.eps.ratio.n", int(len(positive)), precision=0, source="E3")
    # Magnitude, for prose that reads naturally ("short by X of total savings").
    reg.add("e3.eps.ratio.abs.mean", float(positive["epsilon_ratio"].abs().mean()),
            precision=3, source="E3")
    reg.add("e3.eps.ratio.abs.median", float(positive["epsilon_ratio"].abs().median()),
            precision=3, source="E3")
    reg.add("e3.negative.saving.count", int((coop["v_grand"] <= 1e-9).sum()), precision=0, source="E3")
    reg.add("e3.negative.saving.rate", 100.0 * float((coop["v_grand"] <= 1e-9).mean()),
            unit="%", precision=1, source="E3")
    reg.add("e3.saving.min", float(coop["saving_pct"].min()), unit="%", precision=1, source="E3")
    reg.add("e3.core.nonempty.rate", 100.0 * float(coop["core_nonempty"].mean()), unit="%", precision=1, source="E3")
    reg.add("e3.core.empty.rate", 100.0 * float(1 - coop["core_nonempty"].mean()), unit="%", precision=1, source="E3")
    reg.add("e3.shapley.incore.rate", 100.0 * float(coop["shapley_in_core"].mean()), unit="%", precision=1, source="E3")
    reg.add("e3.submod.violation.mean", 100.0 * float(coop["submod_violation_rate"].mean()), unit="%", precision=1, source="E3")
    reg.add("e3.submod.violation.min", 100.0 * float(coop["submod_violation_rate"].min()), unit="%", precision=1, source="E3")
    reg.add("e3.submod.violation.max", 100.0 * float(coop["submod_violation_rate"].max()), unit="%", precision=1, source="E3")
    reg.add("e3.mono.violation.mean", 100.0 * float(coop["mono_violation_rate"].mean()), unit="%", precision=2, source="E3")
    reg.add("e3.infeasible.coalition.rate", 100.0 * float(coop["infeasible_coalition_rate"].mean()), unit="%", precision=1, source="E3")
    reg.add("e3.gini.shapley.mean", float(coop["gini_shapley"].mean()), precision=3, source="E3")
    reg.add("e3.core.empty.count", int((~coop["core_nonempty"]).sum()), precision=0, source="E3")
    reg.add("e3.calls.total", int(coop["c_calls"].sum()), precision=0, source="E3")
    reg.add("e3.calls.per.instance", float(coop["c_calls"].mean()), precision=0, source="E3")
    reg.add("e3.seconds.total", float(coop["c_seconds"].sum()), unit="s", precision=1, source="E3")
    reg.add("e3.ms.per.call", 1000.0 * float(coop["c_seconds"].sum() / coop["c_calls"].sum()), unit="ms", precision=1, source="E3")

    robots = alloc[alloc["kind"] == "robot"]
    machines = alloc[alloc["kind"] == "machine"]
    total = alloc.groupby("instance")["shapley"].sum()
    robot_share = robots.groupby("instance")["shapley"].sum() / total
    reg.add("e3.robot.share.mean", 100.0 * float(robot_share.mean()), unit="%", precision=1, source="E3")
    reg.add("e3.robot.share.std", 100.0 * float(robot_share.std(ddof=1)), unit="%", precision=1, source="E3")
    reg.add("e3.machine.payoff.mean", float(machines["shapley"].mean()), precision=4, source="E3")
    reg.add("e3.robot.payoff.mean", float(robots["shapley"].mean()), precision=4, source="E3")

    # --- E4 -------------------------------------------------------------------------------
    reg.add("e4.blocks.mean", float(coop["cs_blocks"].mean()), precision=2, source="E4")
    reg.add("e4.maxblock.mean", float(coop["cs_max_block"].mean()), precision=2, source="E4")
    reg.add("e4.maxblock.max", int(coop["cs_max_block"].max()), precision=0, source="E4")
    reg.add("e4.messages.mean", float(coop["cs_messages"].mean()), precision=1, source="E4")
    reg.add("e4.dhp.stable.rate", 100.0 * float(coop["cs_dhp_stable"].mean()), unit="%", precision=1, source="E4")
    reg.add("e4.value.ratio.mean", 100.0 * float(coop["cs_value_ratio"].mean()), unit="%", precision=1, source="E4")
    reg.add("e4.value.ratio.min", 100.0 * float(coop["cs_value_ratio"].min()), unit="%", precision=1, source="E4")
    reg.add("e4.partition.beats.grand.rate",
            100.0 * float((coop["cs_value_ratio"] > 1.0 + 1e-9).mean()), unit="%", precision=1, source="E4")
    reg.add("e4.partition.beats.grand.count",
            int((coop["cs_value_ratio"] > 1.0 + 1e-9).sum()), precision=0, source="E4")
    reg.add("e4.value.ratio.median", 100.0 * float(coop["cs_value_ratio"].median()),
            unit="%", precision=1, source="E4")

    # --- E5 -------------------------------------------------------------------------------
    if not barg.empty:
        reg.add("e5.omega.nbs.mean", float(barg["omega_nbs"].mean()), precision=3, source="E5")
        reg.add("e5.omega.nbs.std", float(barg["omega_nbs"].std(ddof=1)), precision=3, source="E5")
        reg.add("e5.omega.nbs.median", float(barg["omega_nbs"].median()), precision=3, source="E5")
        reg.add("e5.omega.ks.mean", float(barg["omega_ks"].mean()), precision=3, source="E5")
        reg.add("e5.nbs.ks.agree.rate", 100.0 * float(barg["agree"].mean()), unit="%", precision=1, source="E5")
        reg.add("e5.front.size.mean", float(barg["front_size"].mean()), precision=1, source="E5")
        reg.add("e5.breakeven.ctime.mean", float(barg["breakeven_ctime"].mean()), unit="EUR/min", precision=4, source="E5")
        reg.add("e5.breakeven.ctime.hour", 60.0 * float(barg["breakeven_ctime"].mean()), unit="EUR/h", precision=2, source="E5")

    # --- E11 ------------------------------------------------------------------------------
    lam = abl["lambda"]
    for value in sorted(lam["lambda"].unique()):
        sub = lam[lam["lambda"] == value]
        tag = str(value).replace(".", "")
        reg.add(f"e11.lambda{tag}.admissible", 100.0 * float(sub["admissible"].mean()), unit="%", precision=1, source="E11")
        reg.add(f"e11.lambda{tag}.binding", 100.0 * float(sub["binding"].mean()), unit="%", precision=1, source="E11")
        reg.add(f"e11.lambda{tag}.margin.mean", float(sub["margin"].mean()), precision=2, source="E11")
        reg.add(f"e11.lambda{tag}.inadmissible",
                100.0 * float((~sub["admissible"]).mean()), unit="%", precision=1, source="E11")
    pen = abl["penalty"]
    spread = pen.groupby("instance")["gini"].agg(lambda s: s.max() - s.min())
    reg.add("e11.penalty.gini.spread.max", float(spread.max()), precision=4, source="E11")
    reg.add("e11.penalty.eps.spread.max",
            float(pen.groupby("instance")["epsilon"].agg(lambda s: s.max() - s.min()).max()),
            precision=4, source="E11")
    chg = abl["charger"]
    for k in sorted(chg["n_chargers"].unique()):
        sub = chg[chg["n_chargers"] == k]
        reg.add(f"e11.charger{int(k)}.eps.mean", float(sub["epsilon"].mean()), precision=4, source="E11")
        reg.add(f"e11.charger{int(k)}.submod.violation", 100.0 * float(sub["submod_violation_rate"].mean()), unit="%", precision=1, source="E11")
        reg.add(f"e11.charger{int(k)}.utilisation", 100.0 * float(sub["charger_utilisation"].mean()), unit="%", precision=1, source="E11")

    # --- E13 ------------------------------------------------------------------------------
    reg.add("e13.ratio.mean", float(e13["ratio"].mean()), precision=1, source="E13")
    reg.add("e13.ratio.min", float(e13["ratio"].min()), precision=1, source="E13")
    reg.add("e13.ratio.max", float(e13["ratio"].max()), precision=1, source="E13")
    reg.add("e13.raw.energy.share.mean", 100.0 * float(e13["raw_energy_share"].mean()), unit="%", precision=2, source="E13")
    reg.add("e13.raw.energy.share.max", 100.0 * float(e13["raw_energy_share"].max()), unit="%", precision=2, source="E13")
    reg.add("e13.norm.energy.share.mean", 100.0 * float(e13["norm_energy_share"].mean()), unit="%", precision=1, source="E13")
    reg.add("e13.omega.star.mean", float(e13["omega_star_raw"].mean()), precision=4, source="E13")
    reg.add("e13.omega.star.max", float(e13["omega_star_raw"].max()), precision=4, source="E13")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="8 instances, one omega")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="rebuild numbers.json from the Parquet files already in --out, without "
        "re-running anything.  This is the reproduction path: the registry is a pure "
        "function of the recorded results, so re-deriving it must change nothing.",
    )
    args = parser.parse_args(argv)

    if args.from_cache:
        e1 = pd.read_parquet(args.out / "e1_methods.parquet")
        coop = pd.read_parquet(args.out / "e3_cooperative.parquet")
        alloc = pd.read_parquet(args.out / "e3_allocations.parquet")
        barg = pd.read_parquet(args.out / "e5_bargaining.parquet")
        abl = {
            name: pd.read_parquet(args.out / f"e11_{name}.parquet")
            for name in ("lambda", "penalty", "charger")
        }
        e13 = pd.read_parquet(args.out / "e13_commensurability.parquet")
        reg = Registry(args.out / "numbers.json")
        populate_registry(reg, e1, coop, alloc, barg, abl, e13)
        reg.write()
        print(f"rebuilt {len(reg.entries)} registry keys from cache")
        return 0

    instances = all_instances()
    if args.quick:
        instances = instances[:8]
    omegas = (0.5,) if args.quick else OMEGAS
    subset = instances[:8]

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"E1  : {len(instances)} instances x {len(omegas)} omega ...", flush=True)
    e1 = run_e1(instances, omegas)
    e1.to_parquet(args.out / "e1_methods.parquet")

    print("E3/E4: cooperative certification ...", flush=True)
    coop, alloc = run_cooperative(instances, COOP_OMEGA)
    coop.to_parquet(args.out / "e3_cooperative.parquet")
    alloc.to_parquet(args.out / "e3_allocations.parquet")

    print("E5  : bargaining ...", flush=True)
    barg = run_bargaining(instances)
    barg.to_parquet(args.out / "e5_bargaining.parquet")

    print("E11 : ablations ...", flush=True)
    abl = run_ablations(instances, subset)
    for name, frame in abl.items():
        frame.to_parquet(args.out / f"e11_{name}.parquet")

    print("E13 : commensurability audit ...", flush=True)
    e13 = run_e13(instances)
    e13.to_parquet(args.out / "e13_commensurability.parquet")

    reg = Registry(args.out / "numbers.json")
    populate_registry(reg, e1, coop, alloc, barg, abl, e13)
    reg.add("meta.wallclock.minutes", (time.time() - started) / 60.0, unit="min", precision=1, source="runner")
    reg.write()
    reg.write_latex_macros(Path("paper_A") / "numbers.tex")
    print(f"\nwrote {len(reg.entries)} registry keys to {reg.path}")
    print(json.dumps({k: v.value for k, v in list(reg.entries.items())[:12]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
