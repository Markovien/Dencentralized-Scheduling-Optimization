"""Rebuild every figure from the recorded results.

Figure inventory for Paper A (ROADMAP.md §5.5), restricted to the blocks this study runs:

F2  worked Gantt with the ToU price band overlaid -- the single most persuasive figure,
    because it shows the deferral lever doing its work against the tariff
F3  potential-function convergence trace
F5  cooperative savings against coalition size, with the least-core radius overlaid
F7  Pareto front with the Nash and Kalai--Smorodinsky points and the disagreement point
F8  induced omega against the instance
F9  critical-difference diagram (Demsar protocol)
F10 scaling: wall-clock per method

Figures are written as PDF (vector, greyscale-legible) into ``paper_A/figures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jsspt_tou.analysis.stats import friedman
from jsspt_tou.benchmark.bilge_ulusoy import build_instance
from jsspt_tou.cooperative.bargaining import (
    build_front,
    kalai_smorodinsky,
    nadir_disagreement,
    nash_bargaining,
)
from jsspt_tou.cooperative.characteristic import CharacteristicFunction, all_players
from jsspt_tou.cooperative.core_lp import least_core
from jsspt_tou.game.best_response import BestResponsePolicy, GameTrace
from jsspt_tou.simulator.engine import Engine

GREY = ["#1b1b1b", "#4d4d4d", "#7a7a7a", "#a6a6a6", "#c8c8c8"]
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "figure.constrained_layout.use": True, "savefig.dpi": 200,
})


def fig_gantt(out: Path, job_set: int = 1, layout: int = 1, omega: float = 0.5) -> None:
    """F2 -- Gantt of the equilibrium schedule over the tariff bands."""
    inst = build_instance(job_set, layout)
    engine = Engine(inst, omega=omega)
    state = engine.reset()
    rng = np.random.default_rng(0)
    policy = BestResponsePolicy(i_max=6)
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

    fig, (ax_price, ax) = plt.subplots(
        2, 1, figsize=(7.2, 4.4), height_ratios=[1, 3], sharex=True
    )
    # Tariff bands stay pale so the activity bars, which carry the information, are the
    # darkest things on the page; band shade increases with price.
    prices = sorted({p.price for p in inst.tariff.periods})
    shade = {p: 0.99 - 0.10 * i / max(len(prices) - 1, 1) for i, p in enumerate(prices)}
    for period in inst.tariff.periods:
        for axis in (ax_price, ax):
            axis.axvspan(period.start, period.end, color=str(shade[period.price]), lw=0)
    ax_price.step(
        [p.start for p in inst.tariff.periods] + [inst.tariff.periods[-1].end],
        [p.price for p in inst.tariff.periods] + [inst.tariff.periods[-1].price],
        where="post", color=GREY[0], lw=1.4,
    )
    ax_price.set_ylabel("tariff\n(EUR/kWh)")
    ax_price.set_ylim(0, max(prices) * 1.25)

    rows: dict[str, int] = {}
    for kind, owner, job, s, e in state.log:
        key = f"M{owner}" if kind == "process" else f"V{owner}"
        rows.setdefault(key, len(rows))
    order = sorted(rows, key=lambda k: (k[0] != "M", k))
    rows = {k: i for i, k in enumerate(order)}
    for kind, owner, job, s, e in state.log:
        key = f"M{owner}" if kind == "process" else f"V{owner}"
        colour = {"process": GREY[0], "transport": GREY[2], "charge": GREY[3]}[kind]
        hatch = {"process": "", "transport": "", "charge": "xxx"}[kind]
        ax.barh(rows[key], e - s, left=s, height=0.5, color=colour,
                edgecolor="white", linewidth=0.9, hatch=hatch)
        if kind == "process" and e - s > 6:
            ax.text((s + e) / 2, rows[key], f"J{job + 1}", ha="center", va="center",
                    color="white", fontsize=7)
    ax.axvline(inst.deadline, color="black", ls="--", lw=1.1)
    ax.text(inst.deadline - 2, len(rows) - 0.35, "deadline $H$ ", fontsize=8,
            va="top", ha="right")
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(list(rows))
    ax.set_xlabel("time (min)")
    ax.set_xlim(0, inst.deadline * 1.02)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GREY[0]),
        plt.Rectangle((0, 0), 1, 1, color=GREY[2]),
        plt.Rectangle((0, 0), 1, 1, color=GREY[3], hatch="xxx"),
    ]
    ax.legend(handles, ["processing", "transport", "charging"], ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.suptitle(f"{inst.instance_id}: $M1$ equilibrium schedule against the ToU profile",
                 fontsize=10)
    fig.savefig(out / "f2_gantt.pdf")
    plt.close(fig)


def fig_potential(out: Path) -> None:
    """F3 -- the potential along the episode."""
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, (job_set, layout) in enumerate([(1, 1), (5, 1), (7, 3)]):
        inst = build_instance(job_set, layout)
        trace = GameTrace()
        Engine(inst, omega=0.5).run(
            BestResponsePolicy(i_max=6, trace=trace), np.random.default_rng(0)
        )
        values = np.cumsum(trace.potential_trajectory)
        ax.plot(np.arange(1, len(values) + 1), values, color=GREY[i], lw=1.3,
                label=inst.instance_id)
    ax.set_xlabel("decision event $q$")
    ax.set_ylabel("cumulative potential $\\sum W$")
    ax.legend(frameon=False)
    fig.savefig(out / "f3_potential.pdf")
    plt.close(fig)


def fig_savings(out: Path, results: Path) -> None:
    """F5 -- savings against coalition size, with the least-core radius overlaid."""
    inst = build_instance(1, 1)
    engine = Engine(inst, omega=0.5)
    cf = CharacteristicFunction(engine=engine, i_max=1)
    cf.enumerate_all()
    players = all_players(engine)
    sizes: dict[int, list[float]] = {}
    for coalition, cost in cf._cache.items():
        sizes.setdefault(len(coalition), []).append(cf.c_empty() - cost)
    lc = least_core(players, cf.v)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = sorted(sizes)
    ax.boxplot([sizes[s] for s in xs], positions=xs, widths=0.55,
               medianprops=dict(color=GREY[0]), boxprops=dict(color=GREY[1]),
               whiskerprops=dict(color=GREY[2]), capprops=dict(color=GREY[2]),
               flierprops=dict(markersize=3, markerfacecolor=GREY[3], markeredgecolor="none"))
    ax.axhline(cf.v(frozenset(players)), color=GREY[0], ls="-", lw=1.0, label="$v(N)$")
    ax.axhspan(cf.v(frozenset(players)) + lc.epsilon, cf.v(frozenset(players)),
               color=GREY[4], alpha=0.7,
               label=f"least-core shortfall $\\varepsilon^\\star={lc.epsilon:.3f}$")
    ax.set_xlabel("coalition size $|S|$")
    ax.set_ylabel("savings $v(S)$")
    ax.set_title(f"{inst.instance_id}: savings by coalition size", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "f5_savings.pdf")
    plt.close(fig)


def fig_front(out: Path) -> None:
    """F7 -- the equilibrium front with the bargaining solutions."""
    inst = build_instance(1, 1)
    anchors = Engine(inst, omega=0.5).anchors
    outcomes = [
        (float(w), Engine(inst, omega=float(w)).run(
            BestResponsePolicy(i_max=4), np.random.default_rng(0)))
        for w in np.linspace(0, 1, 21)
    ]
    front = build_front(outcomes, anchors)
    disagreement = nadir_disagreement(front)
    nbs = nash_bargaining(front, disagreement, anchors)
    ks = kalai_smorodinsky(front, disagreement, anchors)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.scatter([o.c_max for _, o in outcomes], [o.e_cost for _, o in outcomes],
               s=14, color=GREY[3], label="equilibria")
    ax.plot([p.c_max for p in front], [p.e_cost for p in front], "-o", color=GREY[0],
            ms=4, lw=1.2, label="non-dominated front")
    ax.scatter([disagreement.c_max], [disagreement.e_cost], marker="x", s=60,
               color=GREY[0], label="disagreement $d$")
    if nbs:
        ax.scatter([nbs.point.c_max], [nbs.point.e_cost], marker="*", s=150,
                   facecolor="white", edgecolor=GREY[0], lw=1.2,
                   label=f"NBS ($\\omega={nbs.omega_induced:.2f}$)", zorder=5)
    if ks and (not nbs or ks.point.c_max != nbs.point.c_max):
        ax.scatter([ks.point.c_max], [ks.point.e_cost], marker="s", s=60,
                   facecolor="white", edgecolor=GREY[1], label="KS", zorder=5)
    ax.set_xlabel("$C_{\\max}$ (min)")
    ax.set_ylabel("$E_{\\mathrm{cost}}$ (EUR)")
    ax.set_title(f"{inst.instance_id}: bargaining over the equilibrium front", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "f7_front.pdf")
    plt.close(fig)


def fig_cd(out: Path, results: Path) -> None:
    """F9 -- critical-difference diagram over the method ranks."""
    frame = pd.read_parquet(results / "e1_methods.parquet")
    mid = frame[frame["omega"] == 0.5]
    pivot = mid.pivot_table(index="instance", columns="method", values="phi")
    methods = [m for m in ("pi0", "DR-SPT", "DR-ToU", "DR-oracle", "M1a", "M1b", "M2-grand")
               if m in pivot]
    result = friedman({m: pivot[m].tolist() for m in methods})
    ranks = sorted(result.mean_ranks.items(), key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    lo, hi = 0.7, len(methods) + 0.3
    ax.set_xlim(hi, lo)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.78, lo, hi, color="black", lw=1.0)
    for tick in range(1, len(methods) + 1):
        ax.vlines(tick, 0.74, 0.78, color="black", lw=1.0)
        ax.text(tick, 0.66, str(tick), ha="center", fontsize=8)
    for i, (name, rank) in enumerate(ranks):
        side = -1 if i < len(ranks) / 2 else 1
        y = 0.55 - 0.09 * (i if side < 0 else len(ranks) - 1 - i)
        edge = lo if side > 0 else hi
        ax.plot([rank, rank, edge], [0.78, y, y], color="black", lw=0.8)
        ax.text(edge, y, f" {name} ({rank:.2f}) ",
                ha="right" if side > 0 else "left", va="center", fontsize=8)
    ax.hlines(0.92, ranks[0][1], ranks[0][1] + result.critical_difference,
              color="black", lw=2.5)
    ax.text((2 * ranks[0][1] + result.critical_difference) / 2, 0.96,
            f"CD = {result.critical_difference:.2f}", ha="center", fontsize=8)
    fig.suptitle("Critical-difference diagram (Friedman + Nemenyi, $\\alpha=0.05$)",
                 fontsize=10)
    fig.savefig(out / "f9_cd.pdf")
    plt.close(fig)


def fig_runtime(out: Path, results: Path) -> None:
    """F10 -- wall-clock per decision, by method."""
    frame = pd.read_parquet(results / "e1_methods.parquet")
    mid = frame[frame["omega"] == 0.5]
    methods = ["pi0", "DR-SPT", "DR-ToU", "DR-oracle", "M1b", "M1a", "M2-grand"]
    methods = [m for m in methods if m in set(mid["method"])]
    data = [mid[mid["method"] == m]["seconds"].values * 1000.0 for m in methods]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.boxplot(data, tick_labels=methods, widths=0.6,
               medianprops=dict(color=GREY[0]), boxprops=dict(color=GREY[1]),
               whiskerprops=dict(color=GREY[2]), capprops=dict(color=GREY[2]),
               flierprops=dict(markersize=3, markerfacecolor=GREY[3], markeredgecolor="none"))
    ax.set_yscale("log")
    ax.set_ylabel("wall-clock per episode (ms)")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(out / "f10_runtime.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("paper_A/figures"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    fig_gantt(args.out)
    fig_potential(args.out)
    fig_savings(args.out, args.results)
    fig_front(args.out)
    if (args.results / "e1_methods.parquet").exists():
        fig_cd(args.out, args.results)
        fig_runtime(args.out, args.results)
    print("figures written to", args.out)
    for path in sorted(args.out.glob("*.pdf")):
        print("  ", path.name, f"{path.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
