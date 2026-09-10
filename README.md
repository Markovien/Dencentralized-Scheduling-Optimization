# jsspt-tou

Decentralised game-theoretic scheduling of production machines and battery-constrained
autonomous intelligent vehicles (AIVs) under time-of-use electricity tariffs.

Reference implementation and reproducible artefact for *Cooperative and non-cooperative
game-theoretic scheduling of production machines and battery-constrained autonomous vehicles
under time-of-use tariffs* (target venue: *Engineering Applications of Artificial
Intelligence*).

---

## What the problem is

A job shop where every operation is preceded by a transport leg carried out by a
finite-battery AIV, electricity is priced by a time-of-use tariff, the charging station has
finite capacity, and the planning horizon is a **hard deadline**. Two coupled levers: *when*
to draw power, and *when* to recharge.

Decisions are event-driven and decentralised: an idle machine picks a buffered job and a
ToU-aligned deferral; a free vehicle picks a transport task or a charging visit.

## What is in here

| Layer | What it does |
|---|---|
| `src/jsspt_tou/domain/` | instance model, tariff, battery, **units contract**, normalisation anchors, and the single objective |
| `src/jsspt_tou/simulator/` | the event-driven engine every method shares |
| `src/jsspt_tou/exact/` | `EX-CP`, a CP-SAT model built as an **adapter** over `external/` |
| `src/jsspt_tou/baselines/` | 120 composite dispatching rules |
| `src/jsspt_tou/game/` | `M1` — best response and log-linear learning on an exact potential game |
| `src/jsspt_tou/cooperative/` | `M2` — characteristic function, Shapley, core / least core / nucleolus, merge–split, bargaining |
| `src/jsspt_tou/benchmark/` | the Bilge–Ulusoy suite with regenerated tariff profiles, and the N5 congestion instance |
| `src/jsspt_tou/analysis/` | statistics and the `numbers.json` claim registry |

## Quick start

```bash
python -m pip install -e ".[dev]"
make test                       # fast suite
python -m pytest tests -q       # same, plus -m slow for the CP-SAT solves
python -m jsspt_tou.benchmark.bilge_ulusoy --out results/instances
python experiments/run_all.py --quick     # a few minutes
python experiments/run_all.py             # the full benchmark
python experiments/run_n5.py              # the congestion counterexample
python experiments/run_exact.py           # E2: where the exact model stops closing
python experiments/make_numbers.py        # merge registries -> paper_A/numbers.tex
python experiments/check_paper.py         # claim tracing and structure checks
```

## Three results worth knowing before reading the code

**The core is empty.** The natural hypothesis — coordination has diminishing returns, so the
coalition cost is submodular, the savings game convex, and the core non-empty with the
Shapley value inside — is *false* here. A saturating shared resource makes the cost
supermodular. `benchmark/congestion.py` is a minimal instance where this is exact and
hand-checkable, and it is pinned as a regression test in `tests/test_theory.py`: **its core
must come out empty.**

**The raw weighted sum was a makespan objective.** `ω·C_max + (1−ω)·E_cost` adds minutes to
euros; on this benchmark the terms differ by a factor of ~350, so at `ω = 0.5` energy
contributes under 0.3 % of the objective. Everything scalarised here is computed on
normalised quantities with **instance-fixed** anchors — and `tests/test_scalarisation.py`
asserts the anchors never move mid-episode, because adaptive normalisation would silently
invalidate the potential-game theorem.

**Energy-greedy dispatching is the worst method tested.** Deferring into off-peak bands
consumes deadline slack that cannot be recovered. Deferral needs a feasibility safeguard.

## Provenance

`external/jsspt_cpsat_original/` contains the author's own classical JSSPT CP-SAT solver,
**vendored unmodified**. `EX-CP` extends it rather than replacing it, and
`tests/test_exact_regression.py` proves the extension is faithful: with a flat tariff, no
deadline and no battery, `EX-CP` must reproduce the original's makespan *exactly*. See
`docs/provided_assets.md` and `external/jsspt_cpsat_original/PROVENANCE.md`.

## Claim tracing

Every numerical value in the manuscript is written as a `jnum` macro keyed by name, and resolves against
`results/numbers.json`, which is produced by recorded runs. A numeral typed into a results
sentence is a defect; `experiments/check_paper.py` looks for them.

## Reading order

1. `ROADMAP.md` — the research programme and the audit this work implements.
2. `DECISIONS.md` — every ambiguity resolved, and why.
3. `EXPERIMENT_REPORT.md` — what was run, what was measured, what failed.
4. `paper_A/main.tex` — the manuscript.

## Licence

Code MIT; benchmark data CC-BY-4.0. `external/` is the author's prior work — see its
`PROVENANCE.md` before redistributing.
