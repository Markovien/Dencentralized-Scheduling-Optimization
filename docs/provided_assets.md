# Provided-asset intake (WP0.5, ROADMAP.md §3.2.0)

## 1. Inventory

### 1.1 `external/jsspt_cpsat_original/solve_jsspt_cp.py` — the load-bearing asset

CP-SAT model for the classical JSSPT (no tariffs, no batteries, no charger).

* **Entry points.** `load_jsspt_instance(path) -> dict`; `solve_jsspt_cp(instance, last_ttask,
  num_robots, time_limit, num_workers, make_gantt, gantt_path) -> (status_str, C_max, cpu_time)`;
  `_plot_gantt(...)`; a `__main__` CLI taking `--instance --time_limit --last_ttask`.
* **Instance format.** JSON: `instance_id`, `machines_nb`, `robots_nb`, `sigma`
  ((M+1)×(M+1) travel-time matrix, index 0 = L/U station `Z`), `jobs[*].ops[*]` with
  `machine_index` (1-based) and `duration`. Optional `layout_id`, `layout_name`, `power_kw`.
  **Adopted verbatim as this project's canonical instance schema** (roadmap's preferred
  option: the format is already tested), extended with the additive keys listed in §3.
* **Variables.** Per operation: `S_{i,j}`, `C_{i,j}`, `IntervalVar`. Per transport task:
  `TS_t`, `TC_t`, `IntervalVar`. Assignment `robot_assign[r][t] ∈ {0,1}`. Pairwise
  order booleans `y_r_t1_t2`, `y_r_t2_t1` and co-assignment booleans `both_r_t1_t2`.
  `Cmax`.
* **Constraints.** `NoOverlap` per machine; transport-before-processing and
  processing-before-next-transport precedence; final return-to-`Z` transport when
  `last_ttask = 1`; exactly-one robot per transport; big-M sequence-dependent **empty
  travel** between consecutive transports on the same robot; initial positioning from `Z`.
* **Objective.** `Minimize(Cmax)`, with `Cmax = max_i` (end of job *i*'s return transport).
* **Solver options.** `max_time_in_seconds`, `num_search_workers` (default 8).
* **Tests.** None shipped. `tests/test_exact_regression.py` in this repository supplies the
  missing regression harness.

### 1.2 `external/jsspt_cpsat_original/cp_sat_scheduler.py`

The JSSPT-HF (human-fatigue) variant: lookup-table processing-time inflation, fatigue
accumulation and rest blocks, plus a KPI module. Off the ToU path. Two things were reused
conceptually rather than by import: its **rest-block** construction (optional intervals for
a recovery activity interleaved with productive work) is structurally the same object as a
**charging block**, and its KPI helpers fix the schedule-extraction conventions
(`machine_sched`, `robot_sched` tuple layouts) that `EX-CP` reproduces.

### 1.3 What is *not* in the archive

The roadmap's §3.2.0 also expects *published ILP models of classical JSSP under ToU
tariffs* in the same folder. The uploaded archive contains no such papers — it contains the
manuscript draft (`paper_for_new_game_theory_model.pdf`), an RL scaffold
(`rl_scheduler/`), the Bilge–Ulusoy instance spreadsheet and the two solvers above.
`docs/tou_linearisations.md` therefore compares the linearisations from the published
literature reachable without those files, and records the choice made for `EX-CP`. **This
is a declared gap:** if the ILP papers are supplied later, that document must be revisited
before Paper A §4 is finalised.

## 2. Adaptation, not reimplementation

`src/jsspt_tou/exact/cpsat_model.py` builds the *same* skeleton — same interval variables,
same precedence, same exactly-one-robot assignment, same big-M empty-travel sequencing —
and adds on top of it:

| Addition | Roadmap ref | Why the original cannot carry it unchanged |
|---|---|---|
| ToU-indexed **processing** cost | F6 / A8 | the original has no cost term at all |
| Charging as optional intervals per robot | §3.2.1 | no battery in the original |
| `Cumulative` charger capacity `K_CH` | F7 / A9 | no charger in the original |
| Position-indexed **prefix SoC** | F7 / A10 | no battery state |
| Hard deadline `C_max ≤ H` | F11 / A30 | the original minimises `C_max` unconstrained |
| Weighted-sum objective on normalised terms | F10 / A29 | the original is single-objective |

The original's `solve_jsspt_cp` is *called directly* by the regression test, never edited.
Where a constraint block had to be re-expressed rather than reused (the empty-travel
sequencing, because charging intervals now interleave with transports on the same robot),
the port is marked in `cpsat_model.py` with a comment naming the original lines.

## 3. Instance-format bridge

The canonical schema is the original's, plus additive keys that the original ignores:
`tou` (period table), `horizon`, `battery` (capacity, floor, ceiling, depletion rates,
charge block), `charger` (`n_stations`, `power_kw`, location index), `machine_power_kw`.
`src/jsspt_tou/benchmark/bilge_ulusoy.py` emits this schema and
`tests/test_exact_regression.py` round-trips it through the original loader.
