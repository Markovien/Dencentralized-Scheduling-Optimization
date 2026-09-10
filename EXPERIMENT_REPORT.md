# Experiment report

Factual record of what was run, what was measured, what failed and what is incomplete. No
claims of novelty or superiority beyond the stated statistical tests appear here; the
manuscript makes those, and only through registry keys.

**Environment.** Python 3.11.15, 4 CPU cores, no GPU. OR-Tools CP-SAT, SciPy, NumPy, pandas.
Every stochastic component takes an explicit generator; no global seed is read.

---

## What was run

| Block | Question | Scope | Status |
|---|---|---|---|
| E1 | how far from the baselines is each method? | 40 instances x 3 omega x 7 methods | complete |
| E2 | where does the exact model stop closing? | 6 instances x 3 model variants | complete |
| E3 | is the cooperative solution stable? | 40 instances, exact (2^6 coalitions each) | complete |
| E4 | which coalition structures emerge? | 40 instances | complete |
| E5 | what omega does bargaining select? | 40 instances, 11-point front | complete |
| E11 | ablations: lambda, penalty M, charger capacity | 160 / 32 / 24 runs | complete |
| E13 | commensurability audit | 40 instances | complete |
| N5 | the congestion counterexample and its control | 2 instances, exact | complete |
| E6 | empirical price of anarchy | -- | **not delivered**, see below |
| E7 | disruption robustness | -- | **not run**, out of scope for this pass |

## What failed, and what was done about it

### The exact reference does not close the full model (blocks E6)

`EX-CP` reproduces the vendored solver's makespan **exactly** under a flat tariff with no
deadline and no battery, in about 2 s on EX11 -- the provenance regression passes. Adding the
ToU cost and the deadline, and then the state-of-charge recursion, it stops closing: within
60-180 s per instance the bounds returned are far too weak to normalise a price of anarchy
against (on EX11, incumbent 0.25 against a bound of 0.003).

Three things were tried before accepting this: tightening the charging-slot bound from the
naive "loaded rate over the whole horizon" to a transport-based bound (six slots per vehicle
down to two); warm-starting from the `M1` equilibrium schedule; and relaxing the battery
entirely to obtain a valid lower bound. The first two helped the incumbent, none produced a
usable bound.

**Consequence.** No empirical price of anarchy is reported anywhere. `T4` from the roadmap's
theorem inventory is not delivered. The failure is reported as experiment E2 -- the
scalability wall -- because that is what it is evidence of, and as limitation (i) in the
manuscript.

### Three defects found by the tests, not by inspection

* `potential()` evaluated the stage objective with the *stage* evaluator while `utility()`
  used the *rollout* evaluator. Both are individually valid; mixing them breaks the
  exact-potential identity of T1 while leaving both quantities looking plausible.
  `test_theory.py::test_t1_exact_potential_identity` caught it at a residual of 8e-4.
* `next_event` did not clamp the machine branch to the current time, so a machine idle since
  before `state.t` could pull the event clock **backwards**; episodes livelocked at a fixed
  time for 500 stages. Found by instrumenting a stalled episode.
* The ToU overlap variables in `EX-CP` had domains tightened to the period bounds, so
  `max(start, T_{h-1})` could not be represented and the entire model came back INFEASIBLE.

### Two tests that were over-claiming, and were weakened to the truth

* `M1a` was asserted to beat the best of 120 dispatching rules on every instance. It does
  not -- the comparator is a per-instance hindsight oracle. The test now measures the win
  rate and pins it as a majority; the manuscript reports the rate.
* Log-linear learning was asserted to terminate at an exact equilibrium. Its final sweep is a
  single sequential pass, so it terminates at an eps-equilibrium with small non-zero eps. The
  test now bounds eps instead of requiring zero, and the manuscript describes M1b as an
  equilibrium-*selection* device.

## Anomalies worth flagging

* **The deadline is admissible but slack at the published `lambda = 2`.** All 40 instances
  clear the 1.05 feasibility margin, but only ~15 % have a margin at or below 2. The
  constraint is present and enforced; it rarely binds. Reported as an ablation rather than
  fixed by changing `lambda`, because changing it would break comparability with the
  published tariff profiles.
* **The charger is not the congested resource on this benchmark.** Utilisation is around 6 %
  and duplicating it changes neither the least-core radius nor the submodularity violation
  rate. The congestion obstruction is real and measurable there, but its source is the
  bottleneck machine and the two-vehicle fleet. The constructed N5 instance isolates the
  charger version of the same mechanism.
* **The energy-aware dispatching rule ranks last.** It optimises the right quantity locally
  and loses the deadline doing it. This is a result, not a bug: its deadline-feasibility rate
  is reported alongside its objective.

## Truncation and honesty notes

* No run reported here is truncated; every `EX-CP` result carries its status and the E2 table
  reports closure rates rather than treating a `FEASIBLE` incumbent as an optimum.
* `c(S)` is the declared single-sweep surrogate, not an exact coalition optimisation. Its gap
  to the exact value is **not** characterised, for the same reason E6 is not delivered.
* Cooperative certification is exact at `n = 6`: all 2^6 coalitions per instance, no
  sampling, so the Shapley value, the least core and the nucleolus carry no sampling error.
  The sampling estimator is implemented and tested but not exercised in these results.
* Statistical protocol: Friedman omnibus, Nemenyi critical difference, pairwise Wilcoxon with
  Holm-Bonferroni, Vargha-Delaney A12 with every comparison. Deterministic methods on
  deterministic instances are run once per (instance, omega); this is stated rather than
  presented as a 30-seed protocol.
