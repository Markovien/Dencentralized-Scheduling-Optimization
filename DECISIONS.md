# Decisions

Every place the specification in `ROADMAP.md` was ambiguous or where the source draft was
internally inconsistent, together with what was implemented and why. Recorded as the work
was done, in the order the decisions arose.

---

## D1 — Fleet size `|V|` recovered, then verified (erratum A23)

**Ambiguity.** The draft never states the number of vehicles, yet the appendix tariff table
depends on it through the horizon formula.

**Decision.** Recover it: job set 1 / layout 1 gives `2·(176/4 + 128/|V|) = 216.0`, hence
`|V| = 2`. Rather than assume the recovery, `tests/test_domain.py` asserts that the
regenerated horizon reproduces the published value for **all 40** instances. It does.

---

## D2 — Charging station co-located with the L/U station

**Ambiguity.** The draft's location set includes a charging station `CH`, but the
Bilge–Ulusoy layouts give travel times only between the L/U station and the four machines.
There is no data for `CH`.

**Decision.** Place the charger at the L/U station (location index 0). Inventing travel
times to a fifth location would have been a silent change to the benchmark. Recorded in
`Instance.charger_location`, which is a parameter, so a future layout with a separate
charging bay needs no code change.

---

## D3 — Units contract: which of the draft's three battery numbers to keep (erratum A7)

**Inconsistency.** 100 Ah pack, "20 Ah restored in a 5-minute block", and a 1.5 kW charger
cannot all hold. At any bus voltage, 20 Ah in 300 s at 1.5 kW would need a 6.25 V bus.

**Decision.** Keep the **block structure** (5 minutes) and the **capacity** (100 Ah), declare
`U_nom = 48 V`, and *derive* the rest from the roadmap's fast-charge assumption (0 % → 80 %
in 25 min, linear): 3.2 Ah/min ⇒ 16 Ah per block ⇒ 9.216 kW. The charger power is the
quantity dropped, because it is the one the schedule does not depend on directly — the block
length and its charge content are what the scheduler actually manipulates.

---

## D4 — Depletion rates read as mAh/s, not mA/s (erratum A24)

**Inconsistency.** Read literally as milliamperes per second, 7 mA/s drains 0.043 Ah over the
longest horizon in the benchmark: the SoC floor could never bind and the battery axis of the
paper would be inert.

**Decision.** Read them as **mAh per second**. This is testable, not a matter of taste:
`tests/test_domain.py::test_soc_floor_binds_on_most_instances` asserts the floor binds on at
least half the benchmark under a purely reactive charging rule. It binds on 34/40. Under the
mA/s reading it would bind on 0/40.

---

## D5 — The deadline is enforced as `floor(H)` everywhere

**Ambiguity.** `H` is real-valued (e.g. 242.5) while CP-SAT works in integer minutes.

**Decision.** Every model — `EX-CP`, the simulator, the game, the characteristic function —
enforces `C_max ≤ floor(H)`. Using a different rounding in different models would make their
feasible sets differ and silently invalidate every cross-method comparison, which is the
failure mode `domain/objective.py` exists to prevent.

---

## D6 — Weighted sum with upper–lower-bound normalisation, not ε-constraint

**Ambiguity.** ROADMAP.md §3.7.2 proposes the ε-constraint method as the primary
formulation, but a note in the same section directs: *"Ignore 3.7.2 and 3.7.3. Keep the
weighted sum formulation with Upper-Lower Bound Normalization (Nadir–Ideal Scaling)."*

**Decision.** Follow the note. The weighted sum is the primary formulation, computed on
normalised quantities with instance-fixed anchors — which *is* upper–lower-bound
normalisation, so §3.7.3's construction is retained as the mechanism for it while §3.7.2's
ε-constraint/AUGMECON2 sweep is not implemented. Consequences accepted and stated in the
paper: the reported fronts are the supported (convex-hull) points only, and hypervolume/IGD
against an exact front are not reported.

---

## D7 — The null action is a one-event deferral, not a no-op

**Ambiguity.** The draft defines `a_p^0` as "the decision of player p to remain idle" and
uses it as the baseline of the marginal-contribution utility, but never says what idling
*does*.

**Decision (two parts, both load-bearing).**

1. In the *evaluation*, an idling player is made unavailable until the next event, and only
   then does `pi_0` take over for it. If idling merely handed the player's turn to the
   completion policy, `u_p(a_p^0) = 0` would be a *free* option that beats any real action
   whenever `pi_0` is competent.
2. In the *choice set*, the null action is removed whenever the player has productive work.
   It is redundant — deferral is already a first-class action via the delay grid `D` — and
   degenerate: with it in, best-response dynamics idle the shop to a standstill at the first
   event where every clock has caught up. Measured directly: before this change the game
   completed 0 of its episodes; after it, all of them.

A vehicle whose only options are charging keeps the null action, because charging really is
optional.

---

## D8 — `Phi_hat` is a rollout under a fixed completion policy

**Ambiguity.** The draft evaluates the utility both as a stage-indexed accumulation
(`C^q = max(C^{q-1}, max_p R_p)`, `E^q = E^{q-1} + Σ cost_e`) and as
`Phi(Gamma(s_q, alpha))`, a full trajectory.

**Decision.** Implement both (`Engine.evaluator ∈ {"stage", "rollout"}`) and use
`"rollout"` — committing `alpha` and completing with a fixed deterministic `pi_0`. **T1 holds
for either**, since the exact-potential identity needs only that `Phi_hat` be a deterministic
function of `(s_q, alpha)`; what separates them is that the stage form is degenerate as a
decision rule (it accumulates cost already incurred and never credits work done, so the null
action weakly dominates). The stage form is retained because it is what the draft specifies
and because reporting its degeneracy is itself a finding.

---

## D9 — `pi_0` is the fast completion rule, not the best dispatching rule

**Ambiguity.** ROADMAP.md §3.4 defines `c(S)` against "the best dispatching rule" as `pi_0`.

**Decision.** Use a *fast, deadline-directed* rule (`Engine.reference_action`: shortest
buffered operation, earliest-delivery transport, charge only when no transport is
battery-feasible) instead. Two reasons. It sits on the innermost loop of every utility and
every coalition-cost evaluation, so its cost multiplies through the whole programme. And the
energy-aware "best" rule turns out to *miss the deadline* on part of the benchmark, which
would make `c(∅)` a penalised value and inflate every reported saving. Both `pi_0` and the
dispatching family are reported in E1 so the gap is visible, and
`baselines/dispatching.py::REFERENCE_POLICY` carries a docstring saying explicitly that it is
**not** `pi_0`.

---

## D10 — `c(S)` uses the declared single-sweep surrogate

**Specification.** ROADMAP.md §3.4 mandates a single sequential best-response pass rather
than a full optimisation, and requires its gap to `EX-CP` to be measured.

**Decision.** Implemented as specified. The gap to `EX-CP` is **not** measured, because
`EX-CP` does not close the full model within budget (see D12). Stated as limitation (ii) in
the paper rather than left implicit. One measurable symptom is reported instead: the
surrogate violates the monotonicity the exact `c` satisfies by construction, on ~11 % of
sampled pairs.

---

## D11 — The penalised characteristic function keeps `omega`

**Ambiguity.** ROADMAP.md §3.4 writes `c(S) = Ê_cost(S) + M·max(0, Ĉ_max(S) − 1)`, which
drops the makespan term — i.e. it is the `omega = 0` case.

**Decision.** Use `c(S) = Phi(S) + M·max(0, Ĉ_max(S) − 1)` with the weighted normalised
`Phi`. §3.0 requires all three methods and the exact reference to optimise **the same**
objective; using an energy-only cost in the cooperative layer while the game optimises a
weighted one would break exactly the property that makes the comparison tables meaningful.
Reduces to the roadmap's form at `omega = 0`.

---

## D12 — `EX-CP` ports the vehicle-sequencing block rather than reusing it

**Constraint.** The vendored solver sequences transports on a vehicle with pairwise big-M
disjunctions (`solve_jsspt_cp.py` lines 317–364). That fixes relative order but never
adjacency, so it cannot express a prefix state such as state of charge, and it cannot
interleave charging visits with transports.

**Decision.** Replace that one block with a per-vehicle `AddCircuit` over
{depot} ∪ transports ∪ that vehicle's charging slots. Everything else — operation intervals,
machine `NoOverlap`, the precedence structure, the return leg, exactly-one-vehicle-per-task —
is the original's. The flat-tariff regression (`tests/test_exact_regression.py`) proves the
replacement is faithful: with a flat tariff, no deadline and no battery, `EX-CP` reproduces
the original's makespan exactly (98 on EX11, in 1.9 s).

**Consequence, reported not hidden.** The circuit formulation does not close the full
ToU + battery model within a 60–180 s budget, and its bounds are too weak to normalise a
price of anarchy against. This is reported as experiment E2 (the scalability wall) and as
limitation (i). `T4` (empirical PoA) from the roadmap's theorem inventory is therefore **not
delivered** in this version.

---

## D13 — Deadlock guard in the run loop

**Risk R13, materialised.** At an event where every clock has caught up, all players could
choose the null action, nothing would commit, and the clock could not advance — the episode
died with jobs unfinished.

**Decision.** If nothing commits and time cannot advance while jobs remain, force every
active player onto its deadline-directed fallback and set `used_fallback`. The deadlock rate
is a reported metric. On the final benchmark it is 0 % for every method, so the reported
deadline results are not an artefact of the safeguard.

---

## D14 — Bargaining disagreement point is the front's nadir

**Ambiguity.** ROADMAP.md §3.4 Layer 3 names "the payoff pair realised at the non-cooperative
equilibrium of M1" as the disagreement point. Taken literally at a single `omega`, that point
is *on* the front, the bargaining set is empty, and neither NBS nor KS is defined — verified
numerically before changing it.

**Decision.** Use the nadir of the equilibrium front: production's worst outcome is the
makespan it suffers when logistics dictates (`omega = 0`), logistics' worst is the bill it
pays when production dictates (`omega = 1`). Still a non-cooperative threat point, and it is
what actually bounds the negotiation.

---

## D15 — Feasibility margins are reported, not used to reject instances

**Specification.** ROADMAP.md §3.0(1) says WP4 "rejects or re-scales" any instance whose
margin `H / C_max^LB` falls outside `[1.05, 2.0]`.

**Decision.** Report instead. At the published `lambda = 2` all 40 instances are admissible
(margin ≥ 1.05) but only ~15 % are binding (margin ≤ 2.0). Rejecting 85 % of the classical
benchmark would destroy comparability with the published literature, which is the only reason
to use it. The finding — that the published horizon formula produces an admissible but slack
deadline — is reported, and the `lambda` ablation is made load-bearing as erratum A31 asks.

---

## D16 — N5 is a constructed instance; the obstruction is confirmed on the benchmark

**Finding.** The charger-congestion counterexample the roadmap sketches works exactly as
described once the instance is tuned so that each vehicle must charge exactly once and the
cheap window holds exactly one session. But on the *benchmark*, charger utilisation is ~6 %
and duplicating the charger changes nothing.

**Decision.** Report both, and generalise the claim. The counterexample establishes the
mechanism (a saturating shared resource makes the cost supermodular); the benchmark
establishes that the mechanism is active there too, with the bottleneck machine and the
two-vehicle fleet as the saturating resources rather than the charger. The paper states the
general form, which is stronger than the charger-specific one and is what the data support.

---

## D17 — Two cross-model consistency defects, found by an invariant rather than by review

Neither model looked wrong from the inside. Both were found by asking a question that spans
them: **when `EX-CP` proves optimality, its value must be at most that of any schedule the
simulator produces.** On EX12 it was not, and that is impossible if the two agree about the
objective and the feasible set.

**(a) The simulator under-counted idle battery drain.** It charged the idle rate only while a
vehicle *waited at a pickup*, not for the gap between becoming free at its previous drop-off
and departing for the next task. `EX-CP`, whose circuit arcs give adjacency, charged the whole
gap. The simulator's feasible set was therefore strictly larger than the exact model's.
Physically the exact model is right — an idle vehicle draws its idle current wherever it is
standing — so the simulator was fixed, not the model. Both `_preview_transport` and
`_preview_charge` now bill the pre-departure idle interval, and `B_needed` accounts for it.

**(b) `EX-CP` forced every charging slot to be used.** An unused optional slot still has real
start and end variables, so its per-period overlaps are still real quantities. The cost
construction zeroed those overlaps when the slot was inactive — but an interval of positive
length always overlaps *some* period of a profile that tiles the horizon, so requiring zero
overlap in *every* period is unsatisfiable. The solver's only escape was to activate the
slot. The cost is now gated by the slot literal (`cost == sum(overlaps)` if active, `cost == 0`
if not), leaving the overlaps free.

**Consequence for the results.** Both fixes change measured values, so every experiment was
re-run. The invariant is now a test
(`test_objective_consistency.py::test_proven_optimum_is_never_worse_than_a_feasible_heuristic`),
which is the form the roadmap's proof-hygiene rule takes for a cross-model property: a check
that fails if the two models drift apart, rather than a claim that they do not.
