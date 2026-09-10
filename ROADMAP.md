# Research Roadmap
## Decentralized Optimization for Integrated Production–Transportation Scheduling under Time-of-Use Tariffs
### Game-Theoretic and Reinforcement-Learning Approaches — From Algorithm Development to Publication

**Principal investigator:** Kader Sanogo (CESIT, Kedge Business School)
**Co-authors:** Malek Masmoudi, Abdelkader Mekhalef Benhafssa, M'hammed Sahnoun
**Roadmap version:** 1.0 — 27 August 2026
**Execution mode:** autonomous, by three specialised Claude agents under a review gate
**Target venues:** *Engineering Applications of Artificial Intelligence* (Paper A) and *Expert Systems with Applications* (Paper B), both Elsevier
**Assumed environment:** CPU-only, fully open-source solver stack (OR-Tools CP-SAT, HiGHS, PyTorch-CPU)

---

# 0. Executive summary

The draft in `paper_for_new_game_theory_model.zip` contains a **strong, publishable core idea** — decentralised, event-driven scheduling of machines *and* battery-constrained autonomous vehicles under Time-of-Use electricity tariffs, addressed by a potential-game mechanism and an SMDP/RL formulation — wrapped in a manuscript that is **approximately 45 % complete** and carries **two theoretical defects that would trigger rejection at any Q1 venue** (plus a third, presentational, that would draw a major revision).

This roadmap does four things.

1. **Audits** the draft: scope, methodological background, assets, gaps, and a line-by-line theoretical audit (§1). Eleven findings, each with a concrete repair. Four are acceptance-blocking:
   - the claimed optimality bound `½Φ(αᵒᵖᵗ) ≤ Φ(α*) ≤ Φ(αᵒᵖᵗ)` is **directionally impossible** for a minimisation, and Vetta's valid-utility theorem — a *maximisation* result needing a *monotone submodular welfare* — cannot be transposed to a cost objective as written (F1, F3);
   - **`Φ` adds minutes to euros.** On the draft's own EX11 instance the two terms differ by a factor of 110–159, so at ω = 0.5 energy contributes **0.9 %** of the objective and the terms balance only near ω ≈ 0.007. As written, the bi-objective study is a makespan study, and the agents' utilities and the RL reward inherit the same collapse (F10, §3.7.1);
   - **the horizon is a hard deadline that no model enforces** — `C_max ≤ H` appears nowhere, no instance is certified feasible against its own generated `H`, and nothing prevents a decentralised method from deferring itself into a state with no feasible completion (F11, §3.0).
2. **Redirects the defect, then bounds what it can deliver.** The draft's submodularity proof is misapplied to the non-cooperative bound, but it points at the right structure for the new cooperative model: a submodular coalition cost induces a **convex savings game**, which by Shapley (1971) has a non-empty core containing the Shapley value. **That hypothesis, however, fails in this problem.** §3.4 gives an explicit two-robot instance in which the unary charger — the very constraint §3.0 adds to fix the draft's F7 omission — makes the coalition cost *supermodular* and **the core empty**. Congestion in a scarce shared resource is supermodular, and this problem has one by construction. The cooperative model is therefore built on a **certified least-core radius** as its headline stability result (T5), with convexity retained as a conditional proposition on the congestion-free sub-class (P5) and the obstruction itself reported as a structural finding (N5). This is a stronger position than an unconditional core theorem, because it is true.
3. **Specifies the full methodological programme** (§3): a unified problem `P` with the deadline `H` as a first-class constraint and a weighted-sum optimisation approach, an exact CP-SAT reference **adapted from the author's existing JSSPT solver rather than rebuilt** (§3.2.0, with a flat-tariff regression against that solver as the programme's cheapest correctness guarantee), the corrected non-cooperative game **M1**, a new three-layer cooperative game **M2** (decentralised coalition-structure generation → TU savings game with certified least-core stability (T5) and conditional convexity (P5) → Nash-bargaining endogenisation of the trade-off weight ω), and a CTDE attention-based maskable-PPO learner **M3** whose difference reward is *provably the same marginal-contribution utility* that defines M1's potential. Eleven results (T1–T4, P5, T5, N5, T6, T7–T9) with proof strategies.
4. **Defines the autonomous execution system** (§6): three agents — `algo-eng` (code + experiments), `sci-writer` (LaTeX), `referee` (adversarial peer reviewer) — with strict separation of duties, a machine-checkable JSON verdict schema, an independent reproduction check, a claim-tracing check that forbids any number in the paper that does not resolve to a registry key, and a **hard delivery gate**: nothing ships without an `ACCEPT` verdict.

**Venue decision (§2).** One combined article would be an overloaded 45-page manuscript with three method families, eleven formal results and eleven experiment blocks — the classic profile of a desk-reject or a "split this paper" major revision. The roadmap therefore **splits the work into two complementary articles**:

| | **Paper A** | **Paper B** |
|---|---|---|
| **Venue** | **EAAI** (IF ≈ 8.0) | **ESWA** (IF ≈ 7.5) |
| **Title (working)** | *Cooperative and non-cooperative game-theoretic scheduling of production machines and battery-constrained autonomous vehicles under time-of-use tariffs* | *Game-guided multi-agent reinforcement learning for energy-aware integrated job-shop and transport scheduling under time-of-use tariffs* |
| **Core** | M1 + **M2** + exact reference + theory (T1–T4, T5/T6, P5, N5) | M3 + game-guided reward + generalisation (T7–T9) |
| **Why this venue** | EAAI rewards formally grounded AI applied to an engineering system; proofs, mechanism design and equilibrium analysis are in-scope | ESWA's proven recent profile includes RL for FJSP under non-identical ToU tariffs; a learning system with a generalisation study fits its intelligent-systems remit |
| **Submission** | Month 6 | Month 9 (cites A) |

A single-paper fallback (EAAI, M1+M2 only, RL deferred) is specified in §2.6.

**Critical novelty risk, flagged early.** Reference `SANOGO2025111366` (*Computers & Industrial Engineering* 208, 2025) is the authors' own "game theory approach for JSSPT in shared human–robot environments", already using potential games and marginal-contribution utilities. **The non-cooperative game alone is not novel relative to the authors' own prior work.** Paper A's novelty must rest on (i) the ToU + battery/charging dimension, (ii) the cooperative coalitional layer, (iii) the corrected equilibrium theory, and (iv) the exact reference model. §2.3 specifies the delta statement and cross-citation policy that makes this defensible rather than self-plagiarising.

---

# 1. Deep analysis of the research draft

## 1.1 Archive contents

| File | Lines | Status |
|---|---|---|
| `main_text.tex` | 1 480 | Partial manuscript — see §1.4 |
| `cas-refs.bib` | 773 | 69 entries; well-curated, energy/scheduling/game-theory balanced |
| `cover_letter.tex` | 128 | Addressed to *Journal of Manufacturing Systems*, Prof. Lihui Wang, dated 18 Aug 2025 |
| `elsarticle.cls`, 3 × `.bst`, 2 × template `.tex` | — | Elsevier `elsarticle` bundle, unmodified |
| `grabs.pdf` | — | Graphical-abstract placeholder, not populated |

The `\journal{}` field and cover letter target *Journal of Manufacturing Systems*. The instruction for this roadmap redirects to ESWA/EAAI; §2 treats that as a deliberate repositioning and adjusts framing accordingly (JMS rewards manufacturing-system contributions; ESWA/EAAI reward the AI method — the emphasis must shift from "smart green manufacturing" to "decentralised AI mechanism + learning system").

## 1.2 Scope of the work

**Problem class.** Job Shop Scheduling Problem with Transportation (JSSPT), extended along two axes simultaneously:

- **Energy/economic axis** — a Time-of-Use tariff partitions the horizon `H = {h₁,…,h_H}` into peak / mid-peak / off-peak intervals with prices `c_h`. Machines draw a constant 2 kW while processing; robot chargers draw 1.5 kW. Because idle and auxiliary machine power is excluded, *total machine energy consumption is schedule-invariant* — the entire energy-cost lever is **temporal displacement of load**, which is the correct and defensible framing for a ToU study.
- **Resource-feasibility axis** — a fleet `V = {V₁,…,V_v}` of single-load Autonomous Intelligent Vehicles with finite batteries. SoC constrained to [20 %, 80 %] of a 100 Ah capacity; three depletion rates (1 mA/s idle, 3 mA/s empty travel, 7 mA/s loaded travel); 5-minute recharge blocks restoring 20 Ah.

**Job structure.** Job `Jᵢ` is an ordered chain `(T_{i,1}, O_{i,1}, T_{i,2}, O_{i,2}, …, O_{i,n}, T_{i,n+1})` — every operation is preceded by a transport leg, and the final leg returns the part to the Load/Unload station. Makespan is measured to the **last return to L/U**, not to the last operation completion. This is the Bilge–Ulusoy convention and is the correct one for a transport-integrated study.

**Decision structure.** Fully decentralised and event-driven. Machines choose `(job from input buffer, start time)` — the start-time component is the ToU deferral lever. Robots choose `(job, destination)` or `CHARGE` — the charge action is the battery/ToU lever. Decisions are taken only at events (`RobotFree`, `MachineFree`, `ToUChange`, `ChargerFree`), not on a time grid, which keeps the decision count proportional to system activity rather than horizon length.

**Objective.** A scalarised bi-objective cost `Φ = ω·C_max + (1−ω)·E_cost`, `ω ∈ [0,1]`.

**Instances.** Bilge & Ulusoy (1995): 10 job sets × 4 travel-time layouts = 40 instances, 4 machines, ≤ 8 jobs. A ToU profile is generated per instance from a horizon `H = λ(mean machine load + mean transport load)`, `λ = 2`, discretised into six periods at prices {0.14, 0.22, 0.14, 0.22, 0.14, 0.06} €/kWh (Appendix Table `tab:all_instance`, fully populated for all 40 instances).

## 1.3 Methodological background present in the draft

Three distinct formalisms are laid out, in decreasing order of completeness.

**(a) Centralised MILP reference (`§Centralized optimization model`).** A disjunctive time-continuous model: processing/transport start-completion pairs, big-M machine non-overlap `z_{u,v}`, big-M robot non-overlap `y_{a,b,r}` conditioned on assignment `x_{i,k,r}`, block-discrete charging `g_{r,h} ∈ [0,Δ_h]` per robot per tariff interval, and a two-part battery-feasibility relaxation (a per-task conservative condition `β_{i,k}` plus an aggregate energy balance). The authors explicitly flag it as conservative and note that exact interleaving requires position-indexing.

**(b) Non-cooperative game (`§Game theory framework`, `§Game-theoretic formulation`).** `Γ = ⟨P, {A_p}, {u_p}⟩` with `P = V ∪ M`. Utilities follow **Marginal Contribution Utility** (Wolpert–Tumer "wonderful life utility"; Chapman et al. 2009; Marden's distributed welfare games), i.e. the change in `Φ` relative to the null/idle action. Claims: exact potential game with `Φ` as potential; existence of a pure Nash equilibrium; validity of the utility system; submodularity of `Φ`; a 50 %-suboptimality bound via Vetta (ref `1181966`). The game is repeated over stages `q = 1,…,Q`, with stage-indexed makespan `C_max^q = max{C_max^{q−1}, max_p R_p^q(a_p)}` and stage-indexed energy `E_cost^q = E_cost^{q−1} + ΔE_cost^q`. Return functions `R_p^q(·)` and a battery-need predicate `B_needed(a_p)` are fully specified, including the empty-travel + idle-wait + loaded-travel + return-to-charger decomposition — this part is careful and simulator-ready.

**(c) RL framework (`§Reinforcement Learning Framework`).** A semi-Markov decision process `M = (S, E, A, P, r, γ)` with event-conditioned action sets, per-transition reward `r_q = −ω·Δt_q − (1−ω)·ΔE_q + r_q^viol`, time-aware discounting `γ^{Δt_q}`, SMDP Bellman expectation/optimality equations, a return–objective equivalence theorem, a structured observation constructor (global context / local agent / variable-size candidate-action features), explicit feasibility masks for both agent types, attention-based scoring of variable candidate sets, and a full pseudocode listing for event-driven Maskable PPO with a centralised critic and time-aware GAE.

**(d) Appendix.** Two formal proofs (utility-system validity; submodularity of `Φ` split into makespan and energy components), the complete ToU table, four travel-time layouts, ten job sets.

## 1.4 Completeness gap

| Section | State |
|---|---|
| Abstract | Literal placeholder: `Abstract text.` |
| Graphical abstract | Empty |
| Highlights | Present, 4 items — usable |
| Introduction | Complete and well-written |
| **Related work** | **Empty (heading only)** |
| Problem description | Complete |
| Centralised model | Complete |
| Problem modelling (ToU, makespan, energy) | Complete |
| Game framework | Complete but internally inconsistent (§1.5) |
| RL framework | Complete as theory; no implementation, no results |
| **Experiments** | **Empty** |
| **Results and discussion** | **Empty** |
| **Managerial insights** | **Empty** |
| **Conclusion** | **Empty** |
| Appendix proofs | Present but two are defective (§1.5) |
| Appendix data | Complete, but the ToU table has seven defective rows — see erratum A26 |

No figure is included beyond the empty graphical abstract; no algorithm listing for the game (the `\subsection{Best response algorithm implementation}` line is commented out); no experimental results of any kind exist.

## 1.5 Theoretical audit — eleven findings

Severity: **S1** blocks acceptance · **S2** guarantees a major revision · **S3** referee irritation.

---

**F1 — S1 — The optimality bound is directionally impossible.**

The draft states, for a cost `Φ` being minimised:
`½Φ(αᵒᵖᵗ) ≤ Φ(α*) ≤ Φ(αᵒᵖᵗ)`.
The right inequality asserts that an arbitrary Nash equilibrium costs **no more than the optimum**, which would make every equilibrium globally optimal and the optimisation problem vacuous. Vetta's theorem (FOCS 2002, ref `1181966`) is a **maximisation** result: for a valid utility system whose social welfare `f` is *normalised, non-decreasing and submodular*, every Nash equilibrium attains **at least ½ of the optimal welfare**. The mapping welfare → cost is not a sign flip: negating a monotone submodular function yields a *non-increasing supermodular* function, which violates both hypotheses of the theorem.

*Repair.* Three legitimate routes, to be executed in this order:
1. **Price of Stability = 1 (free, exact).** In an exact potential game whose potential *is* `Φ`, any global minimiser of `Φ` is a pure Nash equilibrium. Hence `PoS = 1`: an optimal equilibrium exists. This is the correct, provable version of the draft's remark that "the optimal solution to our problem is always an NE". State it as Theorem T3.
2. **Price of Anarchy — measured, not asserted.** Define `PoA = max_{α* ∈ NE} Φ(α*)/Φ(αᵒᵖᵗ) ≥ 1`. Provide (a) an explicit small worst-case instance giving a *lower bound* on PoA, and (b) an **empirical PoA distribution** over all instances where CP-SAT closes the gap. Empirical PoA with a proven lower bound is a stronger and far more honest contribution than a borrowed bound that does not apply.
3. **Rejected route, recorded so it is not re-attempted.** One could construct an auxiliary *modular* progress welfare over (player, action) pairs so that Vetta's hypotheses hold by construction. This is vacuous: a modular welfare makes the game separable, every Nash equilibrium is then globally optimal, the bound degenerates to 1 rather than ½, and the resulting statement carries no information about `Φ` or about any interaction in the problem. **Do not pursue.** T3 plus a measured empirical PoA is the complete and sufficient answer.

---

**F2 — S3 — Submodularity is stated over an ambiguous ground set, and the natural fix is not union-closed.**

The appendix presents submodularity of `Φ` as a property of **subsets of players** `A ⊆ B ⊆ P`, while the optimisation is over **joint action profiles** and Vetta's ground set is the set of **(player, action) pairs**. In fairness the draft's own `f(𝒫) = max{C_max, max_{p∈𝒫} R_p(α_p)}` already carries the actions along in a fixed profile, so the mathematics is closer to correct than the prose suggests — but the statement as written cannot be quoted in support of any conclusion about actions.

*Repair, in two steps.*
1. **Restate over `Ω = {(p, a) : p ∈ P, a ∈ A_p}` with `S` ranging over all of `2^Ω`**, defining `f(S) = max{C_max^{q−1}, max_{(p,a)∈S} R_p(a)}`. Do **not** restrict `S` to at-most-one-action-per-player: that family is a partition matroid, is **not closed under union**, and `f(B ∪ {x})` is then undefined whenever `x` is a second action for a player already in `B`. Every downstream result — polymatroid structure, Shapley-in-core, any Vetta-style bound — requires submodularity on the full lattice `2^Ω`, so the unrestricted extension is the object to prove things about; the one-action-per-player family enters afterwards, only as the feasible sub-family over which the mechanism actually selects.
2. **The makespan argument then transfers verbatim and correctly**: for `A ⊆ B ⊆ Ω` and `x = (p,a) ∉ B`, `f(A ∪ {x}) − f(A) = (R_x − f(A))⁺ ≥ (R_x − f(B))⁺ = f(B ∪ {x}) − f(B)`,
so `C_max` is **monotone submodular** on `2^Ω`. The energy term `ΔE_cost(S) = Σ_{(p,a)∈S} cost_e(a)` is **modular** and non-decreasing. A non-negative weighted sum of a monotone submodular and a monotone modular function is monotone submodular, so `Φ` is monotone submodular on `2^Ω`.

Severity is S3 rather than S1 because the underlying argument is salvageable as written; what fails is the *use* made of it (F1, F3).

---

**F3 — S1 — The submodularity result is spent on the wrong theorem.**

Having established F2, the draft uses it for a bound that does not hold. Its correct and far more valuable use is on the **cooperative** side, which the current draft does not have.

*Repair — this points at the structure of the new cooperative model.* Let `c(S)` be the optimal cost achievable by coalition `S ⊆ N = M ∪ V`. If `c` were submodular, the **savings game** `v(S) = c(∅) − c(S)` would be **supermodular, i.e. convex** in the sense of Shapley (1971). Convex TU games have a **non-empty core**, the core equals the base polytope of the associated polymatroid, and the **Shapley value lies in the core**. Consequently the cooperative allocation is simultaneously *efficient*, *fair* in Shapley's axiomatic sense, and *stable* — no sub-coalition of machines and robots can profitably secede. That would be a headline-grade theorem, and it is obtained by *redirecting* a proof the authors have already written.

**Caution, established by counterexample (see §3.4).** The implication "submodular cost ⇒ convex savings game ⇒ core ≠ ∅" is itself sound, but its *hypothesis fails in this problem whenever a scarce shared resource creates congestion* — and §3.0 deliberately introduces exactly such a resource by adding the missing unary charger constraint. §3.4 gives an explicit two-robot instance in which `c` is **supermodular**, the savings game is concave, and **the core is empty**. The cooperative model must therefore be built around the *least-core* as its primary stability result, with convexity as a conditional proposition certified per instance. Building Paper A's headline on an unconditional core theorem would be building it on a false statement.

---

**F4 — S2 — The utility sign convention contradicts itself between sections.**

`§Game theory framework` defines `u_p(a_p, α_{−p}) = Φ(a_p, α_{−p}) − Φ(a_p⁰, α_{−p})` — a cost *increase* — and then says rational players *minimise* it. `§Marginal contribution utilities` defines `u_p = Φ(Γ(s_q,(a_p⁰,a_{−p}))) − Φ(Γ(s_q,(a_p,a_{−p})))` — a cost *reduction* — and says players *maximise* it. The Nash-equilibrium definition is given first as `u_p(a*) ≥ u_p(a)` and then re-glossed as `= min_a u_p`. A referee will read this as the authors not knowing which quantity their agents optimise.

*Repair.* Adopt one convention throughout and state it once, in a boxed definition: welfare `W(α) = Φ(α⁰) − Φ(α) ≥ 0` (cost reduction relative to the all-idle profile), agent utility `u_p(α) = W(α) − W(a_p⁰, α_{−p})`, all agents **maximise**. Every subsequent statement — NE, best response, potential — then reads in the same direction. Re-derive the three affected equations mechanically.

---

**F5 — S2 — The potential-game claim is asserted, not proved, and is stated at the wrong scope.**

The draft says the potential property is "well established in the literature and demonstrated in [SANOGO2025111366]". Two problems. First, the proof is one line and should simply be given. Second — and materially — the claim is made for the *dynamic, multi-stage, simulator-mediated* game, where `Φ` depends on the entire downstream rollout `Γ(s_q, a_q)`. A function of a full trajectory is not a function of the current joint action profile alone, and the exact-potential identity does not survive that dependence.

*Repair.* Split into two honest statements.
- **T1 (stage game, exact).** Fix state `s_q` and a *myopic* evaluation `Φ̂(s_q, α)` that depends only on the joint action at that stage. With `u_p = W − W(a_p⁰, ·)`, for any `a_p, a_p'`:
  `u_p(a_p, α_{−p}) − u_p(a_p', α_{−p}) = W(a_p, α_{−p}) − W(a_p', α_{−p})`
  because the `W(a_p⁰, α_{−p})` term is independent of `p`'s action and cancels. Hence the stage game is an **exact potential game** with potential `W`. It has the finite improvement property, so best-response dynamics converge to a pure NE in finitely many steps (Monderer & Shapley 1996).
- **T2 (dynamic game, honest).** The multi-stage game is a sequence of stage potential games. State that per-stage convergence does **not** imply global optimality of the resulting trajectory, and quantify the loss empirically against CP-SAT. Reviewers reward this kind of candour; they punish the alternative.

---

**F6 — S2 — The centralised model and the game optimise different objectives.**

In the MILP, `E_cost = Σ_r Σ_h (P^ch · g_{r,h}/60)·c_h` — **robot charging only**. In the game and in the RL reward, `E_cost` covers **machine processing energy plus robot charging**. The two models are therefore not comparable, and any table reporting "MILP vs game" would be comparing different quantities. Additionally, `ΔE_q` in the RL reward carries only the charging term, so the draft's return-equivalence theorem proves equivalence to the *MILP* objective, not to the objective the game minimises.

*Repair.* Add the machine-processing energy term to the centralised model. Because processing power is constant, the cost of operation `(i,k)` is a **piecewise-constant function of its start time**, expressible either (a) time-indexed with binaries `δ_{i,k,t}`, or (b) continuous with interval-membership binaries `λ_{i,k,h}` and SOS1 linking. Recommendation: keep the continuous disjunctive MILP for structural exposition in the paper, and implement the **CP-SAT model as the computational reference** (§3.2), where ToU cost is naturally an element-constraint on the start-time domain. Then restate the return-equivalence theorem against the corrected `Φ`.

---

**F7 — S2 — Battery feasibility in the MILP is a relaxation presented alongside exact constraints.**

Constraint `(8)` combines a per-task conservative condition and an *aggregate* energy balance over all tasks assigned to a robot. Aggregate balance ignores **ordering**: a robot may satisfy the total-energy inequality yet violate the SoC floor mid-sequence. The model is therefore a **relaxation** — it can return infeasible schedules and its objective is a **lower bound**, not an optimum. The draft's own remark acknowledges this, but the manuscript then reads as if the model were exact.

*Repair.* Two options, both to be implemented: (i) label it explicitly as `LB-MILP` and report it as a *bound*; (ii) implement an exact, position-indexed variant `EX-CP` in CP-SAT with per-prefix SoC tracking and a `cumulative` constraint enforcing charger capacity. Note also that the draft mentions a **single** charging station but imposes **no charger non-overlap constraint anywhere** — a separate, independent modelling omission that must be fixed in every model (MILP, game simulator, RL environment).

---

**F8 — S3 — Modelling inconsistencies and unit hazards.**

- Battery quantities are in **Ah** while tariffs are in **€/kWh**; no voltage is ever stated, so Ah cannot be converted to kWh. The 1.5 kW charger and the "20 Ah per 5 minutes" recharge are not mutually consistent without a stated nominal bus voltage. **Fix: state the nominal voltage (e.g. 48 V), derive the implied energy, and verify the charger power reconciles; publish a units table.**
- Robot traction energy is set to zero in the cost (`cost_e(a_p) = 0` for transport) while battery depletion from the *same* travel is fully modelled. This is internally coherent (traction energy is charged for later, at charging time) but reads as an error; **add one sentence making the accounting explicit**.
- The `[20 %, 80 %]` SoC window is stated as an assumption but only the 20 % floor is ever enforced; the 80 % ceiling appears in no constraint. **Fix or drop.**
- `λ = 2` in the horizon formula is justified as "through empirical testing" with no data. **Fix: add a horizon-sensitivity ablation over λ ∈ {2, 2.5, 3}.**
- Machine set is denoted `M` in the MILP and `𝓜` elsewhere; the RL section reuses `𝓜` for the SMDP tuple. **Fix: single notation table, symbol collision resolved.**

---

**F9 — S2 — Empirical scale is below Q1 expectation.**

Forty instances of ≤ 8 jobs / 4 machines / (typically) 2 robots is the 1995 benchmark. It is essential for comparability and must be kept — but as the *sole* evidence base it will draw "the approach is not demonstrated at industrially relevant scale", which is the single most common referee objection to decentralised-scheduling papers. It also undercuts the paper's own central argument: decentralisation is motivated by centralised methods becoming *computationally prohibitive at scale*, a claim that 8-job instances cannot support.

*Repair.* Retain all 40 classic instances, and add a released, documented generator producing **JSSPT-ToU-Bench** (§5.1): up to ~50 jobs / 20 machines / 10 robots, three layout families, three real tariff structures, 10 seeds — ≈ 450 instances with a Zenodo DOI. This directly evidences the scalability claim and creates a reusable community asset that both papers cite.

---

**F10 — S1 — The two objectives are not commensurable, and the weighted sum silently collapses to a makespan objective.**

`Φ = ω·C_max + (1−ω)·E_cost` adds minutes to euros. On the draft's own EX11 the ratio is **110–159**, so at ω = 0.5 the energy term contributes **0.9 %** of `Φ`, and the two balance only at ω\* ≈ 0.006–0.009. Full arithmetic in §3.7.1. Every downstream object inherits the distortion: the marginal-contribution utilities, the potential the agents descend, and the RL reward are all effectively `C_max` alone. A referee needs two minutes and the paper's own numbers to find this.

*Repair.* Keep the weighted sum approach and normalise objectives using Upper-Lower Bound Normalization (Nadir–Ideal Scaling). You may use instance-fixed anchors wherever a scalar is genuinely required, and monetise for the managerial section. §3.7 in full; erratum A29.

---

**F11 — S1 — The scheduling horizon is a hard deadline, but no model enforces it.**

`H` is the maximum allowed makespan: no schedule may complete after it. The draft carries only the per-action shadow of this (`R_p^q(a_p) ≤ H`, inside the machine return function). `C_max ≤ H` appears in no model; no instance is certified to admit a feasible schedule under its own generated `H`; and no method has a safeguard against deferring itself into a state with no feasible completion. Three consequences are structural rather than cosmetic: instances may be infeasible, greedy and best-response dynamics can deadlock, and the cooperative characteristic function can be `+∞` — which would remove the Shapley value, the core LP and the least-core radius on exactly the instances where coordination matters most.

*Repair.* §3.0 states the deadline and its five design obligations; §3.4 adopts the penalised characteristic function that keeps `c` real-valued; §3.7.3 uses `H` as the principled upper anchor for normalisation. Errata A30, A31.

## 1.6 What is already strong (do not rebuild)

- The **event-driven decision architecture** is the right abstraction and is shared cleanly by all three methods — one simulator serves the game, the cooperative model and the RL environment.
- The **return function `R_p^q(·)` and battery predicate `B_needed(a_p)`** are specified at implementation precision, including the empty-travel + idle-wait + loaded-travel + return-to-charger decomposition. Code them as written.
- The **RL observation/mask design** — global context, local agent features, variable-size candidate-action features with attention scoring and `−∞` logit masking — is current best practice and needs no redesign.
- The **return-equivalence theorem** is correct as a telescoping argument; it only needs its `E_cost` definition harmonised (F6).
- The **makespan submodularity argument** is correct; it is applied to the wrong ground set (F2) and used for the wrong theorem (F3), both repairable without discarding it.
- The **bibliography** is well-curated and venue-appropriate; the layout and job-set appendices are complete and reusable, and the ToU table is reusable once its seven defective rows are regenerated (A26).

## 1.7 Potential contributions, ranked

| # | Contribution | Novelty | Paper |
|---|---|---|---|
| **C1** | **Cooperative scheduling game with certified stability**: characteristic cost function for an integrated production–transport system under ToU, with per-instance least-core certification, and a convexity proposition holding on the congestion-free sub-class | **High** — no prior work analyses core stability for a joint machine/AIV energy-aware scheduling game; the congestion obstruction is itself a reportable structural finding | A |
| **C2** | **Endogenised trade-off weight**: ω derived from a Nash (and Kalai–Smorodinsky) bargaining solution between production and logistics sub-coalitions, with the non-cooperative equilibrium as disagreement point | **High** — replaces an arbitrary hand-tuned scalar with an axiomatically justified one; directly answers the standard "how did you choose ω?" referee question | A |
| **C3** | **Decentralised coalition-structure generation** via merge–split over a hedonic preference induced by sampled Shapley payoffs, converging to a Nash-stable partition | Medium-high | A |
| **C4** | **Corrected equilibrium theory**: exact stage potential (T1), PoS = 1 (T3), worst-case PoA lower bound + measured empirical PoA (T4) | Medium — corrective but rigour-defining | A |
| **C5** | **Exact CP-SAT reference model** with position-indexed SoC tracking, charger cumulative capacity and ToU-indexed processing cost — the first exact model for this exact problem | Medium-high | A |
| **C6** | **Game-guided MARL**: the marginal-contribution utility of M1 *is* the difference reward of Wolpert–Tumer/COMA, so the two methods share one credit-assignment signal; contribution is the **empirical** question of whether that alignment pays off on this SMDP | Medium — the identity is definitional and difference-reward/potential alignment is established (Wolpert & Tumer 1999; Agogino & Tumer; Devlin et al., AAMAS 2014); the transport/battery/ToU evaluation is what is new | B |
| **C7** | **Heterogeneous CTDE maskable PPO with attention over variable candidate sets** for a two-agent-type event-driven SMDP | Medium | B |
| **C8** | **Zero-shot size generalisation and rescheduling-latency study** under machine breakdown, urgent arrivals and tariff-forecast error | Medium-high | B |
| **C9** | **JSSPT-ToU-Bench**: open benchmark suite, generator, simulator and reference results with DOI | Medium (high community value) | A + B |

---

# 2. Publication strategy

## 2.1 Venue analysis

| Criterion | **Expert Systems with Applications** | **Engineering Applications of Artificial Intelligence** |
|---|---|---|
| Publisher / ISSN | Elsevier, 0957-4174 | Elsevier, 0952-1976 |
| Impact factor (2026 listing) | ≈ 7.5 | ≈ 8.0 |
| SJR / h-index | 1.854 / 290 (SCImago) | 1.652 / 149 (SCImago) |
| Dominant topics | AI (39 %), data mining (19 %), machine learning (18 %) | AI (39 %), machine learning (15 %), ANN (15 %) |
| Editorial centre of gravity | Intelligent/expert systems, learning methods, decision support, applied optimisation | AI **methods applied to engineering systems**; strong tradition in MAS, fuzzy, knowledge-based and optimisation methods |
| Direct precedent | *Graph-based reinforced multi-objective optimization for distributed heterogeneous FJSP under nonidentical time-of-use electricity tariffs*, ESWA 2025 — **near-identical problem framing, RL method** | Long-standing MAS/game-theoretic scheduling stream; formal analysis is welcomed rather than tolerated |
| Tolerance for theorems | Moderate — proofs must earn their space by improving the *system* | High — mechanism design, equilibrium analysis and bounds are in-scope contributions |
| Tolerance for "no learning component" | Low — a paper with no learning/expert-system element sits awkwardly | High |

**Reading.** The two journals are close in prestige and overlap heavily, but they reward different centres of gravity. ESWA's own 2025 output contains a paper whose framing is nearly identical to the RL contribution here — that is the strongest possible scope evidence, and it also means the RL paper must clearly differentiate (transport + battery + charging decisions + decentralised execution, none of which that paper has). EAAI is the better home for a paper whose contribution is a *mechanism with proofs*, because the core of the argument is the equilibrium/core analysis rather than a learned model.

## 2.2 Decision

> **Split into two papers. Paper A → EAAI. Paper B → ESWA.**

**Why split.** Combined, the material comprises three method families (M1, M2, M3), an exact reference model, eleven formal results, eleven experiment blocks and two benchmark suites. Compressed into one article this yields ~45 pages in which no single contribution is developed to Q1 depth — the classic profile of either a desk reject or a "the paper attempts too much; consider splitting" major revision. Split, each paper has a single crisp thesis, room for its own related-work section, full experimental treatment and a coherent set of managerial insights.

**Why this assignment and not the reverse.** Paper A's load-bearing contributions (C1–C5) are *theorems about a mechanism*; a referee pool selected for expert/intelligent systems will ask what the "expert system" is. Paper B's load-bearing contributions (C6–C8) are *a learning system and its generalisation*, which is exactly ESWA's demonstrated remit — and the 2025 ESWA precedent proves the problem framing passes scope screening there.

**Sequencing.** Submit A first (Month 6). B (Month 9) cites A as `[Sanogo et al., under review / in press]` for the problem formulation and uses A's equilibria as baselines. This ordering matters: it makes B's game-theoretic baselines citable rather than self-contained, keeping B short.

## 2.3 Anti-salami and self-overlap safeguards

The single largest reputational risk is the authors' own `SANOGO2025111366` (CIE 2025), which already applies potential games with marginal-contribution utilities to JSSPT. Three mandatory safeguards:

1. **Explicit delta paragraph** in Paper A's introduction, naming the prior paper and stating precisely what is new: ToU tariffs and battery/charging decisions as first-class decision variables; the cooperative coalitional layer; corrected equilibrium theory with PoS/PoA; an exact CP-SAT reference. Never paraphrase the prior model without citation.
2. **No text reuse.** Problem-description prose must be re-written, not adapted. Run a similarity check (§6.4, R-7) before delivery.
3. **Distinct-contribution statements in both cover letters**, plus a sentence in Paper B's cover letter declaring Paper A as a companion submission and summarising the non-overlap. Editors respond well to disclosed companion papers and badly to discovered ones.

Additionally: the shared simulator and benchmark (C9) are released **once**, with a DOI, and cited identically by both papers. Reusing an infrastructure asset across companion papers is normal and expected; reusing *results* is not.

## 2.4 Paper A — outline

**Working title.** *Cooperative and non-cooperative game-theoretic scheduling of production machines and battery-constrained autonomous vehicles under time-of-use tariffs*

**Thesis.** A decentralised scheduling mechanism for an integrated job shop with autonomous transport can be given both an equilibrium guarantee and a *measured stability* guarantee. Each stage game is an exact potential game whose price of stability is one, so best-response dynamics are well founded and an optimal equilibrium exists at every stage. On the cooperative side, the induced coalition cost is submodular — hence the savings game convex and the core non-empty with the Shapley value inside — **on the congestion-free sub-class**; where a scarce shared resource (the charger) creates congestion, convexity fails and the core can be empty, so stability is reported as a certified **least-core radius `ε`** per instance. The cooperative solution additionally endogenises the makespan/energy trade-off through bargaining, removing the arbitrary weight ω.

**Target length.** 34–38 pages (`preprint, 3p, review, 12pt` double-spaced review format, ≈ 22–25 pp in journal layout), 10 figures, 8 tables, ~70 references. The per-section budget below sums to 39.5; the writing agent trims §2 and §8 first if it must.

| § | Content | Pages |
|---|---|---|
| 1 | Introduction — Industry 5.0 framing, ToU + battery motivation, **delta vs prior work**, contributions C1–C5, C9 | 3 |
| 2 | Related work — 4 subsections: energy-aware scheduling under ToU; JSSPT with AGV/AIV and battery constraints; game-theoretic and multi-agent scheduling; cooperative game theory in operations and manufacturing. Closes with a **positioning table** (≈ 25 rows × 7 feature columns) and an explicit gap statement | 5 |
| 3 | Problem statement — sets, parameters, assumptions, **units table**, ToU profile generation, worked example figure | 3.5 |
| 4 | Exact reference models — `LB-MILP` (structural, disjunctive) and `EX-CP` (computational, CP-SAT with prefix SoC + charger cumulative + ToU-indexed processing cost); complexity note | 3.5 |
| 5 | **M1** Non-cooperative game — corrected conventions, T1 exact stage potential, T2 dynamic-scope statement, T3 PoS = 1, T4 PoA bound, best-response and log-linear-learning algorithms with complexity | 5 |
| 6 | **M2** Cooperative game — three layers (§3.4): TU savings game with the congestion counterexample N5, conditional convexity P5, certified least-core T5, monotonicity/superadditivity T6; decentralised coalition-structure generation; bargaining-based ω (C2); estimators with sampling-error guarantees | 6 |
| 7 | Experiments — instances, protocol, baselines, statistics | 2 |
| 8 | Results — E1–E7 and E10 (§5.4) | 5 |
| 9 | Managerial insights — peak-shifting policy, fleet sizing, charger investment, internal transfer pricing between production and logistics cost centres | 1.5 |
| 10 | Conclusion and future work (signposting Paper B) | 1 |
| App. | Proofs T1–T4, P5, T5, N5, T6; notation; ToU tables; benchmark description | 4 |

**Highlights (5, ≤ 85 characters each) — draft.**
- Decentralised game-theoretic scheduling of machines and battery-constrained AIVs.
- Each stage game is an exact potential game whose price of stability equals one.
- Charger congestion makes coalition costs supermodular and can empty the core.
- Least-core radius certifies stability; charger congestion breaks convexity.
- Bargaining endogenises the makespan-energy weight; benchmark and code are released.

**Abstract skeleton (≤ 200 words).** Context (Industry 5.0, ToU, AIV batteries) → gap (centralised methods do not scale; existing decentralised methods offer no stability guarantee and hand-tune the trade-off weight) → what is done (unified formulation; exact CP-SAT reference; non-cooperative potential game; **cooperative savings game with certified least-core stability and a bargaining-derived ω**) → theory (stage-game PoS = 1; core non-emptiness on the congestion-free sub-class, certified least-core elsewhere) → evidence (510 instances; X % mean cost reduction vs best dispatching rule; within Y % of CP-SAT optima on closed instances; Z× faster rescheduling under disruption) → implication (deployable decentralised coordination with an auditable internal cost allocation).

## 2.5 Paper B — outline

**Working title.** *Game-guided multi-agent reinforcement learning for energy-aware integrated job-shop and transport scheduling under time-of-use tariffs*

**Thesis.** The marginal-contribution utility that makes the decentralised scheduling game an exact potential game is *identically* the difference reward used for multi-agent credit assignment. Training heterogeneous machine and robot policies on that signal therefore performs policy gradient on the game's potential — yielding a learner that inherits the mechanism's alignment properties, matches or beats equilibrium solutions, and generalises zero-shot to instance sizes never seen in training while rescheduling in milliseconds.

**Target length.** 30–34 pages in review format (≈ 20–22 pp in journal layout), 10 figures, 6 tables, ~65 references. The per-section budget below sums to 33.

| § | Content | Pages |
|---|---|---|
| 1 | Introduction — contributions C6–C9 | 3 |
| 2 | Related work — DRL for shop scheduling; MARL and credit assignment; energy-aware/ToU learning schedulers; action masking and variable action spaces. Positioning table | 5 |
| 3 | Problem and SMDP — event set, state, event-conditioned actions, transition, reward, time-aware discounting; T7 return–objective equivalence (corrected `E_cost`) | 4 |
| 4 | **Game-guided credit assignment** — T8: the difference reward equals the MCU of M1; corollary: per-agent advantage estimates are aligned with the global potential; T9: bias/variance characterisation of the sampled difference reward | 3.5 |
| 5 | Architecture — observation constructor, feasibility masks, attention scoring over variable candidate sets, heterogeneous actors + centralised critic, CTDE PPO with time-aware GAE; **CPU training budget and design choices that follow from it** | 4 |
| 6 | Experiments — training protocol, baselines (dispatching rules, M1 best response, M2 cooperative, NSGA-II, CP-SAT where closable) | 2 |
| 7 | Results — E1 (learning methods only), E8–E10 plus ablations | 6 |
| 8 | Managerial insights — when to deploy a learned policy vs a game mechanism; retraining cadence under tariff change | 1.5 |
| 9 | Conclusion | 1 |
| App. | Proofs T7–T9, hyperparameters, full ablation tables | 3 |

**Differentiation from the ESWA 2025 precedent** (must appear explicitly in §2 of Paper B): that work addresses distributed heterogeneous FJSP under non-identical ToU tariffs with **no transport resource, no battery, no charging decision, and a centralised graph-based optimiser**. Paper B's problem has transport as a first-class scheduled resource, energy-feasibility coupling through SoC, charging as a decision, and fully decentralised execution.

## 2.6 Fallback: single-paper plan

If, after Gate **G5** (§6.5) — the first point at which RL results exist; G4 has already delivered Paper A — the reviewer agent judges the RL results insufficient to carry an independent article — the realistic failure mode under a CPU-only budget — collapse to **one paper for EAAI**: M1 + M2 + exact reference + full theory, with the RL model retained as a **single baseline row** and the SMDP formulation moved to an appendix. This preserves C1–C5 and C9 in full. Do **not** collapse in the other direction: a learning paper without the cooperative theory loses the work's most distinctive contribution.

---

# 3. Methodological programme

## 3.0 Unified problem `P` (shared by both papers, implemented once)

All three methods and the exact reference must optimise **the same** `Φ` over **the same** feasible set, evaluated by **the same** simulator. This is non-negotiable: it is what makes the cross-method comparison tables meaningful, and it is the fix for F6.

**Sets.** Jobs `J`, operations `O = {(i,k)}`, machines `M`, robots `V`, locations `L ⊇ M ∪ {LU, CH}`, tariff intervals `H`. Transport tasks `T = {(i,k) : k = 1,…,mᵢ+1}` with `τ_{i,k} = t_{prev(i,k), dest(i,k)}`.

**Energy cost (corrected — resolves F6).**
```
E_cost = Σ_{(i,k)∈O} ∫_{S_{i,k}}^{C_{i,k}} P_mach · c(t) dt          ← machine processing, ToU-indexed
       + Σ_{r∈V} ∫_{charging intervals of r} P_ch · c(t) dt          ← robot charging, ToU-indexed
```
Both terms are piecewise-constant integrals over the tariff partition and are computed exactly by the simulator.

**Objective.** The primary formulation is an **ε-constraint**, not a weighted sum:
```
minimise   E_cost        subject to   C_max ≤ H
```
The scalarised form `Φ = ω·Ĉ_max + (1−ω)·Ê_cost` is retained **only where a decentralised agent needs a single scalar** — in the game utility and the RL reward — and then strictly over the **normalised** quantities `Ĉ_max, Ê_cost ∈ [0,1]`. §3.7 gives the construction, the reason the raw form is unusable, and the invariance conditions the normalisation must satisfy for T1 and T7 to survive.

**The horizon `H` is a hard deadline, not a display window.**
```
C_max ≤ H                                                  (deadline)
R_p^q(a_p) ≤ H  for every committed action                 (per-action feasibility)
```
`H` is the maximum allowed makespan: **no schedule may complete after it.** The draft already carries the per-action form of this condition in its machine return function, but never states it as a system constraint, and no model enforces it (erratum A30). Making it explicit changes five things, and each is a design obligation rather than a line of algebra:

1. **Instances can be infeasible.** `H` is generated from the λ formula, so it is not guaranteed to admit any feasible schedule. Every instance must carry a certified **feasibility margin** `H / C_max^LB`, where `C_max^LB` is a valid lower bound (critical path + minimum transport + unavoidable charging). WP4 rejects or re-scales any instance with margin < 1, and reports the margin distribution. λ is therefore a **feasibility parameter**, not merely a tariff-shaping one, and its ablation (A18) becomes load-bearing.
2. **Greedy and best-response dynamics can deadlock.** A myopically cheap decision — deferring a machine start into the off-peak band, or sending a robot to charge — can leave no feasible completion. Every decentralised method needs a **feasibility safeguard**: before committing an action, check a lower bound on the remaining work against the time left, and fall back to a deadline-feasible dispatching policy when the bound is violated. Report the **deadlock rate** (episodes needing fallback) as a first-class metric; a method that meets the deadline only by falling back most of the time is not a scheduler.
3. **The deadline supplies the missing upper anchor for normalisation.** `H` is exactly the value of `C_max` that a feasible schedule must not exceed, so `Ĉ_max = (C_max − C_max^LB)/(H − C_max^LB)` maps the entire feasible makespan range onto `[0,1]` with no arbitrary constant. §3.7.
4. **The cooperative characteristic function can be `+∞`.** A small coalition may be unable to meet the deadline while the complement follows `π₀`, which would make `c` non-real-valued and destroy the Shapley machinery. §3.4 must adopt the penalised form below.
5. **Disruption changes meaning.** Under E7, a breakdown can make the deadline unreachable. The KPI is then **deadline-miss rate and lateness**, not degraded makespan.

**Hard constraints.** Operation precedence within a job; transport precedes every operation; machine unary capacity; robot unary capacity; **charger capacity `K_CH` (unary by default — this constraint is absent from the current draft, F7)**; SoC ∈ [`B_min`, `B_max`] at all times with the three depletion rates; no pre-emption; **`C_max ≤ H`**.

**Units contract (resolves F8). Three separate reconciliations, all due at WP1 before any result exists.**

*(i) Voltage and charger power.* Declare nominal bus voltage `U_nom = 48 V`, so `B_max = 100 Ah ⇒ 4.8 kWh`. The draft's 5-minute / 20 Ah recharge block implies 0.96 kWh in 300 s = **11.52 kW**, which contradicts the 1.5 kW charger the draft attributes to its Phihong source. No bus voltage rescues 1.5 kW (it would require 6.25 V). **Resolution: Ajust the charger power and correct the recharge block to 5 min per 16 Ah.** Assume it takes 25 min to go from 0 % to 80 % of SoC under fast-charging,  with the SoC growing linearly Within the [0, 80] range. If it is genuinely needed, cite a fast-charge source.

*(ii) Depletion-rate units — this one silently disables the whole battery axis.* The draft states 1 / 3 / 7 **mA per second**. Read literally against 60 Ah of usable capacity (the [20 %, 80 %] window), 7 mA/s consumes **0.043 Ah over the longest ToU horizon in the appendix table (EX104, 367.5 min)** and would take 357 days to drain the battery. The SoC floor could never bind, no robot would ever need to charge, and the battery/charging dimension — one of the two pillars on which Paper A's novelty rests — would be inert. The intended unit is almost certainly **mAh per second**: at 7 mAh/s the usable 60 Ah drains in 143 min, which binds well inside every horizon in the table. **Resolution: adopt mAh/s, state it explicitly, and add a WP1 test asserting that the SoC floor binds on at least a declared fraction of instances.** If that test fails, the parameters are wrong, not the code. You may change the depletion rates for relevant, literature-supported ones if geniunely needed.

*(iii) `B_needed` dimensional mismatch.* The draft's `B_needed(a_p) = 20 + ec·t_{p,i} + …` adds a bare `20` (Ah, the SoC floor) to terms in mAh/s × s (mAh). **Resolution: express everything in mAh; the floor is 20 000 mAh.**

Publish all three in the units table. Each of these *will* be caught by an energy-literate referee.

## 3.1 Method inventory

| ID | Method | Role | Paper |
|---|---|---|---|
| `EX-CP` | CP-SAT exact model | Optimality reference, PoA denominator | A (B cites) |
| `LB-MILP` | Disjunctive MILP (HiGHS) | Structural exposition + cross-validation on tiny instances | A |
| `DR-*` | Dispatching rules × vehicle rules × charge policies | Cheap baselines, industrial realism | A, B |
| `NSGA2` / `MOEAD` | Multi-objective metaheuristics | Pareto-front baseline | A, B |
| **`M1`** | Non-cooperative potential game, best response / log-linear learning | Core mechanism | A (baseline in B) |
| **`M2`** | Cooperative coalitional game — three layers | **New contribution** | A (baseline in B) |
| **`M3`** | CTDE attention maskable-PPO on the SMDP | Learning system | B |

## 3.2 Exact reference `EX-CP` — **adapted, not rebuilt**

### 3.2.0 Provided assets (read these before writing a line of solver code)

The author supplies, in `C:\Users\HP\Desktop\Work\Dec_Optim_ToU`:

- **a working CP-SAT solver for the classical JSSPT** (job shop with transport, no tariffs, no batteries);
- **published papers presenting ILP models of classical JSSP under ToU tariffs.**

`EX-CP` is an **extension of the provided solver**, not a reimplementation. This is a hard instruction: rebuilding from scratch discards a tested artefact, loses whatever modelling decisions it already encodes, and removes the single cheapest correctness check available to this programme.

**Intake procedure (WP0.5, before WP3 and gated at G0).**

1. **Inventory.** Read every file in the folder. Write `docs/provided_assets.md` recording, for the solver: its entry points, instance format, variable and constraint inventory, objective, the solver options it sets, its tests if any, and its licence/authorship. For each paper: full citation, the ToU-cost linearisation it uses (time-indexed binaries, interval-membership binaries, piecewise-linear, or start-time element constraints), how it handles the horizon, whether it models idle/standby power, its reported instance sizes and solve times.
2. **Comparison table** of the ToU linearisations found, with variable/constraint counts as a function of `|O|` and `|H|`, and a reasoned pick for `EX-CP`. This table is a figure in Paper A §4 and is also what justifies the modelling choice to a referee.
3. **Vendored, unmodified copy** in `external/jsspt_cpsat_original/`, with a `PROVENANCE.md` naming the author and the commit/date received. Never edit files in `external/`.
4. **Adapter layer** in `src/jsspt_tou/exact/cpsat_model.py` that imports or wraps the original and adds the ToU cost, battery/charging, charger capacity and deadline on top of it. Where the original's structure genuinely cannot carry an extension, record why in `DECISIONS.md` and port the affected constraint block explicitly, citing the original lines.
5. **Instance-format bridge**: either adopt the provided solver's instance format as the project's canonical one (preferred — it is already tested), or write a lossless bidirectional converter with a round-trip test.

> **The flat-tariff regression test — the cheapest correctness guarantee in the whole programme.**
> Set every tariff interval to the same price and remove the deadline and battery constraints. The energy term is then schedule-invariant, so `EX-CP` must return **exactly the makespan the provided solver returns** on every classical JSSPT instance the original was validated on. Any discrepancy is a bug in the adaptation, located before any ToU result exists. Make this a permanent test (`tests/test_exact_regression.py`), not a one-off check.
>
> Two companion regressions: with the deadline set to `+∞` the model must reproduce the flat-tariff result; with `ω = 0` in the scalarised variant and a flat tariff, every feasible schedule must tie.

### 3.2.1 The extended model

- Operations as `IntervalVar(start, size=p_{i,k}, end)`; `NoOverlap` per machine.
- Transport tasks as optional intervals `OptionalIntervalVar` per (task, robot) with `ExactlyOne` over robots; `NoOverlap` per robot.
- Charging as a set of optional intervals per robot; `Cumulative(charging intervals, demands=1, capacity=K_CH)` — the missing charger constraint.
- **SoC tracking**: position-indexed per robot via `AddCircuit` or a rank-variable formulation; SoC as integer state in mAh with `AddElement`-linked transitions; prefix constraints `SoC_pos ≥ B_min` at every position (the exact version of F7's relaxation).
- **ToU processing cost**: for each operation, `cost_{i,k} = P_mach · Σ_h c_h · overlap(S_{i,k}, S_{i,k}+p_{i,k}, [T_{h−1}, T_h))`, encoded with per-interval overlap integer variables and linear linking. Same construction for charging cost.
- **Deadline**: `C_max ≤ H` as a hard constraint (`AddLessOrEqual`), and `H` as the upper bound of every `end` variable — which also shrinks every domain and is the single cheapest propagation win available.
- **Objective — Weighted sum**: `minimise E_cost and C_max`, integer-scaled (times in minutes ×10, costs in 10⁻⁴ €) to keep CP-SAT in integer arithmetic.
- **Budget**: 3 600 s wall-clock, 8 workers, per instance. Record `status`, `objective`, `best_bound`, `gap`. Instances where `status = OPTIMAL` form the **PoA-evaluable set**; `status = INFEASIBLE` at `H` marks an instance whose deadline is unachievable and is a WP4 generation defect, not a result. All others contribute bounds only.
- **Expected reach on CPU**: proven optimality on the 4-machine Bilge–Ulusoy family and on generated instances up to ~12 jobs; useful bounds to ~20 jobs. This is exactly the scalability wall the paper argues motivates decentralisation — **report it as evidence, not as a limitation**.

## 3.3 `M1` — non-cooperative potential game (corrected)

**Convention (resolves F4).** `W(α) = Φ(α⁰) − Φ(α)`; `u_p(α) = W(α) − W(a_p⁰, α_{−p})`; **all agents maximise `u_p`**.

**Algorithm M1a — sequential (round-robin) best-response dynamics.**
```
at each event time t_q:
    P_q ← active players (idle machines, free robots)
    initialise α ← (a_p⁰)_{p∈P_q}
    repeat (round-robin over a random permutation of P_q):
        for p in permutation:
            a_p ← argmax_{a ∈ A_p(s_q)} u_p(a, α_{−p})   # evaluated by the simulator's myopic estimator
    until no player improves, or iter > I_max
    commit α, advance simulator to next event
```
Finite improvement property ⇒ termination at a pure NE of the stage game (T1). **The updates must be sequential, not simultaneous:** FIP guarantees termination only when one player deviates at a time; genuinely synchronous best response can cycle even in a potential game. **FIP also requires finite action sets**, which forces the machine action space to be the finite ToU-aligned delay grid `a_m = (i, d), d ∈ 𝒟` used in the draft's later sections, *not* the continuous `(J_i, t_start)` of its earlier ones — see erratum A27. Instrument: iterations to convergence, number of improvement steps, and whether `I_max` binds.

**Algorithm M1b — log-linear learning.** Replace the argmax with a Gibbs choice `Pr(a_p) ∝ exp(u_p(a_p, α_{−p}) / T)` and anneal `T`. As `T → 0` the stationary distribution concentrates on **potential maximisers**, i.e. stochastically stable states — providing an equilibrium-selection mechanism that targets the *best* NE rather than an arbitrary one, and empirically tightening the realised PoA. This is a cheap, well-founded addition that materially strengthens §5 of Paper A.

**Complexity.** Per stage: `O(I_max · Σ_p |A_p| · C_eval)`, where `C_eval` is one myopic simulator evaluation. Report measured `C_eval` and the total decision count.

## 3.4 `M2` — cooperative coalitional game **(new)**

The cooperative model is organised as **three interacting layers**. Layer 2 carries the stability analysis; Layer 1 makes the approach decentralised and tractable; Layer 3 removes the arbitrary weight ω. Layer 2 is presented first because a negative structural result found there — congestion breaks submodularity — determines how the other two layers must be framed.

### Layer 2 (core) — TU savings game and certified stability

**Players.** `N = M ∪ V` (machines and robots), `n = |N|`.

**Characteristic cost function.** For `S ⊆ N`,
```
c(S) = min over joint policies of the members of S  of  Φ,
       with agents in N \ S following a fixed reference policy π₀ (the best dispatching rule),
       evaluated on the shared simulator.
```
`c(∅) = Φ(π₀ everywhere)`. This "coordinate within `S`, default outside" construction is the standard partition-function-free reduction. Two properties of it must be stated in the paper, because referees will look for exactly these admissions: it fixes the externalities from the complement, and — critically — **`c(S)` is a whole-system cost for every `S`, including singletons.**

> **The deadline can make `c(S)` infinite, and that would break everything downstream.** Under §3.0 a schedule must satisfy `C_max ≤ H`. A small coalition coordinating against a `π₀` complement may have no feasible completion, giving `c(S) = +∞`. A characteristic function that is not real-valued has no Shapley value, no core LP and no least-core radius — the entire Layer 2 apparatus stops existing on exactly the instances where coordination matters most.
>
> **Resolution: the penalised characteristic function.**
> ```
> c(S) = Ê_cost(S) + M · max(0, Ĉ_max(S) − 1)          on normalised quantities (§3.7.3)
> ```
> where `Ĉ_max = 1` is precisely the deadline. `M` is finite, so `c` stays real-valued on all of `2^N` and every solution concept remains defined; the penalty is zero on deadline-feasible outcomes, so nothing changes where feasibility holds.
>
> Three obligations follow, and none may be skipped:
> 1. **Calibrate `M`** from the instance, not by taste: `M ≥ E_cost^UB − E_cost^LB` in normalised terms (i.e. `M ≥ 1`) guarantees that any deadline-feasible outcome is preferred to any infeasible one. Use `M = 10` as the default and **report a sensitivity ablation over `M ∈ {2, 5, 10, 50}`** — an allocation that moves with `M` is an artefact, not a result.
> 2. **Re-certify the structural properties.** The penalty term is a max of an affine function, hence convex in the coalition's achievable makespan, and it can change the sign of a marginal contribution. Every submodularity test, least-core LP and superadditivity check in this section must be run on the **penalised** `c`, and the congestion counterexample (N5) re-verified against it.
> 3. **Report the infeasible-coalition rate** — the fraction of sampled coalitions with `Ĉ_max > 1` — per instance. If it is high, the honest reading is that the deadline, not the tariff, is what the coalition structure is organised around, and Paper A should say so.

**Savings game (corrected).**
```
v(S) = c(∅) − c(S),        v(∅) = 0
```
> **Do not use `v(S) = Σ_{i∈S} c({i}) − c(S)`.** That form presupposes `c({i})` is a *standalone* cost. Under the construction above it is a whole-system cost, so the sum double-counts roughly `(|S|−1)·c(∅)`: it gives `v(∅) = −c(∅) ≠ 0`, makes `v({i}) = 0` for every `i` by construction, and leaves `v(N)` measuring a quantity that is not the total savings — which the Shapley efficiency condition `Σφᵢ = v(N)` would then allocate. `v(S) = c(∅) − c(S)` is the correct savings game for this characteristic function and is monotone by construction.

#### The congestion obstruction — established before anything is built on it

> **Proposition (negative).** `c` is **not** submodular in general for problem `P`. Consequently the savings game is not convex in general, and **the core can be empty.**
>
> *Counterexample (in-domain, verified numerically).* Two robots `V1, V2`; one charging station with unary capacity (the `K_CH` constraint that §3.0 correctly adds); one **38.4-minute** off-peak slot at 0.06 €/kWh (the block length fixed by §3.0(i)), otherwise 0.22 €/kWh; each robot needs one 20 Ah charge block = 0.96 kWh at 1.5 kW; `ω = 0`.
> ```
> c(∅) = 0.4224     (both charge at peak, under π₀)
> c({V1}) = c({V2}) = 0.2688     (the coordinating robot takes the cheap slot)
> c({V1,V2}) = 0.2688            (only one robot can use the slot regardless)
>
> Δ_{V1} c(∅)      = c({V1}) − c(∅)        = −0.1536
> Δ_{V1} c({V2})   = c({V1,V2}) − c({V2})  =  0.0000
> Submodularity of a cost requires Δ_{V1}c(A) ≥ Δ_{V1}c(B) for A ⊆ B:  −0.1536 ≥ 0 is FALSE.
> ```
> `c` is **supermodular** here — the second robot to coordinate gains nothing, because the scarce resource is already taken. With the corrected savings game, `v({V1}) = v({V2}) = v({V1,V2}) = 0.1536`, so the core requires `x₁ + x₂ = 0.1536` with `x₁ ≥ 0.1536` and `x₂ ≥ 0.1536` — **empty**. The Shapley value `(0.0768, 0.0768)` is not in it.

The mechanism is general: **any scarce shared resource that saturates converts diminishing returns into congestion, and congestion is supermodular.** The charger is the obvious one here; machine bottlenecks behave the same way. This is not a defect to be engineered around — it is a structural property of the problem, and reporting it is a contribution. It does, however, decide the architecture of Layer 2.

#### What is actually proved and what is measured

> **P5 (conditional proposition — not the headline).** On the **congestion-free sub-class** — instances in which no shared resource saturates over the horizon (formally: charger utilisation below a stated threshold and no machine on the critical path at full utilisation) — `c` is submodular, hence `v` is convex, hence the core is non-empty and `φ(v) ∈ C(v)` (Shapley 1971; Ichiishi 1981 proves the converse and should not be cited for this direction).
>
> *Proof obligations.* (i) State the congestion-free condition precisely and verifiably from a simulation trace. (ii) Prove that submodularity of `Φ` on `2^Ω` (corrected F2) descends to `c` under the min-envelope, **which does not hold automatically** — a pointwise minimum of submodular functions need not be submodular, and `Φ` is non-decreasing in committed (player, action) pairs while `c` is non-increasing in coalition membership, so this is a genuine proof step on two functions of opposite monotonicity, not a restriction. If the proof does not close under a defensible condition, **P5 becomes an empirical claim certified per instance** and nothing else in the programme changes.

> **T5 (headline, empirical and always available). Certified stability via the least core.** For every instance, compute the least-core radius `ε*` and report the distribution of `ε*/v(N)` across the benchmark, together with the fraction of instances whose core is non-empty and the fraction on which the Shapley value is core-stable.

This is the correct headline: it is always computable, it is honest about the congestion obstruction, and — unlike an unconditional core theorem — it is true.

> **T6. `v` is monotone and superadditive**, `v(S ∪ T) ≥ v(S) + v(T)` for disjoint `S, T`, **on the congestion-free sub-class**; a counterexample on the congested class is the one above. Monotonicity `v(S) ≤ v(S')` for `S ⊆ S'` holds unconditionally by construction (a larger coalition can replicate the smaller one's policy).

#### Numerical certification (mandatory)

- **Submodularity sampling test.** Draw triples `(A ⊆ B, x ∉ B)` and check `c(A∪x) − c(A) ≥ c(B∪x) − c(B)`. Report violation rate, worst violation magnitude, and its correlation with charger utilisation — this correlation *is* the evidence for the congestion mechanism, and is a figure in Paper A.
- **Core / least-core LP, in the correct direction for a savings (profit) game:**
  ```
  maximise ε
  s.t.  Σ_{i∈S} xᵢ ≥ v(S) + ε     for all S ⊂ N      (separated by constraint generation)
        Σ_{i∈N} xᵢ = v(N)
  ```
  `ε* ≥ 0` certifies a non-empty core; `ε* < 0` gives the least-core radius, i.e. how far from stable the best allocation is.
  > The cost-game form `Σ_{i∈S} xᵢ ≤ v(S) + ε` with "`ε ≤ 0` certifies membership" is the **wrong direction** for this game and would certify the wrong condition with an inverted sign test. This is an easy error to make and must be asserted in `tests/test_cooperative.py` against a hand-computed three-player example.

#### Shapley value at scale — and its real cost

`φᵢ(v) = Σ_{S ⊆ N\{i}} [|S|!(n−|S|−1)!/n!] · [v(S∪i) − v(S)]` is `O(2ⁿ)`. Implement ApproShapley (Castro et al.) with **stratification by coalition size** and **antithetic permutation pairs**, reporting Hoeffding/CLT confidence intervals with every allocation.

> **The binding constraint is not the combinatorics — it is that every `c(S)` evaluation is itself a scheduling optimisation.** 2 000 permutations at `n = 30` is 60 000 `c(S)` calls, plus 2 000 submodularity triples is 8 000 more: ≈ 68 000 optimisation runs *per instance per ω*. At an optimistic 0.1 s each that is 1.9 h per instance and **≈ 40 days serial over 510 instances**; at 1 s each it is over a year. Any plan that does not budget `c(S)` calls is not a plan.
>
> **Mandatory scoping, declared in the paper:**
> - `c(S)` is evaluated by a **single sequential best-response pass** (M1a with `I_max = 1` restart), not by a full optimisation. Measure and report its wall-clock and its gap to `EX-CP` on small instances, so the surrogate is characterised rather than assumed.
> - Full Shapley + submodularity certification runs on a **declared certification subset**: all instances with `n ≤ 12`, plus 40 instances sampled across the generated suite. Everything else reports coalition-structure results and least-core `ε` only.
> - `c(S)` results are **memoised** across permutations (the dominant saving — most sampled coalitions recur) and evaluated in parallel across cores.
> - Report the total `c(S)` call count and wall-clock hours in the paper. Silent truncation of a certification budget reads as "we certified everything".

**Alternative allocations for comparison.** Nucleolus (LP, exact for `n ≤ 12`), τ-value, equal-split, proportional-to-standalone. Table comparing stability (`ε*`), fairness (Gini), and computation time.

### Layer 1 — Decentralised coalition-structure generation

The grand coalition is efficient **on the congestion-free sub-class** (T6) but requires all-to-all coordination, which contradicts the paper's decentralisation thesis and does not scale. Layer 1 finds a **partition** `CS = {S₁,…,S_K}` of `N` that captures most of the savings at bounded communication.

**Hedonic preference.** Agent `i` prefers coalition `S` to `S'` iff `φᵢ(v|_S) > φᵢ(v|_{S'})` — payoff-based hedonic preferences over the Shapley allocation within each coalition, where `v|_S(T) = c(∅) − c(T)` restricted to `T ⊆ S`. Because congestion can make the grand coalition non-convex (Layer 2), coalition formation is not merely a tractability device here: **partitioning can be genuinely better than the grand coalition when a shared resource saturates**, and E4 should test exactly that.

**Algorithm M2a — merge-and-split (Apt & Witzel / Saad et al.).**
```
CS ← {{i} : i ∈ N}                      # singletons
repeat
    MERGE:  if ∃ S, T ∈ CS with S ∪ T Pareto-improving for all members
                (φᵢ(v|_{S∪T}) ≥ φᵢ(v|_S) ∀i, strict for one)
            then CS ← CS \ {S,T} ∪ {S∪T}
    SPLIT:  if ∃ S ∈ CS and a partition {S₁,…,S_m} of S Pareto-improving
            then CS ← CS \ {S} ∪ {S₁,…,S_m}
until no merge or split applies
```
Guaranteed to terminate (Pareto order over partitions is acyclic) at a **`D_hp`-stable** partition — no group of players can profitably merge or split. State this as a proposition with the standard proof.

**Communication accounting.** Report messages exchanged and maximum coalition size versus centralised coordination — this is the quantitative evidence that the cooperative model remains *decentralised*, and it is exactly what a referee will demand given the framing.

**Scaling guard.** Restrict candidate merges to a **spatial/functional neighbourhood** (machines within a travel-time radius, robots serving them), capping coalition size at `S_max` (default 6). Report the savings lost versus the grand coalition on instances small enough to compute both.

### Layer 3 — Bargaining-derived trade-off weight (contribution C2)

The weight ω is currently a free parameter, which is the standard weak point of scalarised bi-objective scheduling papers. Layer 3 derives it.

**Setup.** Partition the agents into two interest blocs: **production** `M` (payoff decreasing in `C_max`) and **logistics/energy** `V` (payoff decreasing in `E_cost` and in charging disruption). Let the feasible payoff set `U ⊂ ℝ²` be the image of the Pareto front computed by NSGA-II on the shared simulator. Let the **disagreement point** `d = (u_M(α*), u_V(α*))` be the payoff pair realised at the **non-cooperative equilibrium of M1** — the natural threat point, since it is what happens if the two blocs do not coordinate.

**Nash bargaining solution.** `α^NBS = argmax_{u ∈ U, u ≥ d} (u_M − d_M)(u_V − d_V)`. Unique under convexity of `U`; for the non-convex discrete front, take the maximiser over the front and report the convexified relaxation as an upper reference.

**Kalai–Smorodinsky solution.** The point on the front where `(u_M − d_M)/(u_M^ideal − d_M) = (u_V − d_V)/(u_V^ideal − d_V)` — monotonic rather than IIA, and often preferred in operations settings.

**Induced weight.** The supporting hyperplane of the front at `α^NBS` yields `ω^NBS` directly. Report `ω^NBS` per instance and per tariff structure, and characterise how it moves with peak/off-peak spread and fleet size. **This is a genuinely publishable finding in its own right**: it says how much a factory *should* weight energy against throughput, derived rather than assumed, and it converts the standard ω-sensitivity plot from a caveat into a result.

**Axiomatic justification to state.** NBS satisfies Pareto optimality, symmetry, scale invariance and IIA; KS replaces IIA with individual monotonicity. Both are appropriate; report both and let the disagreement between them (if any) be a discussion point.

## 3.5 `M3` — game-guided MARL

**Environment.** Gymnasium-compatible wrapper over the shared simulator, event-driven, with `action_mask` in the observation dict. Implement exactly the observation constructor and masks of the draft (§`sec:obs`) — they need no redesign.

**Architecture (sized for CPU).**
- Shared encoder: global context `g(s)` → 2-layer MLP, `d = 64`.
- Two heterogeneous actor heads (robot, machine), each: local features → query vector `q ∈ ℝ^d`; candidate-action features → key/value `k_j, v_j ∈ ℝ^d`; scores `⟨q, k_j⟩/√d`; masked softmax over the ≤ `K` candidates. Single attention head, no stacking — the variable-size handling is what matters, not depth.
- Centralised critic `V_φ(s)` over the full state (CTDE), 2-layer MLP `d = 128`.
- Parameter count target: **< 250 k**. This is a deliberate CPU-budget choice and must be justified in the paper, not hidden.

**Training.** Maskable PPO exactly as the draft's Algorithm 1, with time-aware discount `γ^{Δt}` and time-aware GAE. 16 vectorised envs, 3–5 M decision steps, ~8–14 h on a 16-core CPU. Curriculum: train on 6–15 job instances, evaluate zero-shot on 20–50.

**Game-guided reward (C6).** Two reward modes, ablated head-to-head:
- `global`: `r_q = −ω Δt_q − (1−ω) ΔE_q` (the draft's reward);
- `difference`: `r_q^p = Φ(a_p⁰, a_{−p}) − Φ(a_p, a_{−p})`, estimated by one counterfactual myopic simulator rollout.

> **T8 (identity, stated honestly as such).** The difference reward of the `difference` mode is **identical by construction** to the marginal-contribution utility `u_p` of M1, once F4's convention is fixed. **Stage-game corollary:** at any single decision stage, the per-agent gradient under `difference` is a gradient on the exact potential of that stage game, so credit assignment is aligned by construction rather than by reward engineering.
>
> **What T8 does *not* say, and must not be written as saying.** The SMDP objective is a time-discounted *sum* of stage rewards, which is **not** the stage potential. Alignment at each stage does not imply that the learned policy optimises a potential of the trajectory — this is the same scope boundary as T2, and Paper B must respect it in exactly the way Paper A does.

**Framing discipline.** T8 is a definitional identity, not a discovery, and difference-reward/potential alignment is established in the literature (Wolpert & Tumer 1999; Agogino & Tumer; Devlin, Yliniemi, Kudenko & Tumer, *Potential-based difference rewards for multiagent reinforcement learning*, AAMAS 2014). **Paper B must cite that literature in §2 and claim the empirical question, not the identity**: does game-derived credit assignment pay off on *this* SMDP — event-driven, two heterogeneous agent types, variable action sets, battery-coupled, ToU-coupled — and at what cost? The honest counterpoint to report: the counterfactual rollout costs one extra simulator evaluation per decision; report the wall-clock overhead against the sample-efficiency gain and let the trade-off be the finding (T9 characterises the bias/variance of the sampled estimator).

**Warm start (optional, ablated).** Behaviour-cloning pre-training on trajectories generated by M1 best response, then PPO fine-tuning. Cheap, usually a large sample-efficiency win, and it makes the two papers concretely complementary.

## 3.6 Theorem inventory

| ID | Statement | Difficulty | Paper |
|---|---|---|---|
| T1 | Stage game with MCU utilities is an **exact potential game**; FIP ⇒ best response converges to a pure NE in finitely many steps | Easy — one-line cancellation + Monderer–Shapley | A |
| T2 | The dynamic game is a sequence of stage potential games; per-stage convergence does not imply trajectory optimality (with a counterexample instance) | Easy but essential for honesty | A |
| T3 | **PoS = 1**: a global minimiser of `Φ` is a pure NE | Easy | A |
| T4 | PoA ≥ ρ for an explicit family of instances; empirical PoA distribution over CP-SAT-closed instances | Medium — needs a constructed worst case | A |
| **P5** | *Conditional:* on the congestion-free sub-class, `c` submodular ⇒ convex savings game ⇒ core ≠ ∅ and Shapley ∈ core | Hard — the min-envelope step is a real proof obligation, not a restriction; may not close | A |
| **T5** | **Certified stability**: least-core radius `ε*` per instance; distribution of `ε*/v(N)`; fraction core-stable; correlation of submodularity violations with charger utilisation | Easy to compute, always available — **this is the headline** | A |
| **N5** | *Negative:* `c` is supermodular under charger congestion; explicit two-robot instance with an empty core | Easy — worked in §3.4; a genuine structural finding | A |
| T6 | `v` is monotone (unconditionally) and superadditive on the congestion-free sub-class | Easy | A |
| T7 | Undiscounted episodic return `= −Φ` on feasible episodes (corrected `E_cost`) | Easy — telescoping; already drafted | B |
| T8 | Difference reward ≡ MCU ⇒ stage-aligned credit assignment (identity, not discovery; scope limited to the stage) | Trivial to state; value is the empirical study it frames | B |
| T9 | Bias/variance of the single-rollout difference-reward estimator; conditions for unbiasedness | Medium | B |

**Assumption additions made explicit.** T1 requires **finite action sets** (hence the finite ToU delay grid `𝒟`) and **sequential** deviations. T3 is a **stage-game** statement: `PoS = 1` for each stage game, *not* for the trajectory. T7 requires feasible episodes and the corrected `E_cost`. P5 requires the congestion-free condition *and* an unproved min-envelope step.

**Proof-hygiene rule for the execution agents.** Every theorem must be accompanied by (a) an explicit assumption list, (b) a machine-checked numerical test in `tests/test_theory.py` that would *fail* if the theorem were false (randomised submodularity checks, potential-identity checks over random action pairs, the core LP against a hand-computed example, the §3.4 congestion counterexample as a regression test asserting the core *is* empty there), and (c) a stated counterexample or limitation where the result does not extend. A theorem without a falsifying test is not delivered.

## 3.7 Bi-objective treatment: commensurability, normalisation, and what the weighted sum can and cannot do

### 3.7.1 The raw weighted sum is unusable as is with numbers

`Φ = ω·C_max + (1−ω)·E_cost` adds **minutes to euros**. On the draft's own instance EX11 (job set 1, layout 1) the two terms are not merely different units, they are two orders of magnitude apart:

| Quantity | Value | Derivation |
|---|---|---|
| Span-weighted mean tariff over the EX11 horizon | **0.140 €/kWh** | prices {0.14, 0.22, 0.14, 0.22, 0.14, 0.06} over spans {27, 45, 27, 27, 18, 72} |
| Machine processing energy | 5.87 kWh → **0.82 €** | 2 kW × 176 min of processing |
| Robot charging energy (2 robots × 2 blocks) | 3.84 kWh → **0.54 €** | 20 Ah = 0.96 kWh per block at 48 V |
| **`E_cost`** | **≈ 1.36 €** | |
| **`C_max`** | **150–216 min** | |
| **Ratio `C_max / E_cost`** | **110 – 159** | |

Consequences, computed from those figures:

| ω | Share of `Φ` contributed by energy (`C_max` = 150 / 216) |
|---|---|
| 0.25 | 2.6 % / 1.9 % |
| 0.50 | 0.90 % / 0.63 % |
| 0.75 | 0.30 % / 0.21 % |

The value of ω at which the two terms contribute equally is **ω\* ≈ 0.009 (C_max = 150) to 0.006 (C_max = 216)**. Everything above roughly ω = 0.02 is, numerically, a pure makespan objective.

> This is not a rounding concern. It means that **as currently written, the draft's entire bi-objective study is a single-objective makespan study**, its ω sweep would produce a flat line over almost all of `[0,1]`, and the roadmap's own earlier grid `ω ∈ {0.25, 0.5, 0.75}` would have sampled three points that are all makespan-only. It also silently corrupts everything downstream: the marginal-contribution utilities are dominated by the makespan term, so the potential the agents descend is effectively `C_max`; the RL reward is dominated by `−Δt`; and the "energy-aware" claim would not survive a referee who does this arithmetic — which takes about two minutes. **Erratum A29.**

**Note: Ignore 3.7.2 and 3.7.3. Keep the weighted sum formulation with Upper-Lower Bound Normalization (Nadir–Ideal Scaling).**
### 3.7.2 Primary formulation: ε-constraint (no scalarisation at all)

Because §3.0 makes `H` a hard deadline, the problem *already is* an ε-constrained one:

```
minimise  E_cost      subject to  C_max ≤ ε,      ε ∈ [C_max^LB, H]
```

This is the right primary formulation on three independent grounds.

- **Commensurability disappears.** There is nothing to add up: one quantity is optimised, the other is constrained. No weight, no units question, no calibration.
- **It reaches the whole Pareto front.** A weighted sum can only return solutions on the **convex hull** of the front; for a discrete scheduling problem the front is non-convex and pitted, so entire regions are unreachable at *any* ω. The ε-constraint method returns supported and unsupported points alike. Use **AUGMECON2** (augmented ε-constraint with a slack term in the objective) to avoid weakly-efficient points, and take this front as the reference for hypervolume and IGD.
- **It matches how the shop actually decides.** A plant manager sets a due date and then buys the cheapest schedule that meets it. The ε sweep *is* the managerial question "what does each extra hour of deadline slack save me?", and §8 of Paper A should present it exactly that way.

The weighted sum survives only as a *presentation* device for the equilibrium mechanism, never as the reference model.

### 3.7.3 Where a scalar is unavoidable: normalisation with instance-fixed anchors

Game utilities and RL rewards need one number. Normalise both objectives onto `[0,1]` first, using anchors that are **closed-form constants of the instance**:

```
Ĉ_max  = (C_max  − C_max^LB) / (H − C_max^LB)                     ∈ [0,1] on feasible schedules
Ê_cost = (E_cost − E_cost^LB) / (E_cost^UB − E_cost^LB)           ∈ [0,1]

C_max^LB  = max( critical path incl. minimum transport,  max-machine load,
                 max-robot load incl. unavoidable charging )
E_cost^LB = (total unavoidable kWh) × min_h c_h
E_cost^UB = (total unavoidable kWh) × max_h c_h
Φ = ω · Ĉ_max + (1 − ω) · Ê_cost
```

Note what the deadline buys: `H` is a *principled* upper anchor for makespan, not an arbitrary constant — it is the largest value a feasible schedule may take. Total energy is schedule-invariant under the draft's assumptions (idle and standby power excluded), so `E_cost^LB` and `E_cost^UB` are the same kWh priced at the cheapest and dearest tariff, and `Ê_cost` measures exactly the quantity the scheduler controls: **where in the tariff profile the load was placed**. Both anchors need no solver call.

> **Invariance condition — T1 and T7 depend on it.** The anchors must be computed **once per instance, before the game or episode begins, and never updated**. Recomputing them from running best-known values (a tempting "adaptive normalisation") makes `Φ` a function of history rather than of the joint action profile: the exact-potential identity of **T1** fails, the FIP convergence argument fails with it, and the telescoping in **T7** no longer sums to `−Φ`. This is a silent, plausible-looking bug that would invalidate two theorems and produce results that still look reasonable. Assert it in `tests/test_theory.py`.

**Dimensional-consistency property test (new, and it should be standard practice).** Rescale the units of either objective — minutes → seconds, euros → cents — recompute the anchors, and assert that the **ranking of any two schedules under `Φ` is unchanged** and that every reported normalised value is identical. A model that fails this test is measuring its own units, not the schedule.

### 3.7.4 The managerial variant: monetise instead of weighting

For Paper A §9 (and only there), report the fully monetised objective:

```
Φ€ = c_time · C_max + E_cost        [€],     c_time = opportunity cost of shop time [€/min]
```

One unit throughout, no ω, and `c_time` is a quantity a plant actually knows. It also reframes the bargaining layer usefully: **Layer 3 then derives `c_time` rather than ω**, and "what is an hour of makespan worth relative to tariff savings?" is a far more publishable managerial finding than "we set ω = 0.5". Report the `c_time` at which the cost-optimal schedule flips from off-peak-seeking to deadline-seeking — that break-even is the paper's headline managerial number.

### 3.7.5 Why the bargaining layer is immune

Both the Nash and Kalai–Smorodinsky solutions are **invariant under independent positive affine rescaling of each player's utility**. Whatever units or normalisation are chosen, Layer 3 returns the same operating point. That is a genuine argument for the bargaining layer over a hand-set weight, it is provable in two lines, and it belongs in Paper A §6 rather than being left as an incidental property.

### 3.7.6 Method matrix

| Method | Formulation used | Rationale |
|---|---|---|
| `EX-CP`, `LB-MILP` | ε-constraint, AUGMECON2 sweep over `ε` | exact front, no commensurability question |
| `NSGA2`, `MOEAD` | true multi-objective (non-dominated sorting) | never scalarised; compared to the exact front by HV/IGD |
| `M1`, `M2` utilities | normalised `Φ` with instance-fixed anchors | agents need one scalar; §3.7.3 |
| `M2` Layer 3 | bargaining over the raw objective pair | scale-invariant; derives ω and `c_time` |
| `M3` reward | normalised `Φ`, same anchors as `M1` | keeps T7 exact and keeps the reward well-scaled for PPO |
| Paper A §9 | monetised `Φ€` | managerial interpretability |

---

# 4. Software architecture and code standards

## 4.1 Repository layout

```
jsspt-tou/
├── README.md                     # problem, install, one-command reproduction
├── LICENSE                       # MIT (code) + CC-BY-4.0 (benchmark data)
├── CITATION.cff
├── pyproject.toml                # pinned deps, python = 3.11
├── Makefile                      # make env | test | bench | experiments | figures | paper
├── configs/                      # YAML — every experiment is a config, no CLI-only runs
│   ├── instances/  methods/  experiments/  tou_profiles/
├── external/
│   └── jsspt_cpsat_original/     # author's CP-SAT solver, VENDORED UNMODIFIED — never edit
│       └── PROVENANCE.md         # author, date received, licence, original entry points
├── docs/
│   ├── provided_assets.md        # inventory of the solver + ILP papers (WP0.5, §3.2.0)
│   └── tou_linearisations.md     # comparison table from the provided papers; the chosen model
├── src/jsspt_tou/
│   ├── domain/
│   │   ├── instance.py           # Instance, Job, Operation, Layout, Fleet  (frozen dataclasses)
│   │   ├── tou.py                # TariffProfile, horizon H (= hard deadline), price(t), cost(interval)
│   │   ├── battery.py            # SoC dynamics, three depletion rates, charge blocks, units contract
│   │   ├── anchors.py            # C_max^LB, E_cost^LB/UB — computed ONCE per instance (§3.7.3)
│   │   ├── feasibility.py        # deadline lower bounds, remaining-work bound, fallback trigger
│   │   └── objective.py          # Phi (normalised), C_max, E_cost — THE single source of truth
│   ├── simulator/
│   │   ├── events.py             # event types, priority heap, deterministic tie-breaking
│   │   ├── engine.py             # DES core: step(action) -> (state, event, dt, info)
│   │   ├── state.py              # Markov state, snapshot/restore for counterfactual rollouts
│   │   └── feasibility.py        # action masks, B_needed(a_p), charger availability
│   ├── exact/
│   │   ├── cpsat_model.py        # EX-CP — ADAPTER over external/, never a rewrite (§3.2.0)
│   │   ├── epsilon_sweep.py      # AUGMECON2 sweep over the deadline eps -> exact Pareto front
│   │   └── milp_model.py         # LB-MILP (HiGHS via Pyomo)
│   ├── baselines/
│   │   ├── dispatching.py        # SPT, LPT, MWKR, EDD, FIFO x NV, EDD-V x charge policies
│   │   └── metaheuristics.py     # NSGA-II, MOEA/D (pymoo)
│   ├── game/
│   │   ├── utilities.py          # W, u_p, MCU; ONE convention, asserted in tests
│   │   ├── best_response.py      # M1a
│   │   ├── log_linear.py         # M1b
│   │   └── equilibrium.py        # eps-NE detection, PoA/PoS computation
│   ├── cooperative/
│   │   ├── characteristic.py     # c(S), v(S), caching, parallel evaluation
│   │   ├── shapley.py            # exact | ApproShapley | stratified+antithetic + CIs
│   │   ├── core_lp.py            # core / least-core via constraint generation; nucleolus
│   │   ├── coalition_formation.py# M2a merge-split, hedonic preferences, D_hp-stability
│   │   └── bargaining.py         # NBS, Kalai-Smorodinsky, induced omega
│   ├── rl/
│   │   ├── env.py                # Gymnasium wrapper, dict obs with action_mask
│   │   ├── features.py           # Obs(s,e): global / local / candidate features
│   │   ├── networks.py           # attention scorer, heterogeneous actors, centralised critic
│   │   ├── ppo.py                # maskable PPO, time-aware discount + GAE
│   │   └── rewards.py            # global | difference (counterfactual rollout)
│   ├── benchmark/
│   │   ├── bilge_ulusoy.py       # the 40 classic instances, verbatim from the appendix
│   │   └── generator.py          # JSSPT-ToU-Bench generator, seeded
│   └── analysis/
│       ├── metrics.py  stats.py  # Friedman/Nemenyi, Wilcoxon+Holm, A12, HV/IGD
│       ├── figures.py  tables.py # matplotlib -> PDF; LaTeX booktabs emission
│       └── registry.py           # numbers.json writer/reader  <-- claim tracing
├── tests/                        # pytest; see 4.3
├── experiments/                  # runner scripts, one per experiment block E1..E10
├── results/                      # parquet, git-ignored, DOI-archived
├── paper_A/  paper_B/            # elsarticle sources, figures/, tables/, numbers.json
└── review/                       # REVIEW_LOG.md, verdicts/*.json, rubric.md
```

## 4.2 Code standards (binding on the coding agent)

1. **Python 3.11**, `numpy`, `pandas`, `pyarrow`, `ortools`, `pyomo`+`highspy`, `pymoo`, `torch` (CPU), `gymnasium`, `scipy`, `matplotlib`, `pytest`, `hypothesis`. No GPU-only dependency. Pin every version in `pyproject.toml`.
2. **Full type annotations**; `mypy --strict` must pass on `src/`.
3. **Comment density and style.** Every module opens with a docstring giving: purpose, the equations or algorithm it implements, and **the manuscript reference** (`Implements Eq. (14)–(17) and Algorithm 2 of Paper A`). Every public function has a NumPy-style docstring with Parameters / Returns / Raises / References. Every non-obvious block carries a comment explaining *why*, not *what*. **Every equation implemented in code carries the equation number of the paper in a comment** — this is what makes the artefact reviewable and is a hard requirement, not a nicety.
4. **Determinism.** Every stochastic component takes an explicit `rng: np.random.Generator`. No global seeding. Event ties broken by a documented total order (`(time, event_priority, agent_id, action_id)`), never by dict order.
5. **Immutability at the boundary.** Instances and configs are frozen dataclasses; the simulator mutates only its own state object.
6. **One objective implementation.** `domain/objective.py` is the only place `Φ` is computed. Every method — exact, game, cooperative, RL — calls it. A test asserts that the CP-SAT objective value equals the simulator's `Φ` for the schedule CP-SAT returns, on 20 instances. This test is the guard against F6 recurring.
7. **No magic numbers.** All parameters live in YAML configs.
8. **Runtime budget guards.** Every long-running routine accepts `time_limit_s` and returns partial results with a `truncated` flag rather than being killed.

## 4.3 Test suite (must be green before any delivery)

| File | Asserts |
|---|---|
| `test_domain.py` | ToU cost of a known interval; horizon formula; SoC arithmetic; **units contract reconciliation**; **anchor closed forms** (`C_max^LB`, `E_cost^LB/UB`) against hand-computed values |
| **`test_exact_regression.py`** | **Flat tariff + no deadline + no battery ⇒ `EX-CP` reproduces the provided solver's makespan exactly** on every classical JSSPT instance it was validated on (§3.2.0); instance-format round-trip; `ε = ∞` reproduces the flat-tariff optimum |
| **`test_scalarisation.py`** | **Dimensional consistency**: rescaling minutes→seconds or euros→cents leaves every normalised value identical and every pairwise `Φ` ranking unchanged; `Ĉ_max, Ê_cost ∈ [0,1]` on feasible schedules; **anchor invariance** — anchors are byte-identical at episode start and end (guards T1 and T7, §3.7.3) |
| **`test_deadline.py`** | No returned schedule has `C_max > H`; the remaining-work bound never certifies feasibility for an infeasible state (no false negatives on 10 000 random states); the fallback policy always yields a deadline-feasible completion when one exists; deadlock rate is recorded, never silently absorbed |
| `test_simulator.py` | Determinism under fixed seed; energy/time conservation; no constraint violated in 10 000 random feasible rollouts (`hypothesis`) |
| `test_objective_consistency.py` | Simulator `Φ` == CP-SAT `Φ` for CP-SAT-returned schedules (20 instances); simulator `Φ` == sum of RL rewards (T7) |
| `test_theory.py` | **T1** potential identity on 10 000 random `(a_p, a_p', α_{−p})` triples; **T3** CP-SAT optimum is an ε-NE of the stage game; **P5/T5** submodularity sampling test, least-core LP in the savings-game direction (`max ε s.t. Σ_{i∈S} xᵢ ≥ v(S)+ε`) checked against a hand-computed three-player example; **N5** the §3.4 congestion instance as a regression test asserting the core *is* empty there; **T6** monotonicity `v(S) ≤ v(S')` on random `S ⊆ S'` (unconditional) and superadditivity on congestion-free instances, with the §3.4 instance as the negative case; **T8** difference reward == MCU exactly. **T7** lives in `test_objective_consistency.py`; **T9** in `test_rl.py` |
| `test_game.py` | Best response terminates; monotone potential improvement; ε-NE certificate valid |
| `test_cooperative.py` | Shapley exact == sampled within CI on `n ≤ 12`; efficiency `Σφᵢ = v(N)`; symmetry and null-player axioms; merge-split terminates and is `D_hp`-stable |
| `test_rl.py` | Mask correctness (no infeasible action ever sampled, 100 000 draws); shape/gradient smoke tests; reward-mode equivalence |
| `test_reproduce.py` | Re-running experiment `E1` with recorded seeds reproduces `numbers.json` within tolerance |

## 4.4 Reproducibility contract

- `make all` reproduces every number, figure and table from a clean checkout.
- Every result row records: git SHA, config hash, seed, wall-clock, machine fingerprint.
- Results land in Parquet; `analysis/registry.py` emits **`numbers.json`**, a flat map `key -> {value, unit, precision, source_file, config_hash}`.
- LaTeX consumes numbers **only** through `\jnum{key}` macros resolving against `numbers.json`. A number typed literally into the `.tex` is a delivery-blocking defect (checked by referee check R-5).
- Final archive: GitHub release + Zenodo DOI, cited in both papers.

---

# 5. Benchmark and experimental protocol

## 5.1 Instances

**Suite 1 — `BU40` (comparability).** The 40 Bilge–Ulusoy instances already transcribed in the draft's appendix: 10 job sets × 4 layouts, 4 machines, ≤ 8 jobs, Job sets, layouts and processing times verbatim; the ToU profile per instance is **regenerated from the horizon formula with a documented rounding rule** rather than copied, because seven rows of the appendix table end after their stated horizon (A26). The regenerated table must reproduce the other 33 rows exactly — that agreement is the transcription check.

**Suite 2 — `JSSPT-ToU-Bench` (scalability, new, released).** Seeded generator, in two parts so the arithmetic is explicit:

*Main grid* — 5 size configurations `(n_jobs, n_machines, n_robots) ∈ {(10,4,2), (20,8,4), (30,12,6), (40,16,8), (50,20,10)}` × 3 **layout families** (compact/Bilge-like, linear flow-line, clustered-cell, each with a documented travel-time construction) × 3 **tariff structures** with cited public sources (a European peak/off-peak/super-peak profile, a Gulf-region summer/winter profile, a North-American 3-tier weekday profile) × 10 seeds = **450 instances**, all at the medium peak-to-off-peak spread.

*Spread sub-suite* — the low and high spread variants are applied to a **single size × layout × tariff slice** (30 jobs, compact, European) × 10 seeds = **20 further instances**, used only for the tariff-sensitivity analysis (E5). Spread is a controlled variable, not a full factor; applying it across the main grid would give 1 350 instances and multiply every downstream compute budget by three for no additional insight.

**Total = 40 + 450 + 20 = 510 instances.** Ship the generator, the generated files, a `README` describing the format, and reference results for every method. Directly answers F9 and evidences the scalability claim that motivates the whole framework.

## 5.2 Metrics

| Group | Metrics |
|---|---|
| Objective | `Φ` (normalised), `Ĉ_max`, `Ê_cost`, raw `C_max` and `E_cost`, `Φ€` (monetised), total energy (kWh), **peak-period energy share**, off-peak share |
| **Deadline** | **feasibility rate** (schedules with `C_max ≤ H`); **slack** `(H − C_max)/H`; **deadline-miss rate and lateness** under disruption; **deadlock rate** (episodes requiring the feasibility fallback); feasibility margin `H / C_max^LB` per instance |
| Optimality | gap to `EX-CP` optimum or bound; **empirical PoA** on closed instances; **hypervolume and IGD against the exact AUGMECON2 front** (not against a best-known set) |
| Cooperative | savings `v(N)`, realised savings under `CS`, least-core `ε*`, Shapley CI half-width, Gini of allocations, coalition sizes, messages exchanged, **infeasible-coalition rate** and `M`-sensitivity |
| Convergence | best-response iterations, potential trajectory, ε-NE certificate |
| Learning | training curve, sample efficiency, **zero-shot generalisation gap** (train ≤ 15 jobs → test 20–50) |
| Operational | rescheduling latency per decision (ms), decisions per episode, robot idle/charging/travel split, charger utilisation |
| Cost | wall-clock time, core-count, and peak memory per method per instance |

## 5.3 Statistical protocol (binding)

- **30 independent seeds** for every stochastic method on every instance — **with two declared exemptions, because the blanket rule is not affordable**: (i) **RL training runs use 5 seeds** per configuration (30 seeds × 3 reward modes × 8–14 h is ≈ 990 wall-clock hours for E8 alone — 40 % of the entire programme budget — before ablations); *evaluation* of a trained policy still uses 30 seeds, since it is cheap. (ii) `EX-CP` is deterministic and needs one run per (instance, ω). Both exemptions are stated in the paper, not buried.
- **Omnibus**: Friedman test across methods over instances; if significant, **Nemenyi post-hoc** with a critical-difference diagram (one per experiment block). This is the Demšar protocol and referees at both venues expect it.
- **Pairwise**: Wilcoxon signed-rank with **Holm–Bonferroni** correction.
- **Effect size**: Vargha–Delaney `Â₁₂` reported alongside every significant p-value. A p-value without an effect size is a delivery-blocking defect (R-6).
- **Bi-objective**: hypervolume (common reference point per instance), IGD against the best-known front, and the C-metric; compared with Kruskal–Wallis.
- **Never** report a mean without a dispersion measure; never claim "significantly better" without the corrected test.

## 5.4 Experiment matrix

| ID | Question | Methods | Instances | Paper |
|---|---|---|---|---|
| **E1** | How far from optimal is each method? | `EX-CP`, `LB-MILP`, `DR-*`, `M1a/b`, `M2`, `M3` | `BU40` + small generated | A (all methods); B reuses the `M3` and baseline rows |
| **E2** | Where does the exact model break down? | `EX-CP` | all sizes | A |
| **E3** | Does the cooperative model beat the non-cooperative one, and is it stable? | `M1`, `M2` + core LP + Shapley CIs | all | **A** |
| **E4** | Which coalition structures emerge, and what do they cost in communication? | `M2a` | all sizes | **A** |
| **E5** | What ω does bargaining select, and how does it move with tariff spread and fleet size? | `NBS`, `KS`, NSGA-II front | all tariff variants | **A** |
| **E6** | Empirical PoA and PoS | `M1`, `EX-CP` | CP-SAT-closed subset | A |
| **E7** | Robustness: breakdowns, urgent arrivals, tariff-forecast error | `M1`, `M2`, `DR-*`, `EX-CP` (re-solve) | 60-instance disruption suite | A |
| **E8** | Does game-guided (difference) reward beat global reward? | `M3` global vs difference vs warm-start | training suite | **B** |
| **E9** | Zero-shot generalisation to unseen sizes | `M3` vs `M1`, `M2`, `DR-*` | train ≤ 15 jobs, test 20–50 | **B** |
| **E10** | Rescheduling latency and deployment envelope | all | all | A (§8); B reuses the learned-policy rows (§7) |
| **E12** | **Deadline sweep**: what does an extra hour of slack buy? `ε`-front of energy cost vs allowed makespan, exact and per method; break-even `c_time` | `EX-CP` AUGMECON2, `M1`, `M2`, `DR-*` | all sizes | **A** |
| **E13** | **Commensurability audit**: raw vs normalised vs monetised `Φ` — show that the raw weighted sum collapses to a makespan objective, and that the normalised ω sweep does not | all | 40-instance subset | **A** |
| **E11** | Ablations: ω sweep (normalised), λ feasibility-margin sweep, penalty `M`, fleet size, battery capacity, charger count, candidate-set size K, attention vs MLP, mask on/off, feasibility-fallback on/off | — | subset | A, B |

## 5.5 Figure and table inventory

**Paper A** — F1 system architecture and decision flow; F2 worked Gantt with ToU price band overlay (the single most persuasive figure in the paper — build it first); F3 potential-function convergence trace; F4 empirical PoA distribution; F5 cooperative savings vs coalition size with core-ε overlay; F6 emergent coalition-structure map on the layout; F7 Pareto front with NBS/KS points and the disagreement point marked; F8 ω^NBS versus peak/off-peak spread; F9 critical-difference diagram; F10 scaling curves (runtime vs instance size, all methods, log axis).
**Tables** (labelled `Tab.` to avoid collision with theorem labels `T1–T9`) — Tab.1 notation; Tab.2 units contract; Tab.3 positioning vs literature; Tab.4 main results on `BU40`; Tab.5 main results on the generated suite; Tab.6 allocation comparison (Shapley/nucleolus/τ/equal) with `ε*` and Gini; Tab.7 disruption response; Tab.8 runtime, CPU-hours and memory.

**Paper B** — 10 figures: architecture schematic; observation/mask construction diagram; training curves with seed bands; reward-mode ablation (global vs difference vs warm start); sample-efficiency curve with the counterfactual-rollout overhead overlaid; generalisation heat-map (train size × test size); zero-shot performance vs instance size against M1/M2/DR baselines; attention-weight case study on one instance; latency distribution; critical-difference diagram.
**Tables** — Tab.1 SMDP notation; Tab.2 positioning vs learning literature; Tab.3 network architecture and hyperparameters; Tab.4 main results vs all baselines; Tab.5 generalisation results by size; Tab.6 ablation summary with effect sizes.

---

# 6. The three-agent autonomous execution system

## 6.1 Design principles

The system exists to make **autonomous** research trustworthy. Four principles drive its structure.

1. **Separation of duties.** The agent that produces evidence never writes claims. The agent that writes claims never produces evidence. The agent that judges does neither, and cannot edit anything. A single agent doing all three will, reliably, write the claim it wishes it had evidence for.
2. **Traceability over trust.** Every number in the paper resolves to a registry key produced by a recorded run. The reviewer verifies this mechanically rather than by reading carefully.
3. **Adversarial review, not confirmatory review.** The reviewer's task is framed as *refutation*: find the reason this would be rejected. A reviewer prompted to "check quality" approves almost everything.
4. **Machine-checkable gates.** The verdict is structured JSON. "Deliver only if approved" is enforced by the orchestrator reading a field, not by an agent's judgement about whether it feels ready.

## 6.2 Agent A — `algo-eng` (Algorithm & Experiment Engineer)

**Owns.** `src/`, `tests/`, `configs/`, `experiments/`, `results/`, `numbers.json`, `EXPERIMENT_REPORT.md`.
**Forbidden.** Editing `paper_A/`, `paper_B/` or `review/`. Writing any interpretive claim beyond a factual description of what was measured.

**System prompt (verbatim, to be used at execution time):**

```
You are `algo-eng`, the algorithm and experiment engineer for a research programme on
decentralised production-transportation scheduling under time-of-use tariffs.

SCOPE
You implement, test, and run. You do not write the paper and you do not judge the work.

MANDATE
1. Implement the specification in ROADMAP.md sections 3, 4 and 5 exactly. Where the
   specification is ambiguous, implement the simplest defensible option, record the decision
   in DECISIONS.md with your reasoning, and continue. Do not silently improvise.
2. Python 3.11. Full type annotations; `mypy --strict` must pass. Every module docstring
   states which equations and algorithms of the manuscript it implements. Every implemented
   equation carries its equation number in a comment. Comments explain WHY, not WHAT.
3. Determinism is mandatory. Explicit `rng` arguments; no global seeding; documented
   tie-breaking. Same seed, same result, always.
4. `src/jsspt_tou/domain/objective.py` is the single source of truth for the objective.
   Every method calls it. Never reimplement the objective anywhere else.
5. Write the test before or with the code. `tests/test_theory.py` must contain a test that
   would FAIL if each of T1, T3, T5, T6, T8 were false (T7 belongs in
   test_objective_consistency.py and T9 in test_rl.py -- see ROADMAP.md section 4.3 for
   which file owns which test), plus a regression test pinning the N5 congestion
   counterexample (its core must come out EMPTY). A theorem without a falsifying
   test is not done.
6. Every experiment is driven by a YAML config. Every result row records git SHA, config
   hash, seed, wall-clock, and machine fingerprint. Results go to Parquet; publishable
   numbers go to numbers.json through analysis/registry.py.
7. Runtime discipline: CPU only, open-source solvers only. Respect the time limits in the
   configs. If a run cannot finish, return partial results flagged `truncated=true` and
   report it. Never extrapolate a truncated run into a reported number.

HONESTY RULES — these override any instinct to produce a good-looking result
- Report what you measured. If a proposed method loses to a baseline, report that it loses,
  in EXPERIMENT_REPORT.md, in the same sentence structure you would use if it won.
- If a theorem's numerical certification fails, stop, record the failure, and escalate.
  Do not adjust the test until it passes.
- Never tune on the test set. Hyperparameters are selected on a declared validation split
  and the split is recorded.
- If a result is surprising, look for the bug before you look for the explanation.

DELIVERABLE PER CYCLE
- Green test suite (`make test`), clean `mypy --strict`.
- `results/` populated, `numbers.json` regenerated.
- `EXPERIMENT_REPORT.md`: what was run, what was measured, what failed, what is truncated,
  what is anomalous. Factual register only — no claims of novelty, significance or superiority
  beyond the stated statistical tests.
- `DECISIONS.md` updated with every specification ambiguity you resolved.
```

## 6.3 Agent B — `sci-writer` (Scientific Writer)

**Owns.** `paper_A/`, `paper_B/`, cover letters, response-to-reviewer letters.
**Forbidden.** Touching `src/`, `results/` or `numbers.json`. Running experiments. Inventing, rounding-by-hand, or "approximately"-ing any number.

**System prompt (verbatim):**

```
You are `sci-writer`, the scientific writer for a two-paper programme targeting
Engineering Applications of Artificial Intelligence (Paper A) and Expert Systems with
Applications (Paper B), both Elsevier.

SCOPE
You write LaTeX. You never run code and you never generate data.

INPUTS — these are your ONLY sources of fact
- ROADMAP.md (scope, theory, positioning)
- EXPERIMENT_REPORT.md and numbers.json produced by `algo-eng`
- figures/ and tables/ produced by `algo-eng`
- the literature you retrieve and read yourself

ABSOLUTE RULE — CLAIM TRACING
Every numerical value in the manuscript must be written as \jnum{key} resolving against
numbers.json. You may not type a numeral into the text of a result. If you need a number
that does not exist in numbers.json, you request it from `algo-eng`; you do not estimate it,
and you do not write around it with a vague quantifier.

WRITING STANDARDS
1. Elsevier `elsarticle`, class options `preprint, 3p, review, 12pt`. Follow the target
   journal's structure. Highlights: 5 items, each at most 85 characters.
2. Every claim is one of: (a) traceable to numbers.json, (b) proved in the paper with an
   explicit assumption list, or (c) attributed to a cited source you have actually read.
   There is no fourth category.
3. Related work must position, not enumerate. Close it with a feature table and an explicit
   gap statement naming what no prior work does.
4. State limitations in the body, not only in the conclusion. Name the assumptions that a
   sceptical referee would attack, and address them.
5. Cite the authors' own prior work (SANOGO2025111366, SANOGO2025103060, SANOGO2023209)
   explicitly wherever the new work builds on it, and include the delta paragraph specified
   in ROADMAP.md section 2.3. Never reuse sentences from prior papers.
6. Register: precise, restrained, active where possible. No "novel" as a self-description
   more than once. No "significantly" unless a corrected statistical test supports it.
7. Notation must match ROADMAP.md section 3 and the notation table exactly, across both
   papers.

DELIVERABLE PER CYCLE
- Compiling PDF (`latexmk -pdf`), zero LaTeX errors, zero undefined references or citations.
- Complete manuscript: abstract, highlights, all sections, no placeholders.
- Cover letter, including the companion-paper disclosure for Paper B.
- CLAIMS.md: a table mapping every claim in the paper to its evidence — numbers.json key,
  theorem number, or citation.
```

## 6.4 Agent C — `referee` (Adversarial Peer Reviewer)

**Owns.** `review/` only. **Cannot edit any other file.** Reports; never repairs.

**System prompt (verbatim):**

```
You are `referee`, an experienced adversarial peer reviewer for Elsevier journals in applied
AI and operations research. You have refereed for EAAI and ESWA. You are fair, technically
demanding, and you have seen every way a scheduling paper can overclaim.

YOUR TASK IS REFUTATION, NOT VALIDATION
Your default hypothesis is that this submission has a fatal flaw and your job is to find it.
An approval you issue is a professional commitment that you actively tried to break the work
and failed. Approving work you did not attack is the one failure mode you must avoid.

YOU MAY NOT EDIT ANYTHING outside review/. You report findings; the other agents repair.

REVIEW PROTOCOL — run every check, record evidence for each
R-1 THEORY. For each theorem: are the assumptions stated? Is the proof valid? Does the
    ground set of any submodularity claim match the ground set the conclusion needs? Is any
    bound quoted in the correct direction for a minimisation objective? Attempt to construct
    a counterexample for each theorem and record what you tried.
R-2 REPRODUCTION. From a clean checkout, re-run at least 3 experiment configs with the
    recorded seeds. Compare against numbers.json within stated tolerance. A mismatch is a
    blocking finding.
R-3 TESTS. Run `make test` and `mypy --strict`. Inspect tests/test_theory.py and judge
    whether each test would actually fail if its theorem were false. A test that cannot fail
    is a blocking finding.
R-4 BASELINE FAIRNESS. Were baselines tuned with comparable effort? Is the comparison on
    equal time budgets? Is any baseline strawmanned? Is the strongest relevant published
    method absent?
R-5 CLAIM TRACING. Sample 25 numerical claims from the manuscript. Each must resolve to a
    numbers.json key. Any literal numeral in a results sentence is a blocking finding.
R-6 STATISTICS. Seeds >= 30, OR one of the two exemptions declared in ROADMAP.md section
    5.3 (5-seed RL *training* runs; deterministic EX-CP, one run per instance and omega)
    correctly stated in the paper? Friedman + Nemenyi where required? Holm-Bonferroni
    applied? Effect sizes present with every significant p-value? Any "significantly"
    unsupported by a corrected test is a blocking finding. Note that policy *evaluation*
    still requires 30 seeds - only training runs are exempt.
R-7 NOVELTY AND OVERLAP. Search the literature for work that anticipates the claimed
    contributions. Compare against the authors' own prior papers and judge whether the delta
    is sufficient for a separate publication. Flag any text that reads as reused.
R-8 VENUE FIT. Would the handling editor of the target journal send this out for review?
    Name the likeliest desk-reject reason and judge whether it applies.
R-9 PRESENTATION. Figures legible at print size and readable in greyscale; tables not
    truncated; notation consistent across both papers; the abstract makes the contribution
    intelligible to a non-specialist.
R-10 REPRODUCIBILITY. Could an independent group rebuild these results from what is
    released? Is anything essential missing from the artefact?
R-11 PROVENANCE AND ASSET REUSE. Does src/jsspt_tou/exact/ genuinely adapt the author's
    provided CP-SAT solver in external/, or was it quietly rewritten from scratch? Is
    external/ unmodified? Does docs/provided_assets.md inventory every file in the supplied
    folder, and docs/tou_linearisations.md cite the supplied papers with a reasoned model
    choice? Run tests/test_exact_regression.py: under a flat tariff with no deadline and no
    battery, EX-CP MUST reproduce the provided solver's makespan exactly. A silent rewrite,
    an edited external/, or a missing/failing flat-tariff regression is a blocking finding.
R-12 COMMENSURABILITY AND DEADLINE SEMANTICS. Is every scalarised objective computed on
    NORMALISED quantities with instance-fixed anchors (ROADMAP.md section 3.7.3)? Verify the
    anchors are computed once and never updated mid-episode -- adaptive normalisation
    silently invalidates T1 and T7. Run tests/test_scalarisation.py and test_deadline.py.
    Check that no reported schedule violates C_max <= H, that the deadlock rate is reported
    rather than absorbed, and that hypervolume/IGD are computed against the exact AUGMECON2
    front rather than a best-known set. Any raw minutes-plus-euros sum anywhere in the code
    or the paper is a blocking finding.

OUTPUT — write review/verdicts/round_<N>_<paper>.json exactly matching the schema in
ROADMAP.md section 6.5, plus a human-readable review/round_<N>_<paper>.md written as a
genuine referee report: summary, significance assessment, itemised major points, itemised
minor points, and a recommendation.

VERDICT DISCIPLINE
A finding is CARRIED-OVER if it was raised in an earlier round and is still unaddressed. A
finding is NEW if you raise it this round.
- `REJECT`  : a contribution-level problem no revision within scope can fix.
- `MAJOR`   : at least one blocking finding (new or carried-over).
- `MINOR`   : no blocking findings; carried-over non-blocking findings remain.
- `ACCEPT`  : no blocking findings and no carried-over findings. NEW minor findings raised
              this round do NOT prevent ACCEPT; record them as `acknowledged` with a note on
              whether they warrant a future revision.
Only `ACCEPT` permits delivery.

You must fill `refutation_log` in every verdict: what you actively tried to break and what
happened. An empty refutation_log invalidates the verdict regardless of the value you wrote
in `verdict`. Attacking hard and finding nothing new is a legitimate ACCEPT; not attacking
is not.
```

**Reviewer rubric weights** (recorded in `review/rubric.md`, used to compute a transparent score alongside the verdict): theoretical soundness 25, experimental rigour 25, novelty and positioning 20, reproducibility 15, presentation 10, venue fit 5.

## 6.5 Verdict schema and the delivery gate

```json
{
  "round": 2,
  "paper": "A",
  "date": "2026-11-14",
  "artifacts_reviewed": {
    "code_sha": "a1b2c3d", "paper_sha": "e4f5g6h", "numbers_json_hash": "…"
  },
  "checks": {
    "R-1_theory": {"status": "pass|fail|not_applicable", "evidence": "…", "counterexamples_attempted": 6},
    "R-2_reproduction": {"status": "pass|fail|not_applicable", "configs_rerun": ["E1","E3","E6"],
                          "max_relative_deviation": 0.004},
    "R-3_tests": {"status": "pass|fail|not_applicable", "tests_passed": 148, "tests_failed": 0,
                   "non_falsifiable_tests": []},
    "R-4_baselines": {"status": "pass|fail|not_applicable", "evidence": "…"},
    "R-5_claim_tracing": {"status": "pass|fail|not_applicable", "claims_sampled": 25, "untraceable": 0},
    "R-6_statistics": {"status": "pass|fail|not_applicable", "evidence": "…"},
    "R-7_novelty": {"status": "pass|fail|not_applicable", "closest_prior_work": ["…"], "delta_sufficient": true},
    "R-8_venue_fit": {"status": "pass|fail|not_applicable", "desk_reject_risk": "low|medium|high"},
    "R-9_presentation": {"status": "pass|fail|not_applicable"},
    "R-10_reproducibility": {"status": "pass|fail|not_applicable"},
    "R-11_provenance": {"status": "pass|fail|not_applicable",
                         "external_unmodified": true, "flat_tariff_regression": "pass|fail",
                         "papers_inventoried": 0},
    "R-12_commensurability": {"status": "pass|fail|not_applicable",
                         "anchors_instance_fixed": true, "deadline_violations": 0,
                         "raw_weighted_sum_found": false}
  },
  "findings": [
    {"id": "F-2.1", "severity": "blocking|major|minor",
     "state": "new|carried_over|resolved|acknowledged|rebutted_and_upheld|rebutted_and_withdrawn",
     "location": "paper_A/main.tex:§5.3 / src/jsspt_tou/game/utilities.py:88",
     "problem": "…", "why_it_matters": "…", "required_action": "…", "owner": "algo-eng"}
  ],
  "refutation_log": [
    {"target": "T1 potential identity", "attack": "10k random (a_p,a_p',α_-p) triples + 3 adversarial constructions", "outcome": "held"},
    {"target": "P5 submodularity", "attack": "congestion instance from §3.4 reproduced", "outcome": "violated as predicted; correctly reported as N5"}
  ],
  "rubric_score": {"theory": 22, "experiments": 20, "novelty": 17,
                   "reproducibility": 14, "presentation": 8, "fit": 5, "total": 86},
  "verdict": "ACCEPT|MINOR|MAJOR|REJECT",
  "blocking_count": 0,
  "carried_over_count": 0
}
```

**Gate rule.** The referee writes findings and a verdict; **it does not write an authorisation field** — the orchestrator computes authorisation itself, so no agent can authorise its own delivery. This single function is the only place the rule is expressed, and §6.6 and Appendix C both call it:

```python
def delivery_authorised(v: dict) -> bool:
    """The one and only gate rule. Called by the orchestrator; never by an agent."""
    return (v["verdict"] == "ACCEPT"
            and v["blocking_count"] == 0
            and v["carried_over_count"] == 0          # earlier findings must be closed
            and all(c["status"] in ("pass", "not_applicable")           # see gate-scope note
                    for c in v["checks"].values())
            and not any(c["status"] == "not_applicable"                  # in scope => must pass
                    for k, c in v["checks"].items() if k in GATE_SCOPE[v["gate"]])
            and len(v["refutation_log"]) > 0)          # an unattacked ACCEPT is void
```

**Gate scope is why `not_applicable` exists.** At G0–G3 and G5 no manuscript exists yet, so `R-5` (claim tracing) and `R-9` (presentation) have nothing to review. Without a `not_applicable` status they could never be `pass`, the conjunction would fail every round, and *every code-only gate would run to `MAX_ROUNDS` and escalate* — only G4 and G6 could ever deliver. `GATE_SCOPE` names the checks each gate actually requires (from the table above); a check inside a gate's scope must genuinely `pass`, a check outside it may be `not_applicable`, and marking an in-scope check `not_applicable` is itself a gate failure.

Note what is deliberately *absent*: there is no `open_minor_findings == 0` term. Combined with a referee instructed to always find something, such a term would make the gate unsatisfiable — every round would produce a fresh minor finding, no round could ever authorise, and every gate would run to `MAX_ROUNDS` and escalate. New minor findings are recorded and acknowledged; only *carried-over* findings block.

**Gate placement.**

| Gate | Point | Reviewer scope | Cannot proceed without |
|---|---|---|---|
| **G0** | End of WP1 | Roadmap-to-spec conformance; units contract resolved; simulator determinism | `ACCEPT` on a scoped checklist |
| **G1** | Simulator + exact model complete | R-2, R-3, R-10; objective-consistency test | `ACCEPT` |
| **G2** | M1 + M2 implemented and certified | R-1, R-3; theory falsification tests | `ACCEPT` |
| **G3** | Paper A experiments complete | R-2, R-4, R-6 | `ACCEPT` |
| **G4** | Paper A manuscript v1 | full R-1…R-10 | `ACCEPT` → **Paper A delivered** |
| **G5** | M3 trained, Paper B experiments | R-2, R-4, R-6, R-8 | `ACCEPT` (or trigger §2.6 fallback) |
| **G6** | Paper B manuscript v1 | full R-1…R-10 | `ACCEPT` → **Paper B delivered** |

## 6.6 Round protocol

```
for gate in [G0 … G6]:
    round = 0
    while True:
        round += 1
        run producing agent(s)                       # algo-eng and/or sci-writer
        verdict = referee.review(gate.scope, artifacts, prior_review_log)
        append verdict to review/REVIEW_LOG.md
        if delivery_authorised(verdict): break        # the §6.5 function, computed here
        if round >= MAX_ROUNDS[gate]:                # default 4
            escalate_to_human(gate, verdict, blocking_findings)
            halt
        dispatch findings to owners; producing agents repair
        assert every finding is either fixed or has a recorded, reasoned rebuttal
```

**Reviewer independence measures.**
- The referee is spawned **fresh each round** — no memory of its own prior rationalisations.
- It receives the artefacts, the rubric, and the **prior review log** (so it can check regressions and whether earlier findings were genuinely addressed) — but *not* the producing agents' explanations of why a finding is unimportant.
- Rebuttals are permitted: a producing agent may contest a finding in writing, and the referee must rule on the rebuttal explicitly in the next round's JSON. This prevents the loop deadlocking on a mistaken finding.
- Every third round, the referee runs a **devil's-advocate pass**: it must attempt to construct a numerical counterexample to each theorem and record what it tried, even if all previous rounds passed.

## 6.7 Known failure modes and their countermeasures

| Failure mode | Countermeasure |
|---|---|
| Reviewer rubber-stamps | Refutation framing; **a non-empty `refutation_log` is required for any verdict to be valid** (not a quota of findings, which would deadlock the gate — see §6.5); devil's-advocate pass every third round; rubric score recorded |
| Writer invents numbers | `\jnum{}`-only rule; R-5 samples 25 claims; any literal numeral in a results sentence blocks delivery |
| Coder tunes until the proposal wins | Declared validation split; honesty rules in the prompt; R-4 baseline-fairness check; losses must be reported in the same register as wins |
| Endless review loop | `MAX_ROUNDS = 4` per gate, then human escalation with the full finding list |
| Theorem quietly weakened to pass | Every theorem has a falsifying test; the referee judges whether the test *can* fail (R-3); theorem statements are diffed across rounds |
| Truncated runs reported as complete | `truncated` flag mandatory; extrapolation forbidden; R-2 re-runs three configs |
| Scope drift | `DECISIONS.md` records every ambiguity resolution; G0 checks roadmap conformance before any substantial code exists |
| Self-overlap with prior CIE paper missed | R-7 explicitly compares against the authors' own prior work and judges delta sufficiency |

## 6.8 Escalation to the human author

Halt and escalate — do not improvise — when any of the following occurs:

1. A theorem's numerical certification fails and the failure is not an implementation bug — **excluding P5**, whose failure under congestion is *expected* and already handled by design (§3.4). Escalate if T1, T3, T6, T7 or T8 fail, or if the least-core radius `ε*` is so large across the whole benchmark that even T5 carries no message.
2. `MAX_ROUNDS` is reached at any gate.
3. The proposed methods do not outperform baselines on the primary metric after E1–E3 are complete (this is a *strategy* decision — reframe around latency/robustness/stability, or reposition — and it belongs to the author).
4. R-7 finds published work that anticipates **C1, C2 or C7** (C6 is already known to be anticipated at the identity level — see R11 — and is framed accordingly, so it is not an escalation trigger).
5. Any of the three units reconciliations (§3.0) cannot be resolved without changing a physical assumption the authors have already published, or the WP1 test showing that the SoC floor binds fails on most instances.
6. Total compute exceeds the declared 2 500 wall-clock-hour cap (§7).
7. The fleet size `|V|` (never stated in the source draft, see erratum A23) cannot be recovered such that the appendix ToU horizons reproduce.

Escalation format: one page — what happened, what was tried, the two or three options with their consequences, and a recommendation.

---

# 7. Work packages and schedule

Effort is given in **agent-sessions** (one focused autonomous working block) and in calendar months assuming part-time supervision by the author.

| WP | Title | Depends on | Sessions | Owner | Gate |
|---|---|---|---|---|---|
| **WP0** | Repository scaffold, environment pinning, CI, `numbers.json` registry, Makefile | — | 2 | algo-eng | — |
| **WP0.5** | **Provided-asset intake**: inventory the author's CP-SAT solver and ToU ILP papers in `Dec_Optim_ToU`; vendor unmodified into `external/`; write `docs/provided_assets.md` and `docs/tou_linearisations.md`; stand up the flat-tariff regression harness | WP0 | **2** | algo-eng | **G0** |
| **WP1** | Domain layer: instance model, ToU profiles, battery/SoC, **units-contract reconciliation**, **deadline semantics and feasibility bounds**, **normalisation anchors**, objective (single source of truth) | WP0.5 | **4** | algo-eng | **G0** |
| **WP2** | Discrete-event simulator: events, engine, state snapshot/restore, feasibility masks, `B_needed`, charger capacity | WP1 | 4 | algo-eng | — |
| **WP3** | `EX-CP` as an **adapter over `external/`** + AUGMECON2 `ε`-sweep + `LB-MILP`; flat-tariff regression; objective-consistency test | WP0.5, WP2 | 4 | algo-eng | **G1** |
| **WP4** | Benchmarks: `BU40` transcription + ToU regeneration (A26); `JSSPT-ToU-Bench` generator and release; **feasibility-margin certification of every instance** (λ calibrated so `H ≥ ρ·C_max^LB`) | WP1 | **3** | algo-eng | — |
| **WP5** | Baselines: dispatching-rule family, charge policies, NSGA-II / MOEA-D | WP2 | 2 | algo-eng | — |
| **WP6** | `M1`: corrected utilities, best response, log-linear learning, ε-NE, PoA/PoS; T1–T4 with falsifying tests | WP2 | 4 | algo-eng | — |
| **WP7** | **`M2`**: characteristic function, Shapley estimators with CIs, least-core LP, nucleolus, merge–split coalition formation, NBS/KS bargaining; N5 counterexample regression test, P5/T5/T6 certification | WP2, **WP5**, WP6 | **8** | algo-eng | **G2** |
| **WP8** | Paper A experiments E1–E7, E10 + E11 ablations; analysis, statistics, figures, tables | WP3–WP7 | 5 | algo-eng | **G3** |
| **WP9** | Literature sweep and positioning table for Paper A (≈ 70 refs) | — (parallel) | 3 | sci-writer | — |
| **WP10** | Paper A manuscript, cover letter, `CLAIMS.md`, response-letter template | WP8, WP9 | 5 | sci-writer | **G4 → deliver A** |
| **WP11** | `M3`: Gymnasium env, features, attention networks, maskable PPO, reward modes, warm start; T7–T9 | WP2, WP6 | 6 | algo-eng | — |
| **WP12** | Paper B experiments E8–E10 + ablations; training runs (CPU budget ≈ 8–14 h per configuration) | WP11 | 5 | algo-eng | **G5** |
| **WP13** | Literature sweep and positioning for Paper B (≈ 65 refs) | — (parallel) | 3 | sci-writer | — |
| **WP14** | Paper B manuscript + cover letter with companion disclosure | WP12, WP13 | 4 | sci-writer | **G6 → deliver B** |
| **WP15** | Artefact release: GitHub, Zenodo DOI, reference results, README | WP8, WP12 | 2 | algo-eng | — |

**Total ≈ 67 agent-sessions.** Review rounds add roughly 25 %.

**Declared compute budget (referenced by §6.8 item 6 and risk R10).** Hard cap **2 500 wall-clock hours** on a single 16-core machine. *Wall-clock, not CPU-hours* — the distinction matters and is easy to get wrong: `EX-CP` runs 8 CP-SAT workers and RL training runs 16 vectorised environments, so the same programme measured in CPU-hours is roughly an order of magnitude larger. Every figure below and every escalation threshold is wall-clock.

| Block | Allocation (wall-clock h) | Basis |
|---|---|---|
| `EX-CP` | 900 | 3 600 s × 510 instances × 5 `ε` values = **2 550 h nominal**, cut to 900 h by an early-exit rule that abandons an instance after two consecutive `ε` values time out, and by a 900 s limit on the four tightened `ε` values (the full hour is spent only on `ε = H`) |
| Cooperative certification | 500 | `c(S)` call budget of §3.4, certification subset only |
| RL training | 600 | 3 reward modes × 5 seeds × ≈ 12 h, plus ablations |
| Baselines, metaheuristics, E11 ablations | 400 | |
| Re-runs for reviewer check R-2 | 100 | |
| **Total** | **2 500** | |

The exact reference sweeps the deadline `ε ∈ {H, 0.95H, 0.90H, 0.85H, 0.80H}` (§3.2.1). Scalarised methods use the **normalised** weight grid `ω ∈ {0.25, 0.5, 0.75}` on the full benchmark and `{0, 0.1, …, 1.0}` for the ω-sensitivity ablation on a 40-instance subset — meaningful only because §3.7.3 normalises first; on the raw objective this grid samples three makespan-only points. Exceeding the cap escalates to the author (§6.8) rather than being absorbed silently.

**Calendar view.**

| Month | Milestone |
|---|---|
| 1 | WP0, **WP0.5**, WP1–WP2 · **G0** |
| 2 | WP3–WP5 · **G1** |
| 3 | WP6 |
| 4 | WP7 · **G2** |
| 5 | WP8 · **G3**; WP9 in parallel |
| 6 | WP10 · **G4** → **Paper A submitted to EAAI** |
| 7 | WP11 |
| 8 | WP12 · **G5**; WP13 in parallel |
| 9 | WP14 · **G6** → **Paper B submitted to ESWA**; WP15 artefact release |

Parallelisation note: WP9 and WP13 (literature) are independent of all code work and should run concurrently with WP6–WP8 and WP11–WP12 respectively. This is the main lever for compressing the calendar.

---

# 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | **Submodularity of `c` fails under congestion**, so P5 is conditional and the core can be empty | **High — already demonstrated by counterexample in §3.4** | Medium (was High) | Already mitigated by design: the headline is T5 (certified least-core), not an unconditional core theorem; N5 is reported as a structural finding; P5 is scoped to the congestion-free sub-class and may be dropped without changing the programme |
| **R2** | Self-overlap with `SANOGO2025111366` judged insufficient by an editor | Medium | High | §2.3 safeguards; cooperative layer carries the novelty; delta paragraph and companion disclosure |
| **R3** | `EX-CP` closes too few instances to support a PoA claim | Medium | Medium | Report PoA on the closed subset and bound-gaps elsewhere; the scalability wall is itself a result (E2) |
| **R4** | RL does not beat M1/M2 on CPU budget | **High** | Medium | Reframe Paper B around *latency*, *zero-shot generalisation* and the T8 alignment result rather than raw objective wins; warm start from M1; §2.6 fallback |
| **R5** | Cooperative certification exceeds budget — the real cost is `c(S)` evaluations, not Shapley combinatorics | **High** | Medium | Best-response surrogate for `c(S)`; memoisation across permutations; declared certification subset (`n ≤ 12` plus 40 sampled instances); stratified + antithetic sampling; `S_max` cap; report call counts and CPU-hours (§3.4) |
| **R6** | Referee demands real factory validation | Medium | Medium | Pre-empt with an industrially parameterised case study built from the cited AIV/charger sources, and an explicit external-validity limitation paragraph |
| **R7** | Units-contract reconciliation changes published assumptions | Low | Medium | Resolve at WP1, before any results exist; document in the units table; escalate if it conflicts with the authors' prior papers |
| **R8** | Two submissions perceived as salami slicing | Low-medium | High | Companion disclosure in both cover letters; genuinely disjoint contributions; shared artefact released once |
| **R9** | Autonomous loop deadlocks at a gate | Medium | Low | `MAX_ROUNDS = 4`, rebuttal mechanism, human escalation with a one-page decision brief |
| **R10** | Compute budget overrun | Medium | Low | Explicit 2 500 wall-clock-hour cap with per-block allocation (§7); < 250 k parameters, 16 vectorised envs, hard step caps, checkpointing; 5-seed exemption for RL training (§5.3); escalation on overrun |
| **R11** | C6/T8 anticipated by the difference-rewards literature | **High — it already is** | Low, if handled | Claim the empirical question, not the identity; cite Wolpert & Tumer (1999), Agogino & Tumer, Devlin et al. (AAMAS 2014) in Paper B §2; C6 is rated Medium, not High (§1.7) |
| **R12** | **The λ-generated horizon `H` is infeasible as a deadline** on some instances, or so loose that it never binds | **High** — untested either way | High: an infeasible instance yields no results; a non-binding deadline removes the constraint that gives the whole study its structure | WP4 certifies the feasibility margin `H/C_max^LB` for every instance and rejects or re-scales outside `[1.05, 2.0]`; λ ablation reports the margin distribution; escalate if the classic `BU40` horizons cannot be made feasible without departing from the published λ formula |
| **R13** | **Decentralised methods deadlock against the deadline** — myopic off-peak deferral leaves no feasible completion | **High** for `M1`/`M3` without a safeguard | High | Mandatory remaining-work bound checked before every commit, plus a deadline-feasible fallback policy; deadlock rate is a reported metric, not an absorbed exception; ablate fallback on/off (E11) |
| **R14** | **The provided CP-SAT solver cannot carry the ToU extension cleanly** (structure, licence, or format mismatch) | Medium | Medium | WP0.5 inventories before committing; port affected constraint blocks explicitly with citation rather than rewriting wholesale; keep `external/` unmodified and the flat-tariff regression green; escalate if the licence forbids derivation |
| **R15** | Normalisation anchors drift mid-episode (adaptive normalisation), silently invalidating T1 and T7 | Medium — it is the natural thing to reach for | High, and hard to detect from results | `anchors.py` returns a frozen dataclass computed at instance load; `test_scalarisation.py` asserts byte-identical anchors at episode start and end; referee check R-12 |

---

# 9. Deliverables checklist

**Code and data**
- [ ] `jsspt-tou` repository, `mypy --strict` clean, test suite green
- [ ] Discrete-event simulator with snapshot/restore
- [ ] `EX-CP` (CP-SAT) and `LB-MILP` (HiGHS) reference models
- [ ] Baseline family: dispatching rules × vehicle rules × charge policies, NSGA-II, MOEA/D
- [ ] `M1` best response + log-linear learning, ε-NE certification, PoA/PoS
- [ ] `M2` characteristic function, Shapley with CIs, core/least-core/nucleolus, merge–split, NBS/KS bargaining
- [ ] `M3` event-driven env, attention actors, centralised critic, maskable PPO, difference reward
- [ ] `JSSPT-ToU-Bench` generator + 470 generated instances (450 main grid + 20 spread sub-suite) + reference results
- [ ] `numbers.json` registry and `\jnum{}` LaTeX integration
- [ ] Zenodo DOI, GitHub release, `CITATION.cff`

**Paper A (EAAI)**
- [ ] Manuscript, abstract, 5 highlights, 10 figures, 8 tables, ≈ 70 references
- [ ] Proofs T1–T4, P5, T5, N5, T6 with assumption lists and falsifying tests
- [ ] Cover letter with delta statement
- [ ] `CLAIMS.md` evidence map
- [ ] `ACCEPT` verdict at G4

**Paper B (ESWA)**
- [ ] Manuscript, abstract, 5 highlights, ≈ 10 figures, 6 tables, ≈ 65 references
- [ ] Proofs T7–T9
- [ ] Cover letter with companion disclosure
- [ ] `CLAIMS.md` evidence map
- [ ] `ACCEPT` verdict at G6

**Process artefacts**
- [ ] `DECISIONS.md`, `EXPERIMENT_REPORT.md`, `review/REVIEW_LOG.md`, all round verdict JSONs

---

# Appendix A — Errata for the current draft

Actionable list for the writing agent, in manuscript order.

| # | Location | Issue | Action |
|---|---|---|---|
| A1 | Front matter | `\journal{Journal of Manufacturing Systems}` | Change to the target journal; rewrite the cover letter |
| A2 | Abstract | `Abstract text.` | Write per §2.4 / §2.5 skeleton |
| A3 | Graphical abstract | Empty | Produce F1 (architecture) as `grabs.pdf` |
| A4 | Highlights | Generic, mention only the non-cooperative model | Replace with the 5 in §2.4 |
| A5 | §2 Related work | Empty | Write per §2.4 structure with positioning table |
| A6 | §3.1 assumptions | 20 %/80 % SoC window; only the floor is enforced | Enforce or drop the ceiling |
| A7 | §3.1 assumptions | 20 Ah in 5 min at 1.5 kW is dimensionally inconsistent | Resolve per §3.0 units contract |
| A8 | §Centralized model, Eq. (cost) | `E_cost` omits machine processing energy | Add ToU-indexed processing cost (F6) |
| A9 | §Centralized model | No charger capacity constraint | Add `Cumulative`/non-overlap on charging (F7) |
| A10 | §Centralized model, Eq. (batt_balance_agg) | Aggregate battery balance is a relaxation | Relabel as `LB-MILP`; add exact `EX-CP` |
| A11 | §Game theory framework | `u_p` defined as cost increase, then minimised | Adopt the single convention of §3.3 (F4) |
| A12 | §Game theory framework | NE stated as `≥` then re-glossed as `min` | Rewrite consistently |
| A13 | §Theoretical properties | Submodularity described over player sets | Re-prove over (player, action) pairs (F2) |
| A14 | §Theoretical properties, bound eq. | `½Φ(αᵒᵖᵗ) ≤ Φ(α*) ≤ Φ(αᵒᵖᵗ)` | Remove; replace with T3 (PoS = 1) + measured PoA (F1) |
| A15 | §Potential-game property | Asserted, cited to the authors' prior paper | Prove as T1; scope to the stage game; add T2 (F5) |
| A16 | §RL, `ΔE_q` | Charging term only | Harmonise with the corrected `Φ`; restate T7 |
| A17 | §RL | No implementation, no results | WP11–WP12 |
| A18 | §ToU profile | `λ = 2` justified as "empirical testing", no data | Add the λ ablation |
| A19 | Notation | `M`/`𝓜` collision; `𝓜` reused for the SMDP tuple | Single notation table |
| A20 | §§5–8 | Experiments, results, insights, conclusion empty | WP8, WP10 |
| A21 | Appendix proofs | Utility-system validity argued informally | Prove with explicit inequalities |
| A22 | Throughout | No figures at all | Produce the §5.5 inventory |
| **A23** | Throughout | **The fleet size `\|V\|` is never stated anywhere in the draft.** It is recoverable only by reverse-engineering the horizon formula (job set 1 / layout 1 gives `2·(176/4 + Σσ/2) = 216.0` ⇒ `\|V\| = 2`). Until it is fixed, "verbatim transcription" of `BU40` and verification of the appendix ToU horizons are not executable | State `\|V\|` per instance in the benchmark files; verify every appendix horizon reproduces from the stated fleet size before WP4 closes |
| **A24** | §3.1 assumptions | Depletion rates given as **mA per second**; read literally the SoC floor can never bind and the entire battery axis is inert (§3.0(ii)) | Adopt **mAh per second**; add the binding-floor test |
| **A25** | §Makespan modeling, `B_needed` | Adds a bare `20` (Ah) to terms in rate × time | Express in mAh; floor = 20 000 mAh |
| **A26** | Appendix Table `tab:all_instance` | **Seven rows have period 6 ending after the stated horizon** — EX32 (191 > 190.5), EX33 (181 > 180.5), EX64 (301 > 300.0), EX71 (285 > 284.0), EX72 (205 > 204.0), EX103 (248 > 247.5), EX104 (368 > 367.5) — contradicting the draft's own `R_p^q(a_p) ≤ H` feasibility condition | Regenerate the table from the horizon formula with a documented rounding rule; add a test asserting the last period ends exactly at `⌈H_len⌉` |
| **A27** | §Game theory framework vs §Stage game vs §Action Space | **Three mutually incompatible machine action spaces**: `(J_i, t_start)` with continuous start time, and `(i, d)` with `d ∈ 𝒟` finite, in different sections | Fix the finite ToU-aligned delay grid `𝒟` everywhere; required for T1's finite-improvement argument (§3.3) |
| **A28** | §Theoretical properties | Cites "Shapley (1971) / Ichiishi (1981)" style attributions loosely | Core non-emptiness and Shapley-in-core for convex games are **Shapley (1971)**; Ichiishi (1981) proves the *converse* (core = set of marginal vectors ⇒ convex). Attribute precisely |
| **A29** | §Game theory framework, Eq. `Φ = ω·C_max + (1−ω)·E_cost`; and everywhere `Φ` appears | **Minutes are added to euros.** On EX11 the terms differ by a factor of 110–159, so at ω = 0.5 energy contributes **0.9 %** of `Φ` and the two terms balance only at **ω\* ≈ 0.006–0.009**. As written the bi-objective study is a makespan study, the ω sweep is flat over almost all of `[0,1]`, and the MCU utilities and RL reward are both dominated by makespan | Adopt the ε-constraint as the primary formulation and normalise before any scalarisation, per §3.7; report the raw-vs-normalised comparison as E13 so the correction is visible rather than silent |
| **A30** | §Problem description; §Makespan modeling; all three models | **The horizon `H` is a hard deadline but is never stated as a constraint.** The condition `R_p^q(a_p) ≤ H` appears only inside the machine return function; no model enforces `C_max ≤ H`, no instance is certified feasible against it, and no method has a safeguard against deadlocking into infeasibility | State `C_max ≤ H` in the problem statement and enforce it in `EX-CP`, `LB-MILP`, the simulator, the game and the RL environment; certify feasibility margins at generation; add the fallback policy and report deadlock rate (§3.0) |
| **A31** | §Problem modeling, λ | λ is presented as a tariff-shaping constant; under A30 it also decides whether an instance is **solvable at all** | Re-frame λ as a feasibility parameter, calibrate against `C_max^LB`, and make its ablation load-bearing rather than incidental |

# Appendix B — Notation contract

`J` jobs · `O` operations · `M` machines · `V` robots · `L` locations · `H` tariff intervals · `T` transport tasks · `N = M ∪ V` players · `Ω` ground set of (player, action) pairs · `Φ` global cost · `W` welfare (cost reduction vs all-idle) · `u_p` agent utility · `ω` trade-off weight · `c(S)` coalition cost · `v(S)` coalition savings · `φ(v)` Shapley value · `CS` coalition structure · `α` joint action profile · `α*` Nash equilibrium · `s_q, e_q, a_q` SMDP state / event / action at decision `q` · `Δt_q` holding time · `γ` base discount per unit time · `R_p^q(·)` return function · `B_needed` required SoC.

Symbol collisions to eliminate: `M` (machines) vs `𝓜` (SMDP tuple) — rename the SMDP tuple to `𝔐`; `T` (transport tasks) vs `T` (temperature in log-linear learning) — rename the temperature to `θ`; `H` (the set of tariff intervals) vs `H` (the scalar horizon length produced by the λ formula) — keep `𝓗` for the set and `H_len` for the scalar; `B` (battery level) vs `𝓑_m` (machine buffer) — keep the calligraphic form for buffers everywhere.

# Appendix C — Agent invocation sketch

```python
# Orchestration skeleton for autonomous execution.
# One gate at a time; the referee's JSON verdict — not any agent's opinion — authorises delivery.

GATES = ["G0","G1","G2","G3","G4","G5","G6"]
MAX_ROUNDS = {g: 4 for g in GATES}

for gate in GATES:
    for rnd in range(1, MAX_ROUNDS[gate] + 1):

        # 1. Producing agents work. Fresh context, roadmap + prior findings as input.
        if gate_needs_code(gate):
            Agent(subagent_type="general-purpose",
                  prompt=ALGO_ENG_PROMPT + scope_for(gate) + open_findings(gate))
        if gate_needs_paper(gate):
            Agent(subagent_type="general-purpose",
                  prompt=SCI_WRITER_PROMPT + scope_for(gate) + open_findings(gate))

        # 2. Referee reviews. Fresh context every round; gets artefacts + prior review log,
        #    never the producers' justifications.
        verdict = Agent(subagent_type="general-purpose",
                        prompt=REFEREE_PROMPT + scope_for(gate) + review_log(gate),
                        schema=VERDICT_SCHEMA)

        append_review_log(gate, rnd, verdict)

        # 3. The gate. Mechanical, not judgemental — the §6.5 function, nothing else.
        if delivery_authorised(verdict):
            deliver(gate)          # only here does anything leave the repository
            break
    else:
        escalate_to_human(gate, last_verdict)   # MAX_ROUNDS exhausted
        break
```

---

*End of roadmap.*
