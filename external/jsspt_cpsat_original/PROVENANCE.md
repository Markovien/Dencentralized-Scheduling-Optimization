# Provenance — author-provided CP-SAT assets

**Do not edit any file in this directory.** Everything here is a verbatim copy of an
artefact supplied by the author. `EX-CP` (`src/jsspt_tou/exact/cpsat_model.py`) is an
*adapter* over these files, per ROADMAP.md §3.2.0 and referee check R-11.

| File | Origin | Received | Role |
|---|---|---|---|
| `solve_jsspt_cp.py` | `Intelliegent_warehouse.zip`, authored by Kader Sanogo | 2026-09-10 (with this session's uploads) | Classical JSSPT CP-SAT solver (transport, empty travel, robot NoOverlap, return-to-L/U). **This is the reference `EX-CP` extends.** |
| `cp_sat_scheduler.py` | same archive, header dated 19 Feb 2026 | 2026-09-10 | JSSPT-HF variant (human fatigue). Not on the ToU path; retained for the asset inventory and because its `Cumulative`-style resource handling informed the charger model. |
| `example_instance.json` | same archive (`temp_solver_instance.json`) | 2026-09-10 | Example of the canonical instance JSON schema, adopted as the project format. |

Authorship: Kader Sanogo. No licence file accompanied the archive; the files are used here
as the author's own prior work, extended by the author's project. If the repository is
released publicly, add an explicit licence statement for this directory.

## The regression contract

`tests/test_exact_regression.py` asserts that with a flat tariff, no deadline and no battery
constraints, `EX-CP` returns **exactly** the makespan `solve_jsspt_cp.solve_jsspt_cp`
returns. Any divergence is a bug in the adaptation, not in the original.
