"""Provenance regression (referee check R-11).

The cheapest correctness guarantee in the whole programme: with a **flat tariff, no
deadline and no battery**, the energy term is schedule-invariant, so ``EX-CP`` must return
*exactly* the makespan the author's own solver in ``external/`` returns.  Any discrepancy
is a bug in the adaptation, located before any ToU result exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from jsspt_tou.benchmark.bilge_ulusoy import build_instance
from jsspt_tou.domain.instance import Instance
from jsspt_tou.exact.cpsat_model import solve_exact

EXTERNAL = Path(__file__).resolve().parents[1] / "external" / "jsspt_cpsat_original"


def _original_solver():
    sys.path.insert(0, str(EXTERNAL))
    from solve_jsspt_cp import solve_jsspt_cp  # noqa: E402

    return solve_jsspt_cp


def test_external_directory_is_present_and_unmodified_in_spirit() -> None:
    assert (EXTERNAL / "solve_jsspt_cp.py").exists()
    assert (EXTERNAL / "PROVENANCE.md").exists()
    source = (EXTERNAL / "solve_jsspt_cp.py").read_text()
    assert "def solve_jsspt_cp(" in source
    assert "jsspt_tou" not in source, "external/ must not import project code"


def test_instance_format_round_trips_through_the_original_schema() -> None:
    inst = build_instance(1, 1)
    payload = json.loads(json.dumps(inst.to_json()))
    restored = Instance.from_json(payload)
    assert restored.instance_id == inst.instance_id
    assert restored.n_machines == inst.n_machines
    assert restored.n_robots == inst.n_robots
    assert restored.sigma == inst.sigma
    assert [[(o.machine, o.duration) for o in j] for j in restored.jobs] == [
        [(o.machine, o.duration) for o in j] for j in inst.jobs
    ]
    assert restored.horizon == pytest.approx(inst.horizon)
    # and the original loader accepts it unchanged
    for job in payload["jobs"]:
        for op in job["ops"]:
            assert isinstance(op["duration"], int)
    for row in payload["sigma"]:
        for x in row:
            assert isinstance(x, int)


@pytest.mark.slow
@pytest.mark.parametrize("job_set,layout", [(1, 1), (5, 1), (1, 2)])
def test_flat_tariff_regression_reproduces_the_original_makespan(
    job_set: int, layout: int
) -> None:
    solve_jsspt_cp = _original_solver()
    inst = build_instance(job_set, layout)
    ours = solve_exact(
        inst,
        omega=1.0,
        time_limit_s=180.0,
        enforce_deadline=False,
        enforce_battery=False,
        flat_tariff=True,
    )
    status, c_max, _ = solve_jsspt_cp(
        inst.to_json(), last_ttask=1, time_limit=180.0, make_gantt=False
    )
    assert status == "OPTIMAL"
    assert ours.status == "OPTIMAL"
    assert ours.outcome is not None
    assert int(ours.outcome.c_max) == int(c_max), (
        f"{inst.instance_id}: EX-CP {ours.outcome.c_max} vs original {c_max}"
    )
