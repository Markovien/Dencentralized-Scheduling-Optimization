"""Problem instance: the frozen description of one JSSPT-ToU problem.

Implements the sets, parameters and transport-task construction of Paper A §3
("Problem statement").  Instances are immutable; the simulator mutates only its own state.

The JSON schema is the author's own, taken verbatim from
``external/jsspt_cpsat_original/solve_jsspt_cp.py`` (keys ``machines_nb``, ``robots_nb``,
``sigma``, ``jobs[*].ops[*]`` with ``machine_index`` 1-based and ``duration``), extended
with additive keys the original loader ignores: ``tou``, ``horizon``, ``battery``,
``charger``, ``machine_power_kw``.  Adopting the tested format rather than inventing one
is the roadmap's preferred option (§3.2.0, step 5).

Location indexing follows the original: ``0`` is the load/unload station ``Z`` (= L/U),
``1..M`` are the machines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from jsspt_tou.domain.battery import BatterySpec
from jsspt_tou.domain.tou import TariffProfile, horizon_length

def _as_int(x: float) -> float | int:
    """Emit whole numbers as ``int`` so the vendored loader (which calls ``int(...)`` on
    durations and indexes ``sigma`` into CP-SAT domains) accepts the file unchanged."""
    return int(x) if float(x).is_integer() else x


LU: Final[int] = 0
"""Location index of the load/unload station."""

MACHINE_POWER_KW_DEFAULT: Final[float] = 2.0
"""Constant machine draw while processing, per the draft's assumption list."""


@dataclass(frozen=True, slots=True)
class Operation:
    """One operation ``O_{i,k}``: the ``k``-th step of job ``i``."""

    job: int
    index: int
    machine: int  # 1..M, also the location index
    duration: float


@dataclass(frozen=True, slots=True)
class TransportTask:
    """One transport task ``T_{i,k}``: move job ``i`` from ``origin`` to ``dest``.

    ``index == len(ops)`` is the final return leg to the L/U station, which is what the
    makespan is measured to (the Bilge--Ulusoy convention).
    """

    job: int
    index: int
    origin: int
    dest: int
    duration: float

    @property
    def is_return(self) -> bool:
        return self.dest == LU


@dataclass(frozen=True, slots=True)
class Instance:
    """A complete JSSPT-ToU instance.

    Attributes
    ----------
    instance_id:
        e.g. ``"EX11"`` -- job set 1, layout 1.
    n_machines, n_robots:
        Fleet sizes.  ``n_robots`` is recovered from the published horizons (erratum A23).
    sigma:
        ``(M+1) x (M+1)`` travel-time matrix; index 0 is the L/U station.
    jobs:
        Per job, the ordered tuple of its operations.
    tariff:
        The regenerated ToU profile (erratum A26).
    battery:
        Battery/charger parameters under the units contract.
    n_chargers:
        Charger capacity ``K_CH``.  **Unary by default** -- the constraint absent from
        every model in the draft (finding F7, erratum A9), and the very resource whose
        congestion breaks submodularity (result N5).
    charger_location:
        Location index of the charging station.  The Bilge--Ulusoy layouts define travel
        times only between the L/U station and the four machines, so the charger is
        co-located with L/U; recorded in ``DECISIONS.md``.
    machine_power_kw:
        Constant processing draw.
    lam:
        Horizon scaling factor actually used to build ``tariff``.
    """

    instance_id: str
    n_machines: int
    n_robots: int
    sigma: tuple[tuple[float, ...], ...]
    jobs: tuple[tuple[Operation, ...], ...]
    tariff: TariffProfile
    battery: BatterySpec
    n_chargers: int = 1
    charger_location: int = LU
    machine_power_kw: float = MACHINE_POWER_KW_DEFAULT
    lam: float = 2.0
    layout_id: int = 0
    transports: tuple[TransportTask, ...] = field(default=(), compare=False)

    # -- construction -----------------------------------------------------------------------
    @staticmethod
    def build(
        instance_id: str,
        n_machines: int,
        n_robots: int,
        sigma: Sequence[Sequence[float]],
        job_ops: Sequence[Sequence[tuple[int, float]]],
        battery: BatterySpec | None = None,
        n_chargers: int = 1,
        lam: float = 2.0,
        layout_id: int = 0,
        machine_power_kw: float = MACHINE_POWER_KW_DEFAULT,
        tariff: TariffProfile | None = None,
    ) -> "Instance":
        """Assemble an instance from raw job/layout data, generating the ToU profile."""
        jobs = tuple(
            tuple(
                Operation(job=i, index=k, machine=int(m), duration=float(d))
                for k, (m, d) in enumerate(ops)
            )
            for i, ops in enumerate(job_ops)
        )
        sigma_t = tuple(tuple(float(x) for x in row) for row in sigma)
        transports = _build_transports(jobs, sigma_t)
        spec = battery if battery is not None else BatterySpec.published()
        if tariff is None:
            total_p = sum(op.duration for job in jobs for op in job)
            total_t = sum(t.duration for t in transports)
            horizon = horizon_length(total_p, total_t, n_machines, n_robots, lam)
            tariff = TariffProfile.generate(horizon)
        return Instance(
            instance_id=instance_id,
            n_machines=n_machines,
            n_robots=n_robots,
            sigma=sigma_t,
            jobs=jobs,
            tariff=tariff,
            battery=spec,
            n_chargers=n_chargers,
            charger_location=LU,
            machine_power_kw=machine_power_kw,
            lam=lam,
            layout_id=layout_id,
            transports=transports,
        )

    # -- derived quantities -------------------------------------------------------------------
    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    @property
    def operations(self) -> tuple[Operation, ...]:
        return tuple(op for job in self.jobs for op in job)

    @property
    def horizon(self) -> float:
        """The scheduling horizon ``H``.  A **hard deadline** (ROADMAP.md §3.0)."""
        return self.tariff.horizon

    @property
    def deadline(self) -> float:
        """The enforced deadline, ``floor(H)``.

        Every model -- ``EX-CP``, the simulator, the game, the cooperative characteristic
        function -- uses this same integer value, so their feasible sets coincide exactly.
        Recorded in ``DECISIONS.md``.
        """
        return float(int(self.horizon))

    @property
    def total_processing(self) -> float:
        return sum(op.duration for op in self.operations)

    @property
    def total_transport(self) -> float:
        return sum(t.duration for t in self.transports)

    def travel(self, a: int, b: int) -> float:
        """Travel time between two locations."""
        return self.sigma[a][b]

    def transports_of_job(self, job: int) -> tuple[TransportTask, ...]:
        return tuple(t for t in self.transports if t.job == job)

    # -- serialisation -------------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        """Canonical JSON, readable by the vendored original loader."""
        return {
            "instance_id": self.instance_id,
            "layout_id": self.layout_id,
            "machines_nb": self.n_machines,
            "robots_nb": self.n_robots,
            "sigma": [[_as_int(x) for x in row] for row in self.sigma],
            "jobs": [
                {
                    "job_id": f"J{i + 1}",
                    "job_index": i + 1,
                    "n_ops": len(ops),
                    "ops": [
                        {
                            "op_id": f"O{i + 1}_{k + 1}",
                            "op_index": k + 1,
                            "machine_id": f"M{op.machine}",
                            "machine_index": op.machine,
                            "duration": _as_int(op.duration),
                            "power_kw": self.machine_power_kw,
                        }
                        for k, op in enumerate(ops)
                    ],
                }
                for i, ops in enumerate(self.jobs)
            ],
            "horizon": self.horizon,
            "deadline": self.deadline,
            "lambda": self.lam,
            "machine_power_kw": self.machine_power_kw,
            "tou": [
                {"start": p.start, "end": p.end, "price": p.price}
                for p in self.tariff.periods
            ],
            "battery": {
                "capacity_mah": self.battery.capacity_mah,
                "floor_mah": self.battery.floor_mah,
                "ceiling_mah": self.battery.ceiling_mah,
                "start_mah": self.battery.start_mah,
                "idle_mah_min": self.battery.idle_mah_min,
                "empty_mah_min": self.battery.empty_mah_min,
                "loaded_mah_min": self.battery.loaded_mah_min,
                "charge_rate_mah_min": self.battery.charge_rate_mah_min,
                "charge_block_min": self.battery.charge_block_min,
                "charger_power_kw": self.battery.charger_power_kw,
            },
            "charger": {
                "n_stations": self.n_chargers,
                "location": self.charger_location,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Instance":
        """Inverse of :meth:`to_json` (round-tripped in ``tests/test_exact_regression.py``)."""
        job_ops = [
            [(int(op["machine_index"]), float(op["duration"])) for op in job["ops"]]
            for job in data["jobs"]
        ]
        bat = data.get("battery")
        spec = (
            BatterySpec(
                capacity_mah=bat["capacity_mah"],
                floor_mah=bat["floor_mah"],
                ceiling_mah=bat["ceiling_mah"],
                start_mah=bat["start_mah"],
                idle_mah_min=bat["idle_mah_min"],
                empty_mah_min=bat["empty_mah_min"],
                loaded_mah_min=bat["loaded_mah_min"],
                charge_rate_mah_min=bat["charge_rate_mah_min"],
                charge_block_min=bat["charge_block_min"],
                charger_power_kw=bat["charger_power_kw"],
            )
            if bat
            else BatterySpec.published()
        )
        tariff: TariffProfile | None = None
        if "tou" in data and "horizon" in data:
            from jsspt_tou.domain.tou import TariffPeriod

            tariff = TariffProfile(
                periods=tuple(
                    TariffPeriod(start=p["start"], end=p["end"], price=p["price"])
                    for p in data["tou"]
                ),
                horizon=float(data["horizon"]),
            )
        charger = data.get("charger", {})
        return Instance.build(
            instance_id=str(data.get("instance_id", "unknown")),
            n_machines=int(data["machines_nb"]),
            n_robots=int(data["robots_nb"]),
            sigma=data["sigma"],
            job_ops=job_ops,
            battery=spec,
            n_chargers=int(charger.get("n_stations", 1)),
            lam=float(data.get("lambda", 2.0)),
            layout_id=int(data.get("layout_id", 0)),
            machine_power_kw=float(data.get("machine_power_kw", MACHINE_POWER_KW_DEFAULT)),
            tariff=tariff,
        )

    @staticmethod
    def load(path: Path) -> "Instance":
        return Instance.from_json(json.loads(Path(path).read_text()))


def _build_transports(
    jobs: tuple[tuple[Operation, ...], ...],
    sigma: tuple[tuple[float, ...], ...],
) -> tuple[TransportTask, ...]:
    """Construct ``T = {(i,k) : k = 1..m_i+1}``, mirroring the vendored solver's loop."""
    tasks: list[TransportTask] = []
    for i, ops in enumerate(jobs):
        for k, op in enumerate(ops):
            origin = LU if k == 0 else ops[k - 1].machine
            tasks.append(
                TransportTask(
                    job=i,
                    index=k,
                    origin=origin,
                    dest=op.machine,
                    duration=sigma[origin][op.machine],
                )
            )
        last = ops[-1].machine
        tasks.append(
            TransportTask(
                job=i,
                index=len(ops),
                origin=last,
                dest=LU,
                duration=sigma[last][LU],
            )
        )
    return tuple(tasks)
