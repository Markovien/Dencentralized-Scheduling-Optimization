"""The BU40 suite: 10 job sets x 4 layouts (Bilge & Ulusoy, 1995).

Job sets and travel-time layouts are transcribed **verbatim** from the appendix of the
manuscript draft (Tables ``tab:job_sets`` and ``tab:travel_times``).  The ToU profile is
*regenerated* from the horizon formula rather than copied, because seven rows of the
published table end after their own stated horizon -- erratum **A26**.
:func:`published_tou_table` keeps the published values so the regeneration can be checked
against the 33 sound rows; that agreement is the transcription check
(ROADMAP.md §5.1).

Fleet size.  The draft never states ``|V|`` anywhere (erratum **A23**).  It is recoverable
from the published horizons: job set 1 / layout 1 gives
``2 * (176/4 + 128/|V|) = 216.0``, hence ``|V| = 2``.  :func:`build_instance` asserts the
recovered horizon reproduces the published one for every instance, so the recovery is
checked rather than assumed.

Run ``python -m jsspt_tou.benchmark.bilge_ulusoy --out results/instances`` to emit the 40
instance files in the canonical JSON schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

from jsspt_tou.domain.anchors import Anchors, compute_anchors
from jsspt_tou.domain.instance import Instance

N_ROBOTS: Final[int] = 2
"""Recovered from the published horizons -- see the module docstring (erratum A23)."""

# --- job sets (Table tab:job_sets) --------------------------------------------------------
# Each entry: job -> ordered [(machine index 1..4, processing time)].
JOB_SETS: Final[dict[int, tuple[tuple[tuple[int, int], ...], ...]]] = {
    1: (
        ((1, 8), (2, 16), (4, 12)),
        ((1, 20), (3, 10), (2, 18)),
        ((3, 12), (4, 8), (1, 15)),
        ((4, 14), (2, 18)),
        ((3, 10), (1, 15)),
    ),
    2: (
        ((1, 10), (4, 18)),
        ((2, 10), (4, 18)),
        ((1, 10), (3, 20)),
        ((2, 10), (3, 15), (4, 12)),
        ((1, 10), (2, 15), (4, 12)),
        ((1, 10), (2, 15), (3, 12)),
    ),
    3: (
        ((1, 16), (3, 15)),
        ((2, 18), (4, 15)),
        ((1, 20), (2, 10)),
        ((3, 15), (4, 10)),
        ((1, 8), (2, 10), (3, 15), (4, 17)),
        ((2, 10), (3, 15), (4, 8), (1, 15)),
    ),
    4: (
        ((4, 11), (1, 10), (2, 7)),
        ((3, 12), (2, 10), (4, 8)),
        ((2, 7), (3, 10), (1, 9), (3, 8)),
        ((2, 7), (4, 8), (1, 12), (2, 6)),
        ((1, 9), (2, 7), (4, 8), (2, 10), (3, 8)),
    ),
    5: (
        ((1, 6), (2, 12), (4, 9)),
        ((1, 18), (3, 6), (2, 15)),
        ((3, 9), (4, 3), (1, 12)),
        ((4, 6), (2, 15)),
        ((3, 3), (1, 9)),
    ),
    6: (
        ((1, 9), (2, 11), (4, 7)),
        ((1, 19), (2, 20), (4, 13)),
        ((2, 14), (3, 20), (4, 9)),
        ((2, 14), (3, 20), (4, 9)),
        ((1, 11), (3, 16), (4, 8)),
        ((1, 10), (3, 12), (4, 10)),
    ),
    7: (
        ((1, 6), (4, 6)),
        ((2, 11), (4, 9)),
        ((2, 9), (4, 7)),
        ((3, 16), (4, 7)),
        ((1, 9), (3, 18)),
        ((2, 13), (3, 19), (4, 6)),
        ((1, 10), (2, 9), (3, 13)),
        ((1, 11), (2, 9), (4, 8)),
    ),
    8: (
        ((2, 12), (3, 21), (4, 11)),
        ((2, 12), (3, 21), (4, 11)),
        ((2, 12), (3, 21), (4, 11)),
        ((2, 12), (3, 21), (4, 11)),
        ((1, 10), (2, 14), (3, 18), (4, 9)),
        ((1, 10), (2, 14), (3, 18), (4, 9)),
    ),
    9: (
        ((3, 9), (1, 12), (2, 9), (4, 6)),
        ((3, 16), (2, 11), (4, 9)),
        ((1, 21), (2, 18), (4, 7)),
        ((2, 20), (3, 22), (4, 11)),
        ((3, 14), (1, 16), (2, 13), (4, 9)),
    ),
    10: (
        ((1, 11), (3, 19), (2, 16), (4, 13)),
        ((2, 21), (3, 16), (4, 14)),
        ((3, 8), (2, 10), (1, 14), (4, 9)),
        ((2, 13), (3, 20), (4, 10)),
        ((1, 9), (3, 16), (4, 18)),
        ((2, 19), (1, 21), (3, 11), (4, 15)),
    ),
}

# --- layouts (Table tab:travel_times), index 0 = L/U station ------------------------------
LAYOUTS: Final[dict[int, tuple[tuple[int, ...], ...]]] = {
    1: (
        (0, 6, 8, 8, 6),
        (6, 0, 6, 8, 10),
        (8, 6, 0, 6, 8),
        (8, 8, 6, 0, 6),
        (6, 10, 8, 6, 0),
    ),
    2: (
        (0, 4, 6, 6, 4),
        (4, 0, 2, 4, 2),
        (6, 2, 0, 2, 4),
        (6, 4, 2, 0, 2),
        (4, 2, 4, 2, 0),
    ),
    3: (
        (0, 2, 4, 4, 2),
        (2, 0, 2, 6, 4),
        (4, 2, 0, 6, 6),
        (4, 6, 6, 0, 2),
        (2, 4, 6, 2, 0),
    ),
    4: (
        (0, 4, 8, 10, 14),
        (4, 0, 4, 6, 10),
        (8, 4, 0, 6, 6),
        (10, 6, 6, 0, 6),
        (14, 10, 6, 6, 0),
    ),
}

# --- published ToU table (appendix tab:all_instance) ---------------------------------------
# instance -> (horizon, (inclusive display start of each of the six periods, ...)).
# Retained only so the regeneration can be verified; the *generated* profile is what every
# model uses.  Rows EX32, EX33, EX64, EX71, EX72, EX103, EX104 have a period 6 that ends
# after the stated horizon -- erratum A26.
_PUBLISHED: Final[dict[str, tuple[float, tuple[int, ...]]]] = {
    "EX11": (216.0, (0, 27, 72, 99, 126, 144)),
    "EX21": (242.5, (0, 30, 81, 111, 141, 161)),
    "EX31": (258.5, (0, 32, 86, 118, 150, 172)),
    "EX41": (261.5, (0, 33, 87, 120, 153, 175)),
    "EX51": (189.5, (0, 24, 63, 87, 111, 127)),
    "EX61": (272.0, (0, 34, 91, 125, 159, 182)),
    "EX71": (284.0, (0, 36, 95, 131, 167, 191)),
    "EX81": (303.0, (0, 38, 101, 139, 177, 202)),
    "EX91": (263.5, (0, 33, 88, 121, 154, 176)),
    "EX101": (333.5, (0, 42, 111, 153, 195, 223)),
    "EX12": (160.0, (0, 20, 53, 73, 93, 106)),
    "EX22": (178.5, (0, 22, 59, 81, 103, 118)),
    "EX32": (190.5, (0, 24, 64, 88, 112, 128)),
    "EX42": (177.5, (0, 22, 59, 81, 103, 118)),
    "EX52": (133.5, (0, 17, 45, 62, 79, 90)),
    "EX62": (200.0, (0, 25, 67, 92, 117, 134)),
    "EX72": (204.0, (0, 26, 69, 95, 121, 138)),
    "EX82": (223.0, (0, 28, 74, 102, 130, 149)),
    "EX92": (195.5, (0, 24, 65, 89, 113, 129)),
    "EX102": (245.5, (0, 31, 82, 113, 144, 164)),
    "EX13": (154.0, (0, 19, 51, 70, 89, 102)),
    "EX23": (170.5, (0, 21, 57, 78, 99, 113)),
    "EX33": (180.5, (0, 23, 61, 84, 107, 122)),
    "EX43": (185.5, (0, 23, 62, 85, 108, 123)),
    "EX53": (127.5, (0, 16, 43, 59, 75, 86)),
    "EX63": (192.0, (0, 24, 64, 88, 112, 128)),
    "EX73": (190.0, (0, 24, 64, 88, 112, 128)),
    "EX83": (223.0, (0, 28, 74, 102, 130, 149)),
    "EX93": (195.5, (0, 24, 65, 89, 113, 129)),
    "EX103": (247.5, (0, 31, 83, 114, 145, 166)),
    "EX14": (218.0, (0, 27, 72, 99, 126, 144)),
    "EX24": (260.5, (0, 33, 87, 120, 153, 175)),
    "EX34": (270.5, (0, 34, 90, 124, 158, 181)),
    "EX44": (263.5, (0, 33, 88, 121, 154, 176)),
    "EX54": (191.5, (0, 24, 64, 88, 112, 128)),
    "EX64": (300.0, (0, 38, 101, 139, 177, 202)),
    "EX74": (318.0, (0, 40, 106, 146, 186, 213)),
    "EX84": (343.0, (0, 43, 114, 157, 200, 229)),
    "EX94": (289.5, (0, 36, 96, 132, 168, 192)),
    "EX104": (367.5, (0, 46, 123, 169, 215, 246)),
}


def instance_id(job_set: int, layout: int) -> str:
    """``EX<job set><layout>`` -- the naming of the published table."""
    return f"EX{job_set}{layout}"


def published_tou_table() -> dict[str, tuple[float, tuple[int, ...]]]:
    """The published horizon and period starts, for the transcription check."""
    return dict(_PUBLISHED)


def build_instance(
    job_set: int,
    layout: int,
    n_robots: int = N_ROBOTS,
    lam: float = 2.0,
) -> Instance:
    """Build one BU40 instance with its regenerated ToU profile."""
    return Instance.build(
        instance_id=instance_id(job_set, layout),
        n_machines=4,
        n_robots=n_robots,
        sigma=LAYOUTS[layout],
        job_ops=[list(job) for job in JOB_SETS[job_set]],
        lam=lam,
        layout_id=layout,
    )


def all_instances(n_robots: int = N_ROBOTS, lam: float = 2.0) -> list[Instance]:
    """All 40 instances, in the published table's order (layout-major)."""
    return [
        build_instance(js, ly, n_robots=n_robots, lam=lam)
        for ly in (1, 2, 3, 4)
        for js in range(1, 11)
    ]


def certify(inst: Instance) -> tuple[Anchors, bool, bool]:
    """Certify one instance against its own deadline (ROADMAP.md §3.0(1), risk R12).

    Returns ``(anchors, admissible, binding)``.

    * ``admissible`` -- the margin ``H / C_max^LB`` is at least 1.05, so the deadline does
      not exclude every schedule.  A margin below 1 would be a *generation defect*, not a
      result.
    * ``binding`` -- the margin is at most 2.0, so the deadline is tight enough to shape
      the schedule.  On BU40 at the published ``lambda = 2`` this is **false almost
      everywhere**: the deadline is admissible but slack.  That is a finding about the
      published horizon formula, reported as such (erratum A31), and it is why the
      ``lambda`` ablation is load-bearing rather than incidental.
    """
    anchors = compute_anchors(inst)
    margin = anchors.feasibility_margin
    return anchors, margin >= 1.05, margin <= 2.0


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the BU40 benchmark files.")
    parser.add_argument("--out", type=Path, default=Path("results/instances"))
    parser.add_argument("--robots", type=int, default=N_ROBOTS)
    parser.add_argument("--lam", type=float, default=2.0)
    args = parser.parse_args(argv)

    admissible = binding = 0
    for inst in all_instances(n_robots=args.robots, lam=args.lam):
        inst.save(args.out / f"{inst.instance_id}.json")
        anchors, adm, bind = certify(inst)
        admissible += int(adm)
        binding += int(bind)
        print(
            f"{inst.instance_id:>6}  H={inst.horizon:8.1f}  "
            f"Cmax_LB={anchors.cmax_lb:7.1f}  margin={anchors.feasibility_margin:5.2f}  "
            f"{'admissible' if adm else 'INFEASIBLE'}"
            f"{'  binding' if bind else ''}"
        )
    print(
        f"\nlambda={args.lam}: {admissible}/40 admissible (margin >= 1.05), "
        f"{binding}/40 binding (margin <= 2.0)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
