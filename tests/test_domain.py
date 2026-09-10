"""Domain-layer tests: units contract, ToU regeneration, anchors, benchmark transcription.

Covers ROADMAP.md §4.3 row ``test_domain.py`` plus the WP1 obligations of §3.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from jsspt_tou.baselines.dispatching import DispatchingPolicy
from jsspt_tou.benchmark.bilge_ulusoy import (
    all_instances,
    build_instance,
    certify,
    published_tou_table,
)
from jsspt_tou.domain.anchors import compute_anchors
from jsspt_tou.domain.battery import BatterySpec, b_needed
from jsspt_tou.domain.tou import TariffProfile, horizon_length
from jsspt_tou.domain.units import charge_rate_mah_per_min, charger_power_kw, mah_to_kwh
from jsspt_tou.simulator.engine import Engine


# --- units contract (F8 / A7 / A24 / A25) -------------------------------------------------
def test_units_reconciliation_is_self_consistent() -> None:
    """The charging block, the charger power and the pack energy must agree.

    Reconciliation (i) of ROADMAP.md §3.0.  The draft's own triple -- 100 Ah, 20 Ah per
    5 minutes, 1.5 kW -- has no solution at any bus voltage.
    """
    spec = BatterySpec.published()
    assert spec.capacity_mah == 100_000.0
    assert mah_to_kwh(spec.capacity_mah) == pytest.approx(4.8)  # 100 Ah at 48 V
    # 0 % -> 80 % in 25 min  =>  3.2 Ah/min  =>  16 Ah per 5-minute block
    assert charge_rate_mah_per_min(spec.capacity_mah) == pytest.approx(3200.0)
    assert spec.block_mah == pytest.approx(16_000.0)
    # and the implied charger power must reproduce the block's energy
    assert charger_power_kw(spec.capacity_mah) == pytest.approx(9.216)
    assert spec.block_kwh == pytest.approx(spec.charger_power_kw * 5.0 / 60.0)
    assert spec.block_kwh == pytest.approx(mah_to_kwh(spec.block_mah))


def test_depletion_rates_are_read_as_mah_per_second() -> None:
    """Reconciliation (ii): read as mA/s the battery axis would be inert.

    At 7 mAh/s the usable 60 Ah window drains in about 143 minutes, well inside every
    horizon in the benchmark.  Read literally as mA/s it would take a year.
    """
    spec = BatterySpec.published()
    assert spec.loaded_mah_min == pytest.approx(420.0)
    minutes_to_empty = spec.usable_mah / spec.loaded_mah_min
    assert 140.0 < minutes_to_empty < 145.0


def test_b_needed_is_dimensionally_consistent() -> None:
    """Reconciliation (iii) / erratum A25: the floor enters as 20 000 mAh, not as ``20``."""
    spec = BatterySpec.published()
    need = b_needed(spec, travel_to_pickup=10.0, wait_at_pickup=5.0, loaded_travel=8.0,
                    travel_to_charger=6.0)
    expected = 20_000.0 + 180.0 * 10 + 60.0 * 5 + 420.0 * 8 + 180.0 * 6
    assert need == pytest.approx(expected)
    assert need > spec.floor_mah


def test_soc_ceiling_is_enforced() -> None:
    """Erratum A6: the draft states a [20 %, 80 %] window but constrains only the floor."""
    spec = BatterySpec.published()
    assert spec.charge(spec.ceiling_mah - 1.0, n_blocks=5) == spec.ceiling_mah


def test_soc_floor_binds_on_most_instances() -> None:
    """WP1 obligation: if the floor never binds, the parameters are wrong, not the code."""
    policy = DispatchingPolicy(
        machine_rule="SPT", vehicle_rule="NT", charge_rule="reactive", delay_rule="asap"
    )
    charging = 0
    instances = all_instances()
    for inst in instances:
        outcome = Engine(inst, omega=0.5).run(policy, np.random.default_rng(0))
        charging += int(outcome.charge_blocks > 0)
    assert charging >= int(0.5 * len(instances)), (
        f"the SoC floor binds on only {charging}/{len(instances)} instances; "
        "the depletion-rate units are probably wrong"
    )


# --- ToU regeneration (A26) and the horizon (A23) -------------------------------------------
def test_horizon_reproduces_every_published_value() -> None:
    """Recovering ``|V| = 2`` is checked, not assumed (erratum A23)."""
    published = published_tou_table()
    for inst in all_instances():
        assert inst.horizon == pytest.approx(published[inst.instance_id][0]), inst.instance_id


def test_regenerated_periods_match_the_published_table() -> None:
    """The transcription check of ROADMAP.md §5.1: periods 1--5 must reproduce exactly."""
    published = published_tou_table()
    for inst in all_instances():
        starts = tuple(int(p.start) for p in inst.tariff.periods)
        assert starts == published[inst.instance_id][1], inst.instance_id


def test_last_period_never_ends_after_the_horizon() -> None:
    """Erratum A26: seven published rows violate this; the regenerated table cannot."""
    for inst in all_instances():
        assert inst.tariff.periods[-1].end <= inst.horizon + 1e-9
        assert inst.tariff.periods[0].start == 0.0
        for a, b in zip(inst.tariff.periods, inst.tariff.periods[1:]):
            assert a.end == b.start  # the profile tiles [0, H) with no gap or overlap


def test_tariff_cost_is_the_piecewise_integral() -> None:
    profile = TariffProfile(
        periods=TariffProfile.generate(120.0).periods, horizon=120.0
    )
    total = profile.cost(2.0, 0.0, 120.0)
    by_hand = sum(2.0 * p.length / 60.0 * p.price for p in profile.periods)
    assert total == pytest.approx(by_hand)


def test_horizon_formula() -> None:
    # job set 1 / layout 1: 2 * (176/4 + 128/2) = 216
    assert horizon_length(176.0, 128.0, 4, 2, 2.0) == pytest.approx(216.0)


# --- anchors and feasibility margins ---------------------------------------------------------
def test_anchor_bounds_are_valid_and_ordered() -> None:
    for inst in all_instances():
        a = compute_anchors(inst)
        assert a.cmax_lb > 0
        assert a.cmax_ub == inst.deadline
        assert a.ecost_lb < a.ecost_ub
        assert a.kwh_lb <= a.kwh_ub
        assert a.charge_blocks_lb <= a.charge_blocks_ub


def test_every_instance_is_deadline_admissible() -> None:
    """ROADMAP.md §3.0(1) / risk R12: a margin below 1 is a generation defect."""
    for inst in all_instances():
        _, admissible, _ = certify(inst)
        assert admissible, f"{inst.instance_id} has no deadline-feasible schedule"


def test_lambda_controls_whether_the_deadline_binds() -> None:
    """Erratum A31: ``lambda`` is a *feasibility* parameter, not only a tariff-shaping one.

    At the published ``lambda = 2`` the deadline is admissible everywhere but binding
    almost nowhere; tightening it makes the constraint real.  This is a reported finding,
    which is why the ablation is load-bearing.
    """
    loose = sum(certify(build_instance(j, l, lam=2.0))[2] for j in range(1, 11) for l in (1, 2, 3, 4))
    tight = sum(certify(build_instance(j, l, lam=1.25))[2] for j in range(1, 11) for l in (1, 2, 3, 4))
    assert tight > loose
