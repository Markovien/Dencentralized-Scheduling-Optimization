"""The units contract.

Implements ROADMAP.md §3.0 ("Units contract"), resolving findings F8, A7, A24 and A25 of
the manuscript audit.  Every physical quantity in this project is expressed in exactly one
unit, declared here once, and every conversion goes through this module.

The three reconciliations the audit demands, and how each is resolved
--------------------------------------------------------------------

(i) *Voltage and charger power.*  The draft states a 100 Ah battery, a 5-minute recharge
    block restoring 20 Ah, and a 1.5 kW charger, without ever stating a bus voltage.  No
    voltage reconciles those three numbers: 20 Ah in 300 s at 1.5 kW would require a
    6.25 V bus.  We declare the nominal bus voltage ``U_NOM_V = 48`` (standard for
    industrial AIVs), and adopt the roadmap's resolution: fast charging takes the pack
    from 0 % to 80 % SoC in 25 minutes, linearly.  That fixes the block at

        80 Ah / 25 min = 3.2 Ah/min  =>  16 Ah per 5-minute block,

    and the implied charger power at

        16 Ah x 48 V = 0.768 kWh in 5 min  =>  9.216 kW.

    The 1.5 kW figure of the draft is therefore *not* used; it is inconsistent with the
    recharge block the draft itself specifies, and the block is the quantity the schedule
    actually depends on.

(ii) *Depletion-rate units.*  Read literally as mA/s against a 60 Ah usable window, the
    draft's 1/3/7 rates would take a year to drain the battery and the SoC floor could
    never bind -- the battery axis of the paper would be inert.  We adopt **mAh/s**, as
    the roadmap directs: at 7 mAh/s the usable 60 Ah drains in 143 min, which binds well
    inside every horizon of the benchmark.  ``tests/test_domain.py`` asserts the floor
    binds on a declared fraction of instances; if that test fails, the parameters are
    wrong, not the code.

(iii) *``B_needed`` dimensional mismatch.*  The draft adds a bare ``20`` (Ah, the SoC
    floor) to terms in rate x time (mAh).  Everything here is in **mAh**; the floor is
    20 000 mAh.

Canonical units
---------------
============  =========================================================
time          minutes (float; all benchmark data are integer minutes)
energy        kWh
money         EUR
tariff price  EUR/kWh
charge        mAh
power         kW
============  =========================================================
"""

from __future__ import annotations

from typing import Final

# --- electrical ------------------------------------------------------------------------
U_NOM_V: Final[float] = 48.0
"""Nominal battery-bus voltage [V].  Declared, not derived: reconciliation (i)."""

MAH_PER_AH: Final[float] = 1000.0

SECONDS_PER_MINUTE: Final[float] = 60.0
MINUTES_PER_HOUR: Final[float] = 60.0


def mah_to_kwh(mah: float, voltage_v: float = U_NOM_V) -> float:
    """Convert a charge quantity in mAh to energy in kWh at the nominal bus voltage.

    Parameters
    ----------
    mah:
        Charge [mAh].
    voltage_v:
        Bus voltage [V].  Defaults to :data:`U_NOM_V`.

    Returns
    -------
    float
        Energy [kWh].
    """
    return mah * voltage_v / 1.0e6


def kwh_to_mah(kwh: float, voltage_v: float = U_NOM_V) -> float:
    """Inverse of :func:`mah_to_kwh`."""
    return kwh * 1.0e6 / voltage_v


def rate_mah_per_second_to_per_minute(rate_mah_s: float) -> float:
    """Convert a depletion rate given in mAh/s (reconciliation (ii)) to mAh/min."""
    return rate_mah_s * SECONDS_PER_MINUTE


def energy_kwh(power_kw: float, duration_min: float) -> float:
    """Energy [kWh] drawn by a constant load of ``power_kw`` for ``duration_min`` minutes."""
    return power_kw * duration_min / MINUTES_PER_HOUR


# --- the reconciled charging block -------------------------------------------------------
FAST_CHARGE_FULL_MINUTES: Final[float] = 25.0
"""Minutes to take the pack from 0 % to 80 % SoC under fast charging (roadmap §3.0(i))."""

FAST_CHARGE_SOC_SPAN: Final[float] = 0.80
"""Fraction of capacity covered by :data:`FAST_CHARGE_FULL_MINUTES` (0 % -> 80 %)."""


def charge_rate_mah_per_min(capacity_mah: float) -> float:
    """Linear fast-charging rate [mAh/min] implied by the 0 %->80 %-in-25-min assumption."""
    return capacity_mah * FAST_CHARGE_SOC_SPAN / FAST_CHARGE_FULL_MINUTES


def charger_power_kw(capacity_mah: float, voltage_v: float = U_NOM_V) -> float:
    """Charger power [kW] implied by the reconciled charging rate.

    This *replaces* the draft's 1.5 kW figure, which cannot be reconciled with the
    recharge block the draft specifies (reconciliation (i)).
    """
    rate_mah_min = charge_rate_mah_per_min(capacity_mah)
    kwh_per_min = mah_to_kwh(rate_mah_min, voltage_v)
    return kwh_per_min * MINUTES_PER_HOUR
