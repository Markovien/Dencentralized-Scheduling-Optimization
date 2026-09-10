"""Battery model: SoC dynamics, depletion rates, charging blocks, ``B_needed``.

Implements the battery assumptions of Paper A §3 with the units contract of
:mod:`jsspt_tou.domain.units` applied (errata A6, A7, A24, A25).

Everything is in **mAh**.  The three depletion rates are read as **mAh/s** and stored as
mAh/min.  The SoC ceiling (80 %) is *enforced* here, not merely asserted -- erratum A6
notes the draft states the [20 %, 80 %] window but constrains only the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from jsspt_tou.domain.units import (
    MAH_PER_AH,
    charge_rate_mah_per_min,
    charger_power_kw,
    rate_mah_per_second_to_per_minute,
)

CHARGE_BLOCK_MINUTES: Final[float] = 5.0
"""Length of one charging block [min].  The draft's block; its *content* is re-derived."""


@dataclass(frozen=True, slots=True)
class BatterySpec:
    """Battery and charger parameters, in the canonical units.

    Attributes
    ----------
    capacity_mah:
        Full capacity (100 % SoC).  100 Ah = 100 000 mAh in the published setting.
    floor_mah, ceiling_mah:
        The enforced [20 %, 80 %] operating window.
    start_mah:
        Initial SoC of every robot at t = 0.
    idle_mah_min, empty_mah_min, loaded_mah_min:
        Depletion rates while idle, travelling empty, and travelling loaded.
    charge_rate_mah_min:
        Linear fast-charging rate, from the 0 %->80 %-in-25-min reconciliation.
    charge_block_min:
        Duration of one indivisible charging block.
    charger_power_kw:
        Implied charger power, used to price charging against the tariff.
    """

    capacity_mah: float
    floor_mah: float
    ceiling_mah: float
    start_mah: float
    idle_mah_min: float
    empty_mah_min: float
    loaded_mah_min: float
    charge_rate_mah_min: float
    charge_block_min: float
    charger_power_kw: float

    # -- construction ---------------------------------------------------------------------
    @staticmethod
    def published(
        capacity_ah: float = 100.0,
        floor_frac: float = 0.20,
        ceiling_frac: float = 0.80,
        idle_mah_s: float = 1.0,
        empty_mah_s: float = 3.0,
        loaded_mah_s: float = 7.0,
    ) -> "BatterySpec":
        """The paper's parameterisation, with the units contract applied.

        The rates are the draft's 1 / 3 / 7, read as **mAh per second** (erratum A24).
        The charging block and charger power are *derived*, not taken from the draft's
        mutually inconsistent 20 Ah / 5 min / 1.5 kW triple (erratum A7).
        """
        capacity_mah = capacity_ah * MAH_PER_AH
        rate = charge_rate_mah_per_min(capacity_mah)
        return BatterySpec(
            capacity_mah=capacity_mah,
            floor_mah=floor_frac * capacity_mah,
            ceiling_mah=ceiling_frac * capacity_mah,
            start_mah=ceiling_frac * capacity_mah,
            idle_mah_min=rate_mah_per_second_to_per_minute(idle_mah_s),
            empty_mah_min=rate_mah_per_second_to_per_minute(empty_mah_s),
            loaded_mah_min=rate_mah_per_second_to_per_minute(loaded_mah_s),
            charge_rate_mah_min=rate,
            charge_block_min=CHARGE_BLOCK_MINUTES,
            charger_power_kw=charger_power_kw(capacity_mah),
        )

    # -- derived quantities ----------------------------------------------------------------
    @property
    def usable_mah(self) -> float:
        """Charge available between the ceiling and the floor."""
        return self.ceiling_mah - self.floor_mah

    @property
    def block_mah(self) -> float:
        """Charge restored by one block."""
        return self.charge_rate_mah_min * self.charge_block_min

    @property
    def block_kwh(self) -> float:
        """Energy drawn from the grid by one block [kWh]."""
        return self.charger_power_kw * self.charge_block_min / 60.0

    def drain(self, soc_mah: float, minutes: float, mode: str) -> float:
        """SoC after ``minutes`` in ``mode`` in {'idle', 'empty', 'loaded'}."""
        rate = {
            "idle": self.idle_mah_min,
            "empty": self.empty_mah_min,
            "loaded": self.loaded_mah_min,
        }[mode]
        return soc_mah - rate * minutes

    def blocks_to_full(self, soc_mah: float) -> int:
        """Number of blocks that take ``soc_mah`` up to the ceiling (erratum A6)."""
        deficit = max(0.0, self.ceiling_mah - soc_mah)
        return int(math.ceil(deficit / self.block_mah - 1e-9))

    def charge(self, soc_mah: float, n_blocks: int) -> float:
        """SoC after ``n_blocks`` charging blocks, clipped at the 80 % ceiling."""
        return min(self.ceiling_mah, soc_mah + n_blocks * self.block_mah)


def b_needed(
    spec: BatterySpec,
    travel_to_pickup: float,
    wait_at_pickup: float,
    loaded_travel: float,
    travel_to_charger: float,
) -> float:
    """Minimum SoC [mAh] required to commit to a transport action.

    Implements the draft's ``B_needed`` predicate with its dimensional error repaired
    (erratum A25): the SoC floor enters as 20 000 mAh, not as a bare ``20``.

        B_needed = floor
                 + empty_rate  * travel_to_pickup
                 + idle_rate   * wait_at_pickup
                 + loaded_rate * loaded_travel
                 + empty_rate  * travel_from_destination_to_charger

    The final term is what makes the predicate *safe*: after finishing the task the robot
    must still be able to reach the charger without crossing the floor.
    """
    return (
        spec.floor_mah
        + spec.empty_mah_min * travel_to_pickup
        + spec.idle_mah_min * max(0.0, wait_at_pickup)
        + spec.loaded_mah_min * loaded_travel
        + spec.empty_mah_min * travel_to_charger
    )
