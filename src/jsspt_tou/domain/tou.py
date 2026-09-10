"""Time-of-Use tariff profile and the scheduling horizon.

Implements the ToU profile of Paper A §3 (draft: "ToU pricing profile") and the horizon
formula, and fixes the regeneration rule that repairs erratum **A26** (seven rows of the
draft's appendix table end *after* their own stated horizon).

Horizon
-------
    H = lambda * ( (1/|M|) * sum_{i,k} p_{i,k}  +  (1/|V|) * sum_{i,k} tau_{i,k} )

with ``lambda = 2``.  Recovering ``|V| = 2`` from this formula on job set 1 / layout 1
(2*(176/4 + 128/2) = 216.0, matching the published EX11 horizon) is what resolves erratum
**A23** -- the fleet size is stated nowhere in the draft.

Under ROADMAP.md §3.0 the horizon is a **hard deadline**, not a display window:
no schedule may complete after it.  :mod:`jsspt_tou.domain.feasibility` and every model
enforce ``C_max <= deadline``.

Regeneration rule for the six tariff periods (erratum A26)
----------------------------------------------------------
The published profile has six periods with prices (0.14, 0.22, 0.14, 0.22, 0.14, 0.06)
EUR/kWh and length fractions ``(3, 5, 3, 3, 2, 8) / 24`` of the horizon.  We regenerate as

    len_k = round_half_up(H * f_k)   for k = 1..5,      period 6 = [b_5, H).

with round-half-up (not banker's rounding: three published rows -- EX64, EX72, EX74 --
land exactly on a half-minute and the published table rounds them up).

Periods therefore tile ``[0, H)`` exactly and the last period **never ends after H**.
``tests/test_domain.py`` checks the regenerated boundaries of periods 1--5 against all 40
published rows (the transcription check demanded by ROADMAP.md §5.1) and asserts the
last-period property that the seven defective rows violate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

LAMBDA_DEFAULT: Final[float] = 2.0
"""Horizon scaling factor.  A *feasibility* parameter, not only a tariff-shaping one
(erratum A31): it decides whether an instance admits any deadline-feasible schedule."""

PERIOD_FRACTIONS: Final[tuple[float, ...]] = (
    3.0 / 24.0,
    5.0 / 24.0,
    3.0 / 24.0,
    3.0 / 24.0,
    2.0 / 24.0,
    8.0 / 24.0,
)
PERIOD_PRICES: Final[tuple[float, ...]] = (0.14, 0.22, 0.14, 0.22, 0.14, 0.06)


def _round_half_up(x: float) -> int:
    """Round half away from zero, as the published appendix table does."""
    return int(math.floor(x + 0.5))


@dataclass(frozen=True, slots=True)
class TariffPeriod:
    """A half-open tariff interval ``[start, end)`` at a constant price."""

    start: float
    end: float
    price: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TariffProfile:
    """A piecewise-constant ToU price curve tiling ``[0, horizon)``."""

    periods: tuple[TariffPeriod, ...]
    horizon: float

    # -- construction ---------------------------------------------------------------------
    @staticmethod
    def generate(
        horizon: float,
        fractions: Sequence[float] = PERIOD_FRACTIONS,
        prices: Sequence[float] = PERIOD_PRICES,
    ) -> "TariffProfile":
        """Regenerate the six-period profile for a horizon (erratum A26 rule, above)."""
        if len(fractions) != len(prices):
            raise ValueError("fractions and prices must have the same length")
        bounds: list[float] = [0.0]
        for frac in fractions[:-1]:
            nxt = bounds[-1] + float(_round_half_up(horizon * frac))
            bounds.append(min(nxt, horizon))
        bounds.append(horizon)
        periods = tuple(
            TariffPeriod(start=bounds[k], end=bounds[k + 1], price=float(prices[k]))
            for k in range(len(prices))
            if bounds[k + 1] > bounds[k]
        )
        return TariffProfile(periods=periods, horizon=horizon)

    # -- queries --------------------------------------------------------------------------
    def price_at(self, t: float) -> float:
        """Price [EUR/kWh] in force at time ``t``.

        Times at or beyond the horizon are priced at the last period's rate; this can only
        be reached by an infeasible schedule (``C_max > H``), which the deadline constraint
        forbids and the penalty term of :mod:`jsspt_tou.domain.objective` charges for.
        """
        for period in self.periods:
            if period.start <= t < period.end:
                return period.price
        return self.periods[-1].price

    def cost(self, power_kw: float, start: float, end: float) -> float:
        """Exact ToU cost [EUR] of a constant load over ``[start, end)``.

        Implements the piecewise-constant integral

            cost = integral_{start}^{end} P * c(t) dt

        of ROADMAP.md §3.0 ("Energy cost (corrected)"), used for both machine processing
        (resolving F6/A8 -- the draft's centralised model priced only robot charging) and
        robot charging.
        """
        if end <= start:
            return 0.0
        total = 0.0
        for period in self.periods:
            lo = max(start, period.start)
            hi = min(end, period.end)
            if hi > lo:
                total += power_kw * (hi - lo) / 60.0 * period.price
        # Any residue beyond the horizon is priced at the final period's rate.
        if end > self.horizon:
            lo = max(start, self.horizon)
            total += power_kw * (end - lo) / 60.0 * self.periods[-1].price
        return total

    @property
    def min_price(self) -> float:
        return min(p.price for p in self.periods)

    @property
    def max_price(self) -> float:
        return max(p.price for p in self.periods)

    def boundaries(self) -> tuple[float, ...]:
        """Period start times -- the natural ToU-aligned delay grid for machine actions."""
        return tuple(p.start for p in self.periods)

    def as_rows(self) -> list[dict[str, float]]:
        """Rows for the published appendix table (inclusive display ends)."""
        rows: list[dict[str, float]] = []
        for period in self.periods:
            rows.append(
                {
                    "start": float(period.start),
                    "end": float(math.ceil(period.end) - 1),
                    "price": period.price,
                }
            )
        return rows


def horizon_length(
    total_processing: float,
    total_transport: float,
    n_machines: int,
    n_robots: int,
    lam: float = LAMBDA_DEFAULT,
) -> float:
    """Scheduling horizon ``H`` [min] (Paper A Eq. for the ToU horizon).

    Parameters
    ----------
    total_processing:
        Sum of all operation processing times.
    total_transport:
        Sum of all transport-task durations, including the final return to L/U.
    n_machines, n_robots:
        Fleet sizes.  ``n_robots = 2`` for the Bilge--Ulusoy suite; see erratum A23.
    lam:
        Scaling factor, ``lambda = 2`` in the published profile.
    """
    return lam * (total_processing / n_machines + total_transport / n_robots)
