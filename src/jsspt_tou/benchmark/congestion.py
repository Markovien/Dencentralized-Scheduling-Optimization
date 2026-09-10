"""The N5 congestion instance: an in-domain case with an **empty core**.

Result **N5** of Paper A (ROADMAP.md §3.4).  The instance is deliberately minimal so the
arithmetic can be checked by hand, and it is used as a *regression test*: if a future change
to the simulator or the characteristic function made its core non-empty, something has
broken.

The mechanism
-------------
Two robots, **one** charging station, and exactly one cheap tariff window long enough for
**one** charging session.

* under the reference policy both robots charge in the expensive band;
* a single coordinating robot shifts its session into the cheap window and saves;
* when *both* coordinate, only one of them can still have the window -- the second gains
  nothing.

So the second robot's marginal saving is smaller in the larger coalition, i.e. the cost
function is **supermodular**, the savings game is concave, and

    v({V1}) = v({V2}) = v({V1,V2}) = x  >  0
    core:  x1 + x2 = x,  x1 >= x,  x2 >= x   ==>   empty.

The Shapley value ``(x/2, x/2)`` is not in it.  The mechanism is general: *any* scarce
shared resource that saturates converts diminishing returns into congestion, and congestion
is supermodular.  The charger is the obvious one here; a bottleneck machine behaves the
same way.  This is not a defect to engineer around -- it is a structural property of the
problem, and it is why Paper A's headline stability result is the **certified least-core
radius** (T5) rather than an unconditional core theorem, and why submodularity survives
only as the conditional proposition **P5** on the congestion-free sub-class.

Machine power is set to zero in this instance so that *all* energy is charging energy and
the congestion mechanism is visible in isolation.  It is an isolating construction, not a
claim about real shops; the same phenomenon is measured on the unmodified Bilge--Ulusoy
instances in experiment E3.
"""

from __future__ import annotations

from typing import Final

from jsspt_tou.domain.battery import BatterySpec
from jsspt_tou.domain.instance import Instance
from jsspt_tou.domain.tou import TariffPeriod, TariffProfile

CHEAP_START: Final[float] = 30.0
CHEAP_LENGTH: Final[float] = 15.0
CHEAP_END: Final[float] = CHEAP_START + CHEAP_LENGTH
CHEAP_PRICE: Final[float] = 0.06
PEAK_PRICE: Final[float] = 0.22
HORIZON: Final[float] = 160.0
START_SOC_MAH: Final[float] = 24_000.0


def congestion_battery() -> BatterySpec:
    """A battery tuned so that one visit to the charger is exactly one block.

    ``ceiling - soc_at_charger`` is comfortably below one block's charge for every arrival
    time the instance can produce, so the number of blocks per visit is 1 regardless of how
    long a robot has queued.  That removes the only nuisance variable and leaves the
    *timing* of the single session as the whole decision -- which is the point.
    """
    return BatterySpec(
        capacity_mah=100_000.0,
        floor_mah=20_000.0,
        ceiling_mah=40_000.0,
        start_mah=START_SOC_MAH,
        idle_mah_min=60.0,
        empty_mah_min=180.0,
        loaded_mah_min=420.0,
        charge_rate_mah_min=3_200.0,
        charge_block_min=CHEAP_LENGTH,
        charger_power_kw=9.216,
    )


def congestion_tariff() -> TariffProfile:
    """One cheap window of exactly one charging block, in an otherwise expensive horizon."""
    return TariffProfile(
        periods=(
            TariffPeriod(0.0, CHEAP_START, PEAK_PRICE),
            TariffPeriod(CHEAP_START, CHEAP_END, CHEAP_PRICE),
            TariffPeriod(CHEAP_END, HORIZON, PEAK_PRICE),
        ),
        horizon=HORIZON,
    )


def congestion_instance(
    processing: float = 20.0, travel: float = 10.0, n_chargers: int = 1
) -> Instance:
    """Two robots, two single-operation jobs, one charger, one cheap window.

    The initial SoC is set so that **each robot must charge exactly once**, and the cheap
    window is exactly one charging block long, so the two robots' sessions cannot both fit
    inside it.  Everything else is symmetric, which is what makes the resulting TU game
    checkable by hand.
    """
    sigma = [
        [0.0, travel, travel],
        [travel, 0.0, travel],
        [travel, travel, 0.0],
    ]
    return Instance.build(
        instance_id="N5-congestion" if n_chargers == 1 else "N5-relaxed",
        n_machines=2,
        n_robots=2,
        sigma=sigma,
        job_ops=[[(1, processing)], [(2, processing)]],
        battery=congestion_battery(),
        n_chargers=n_chargers,
        machine_power_kw=0.0,
        tariff=congestion_tariff(),
    )


def relaxed_instance(processing: float = 20.0, travel: float = 10.0) -> Instance:
    """The same instance with **two** chargers -- the congestion-free control.

    The only difference is ``K_CH``.  Both robots can then take the cheap window, marginal
    savings stop shrinking, and the core becomes non-empty.  Running the pair side by side
    is what isolates *congestion* as the cause rather than anything else about the instance,
    and it is the empirical content of proposition **P5**.
    """
    return congestion_instance(processing, travel, n_chargers=2)
