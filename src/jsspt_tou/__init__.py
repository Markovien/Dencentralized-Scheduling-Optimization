"""jsspt-tou: decentralised scheduling of machines and battery-constrained AIVs under ToU tariffs.

Reference implementation for *Cooperative and non-cooperative game-theoretic scheduling of
production machines and battery-constrained autonomous vehicles under time-of-use tariffs*
(target venue: Engineering Applications of Artificial Intelligence).

Layer map (ROADMAP.md §4.1):
    domain/       instance, tariff, battery, normalisation anchors, objective
    simulator/    event-driven engine shared by every method
    exact/        EX-CP (adapter over external/jsspt_cpsat_original)
    baselines/    dispatching rules and charge policies
    game/         M1  -- non-cooperative potential game
    cooperative/  M2  -- coalitional game, the new contribution
    benchmark/    BU40 transcription and ToU regeneration
    analysis/     metrics, statistics, numbers.json registry
"""

__version__ = "0.1.0"
