"""``M2`` Layer 1 — decentralised coalition-structure generation by merge and split.

Implements Paper A §6, Layer 1 (ROADMAP.md §3.4).

The grand coalition requires all-to-all coordination, which contradicts the paper's own
decentralisation thesis and does not scale.  Layer 1 finds a *partition*
``CS = {S_1, ..., S_K}`` of ``N`` that captures most of the savings at bounded
communication.

Because congestion can make the grand coalition non-convex (result N5), coalition formation
here is **not merely a tractability device**: when a shared resource saturates, partitioning
can be genuinely better than the grand coalition, and experiment E4 tests exactly that.

Hedonic preference
------------------
Agent ``i`` prefers coalition ``S`` to ``S'`` iff ``phi_i(v|_S) > phi_i(v|_{S'})``, where
``v|_S(T) = c(0) - c(T)`` restricted to ``T subset S`` -- payoff-based hedonic preferences
over the Shapley allocation *within* each coalition.

Merge and split (Apt & Witzel; Saad et al.) terminates because the Pareto order over
partitions is acyclic: every accepted move strictly improves at least one member and harms
none, and no partition can recur.  The fixed point is ``D_hp``-stable: no group can
profitably merge and no coalition can profitably split.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

from jsspt_tou.cooperative.shapley import exact_shapley
from jsspt_tou.simulator.state import Player

ValueFn = Callable[[frozenset[Player]], float]


@dataclass(frozen=True, slots=True)
class CoalitionStructure:
    """A partition of the player set, with its realised value and communication cost."""

    blocks: tuple[frozenset[Player], ...]
    payoff: dict[Player, float]
    value: float
    messages: int
    merges: int
    splits: int

    @property
    def max_block_size(self) -> int:
        return max((len(b) for b in self.blocks), default=0)

    def label(self) -> str:
        return " | ".join(
            "{" + ",".join(str(p) for p in sorted(b)) + "}" for b in sorted(
                self.blocks, key=lambda b: sorted(b)[0] if b else Player("machine", 0)
            )
        )


def restricted_shapley(
    block: frozenset[Player], v: ValueFn
) -> dict[Player, float]:
    """Shapley payoffs of the sub-game ``v|_block``."""
    members = sorted(block)
    if not members:
        return {}
    return exact_shapley(members, lambda s: v(frozenset(s))).payoff


def merge_split(
    players: Sequence[Player],
    v: ValueFn,
    s_max: int = 6,
    max_rounds: int = 50,
) -> CoalitionStructure:
    """**M2a** — merge-and-split to a ``D_hp``-stable partition.

    Parameters
    ----------
    s_max:
        Scaling guard: coalitions larger than this are never formed.  Reported alongside
        the savings lost versus the grand coalition on instances small enough to compute
        both, so the guard is characterised rather than hidden.

    Returns
    -------
    CoalitionStructure
        Including the **message count** -- the quantitative evidence that the cooperative
        model is still decentralised, which is exactly what a referee will demand given the
        framing.  One message is counted per pairwise merge test and per split test, i.e.
        per bilateral negotiation actually conducted.
    """
    blocks: list[frozenset[Player]] = [frozenset({p}) for p in players]
    payoffs = {b: restricted_shapley(b, v) for b in blocks}
    messages = 0
    merges = 0
    splits = 0

    for _ in range(max_rounds):
        changed = False

        # MERGE: accept a union that is Pareto-improving for every member.
        for a, b in itertools.combinations(list(blocks), 2):
            if len(a) + len(b) > s_max:
                continue
            messages += 1
            union = a | b
            union_pay = restricted_shapley(union, v)
            if _pareto_improves(union_pay, {**payoffs[a], **payoffs[b]}):
                blocks.remove(a)
                blocks.remove(b)
                blocks.append(union)
                payoffs.pop(a, None)
                payoffs.pop(b, None)
                payoffs[union] = union_pay
                merges += 1
                changed = True
                break
        if changed:
            continue

        # SPLIT: accept a partition of one block that is Pareto-improving for every member.
        for block in list(blocks):
            if len(block) < 2:
                continue
            found = False
            for part in _binary_partitions(block):
                messages += 1
                new_pay: dict[Player, float] = {}
                for piece in part:
                    new_pay.update(restricted_shapley(piece, v))
                if _pareto_improves(new_pay, payoffs[block]):
                    blocks.remove(block)
                    payoffs.pop(block, None)
                    for piece in part:
                        blocks.append(piece)
                        payoffs[piece] = restricted_shapley(piece, v)
                    splits += 1
                    changed = found = True
                    break
            if found:
                break
        if not changed:
            break

    payoff = {p: val for block in blocks for p, val in payoffs[block].items()}
    return CoalitionStructure(
        blocks=tuple(blocks),
        payoff=payoff,
        value=sum(v(b) for b in blocks),
        messages=messages,
        merges=merges,
        splits=splits,
    )


def _pareto_improves(
    new: dict[Player, float], old: dict[Player, float], tol: float = 1e-9
) -> bool:
    """Weakly better for everyone and strictly better for at least one."""
    strict = False
    for p, val in new.items():
        prev = old.get(p, 0.0)
        if val < prev - tol:
            return False
        if val > prev + tol:
            strict = True
    return strict


def _binary_partitions(block: frozenset[Player]) -> list[tuple[frozenset[Player], ...]]:
    """All two-way splits of a block (the standard split neighbourhood)."""
    members = sorted(block)
    out: list[tuple[frozenset[Player], ...]] = []
    for size in range(1, len(members) // 2 + 1):
        for combo in itertools.combinations(members, size):
            left = frozenset(combo)
            right = block - left
            if not right:
                continue
            if size * 2 == len(members) and members[0] not in left:
                continue  # each bipartition once
            out.append((left, right))
    return out


def is_dhp_stable(
    structure: CoalitionStructure, v: ValueFn, s_max: int = 6
) -> bool:
    """Verify ``D_hp``-stability of a partition: no profitable merge and no profitable split."""
    payoffs = {b: restricted_shapley(b, v) for b in structure.blocks}
    for a, b in itertools.combinations(structure.blocks, 2):
        if len(a) + len(b) > s_max:
            continue
        if _pareto_improves(restricted_shapley(a | b, v), {**payoffs[a], **payoffs[b]}):
            return False
    for block in structure.blocks:
        for part in _binary_partitions(block):
            new_pay: dict[Player, float] = {}
            for piece in part:
                new_pay.update(restricted_shapley(piece, v))
            if _pareto_improves(new_pay, payoffs[block]):
                return False
    return True
