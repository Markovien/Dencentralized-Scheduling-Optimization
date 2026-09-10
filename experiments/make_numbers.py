"""Merge every registry shard into ``numbers.json`` and emit ``paper_A/numbers.tex``.

Each experiment writes its own shard so blocks can be re-run independently; this collects
them into the single map the manuscript resolves against.  A key defined twice by two shards
is an error, not a silent overwrite -- two runs disagreeing about the same number is exactly
the situation claim tracing exists to catch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsspt_tou.analysis.registry import Entry, Registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("paper_A/numbers.tex"))
    args = parser.parse_args(argv)

    merged = Registry(args.results / "numbers.json")
    shards = sorted(args.results.glob("numbers*.json"))
    seen: dict[str, str] = {}
    for shard in shards:
        raw = json.loads(shard.read_text())
        for key, payload in raw.items():
            if key in seen and seen[key] != shard.name:
                raise KeyError(f"key {key!r} defined by both {seen[key]} and {shard.name}")
            seen[key] = shard.name
            merged.entries[key] = Entry(**payload)
    merged.write()
    merged.write_latex_macros(args.out)
    print(f"merged {len(shards)} shards -> {len(merged.entries)} keys -> {args.out}")
    missing = [k for k in merged.entries if merged.entries[k].value is None]
    if missing:
        print("WARNING: keys with no value:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
