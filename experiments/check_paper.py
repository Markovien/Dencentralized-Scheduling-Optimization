"""Static checks on the manuscript that a LaTeX build cannot do for us.

Three of them enforce contracts from ROADMAP.md §6.4:

* **R-5 claim tracing** -- every ``\\jnum{key}`` resolves against ``numbers.json``, and no
  results sentence contains a literal numeral that should have been a key.
* **citations** -- every ``\\cite``/``\\citep``/``\\citet`` key exists in a bibliography file.
* **structure** -- environments and references balance, and no label is defined twice or
  referenced without being defined.

Run ``python experiments/check_paper.py`` before delivery.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

JNUM = re.compile(r"\\jnum\{([^}]+)\}")
CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|eqref)\{([^}]+)\}")
BIBKEY = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
BEGIN = re.compile(r"\\begin\{(\w+\*?)\}")
END = re.compile(r"\\end\{(\w+\*?)\}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=Path("paper_A/main.tex"))
    parser.add_argument("--numbers", type=Path, default=Path("results/numbers.json"))
    parser.add_argument("--bib", type=Path, nargs="*", default=[
        Path("paper_A/cas-refs.bib"), Path("paper_A/added-refs.bib")
    ])
    args = parser.parse_args(argv)

    text = args.tex.read_text()
    problems: list[str] = []

    # --- R-5: claim tracing -------------------------------------------------------------
    used = set(JNUM.findall(text))
    if args.numbers.exists():
        available = set(json.loads(args.numbers.read_text()))
    else:
        available = set()
        problems.append(f"{args.numbers} does not exist -- run the experiments first")
    missing = sorted(used - available)
    if missing:
        problems.append(f"{len(missing)} \\jnum keys with no registry entry: {missing}")
    unused = sorted(available - used)

    # literal numerals inside the results sections
    body = text.split(r"\section{Results}")[-1].split(r"\appendix")[0]
    body = re.sub(r"\\jnum\{[^}]+\}", "JNUM", body)
    body = re.sub(r"\\(?:label|ref|eqref|cite[tp]?)\*?\{[^}]*\}", "", body)
    body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", body, flags=re.S)
    suspicious = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or stripped.startswith("\\"):
            continue
        for match in re.finditer(r"(?<![\w\\{])\d+(?:\.\d+)?", stripped):
            token = match.group(0)
            # section numbers, footnote markers and the small integers used in prose
            # ("two vehicles", "$n=6$") are written as words or inside math; a bare decimal
            # or a number above 10 in a results sentence is what we are looking for.
            if "." in token or float(token) > 10:
                suspicious.append(f"{stripped[:90]!r} -> {token}")
    if suspicious:
        problems.append(
            "literal numerals in results prose (should be \\jnum keys):\n    "
            + "\n    ".join(suspicious[:20])
        )

    # --- citations ------------------------------------------------------------------------
    bibkeys: set[str] = set()
    for path in args.bib:
        if path.exists():
            bibkeys |= set(BIBKEY.findall(path.read_text()))
    cited: set[str] = set()
    for group in CITE.findall(text):
        cited |= {k.strip() for k in group.split(",")}
    unknown = sorted(cited - bibkeys)
    if unknown:
        problems.append(f"citations with no bibliography entry: {unknown}")

    # --- structure --------------------------------------------------------------------------
    begins = BEGIN.findall(text)
    ends = END.findall(text)
    for env in set(begins) | set(ends):
        if begins.count(env) != ends.count(env):
            problems.append(
                f"environment {env!r} unbalanced: {begins.count(env)} begin, {ends.count(env)} end"
            )
    if text.count("{") != text.count("}"):
        problems.append(f"brace imbalance: {text.count('{')} open, {text.count('}')} close")

    labels = LABEL.findall(text)
    dupes = sorted({l for l in labels if labels.count(l) > 1})
    if dupes:
        problems.append(f"duplicate labels: {dupes}")
    dangling = sorted(set(REF.findall(text)) - set(labels))
    if dangling:
        problems.append(f"references with no label: {dangling}")

    # --- report -----------------------------------------------------------------------------
    print(f"tex        : {args.tex} ({len(text.splitlines())} lines)")
    print(f"jnum keys  : {len(used)} used, {len(available)} available, {len(missing)} missing")
    print(f"citations  : {len(cited)} distinct, {len(unknown)} unknown")
    print(f"labels     : {len(labels)} defined, {len(dangling)} dangling references")
    if unused:
        print(f"note       : {len(unused)} registry keys not cited in the paper")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(" -", p)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
