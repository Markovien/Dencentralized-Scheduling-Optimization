PY ?= python3
export PYTHONPATH := src

.PHONY: env test test-all typecheck bench experiments exact n5 numbers figures claims check paper all clean

env:            ## install runtime + dev dependencies
	$(PY) -m pip install -e ".[dev]"

test:           ## fast suite (skips the CP-SAT solves)
	$(PY) -m pytest -q tests -m "not slow"

test-all:       ## everything, including the flat-tariff provenance regression
	$(PY) -m pytest -q tests

typecheck:
	$(PY) -m mypy --strict src

bench:          ## (re)generate the BU40 benchmark files and report feasibility margins
	$(PY) -m jsspt_tou.benchmark.bilge_ulusoy --out results/instances

experiments:    ## E1, E3, E4, E5, E11, E13 over the full benchmark
	$(PY) experiments/run_all.py --out results

exact:          ## E2: where the exact model stops closing
	$(PY) experiments/run_exact.py --out results

n5:             ## the congestion counterexample and its control
	$(PY) experiments/run_n5.py --out results

numbers:        ## merge every registry shard -> paper_A/numbers.tex
	$(PY) experiments/make_numbers.py

figures:        ## rebuild figures from results/
	$(PY) experiments/make_figures.py

claims:         ## regenerate the evidence map paper_A/CLAIMS.md
	$(PY) experiments/make_claims.py

check: claims   ## claim tracing, citations and structure
	$(PY) experiments/check_paper.py

paper:
	cd paper_A && latexmk -pdf main.tex

all: test bench experiments exact n5 numbers figures claims check

clean:
	rm -rf results/*.parquet results/numbers*.json paper_A/figures/*.pdf \
	       paper_A/numbers.tex .pytest_cache .mypy_cache
