PY ?= python3

.PHONY: env test typecheck bench experiments figures paper all clean

env:            ## install runtime + dev dependencies
	$(PY) -m pip install -e ".[dev]"

test:           ## full test suite (must be green before any delivery)
	$(PY) -m pytest -q tests

typecheck:
	$(PY) -m mypy --strict src

bench:          ## (re)generate the BU40 benchmark files
	$(PY) -m jsspt_tou.benchmark.bilge_ulusoy --out results/instances

experiments:    ## run every experiment block into results/
	$(PY) experiments/run_all.py

figures:        ## rebuild figures and LaTeX tables from results/
	$(PY) experiments/make_figures.py

paper:
	cd paper_A && latexmk -pdf main.tex

all: test bench experiments figures

clean:
	rm -rf results/*.parquet results/figures results/tables .pytest_cache
