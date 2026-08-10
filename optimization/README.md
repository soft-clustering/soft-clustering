# SCPP optimization study

A scientific software engineering study of the algorithm implementations in
SCPP: audit, profile, optimize, verify, benchmark, report.

Nothing in this directory is part of the distributed package. It is not on the
production import path, and `soft_clustering/` never imports from here.

## Layout

```text
optimization/
├── README.md                  this file
├── inventory.md               generated audit of all 40 algorithms
├── harness.py                 shared registry of runnable, scalable configurations
├── audit_static.py            AST audit: loop structure and numerical call sites
├── snapshot_originals.py      freezes pre-optimization code as the reference build
├── run_case.py                fits one configuration in isolation, emits JSON
├── sweep.py                   drives run_case.py with per-case timeouts
├── compare.py                 optimized vs reference correctness comparison
├── analyze.py                 raw measurements -> results.csv/json -> tables + figures
├── make_inventory.py          regenerates inventory.md from measured data
│
├── original/scpp_original/    preserved reference implementations (importable)
├── profiling/                 per-algorithm profiling reports
├── correctness/               correctness comparison output
├── benchmarks/
│   ├── raw/*.jsonl            every individual measurement
│   ├── results.csv            consolidated, machine-readable
│   └── results.json           consolidated + correctness summary
├── figures/                   PNG + vector PDF, all regenerated from results.csv
└── reports/
    ├── optimization_report.md the study
    └── tables/                Markdown + LaTeX tables
```

## Reproducing the study

All commands run from the repository root. `PY` is any interpreter with the
`dev` extra installed (`pip install -e ".[dev]"`).

```bash
# 1. Static audit of every algorithm module
python optimization/audit_static.py

# 2. Baseline survey across every runnable estimator (n = 200)
python optimization/sweep.py --out optimization/benchmarks/raw/smoke.jsonl \
       --n 200 --timeout 45

# 3. Detailed profiling of the slowest
python optimization/sweep.py --out optimization/benchmarks/raw/profile_top.jsonl \
       --cases SoftDBSCANGM,MBMM,KFCCL,KMART --n 200 --profile --memory --timeout 180

# 4. Freeze the pre-optimization code as the reference build
python optimization/snapshot_originals.py _softdbscangm _mbmm --from-git <pre-opt-ref>

# 5. Correctness: optimized vs reference
python optimization/compare.py SoftDBSCANGM --sizes 60,120,240 --seeds 0,1,2 \
       --out optimization/correctness/softdbscangm.json
python optimization/compare.py MBMM --sizes 200,600 --seeds 0,1,2 \
       --out optimization/correctness/mbmm.json

# 6. Paired benchmarks (--module selects the build)
python optimization/sweep.py --out optimization/benchmarks/raw/sdbg_original.jsonl \
       --cases SoftDBSCANGM --sizes 60,120,240 --repeats 3 --warmup 1 --memory \
       --module scpp_original --timeout 600
python optimization/sweep.py --out optimization/benchmarks/raw/sdbg_optimized.jsonl \
       --cases SoftDBSCANGM --sizes 60,120,240,480,960 --repeats 3 --warmup 1 --memory

# 7. Consolidate, and regenerate every table and figure
python optimization/analyze.py
python optimization/make_inventory.py
```

Step 7 rebuilds `results.csv`, `results.json`, every table under
`reports/tables/` and every figure under `figures/` from the raw measurements,
so no artefact can drift from the data it describes.

## The reference build

`optimization/original/scpp_original/` holds the algorithm modules exactly as
they stood before optimization, plus the unmodified `_base.py` they depend on.
It is a standalone importable package:

```python
import sys; sys.path.insert(0, "optimization/original")
import scpp_original
reference = scpp_original.SoftDBSCANGM()
```

It is never imported by `soft_clustering/`, and `tests/test_optimization_equivalence.py`
asserts that separation.

## Regression guard

`tests/test_optimization_equivalence.py` fits each optimized estimator
alongside its preserved reference and requires the outputs to agree. It runs as
part of the normal suite and skips cleanly when the reference build is absent
(for example when testing an installed wheel).

## Status

See `inventory.md` for per-algorithm coverage and
`reports/optimization_report.md` for the results, including an explicit account
of what was **not** optimized and why.
