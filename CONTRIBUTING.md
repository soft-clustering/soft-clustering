# Contributing to Soft Clustering

Thank you for considering contributing to **Soft Clustering**! 🎉

All contributions are welcome, including:

- Bug reports and fixes
- New features or algorithm implementations
- Documentation improvements
- Example code or tutorials
- Performance optimizations

## How to Contribute

1. **Fork** the repository and create a new branch for your changes.
2. Make your changes following the existing code style.
3. Write or update tests.
4. Run the checks below; they are the same ones CI runs.
5. Submit a **Pull Request** with a clear description of your changes.

## Development Setup

```bash
git clone https://github.com/soft-clustering/soft-clustering.git
cd soft-clustering
pip install -e ".[dev,deep]"   # drop [deep] if you do not need the PyTorch models
```

The extras are separated by what they are needed *for*:

| Extra | Provides | Needed for |
| --- | --- | --- |
| *(none)* | numpy, scipy, typeguard | Fitting any non-deep estimator |
| `deep` | torch, torch_geometric | CDCGS, DMoN, MMSB, NOCD, RDFKC |
| `bench` | scikit-learn, pandas, psutil, tabulate | Running `soft_clustering.benchmarking` |
| `docs` | sphinx, sphinx-rtd-theme, myst-parser | Building the documentation |
| `dev` | pytest, pytest-cov, matplotlib, and the `bench` set | Running the test suite |

Nothing outside the base dependencies may be imported at module level in
`soft_clustering/` without a guard — the package must stay importable on a bare
install. See `soft_clustering/benchmarking/_optional.py` for the pattern.

## Running the checks

```bash
pytest                                                    # tests
pytest --cov=soft_clustering --cov-report=term-missing    # tests with coverage
ruff check soft_clustering tests example tools            # lint
black --check soft_clustering tests example tools         # formatting
sphinx-build -b html -W docs/source docs/_build/html      # docs, warnings are errors
```

Notes:

- CI pins `ruff` and `black` to the versions in `.github/workflows/lint.yml`.
  Use those versions locally so formatting does not ping-pong.
- The docs build with `-W`, so a broken reference or a skipped heading level
  fails the build. Run it before opening a PR.
- Coverage is reported to Codecov. New or modified lines are expected to be
  covered; see `codecov.yml` for the thresholds.
- Many tests use fixtures from `tests/conftest.py`.
- The five deep-learning estimators are skipped unless the `deep` extra is
  installed.

## Adding an estimator

Estimators inherit from `BaseSoftClusterer` (`soft_clustering/_base.py`), which
reconciles the differing fit signatures and publishes canonical fitted
attributes. Read the module docstring there first — it is the contract.

After `fit`, every estimator exposes `memberships_` of shape
`(n_samples, n_clusters)`, `labels_`, `centers_` (or `None`), and `n_clusters`.
You do not write that plumbing; you declare where your solution lives:

```python
@typechecked
class MyMethod(BaseSoftClusterer):
    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _partition_constrained = True   # False for possibilistic / typicality methods
```

If your estimator stores its memberships under a name not already in
`BaseSoftClusterer._membership_attrs`, add it there — that tuple is the single
registry, and the benchmarking code reads it rather than keeping its own copy.

Then:

1. Register the class in `_ESTIMATORS` in `soft_clustering/__init__.py`, and in
   `DEEP_ESTIMATORS` if it needs the `deep` extra.
2. Add an entry to `CASES` in `tests/test_protocol.py`; the shared conformance
   checks then apply automatically.
3. Add a page under `docs/source/` and list it in the `index.rst` toctree.
4. Add a runnable script under `example/`.
5. Cite the defining paper in the module docstring.

## Code Style

- `black` for formatting, `ruff` for linting; both are configured in
  `pyproject.toml` and enforced in CI.
- Follow PEP 8 where the algorithm code allows. Single-letter names matching a
  paper's notation are fine — that exception is already configured in ruff.
- Docstrings state the shape and semantics of every fitted attribute.

## Reporting Bugs

Please open an issue using the **Bug report** form, which asks for the version,
environment, and a minimal reproducible example. For security problems, do not
open a public issue — see [SECURITY.md](.github/SECURITY.md).

---

Happy coding! We look forward to your contributions.
