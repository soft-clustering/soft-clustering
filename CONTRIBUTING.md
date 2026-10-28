# Contributing to SCPP

Bug reports, fixes, new algorithms, documentation, examples and performance
work are all welcome.

## Workflow

1. Fork the repository and create a branch.
2. Make the change, with tests (see below for what an algorithm needs).
3. Run the checks below; they are the ones CI runs.
4. Open a pull request that says what changed and why. If any code is
   adapted from another project, say so, check that its license permits it,
   and keep its notice (see `THIRD_PARTY_NOTICES.md`). A change to what an
   algorithm computes must be described as such, with the evidence that the
   new behaviour is correct.

## Development setup

```bash
git clone https://github.com/soft-clustering/soft-clustering.git
cd soft-clustering
pip install -e ".[dev,deep]"   # drop deep if you do not need the PyTorch models
pip install pre-commit && pre-commit install   # optional: lint on commit
```

| Extra | Provides | Needed for |
| --- | --- | --- |
| *(none)* | numpy, scipy, scikit-learn, typeguard | every estimator except the deep ones |
| `deep` | torch | CDCGS, DMoN, NOCD, RDFKC |
| `bench` | pandas, psutil, tabulate | `soft_clustering.benchmarking` |
| `baselines` | scikit-fuzzy, evclust, networkx | external agreement tests |
| `docs` | sphinx, sphinx-rtd-theme, myst-parser | building the documentation |
| `dev` | pytest, pytest-cov, the `bench` and `baselines` sets, matplotlib | the test suite |

Nothing outside the base dependencies may be imported at module level in
`soft_clustering/` except in the four deep estimators, which the package
imports lazily. See `soft_clustering/benchmarking/_optional.py` for the
pattern.

## Checks

```bash
pytest                                                      # tests (deep tests skip without torch)
pytest --cov=soft_clustering --cov-report=term-missing      # with coverage
ruff check soft_clustering tests example tools benchmarks   # lint
black --check soft_clustering tests example tools benchmarks
mypy                                                        # the whole package, see pyproject.toml
sphinx-build -b html -W docs/source docs/_build/html        # docs; warnings are errors
python tools/algorithm_table.py --write                     # after editing the registry
```

CI pins ruff, black and mypy to the versions in `.github/workflows/lint.yml`;
`.pre-commit-config.yaml` uses the same ones.

## Adding an algorithm

Read the module docstring of `soft_clustering/_base.py` first: it is the
contract. In short, after `fit` every estimator exposes `memberships_`,
`labels_`, `centers_` (or `None`) and `n_clusters_`, and `fit_predict`
returns `memberships_`. You do not write that plumbing; you declare where the
solution lives.

```python
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._validation import check_array, check_generator, check_n_clusters


@typechecked
class MyMethod(BaseSoftClusterer):
    r"""One-line summary.

    The objective and update equations, in LaTeX. Parameters, Attributes,
    References (with DOI) and Notes (any deviation from the reference)
    sections in NumPy style.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)
    _partition_constrained = True    # False if rows need not sum to one
    _supports_out_of_sample = False  # True only with a genuine rule for new data

    def __init__(self, n_clusters: int = 3, max_iter: int = 100,
                 random_state: int | None = None):
        self.n_clusters = n_clusters      # store parameters unchanged;
        self.max_iter = max_iter          # validate them in fit
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X, estimator="MyMethod")
        K = check_n_clusters(self.n_clusters, X.shape[0], estimator="MyMethod")
        rng = check_generator(self.random_state)   # never the global RNG
        ...
        self.memberships_ = U
        self.centers_ = V
        return self
```

Rules that the test suite enforces:

- The cluster count is the constructor argument `n_clusters`.
- Every source of randomness is controlled by `random_state`; a seeded fit is
  reproducible, refits identically, and neither reads nor reseeds NumPy's
  (or PyTorch's) global random state.
- `fit` validates its input with `check_array` (NaN/inf raise) and never
  modifies it.
- `_partition_constrained` is truthful in both directions.

Then:

1. Register the class in `_ESTIMATORS` in `soft_clustering/__init__.py`, and in
   `DEEP_ESTIMATORS` if it needs PyTorch.
2. Describe it in `soft_clustering/_registry.py` (method, family, input,
   output kind, how K is obtained, reference, and an honest verification
   status), then run `python tools/algorithm_table.py --write`. The
   documentation page is generated from the registry and the docstring.
3. Add a case to `CASES` in `tests/test_protocol.py` (conformance) and to
   `tests/test_recovery.py` (planted-structure recovery).
4. Add `tests/test_<name>.py` testing the method's own equations and
   invariants, for example a monotone objective.
5. If an independent implementation exists, add an agreement test to
   `tests/test_external_agreement.py`; only then may the status be
   `validated`.
6. Add `example/example_<name>.py`.

## Optimized implementations

KFCM, MBMM, SoftDBSCANGM and KMART carry vectorised implementations from the
optimization study in `optimization/`. If you change one of them,
`tests/test_optimization_equivalence.py` must still pass: it compares the
estimator with the preserved pre-optimization code in
`optimization/original/scpp_original/`, which nothing in `soft_clustering/` may
import. An optimization must preserve the objective, update rules,
initialisation, convergence test and public API; anything else is an
algorithmic change and must be presented as one.

## Reporting bugs

Open an issue with the **Bug report** form (version, environment, minimal
reproducible example). Report security problems privately as described in
[SECURITY.md](.github/SECURITY.md), not in a public issue.
