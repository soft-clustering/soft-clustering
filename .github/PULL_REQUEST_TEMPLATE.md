## Summary

<!-- What does this change do, and why? Link the issue it closes. -->

## Checklist

- [ ] Tests added or updated under `tests/`, and `pytest` passes locally.
- [ ] Coverage of the changed lines is not reduced (`pytest --cov=soft_clustering --cov-report=term-missing`).
- [ ] `ruff check` and `black --check` pass.
- [ ] Public API changes are reflected in `soft_clustering/__init__.py` (`__all__`) and in `docs/source/`.
- [ ] A runnable script is included in `example/` for any new estimator.
- [ ] Docstrings state the shape and semantics of every fitted attribute.

## New estimator (delete if not applicable)

- [ ] Primary reference cited in the module docstring.
- [ ] Membership matrix is exposed with shape `(n_samples, n_clusters)`.
- [ ] Tests cover output shape, the partition constraint (where the method defines one),
      invalid hyperparameters, and reproducibility under a fixed `random_state`.
