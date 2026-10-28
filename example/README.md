# Examples

One short, runnable script per estimator. Each follows the same workflow:

```text
make or load data  ->  construct the estimator  ->  fit  ->  read memberships_  ->  inspect
```

Run any of them against an installed package, for example:

```bash
pip install soft-clustering
python example/example_fcm.py
```

The scripts for CDCGS, DMoN, NOCD and RDFKC need the `deep` extra
(`pip install "soft-clustering[deep]"`). All randomness is seeded, so the
output is reproducible. `tests/test_examples.py` runs every script, so they
stay in step with the API.

Scripts worth reading first:

| Script | Shows |
| --- | --- |
| `example_fcm.py` | the basic workflow, prototypes, the objective, out-of-sample memberships |
| `example_ecm.py` | a credal partition: mass on a pair of clusters for an ambiguous point, on the empty set for an outlier |
| `example_pcm.py`, `example_pfcm.py` | typicalities: an outlier is atypical of every cluster |
| `example_afcm.py` | non-convex clusters that fuzzy c-means cannot separate |
| `example_bnmf.py` | relevance determination pruning unneeded communities |
| `example_smcla.py` | a soft consensus of several clusterings |
| `example_afcmaptive.py` | image segmentation under a smooth intensity bias |
