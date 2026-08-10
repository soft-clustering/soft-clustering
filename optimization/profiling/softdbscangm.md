# Profiling — SoftDBSCANGM

**Module:** `soft_clustering/_softdbscangm.py`
**Family:** density-based, Gaussian-mixture refinement
**Reference preserved at:** `optimization/original/scpp_original/_softdbscangm.py`

## Method

`cProfile` over a single `fit`, inputs from `optimization/harness.py`
(`features(n, d=8, k=3)`, three well-separated isotropic blobs). Sorted by
`tottime`. Raw record: `optimization/benchmarks/raw/profile_slow.jsonl`.

Profiled at n = 80 rather than the default 600 because the reference
implementation does not finish at larger sizes in reasonable time — which is
itself the finding.

## 1. Dominant functions (n = 80, single fit)

| tottime | cumtime | calls | function |
| ---: | ---: | ---: | --- |
| 2178.1 ms | 3607.3 ms | 1,036,800 | `scipy/spatial/distance.py:994 mahalanobis` |
| 598.1 ms | 4243.1 ms | 1,036,800 | `_softdbscangm.py:91 <genexpr>` |
| 470.5 ms | 686.8 ms | 1,036,801 | `numpy/.../shape_base.py:78 atleast_2d` |
| 381.1 ms | 557.4 ms | 2,073,600 | `scipy/spatial/distance.py:289 _validate_vector` |
| 176.4 ms | 176.4 ms | 2,073,769 | `numpy.asarray` |
| 130.4 ms | 130.4 ms | 2,073,920 | `multiarray.py:748 dot` |
| 113.3 ms | 113.3 ms | 12,800 | `builtins.sum` |

## 2. Dominant lines

`fit`, step 5:

```python
for i in range(N):
    for j in range(k):
        d = mahalanobis(X[i], self.centers[j], self.cov_inv[j])
        d = max(d, 1e-10)
        denom = sum(
            (d / max(mahalanobis(X[i], self.centers[t], self.cov_inv[t]), 1e-10))
            ** (2 / (self.m - 1))
            for t in range(k)
        )
        self.U[i, j] = 1.0 / denom
```

## 3. Runtime contribution

The membership update accounts for essentially all of the runtime. Of the
4.36 s spent in `fit`, 4.24 s is inside the generator expression on line 91 and
its `mahalanobis` calls. `atleast_2d`, `_validate_vector` and `asarray` — 1.03 s
combined — are pure SciPy call overhead, incurred once per scalar distance.

Two compounding problems:

1. **Repeated work.** Only `N * k` distinct distances exist per iteration, but
   the inner `t` loop recomputes the full row for every `j`, so `N * k**2`
   distances are evaluated. At n = 80 that is 1,036,800 calls where 12,800
   distinct values were needed — an 81x redundancy factor, equal to `k`.
2. **Superlinear `k`.** Every DBSCAN noise point becomes its own component
   (`k = len(cluster_ids) + len(noise_ids)`), so on data where DBSCAN labels
   most points as noise, `k` grows with `N`. The measured membership shapes
   confirm it: `(60, 60)`, `(120, 120)`, `(240, 240)`. The cost is therefore
   `O(N * k**2) = O(N**3)` scalar SciPy calls.

Measured scaling of the reference confirms the cubic term:

| n | reference fit time | ratio vs previous |
| ---: | ---: | ---: |
| 60 | 3,410 ms | — |
| 120 | 26,695 ms | 7.83x for 2x n |
| 240 | > 600,000 ms (timeout) | — |

7.83x per doubling is 2^2.97 — cubic to within measurement error.

## 4. Memory contribution

Not a memory problem. Peak traced allocation was 0.15 MB at n = 60 and 0.45 MB
at n = 120: the reference allocates scalars, not large intermediates. Any
optimization must avoid *introducing* a memory problem — specifically, the
obvious `(N, k, k)` ratio tensor would be `O(N**3)` in memory and is rejected
for that reason.

## 5. Observed bottlenecks

| # | Bottleneck | Evidence |
| --- | --- | --- |
| B1 | `N * k**2` scalar `mahalanobis` calls per iteration | 1,036,800 calls, 2.18 s tottime |
| B2 | SciPy per-call validation overhead | `_validate_vector` 2,073,600 calls, 0.38 s |
| B3 | `list(...).index()` inside the step-2 init loop | `O(N**2)`; not visible at n = 80 but scales |
| B4 | `k` separate `scipy.linalg.inv` calls per iteration | one per component per iteration |

## 6. Proposed optimizations

| # | Change | Type | Rationale |
| --- | --- | --- | --- |
| O1 | Compute the `N * k` distance matrix once per iteration, batched `einsum` | implementation | removes the `k`-fold redundancy and all per-call SciPy overhead |
| O2 | Rewrite the ratio sum as `d**-p / sum_t d_t**-p` | implementation (algebraic identity) | avoids the `(N, k, k)` tensor entirely |
| O3 | Rescale distances by the per-sample minimum before exponentiating | implementation | the factor cancels exactly; keeps powers in [0, 1] so small `m` cannot overflow |
| O4 | `searchsorted` for the init column mapping | implementation | same columns, `O(N log k)` instead of `O(N**2)` |
| O5 | Batched `np.linalg.inv` over the stacked `(k, d, d)` array | implementation | one LAPACK call per iteration |
| O6 | Block the `(rows, k, d)` temporary to ~32 MB | implementation | keeps memory bounded now that `k` grows with `N` |

None of these changes the objective, the update rules, the initialisation, the
clamping constants or the convergence test. O2 and O3 are algebraic identities:

    sum_t (d_j / d_t)**p  =  d_j**p * sum_t d_t**-p
    =>  1 / sum_t (d_j/d_t)**p  =  d_j**-p / sum_t d_t**-p

and multiplying numerator and denominator by `min_t(d_t)**p` leaves the value
unchanged while bounding every power by 1.

## 7. Expected effect

Removal of an `O(N)` factor: from `O(N * k**2)` Python-level SciPy calls to
`O(N * k * d**2)` BLAS work. Expected to be dramatic — orders of magnitude —
and to convert an algorithm that times out at n = 240 into one that is usable.

Measured outcome is in `optimization/reports/optimization_report.md`; the
correctness evidence is `optimization/correctness/softdbscangm.json`.
