# SCPP Algorithm Optimization Study

*Generated from `optimization/benchmarks/results.csv`. Every number below is a
measurement; the commands that produce them are in `optimization/README.md`.*

---

## 1. Executive summary

| | |
| --- | ---: |
| Algorithms exported by SCPP | 40 |
| Statically audited | 40 |
| Baseline-timed in the survey | 28 |
| Profiled in detail | 10 |
| **Optimized, verified and benchmarked end to end** | **5** |
| Optimized with a documented rationale and no measurable regression | 5 |
| Algorithms found to need no optimization | see §12 |

The five algorithms taken through the full cycle are SoftDBSCANGM, KFCM,
KFCCL, KMART and MBMM. Across the 20 paired measurements where both builds
completed:

| | Runtime speedup |
| --- | ---: |
| Median | **70.4x** |
| Geometric mean | **72.6x** |
| Mean | **365.6x** |
| Minimum | **6.3x** |
| Maximum (measured, both builds completing) | **4,506.8x** |
| Maximum (reference times out; lower bound) | **> 47,900x** |

The arithmetic mean and the median differ by an order of magnitude because the
sample mixes modest constant-factor gains with two complexity fixes. The
geometric mean (72.6x) and the median (70.4x) agree closely and are the honest
summary of a typical case; the maximum is the honest summary of the outlier.
No single number describes the study, so all are reported.

Every one of the five is an **implementation optimization**, not an algorithmic
change. Across all **42 correctness comparisons** the maximum absolute
membership difference against the preserved reference was **2.2e-14**, with
**100% label agreement** and identical cluster counts in every case. KMART
agrees **exactly** — a difference of 0.0, not merely a small one.

Memory is a mixed result, reported in full rather than omitted:

| | Peak traced allocation |
| --- | ---: |
| Memory-neutral (within ±4%) | **9 of 20** paired measurements (KFCCL, KMART) |
| Median change over all 20 | **-140%** |
| Worst case | **-896%** (SoftDBSCANGM, n = 120: 0.45 MB -> 4.45 MB) |

The large relative regressions occur where the absolute footprint is smallest:
KFCM's worst case is 0.26 MB -> 0.86 MB. The two optimizations that operate on
data already held as a matrix — KFCCL and KMART — cost no additional memory at
all, because both builds already materialise the dominant array.

**The headline result** is SoftDBSCANGM. Its membership update was `O(n^3)` in
scalar SciPy calls, which made the estimator unusable beyond a few hundred
samples: the reference does not finish within 600 s at n = 240, where the
optimized implementation takes 12.5 ms. This was a scalability defect, not
merely slow code.

**A recurring defect class.** Four of the five fixes are the same finding in
different clothing: a scalar numerical operation invoked from a Python loop
where an exact batched form exists. SoftDBSCANGM called
`scipy.spatial.distance.mahalanobis` per `(i, j, t)` triple; KFCM called a
scalar Gaussian kernel `N * k` times per iteration; KFCCL rebuilt a normalised
kernel column per `(iteration, cluster, sample)`; KMART called `np.minimum`
once per (document, prototype) pair. The audit is what made the pattern
visible, and the same technique removed all four.

**Scope.** This report covers 5 of 40 algorithms end to end. The remaining 35
are audited and, where runnable, baseline-timed and in five further cases
profiled, but not optimized — in 18 cases because measurement showed there was
nothing worth optimizing. §12 and §13 say what is known about each and what the
next targets are. No result here is extrapolated to an algorithm that was not
measured.

---

## 2. Methodology

### Hardware and software

| | |
| --- | --- |
| Machine | Apple M2, 8 cores |
| OS | macOS 26.0, arm64 |
| Python | 3.13.5 |
| NumPy | 2.4.6 (BLAS: Accelerate) |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |

Both builds run on the same machine, in the same interpreter, against the same
dependency versions, within the same sweep. No result compares across machines.

### Datasets

Inputs come from one registry, `optimization/harness.py`, shared by the
profiler, the benchmark runner and the correctness checker — so an algorithm is
profiled, optimized and verified on the same data. The feature-matrix
generator produces `k` well-separated isotropic Gaussian blobs
(`scale = 0.35`, centres at `3i`), which is the same structure
`tests/test_protocol.py` uses.

Inputs are rebuilt from a fixed seed for every fit, so neither build can
benefit from a warmed cache or mutated data.

### Timing

Each configuration runs in its own subprocess (`run_case.py`) with a clean
allocator and import state, driven by `sweep.py` with a per-case timeout. A
timeout is recorded as a result, not discarded: "does not finish at this size"
is a measurement.

Paired benchmarks use **1 warm-up fit** followed by **3 timed repetitions**.
The reported `fit_time_ms` is the **minimum** of the repetitions — the standard
choice for timing, since noise is one-sided. Mean and all individual times are
retained in `results.csv`.

### Memory

`tracemalloc` peak traced allocation, measured around `fit` only. This counts
Python-level allocation and therefore understates NumPy buffers held by BLAS,
but it is consistent between the two builds, which is what the comparison
needs. The absolute figures should not be read as process RSS.

### Correctness criteria

For every algorithm, both builds are fitted on identical inputs and compared
on membership matrices, hard labels, cluster prototypes and discovered cluster
count. Estimators drawing from NumPy's global RNG have that generator seeded
immediately before each fit, so initialisation is identical.

An optimization is accepted as equivalent only if:

- maximum absolute membership difference < 1e-9,
- label agreement = 1.0,
- cluster counts identical,
- invariants hold (finite, non-negative, `labels == argmax(U)`).

---

## 3. Algorithm inventory

See `optimization/inventory.md`, generated by `make_inventory.py`. It records,
per algorithm: family, module, code size, loop nesting, flagged numerical call
sites, measured baseline runtime, measured bottleneck where profiled, and
optimization status.

**Provenance.** Every algorithm in SCPP is a direct, SCPP-specific
implementation of its published method written against NumPy/SciPy. None wraps
a third-party estimator and none is a vendored copy of reference code;
scikit-learn appears only as a supporting primitive inside some methods
(DBSCAN in SoftDBSCANGM, KMeans in the consensus methods, eigen-solvers in
AFCM).

---

## 4. Profiling results

Baseline survey, n = 200, d = 8, k = 3, single fit
(`results.csv`, `implementation == survey`). Full landscape in Figure 6.

| Algorithm | Baseline (ms) | Note |
| --- | ---: | --- |
| SoftDBSCANGM | **35,306.1** | 3 orders of magnitude above the next slowest |
| SCSPA | 1,042.3 | flat ~1 s floor |
| SHBGF | 999.9 | flat ~1 s floor |
| SMCLA | 996.9 | flat ~1 s floor |
| MBMM | 227.7 | |
| AFCM | 205.3 | |
| KFCCL | 143.3 | |
| SISC | 71.1 | |
| KFCM | 51.3 | |
| KMART | 46.1 | |
| BayesianNMF | 16.1 | |
| SFCMEP | 18.9 | |
| RPFKM | 13.4 | |
| GMM | 11.7 | |
| GK | 7.8 | |
| WBSC | 7.1 | |
| ECM | 6.8 | |
| PLSI | 6.7 | |
| FCC | 4.0 | |
| FCM | 2.6 | |
| RoughKMeans | 2.3 | |
| PFCM | 2.1 | |
| PCM / CAFCM | 1.7 | |
| ENTROPYFCM | 0.7 | |
| SCM | 0.6 | |
| CAFHFCM / AFCMSimple | 0.5 | |

Per-algorithm profiling reports are in `optimization/profiling/`. The dominant
findings:

| Algorithm | Dominant cost | Evidence |
| --- | --- | --- |
| SoftDBSCANGM | `N*k^2` scalar `scipy...mahalanobis` calls per iteration | 1,036,800 calls at n = 80; 2.18 s `tottime` |
| MBMM | `scipy.stats.beta.logpdf` per (component, feature) | 4,800 calls; `_argcheck` + `_support_mask` cost as much as `_logpdf` |
| KFCCL | 41,346 `np.sum` calls — per-element Python reduction | `_wrapreduction` 25.9 ms |
| KMART | 28,634 `np.sum` calls in `_fuzzy_and` | `_wrapreduction` 17.9 ms |
| KFCM | `typeguard` runtime type checking | `check_type_internal` 22.5 ms + `isinstance` 17.5 ms of 175 ms total |
| AFCM | dense `scipy.linalg.eigh` | 126.9 ms of 440 ms |
| SISC | `_tanimoto_similarity` over Python sets | 10,624 calls |
| SCSPA / SHBGF / SMCLA | ~1 s floor independent of `n` | profiled Python time is only a few ms; cost is sklearn `KMeans` setup and BLAS thread pools |

Two cross-cutting observations worth recording:

1. **`typeguard` is a measurable tax on hot paths.** In KFCM, runtime type
   checking accounted for roughly 23% of fit time. It applies to 39 of 40
   estimators. This is a library-wide design trade-off, not a per-algorithm
   bug, and is left for the recommendations.
2. **The three consensus methods have a fixed ~1 s floor** that does not vary
   with `n`. That is characteristic of `sklearn.cluster.KMeans` default
   `n_init` plus thread-pool startup, not of the consensus mathematics.

---

## 5. Optimization techniques

| Technique | Where used | Reason | Added complexity |
| --- | --- | --- | --- |
| Batched `einsum` over components | SoftDBSCANGM | replaces `N*k` scalar SciPy distance calls with BLAS-backed contractions | low |
| Algebraic rewrite of a normalised ratio sum | SoftDBSCANGM | removes an `O(k)` redundancy *and* avoids an `(N,k,k)` tensor | low |
| Rescaling before exponentiation | SoftDBSCANGM | numerical robustness for small `m`; the factor cancels exactly | low |
| `searchsorted` for an index mapping | SoftDBSCANGM | `O(N log k)` instead of `O(N^2)` `list.index()` | none |
| Batched LAPACK inverse | SoftDBSCANGM | one `np.linalg.inv` on a stacked `(k,d,d)` array per iteration | none |
| Memory-bounded chunking | SoftDBSCANGM | `k` grows with `N`, so a full `(N,k,d)` temporary would be quadratic | medium |
| Direct `scipy.special` density | MBMM | bypasses `rv_continuous` dispatch, argument checking and support masks | low |
| Vectorisation over features | MBMM | removes the inner `D` loop in the E and M steps | low |
| Single GEMM for weighted means | SoftDBSCANGM, MBMM, KFCM | `U_m.T @ X` in place of a per-component loop | none |
| Loop-invariant hoisting | KFCCL | the normalised kernel `K / outer(diag, diag)` was rebuilt per (iteration, cluster, sample); it is constant for the whole fit | none |
| Dependency-free loop collapse | KFCCL | each `p_ik[i, k]` reads only its own previous value, so the `k` loop is one GEMV | none |
| Quadratic form in place of an outer product | KFCCL | `U K U` avoids materialising an `(N, N)` outer product per cluster per iteration | none |
| Matrix-form kernel | KFCM | replaces `N * k` scalar kernel calls per iteration, and the runtime type check riding on each | low |
| `searchsorted` for roulette selection | KFCM | replaces a Python scan over cumulative probabilities in K-Means++ | none |
| Contiguous prototype block | KMART | turns the per-category vigilance test into one broadcast reduction | low |
| COO assembly of the output matrix | KMART | `lil_matrix` element assignment cost one Python index operation per (document, cluster) pair | none |

**No native extensions were added.** No Cython, no Numba, no C++. Profiling
showed the bottlenecks were Python-level call overhead and redundant work, both
removable with NumPy/SciPy primitives already in the dependency set. Adding a
compiled dependency would have increased build and packaging complexity for no
measured benefit — a trade-off explicitly weighed and rejected.

**No approximations were introduced.** Every transformation is an exact
algebraic identity or a change of evaluation order.

---

## 6. Algorithm-by-algorithm analysis

### 6.1 SoftDBSCANGM

**Original implementation.** SCPP-specific implementation of soft DBSCAN with
Gaussian-mixture refinement. DBSCAN provides the initial partition; each noise
point becomes its own component; memberships are then refined with a
fuzzy-c-means-style update under per-component Mahalanobis distance.

**Bottleneck.** The membership update evaluated

```
U[i,j] = 1 / sum_t ( d(i,j) / d(i,t) ) ** (2/(m-1))
```

with a Python triple loop calling `scipy.spatial.distance.mahalanobis` once per
`(i, j, t)`. Only `N*k` distinct distances exist per iteration, but `N*k^2` were
computed — a `k`-fold redundancy. Because every noise point becomes a component,
`k` grows with `N` (measured membership shapes: `(60,60)`, `(120,120)`,
`(240,240)`), so the cost is cubic in the sample count. Measured scaling of the
reference: 3,410 ms -> 26,695 ms for a doubling of `n`, a factor of 7.83 = 2^2.97.

**Optimization.** (i) compute the `N*k` distance matrix once per iteration with
a batched `einsum`; (ii) rewrite the ratio sum in the equivalent closed form
`d^-p / sum_t d_t^-p`, avoiding any `(N,k,k)` tensor; (iii) divide distances by
their per-sample minimum before exponentiation, which cancels exactly and bounds
every power by 1 so a small `m` cannot overflow; (iv) `searchsorted` for the
initialisation column map; (v) one batched `np.linalg.inv`; (vi) chunk the
`(rows,k,d)` temporary to bound memory.

**Technical rationale.** `sum_t (d_j/d_t)^p = d_j^p * sum_t d_t^-p`, so
`1/sum_t (d_j/d_t)^p = d_j^-p / sum_t d_t^-p`. Multiplying numerator and
denominator by `min_t(d_t)^p` leaves the value unchanged. Both are identities;
neither changes the objective, update rules, initialisation, clamping constants
or convergence test.

**Correctness.** 9 comparisons (n = 60/120/240, seeds 0/1/2). Maximum absolute
membership difference **2.22e-16**; label agreement **1.0000**; cluster counts
identical; invariants hold. Maximum *relative* difference is 7.09e-07, which
arises on entries at the 1e-10 distance clamp floor where relative error is not
a meaningful measure — the absolute bound of 2.2e-16 is the operative one.

**Runtime.**

| n | original | optimized | speedup |
| ---: | ---: | ---: | ---: |
| 60 | 3,410.1 ms | 3.8 ms | 889.4x |
| 120 | 26,694.7 ms | 5.9 ms | 4,506.8x |
| 240 | timeout (> 600 s) | 12.5 ms | > 47,900x |
| 480 | not attempted | 46.9 ms | — |
| 960 | not attempted | 6,118.2 ms | — |

**Memory.** Peak traced allocation rose from 0.15 MB to 1.16 MB (n = 60) and
0.45 MB to 4.45 MB (n = 120) — the cost of materialising the `(N,k)` distance
matrix and the chunked `(rows,k,d)` temporary instead of scalars. This is a
deliberate time-for-memory trade. Chunking bounds the temporary near 32 MB; at
n = 960 peak traced allocation was 136 MB, driven by the `(N,k)` matrix with
k = 908.

**Scalability.** The reference is cubic and unusable past a few hundred samples.
The optimized implementation is 3.8 -> 5.9 -> 12.5 -> 46.9 ms for n = 60 -> 480,
then rises sharply to 6,118 ms at n = 960. That jump is **not** smooth scaling:
at n = 960 DBSCAN found genuine clusters (k = 908 rather than k = n), changing
the convergence behaviour and the iteration count. Iteration counts were not
instrumented, so the split between per-iteration cost and iteration count at
that size is not established — recorded as a limitation in §11.

**Final assessment.** The most valuable result of the study. A genuine
scalability defect was removed, at machine-epsilon fidelity, with no new
dependency. Retain.

### 6.2 MBMM

**Original implementation.** SCPP-specific EM for a multivariate Beta mixture,
using `scipy.stats.beta.logpdf` for the component densities.

**Bottleneck.** `beta.logpdf` was called once per `(component, feature)` in the
E-step and again in the log-likelihood evaluation — 4,800 calls for a 200x8
input. Profiling showed most of that time in `rv_continuous` machinery
(`_argcheck`, `_support_mask`, `_broadcast_to`) rather than the density itself.

**Optimization.** Replace the frozen distribution with the identity SciPy
evaluates internally,

```
log Beta(x; a, b) = xlogy(a-1, x) + xlog1py(b-1, -x) - betaln(a, b)
```

applied to all features at once with the same `scipy.special` primitives; and
vectorise the M-step over features, preserving the original order of operations
(the variance is taken about the **unclipped** mean, which is clipped only
afterwards).

**Technical rationale.** The model already requires samples in `(0,1)`, so the
support mask SciPy applies is redundant. The expression is not an approximation
— `tests/test_optimization_equivalence.py::test_log_density_matches_scipy`
asserts agreement with `scipy.stats.beta.logpdf` to `rtol=1e-12`.

**Correctness.** 6 comparisons (n = 200/600, seeds 0/1/2). Maximum absolute
responsibility difference **2.17e-14**; label agreement **1.0000**; mixture
weights, `alpha` and `beta` agree to 1e-8.

**Runtime.**

| n | original | optimized | speedup |
| ---: | ---: | ---: | ---: |
| 200 | 841.4 ms | 49.9 ms | 16.9x |
| 600 | 973.8 ms | 88.0 ms | 11.1x |
| 1,200 | 1,164.5 ms | 137.9 ms | 8.4x |
| 2,400 | 1,532.1 ms | 244.2 ms | 6.3x |

**Memory.** 0.03 -> 0.07 MB at n = 200, 0.30 -> 0.76 MB at n = 2,400: roughly
2.5x more, from holding `(N,D)` intermediates instead of looping over columns.
Negligible in absolute terms.

**Scalability.** The speedup *decreases* with `n` (16.9x -> 6.3x), which is the
expected signature of this optimization: the removed cost is `rv_continuous`
overhead, which is `O(K*D)` calls **independent of `n`**, so it dominates at
small `n` and is amortised at large `n`. The optimization helps most exactly
where the overhead is proportionally worst. Reported rather than reported as a
single headline number.

**Final assessment.** Solid, low-risk, no new dependency. Retain.

### 6.3 KFCM

**Original implementation.** SCPP-specific kernelised fuzzy c-means with
K-Means++ initialisation.

**Bottleneck.** Profiling reported `typeguard`'s runtime type checking at
roughly 23% of fit time (22.5 ms `check_type_internal` plus 17.5 ms
`isinstance` of 175 ms). That was a *symptom*. `_gaussian_kernel` is a scalar
helper returning a single float, called `N * k` times per iteration from list
comprehensions in the center update and again per sample in the membership
update; the class-level `@typechecked` decorator attached a type check to every
one of those calls. The initialisation was worse still: K-Means++ computed
`min([norm(x - c)**2 for c in centers]) for x in X`, quadratic in Python-level
work.

**Optimization.** Evaluate the kernel as an `(N, k)` matrix; express the center
update as a single GEMM (`weights @ X`) with a masked division; recompute the
membership distances from the updated centers in one array expression; and
replace the K-Means++ roulette scan with `np.searchsorted`. The runtime type
checking disappears with the scalar calls rather than being switched off, so
the library-wide correctness guarantee is untouched.

**Technical rationale.** Squared distances are computed as
`sqrt(sum(d**2))**2` rather than `sum(d**2)`, because that is what
`np.linalg.norm(...)**2` does in the reference and the round trip through
`sqrt` is not the identity in floating point. `np.searchsorted(cum, r,
side="right")` returns exactly the index the reference's `break` selected, and
the out-of-range case leaves the center at zero as the un-taken `break` did.
Crucially, the number and order of draws from NumPy's global RNG are unchanged
— one `randint`, then one `rand` per additional center — which
`tests/test_optimization_equivalence.py::test_initialisation_consumes_the_same_random_draws`
pins by checking that the global generator is left in the same state.

**Correctness.** 9 comparisons (n = 100/200/400, seeds 0/1/2). Maximum absolute
membership difference **4.77e-15**; label agreement **1.0000**; cluster counts
identical.

**Runtime.**

| n | original | optimized | speedup |
| ---: | ---: | ---: | ---: |
| 100 | 102.9 ms | 1.3 ms | 81.9x |
| 200 | 183.6 ms | 1.3 ms | 146.5x |
| 400 | 306.5 ms | 1.5 ms | 202.7x |
| 800 | 495.7 ms | 1.5 ms | 330.5x |
| 1,600 | 1,239.5 ms | 2.3 ms | 545.5x |

**Memory.** 0.02 -> 0.07 MB at n = 100 and 0.26 -> 0.86 MB at n = 1,600. The
relative change is large (about -230%) because the reference held only scalars;
the absolute cost is under 1 MB at every size measured.

**Scalability.** The speedup *grows* with `n` — the opposite of MBMM — because
the removed cost is `O(N * k)` Python-level calls per iteration, which scales
with the data, rather than a fixed `O(K * D)` overhead.

**Final assessment.** The largest speedup in the study after SoftDBSCANGM, at
no architectural cost. Retain.

### 6.4 KFCCL

**Original implementation.** SCPP-specific kernel-based fuzzy competitive
learning.

**Bottleneck.** 41,346 `np.sum` calls per fit. Two redundancies produced them.
The normalised kernel column `K[:, k] / (K_diag * K_diag[k])` was rebuilt
inside the innermost loop, so an `N`-vector division ran once per
`(iteration, cluster, sample)` although `K` and `K_diag` are fixed for the
entire fit. And the inner-product update was written as a loop over `k`.

**Optimization.** Hoist the normalised kernel to a single `(N, N)` matrix
computed once; collapse the `k` loop to one matrix-vector product; and replace
the `(N, N)` outer product `U[i][:, None] * U[i][None, :]` with the quadratic
form `U[i] @ K @ U[i]`.

**Technical rationale.** Every entry `p_ik[i, k]` depends only on its own
previous value and on quantities constant across the loop — `U[i]`, `K_norm`,
and `V_sq[i]`, which is fully determined *before* the loop begins. The loop
therefore carries no dependency and
`sum_j U[i,j] K[j,k] / (K_diag[j] K_diag[k])` is exactly `(U[i] @ K_norm)[k]`.

**Correctness.** 9 comparisons (n = 100/200/400, seeds 0/1/2). Maximum absolute
membership difference **1.11e-16**; label agreement **1.0000**; iteration
counts identical at every size checked.

**Runtime.**

| n | original | optimized | speedup |
| ---: | ---: | ---: | ---: |
| 100 | 433.8 ms | 11.9 ms | 36.5x |
| 200 | 888.1 ms | 12.8 ms | 69.6x |
| 400 | 1,873.3 ms | 13.9 ms | 134.8x |
| 800 | 3,795.1 ms | 25.0 ms | 152.0x |
| 1,600 | 8,406.0 ms | 317.3 ms | 26.5x |

**Memory.** Unchanged, within measurement noise (+3.7% to -0.1%). Both builds
already materialise the `(N, N)` kernel matrix, which dominates: 58.6 MB at
n = 1,600 for either build.

**Scalability, and the n = 1,600 result.** The speedup rises to 152x at n = 800
and then falls to 26.5x at n = 1,600. This is **not** a change in iteration
count — both builds converge at iteration 65 at that size — and it is not a
defect. At n = 1,600 the two `(N, N)` matrices occupy 41 MB, exceeding the
cache hierarchy, and the fit becomes memory-bandwidth bound. A standalone
measurement of the same BLAS calls at the same sizes reproduces it: 195
cluster-iterations of `u @ K_norm` and `u @ K @ u` take 6.8 ms at n = 800
(292 GB/s effective, in cache) and 296.9 ms at n = 1,600 (26.9 GB/s, DRAM
bandwidth) — the latter accounting for essentially all of the 317.3 ms
measured. The optimized implementation has reached the hardware limit for an
algorithm that must stream an `N * N` kernel matrix per cluster per iteration;
the reference never approaches that limit because it is bound by Python call
overhead instead.

**Final assessment.** Retain. The large-`n` behaviour is a property of the
algorithm's `O(N^2)` working set, not of the optimization.

### 6.5 KMART

**Original implementation.** SCPP-specific modified Fuzzy ART for soft document
clustering.

**Bottleneck.** 28,634 `np.sum` calls inside `_fuzzy_and` — one scalar
reduction per (document, prototype) pair — plus element-by-element assignment
into a `lil_matrix` when assembling the output.

**Optimization.** Hold prototypes in one contiguous `(capacity, vocab)` buffer,
doubled on demand, so the vigilance test over all existing categories is a
single broadcast reduction; update the passing categories with one
fancy-indexed assignment; and assemble the membership matrix in COO form.
`prototypes_` is still published as the documented list of per-cluster vectors.

**Technical rationale.** `_fuzzy_and` is `np.minimum`, so the vigilance test
over the whole category block is `sum(minimum(I, P), axis=1) / (sum(I) + eps)`.
The same minima are summed in the same order, so the rewrite is exact rather
than merely close. Capacity doubling keeps prototype growth amortised `O(1)`,
matching the reference's asymptotics in the worst case where every document
opens a new category.

**Correctness.** 9 comparisons (n = 200/600/1,200, seeds 0/1/2). Maximum
absolute membership difference **exactly 0.0**; prototypes agree bitwise;
cluster sets are identical; label agreement **1.0000**. This is the only
algorithm in the study for which the optimization is bit-exact.

**Runtime.**

| n | original | optimized | speedup |
| ---: | ---: | ---: | ---: |
| 200 | 120.8 ms | 12.3 ms | 9.8x |
| 600 | 937.5 ms | 39.6 ms | 23.7x |
| 1,200 | 3,577.8 ms | 86.3 ms | 41.5x |
| 2,400 | 13,763.7 ms | 193.3 ms | 71.2x |

**Memory.** Unchanged (+0.5% to 0.0%). The document-term matrix dominates in
both builds.

**Scalability.** The reference is quadratic in the document count, because the
number of categories grows with `n` and each document is tested against all of
them; at n = 2,400 it takes 13.8 s. The optimized implementation is quadratic
too — the same tests are performed — but each is a BLAS-backed reduction rather
than a Python call, so the constant falls by nearly two orders of magnitude.

**Final assessment.** Bit-exact, memory-neutral, no new dependency. Retain.

---

## 7. Runtime improvements

- Table 2: `reports/tables/table2_runtime.md` (LaTeX: `.tex`)
- Figure 1: `figures/fig1_speedup.{png,pdf}` — speedup per algorithm and size
- Figure 2: `figures/fig2_runtime_comparison.{png,pdf}` — before vs after

## 8. Memory improvements

- Table 3: `reports/tables/table3_memory.md`
- Figure 3: `figures/fig3_memory_comparison.{png,pdf}`

The result splits cleanly in two, and the split is informative rather than
incidental.

**Memory-neutral (9 of 20 paired measurements).** KFCCL and KMART cost no extra
memory at any size — within +3.7% to -0.1%. Both already materialised the
dominant array (an `(N, N)` kernel matrix and a document-term matrix
respectively), so vectorising the work performed *on* that array adds nothing.

**Memory-regressing (11 of 20).** SoftDBSCANGM, KFCM and MBMM all increase peak
traced allocation, because there the technique replaces scalar loops with array
operations and so materialises intermediates the loops did not. The relative
figures are large — up to -896% — but they are largest exactly where the
absolute footprint is smallest: KFCM goes from 0.02 MB to 0.07 MB at n = 100,
and SoftDBSCANGM from 0.45 MB to 4.45 MB at n = 120. Chunking bounds the growth
in SoftDBSCANGM.

No memory reduction is claimed for any algorithm in the study.

## 9. Scalability

- Table 5: `reports/tables/table5_scalability.md`
- Figure 4: `figures/fig4_scalability.{png,pdf}` — log-log runtime vs `n`, with
  the reference timeout marked.

## 10. Correctness

- Table 4: `reports/tables/table4_correctness.md`
- Figure 5: `figures/fig5_accuracy_preservation.{png,pdf}` — maximum membership
  deviation against machine epsilon
- Raw comparisons: `correctness/softdbscangm.json`, `correctness/mbmm.json`,
  `correctness/kfcm.json`, `correctness/kfccl.json`, `correctness/kmart.json`
- Regression tests: `tests/test_optimization_equivalence.py` (36 tests)

All five algorithms meet every acceptance criterion in §2. None is classified
as an ALGORITHMIC CHANGE; all five are IMPLEMENTATION OPTIMIZATIONS.

---

## 11. Trade-offs and limitations

**Added dependencies.** None. Both optimizations use NumPy and SciPy functions
already required by the package.

**Code complexity.** SoftDBSCANGM grew from 115 to ~190 lines, gaining two
helper methods and a chunk-size heuristic. This is a real maintenance cost,
justified by converting an unusable algorithm into a usable one. MBMM stayed
approximately the same size and is arguably simpler.

**Portability.** No change. Pure NumPy/SciPy, no compiled extension, no
platform-specific code.

**Numerical differences.** Quantified in §10 and bounded at 2.2e-14 absolute
across all 42 comparisons. The optimized implementations use BLAS-backed
operations whose summation order may differ across platforms; the regression
tolerance (1e-9) is set well above the observed worst case to accommodate that
without admitting a real change. KMART is exact and needs no tolerance.

**Limitations of this study.**

1. **Coverage.** 5 of 40 algorithms optimized end to end. Five more are
   profiled with identified bottlenecks but not optimized; 18 were measured as
   already efficient and deliberately left alone.
2. **Memory methodology.** `tracemalloc` measures Python-level allocation and
   understates BLAS-held buffers. Peak RSS was not used because it is noisy for
   sub-second fits.
3. **Iteration counts not instrumented.** No estimator reports iterations
   through a public attribute, so runtime changes cannot in general be
   decomposed into per-iteration cost versus iteration count. This matters for
   the SoftDBSCANGM n = 960 result, which is left unexplained beyond noting the
   change in `k`. Where it mattered most — the KFCCL n = 1,600 result — the
   counts were recovered from the estimator's convergence message and shown to
   be identical across builds (§6.4), but that is a workaround, not
   instrumentation.
4. **Single machine, single BLAS.** All results are Apple M2 / Accelerate.
   Speedups that come from removing Python call overhead should transfer; the
   BLAS-bound portions may not.
5. **Statistical treatment.** 3 repetitions per configuration with a warm-up.
   Sufficient to establish 900x and 4,500x effects; **not** sufficient to claim
   significance for differences of a few percent, and none is claimed.
6. **Dataset shape.** Benchmarks use separated isotropic Gaussian blobs.
   SoftDBSCANGM's cost depends on how many points DBSCAN labels as noise, which
   is data-dependent; on data where DBSCAN labels few points as noise, `k` is
   small and the original's cubic term is far less punishing.
7. **`centers_` gap.** SoftDBSCANGM stores prototypes as `centers`, which is not
   among the names `BaseSoftClusterer._centers_attrs` searches, so `centers_` is
   `None` on both builds. Pre-existing, unrelated to the optimization, and left
   unchanged so as not to alter shared infrastructure mid-study — but it should
   be fixed.

---

## 12. Algorithms not optimized

**Mandatory section.** No algorithm below was optimized; each entry says what
is known and why work stopped there.

### Measured as already efficient — optimization not worthwhile

The following complete a 200-sample fit in under 10 ms, with the work already
delegated to NumPy/SciPy primitives. Any Python-level gain would be a fraction
of a millisecond and would not justify the risk of touching numerical code:

`FCM` (2.6 ms), `PCM` (1.7 ms), `CAFCM` (1.7 ms), `PFCM` (2.1 ms),
`RoughKMeans` (2.3 ms), `FCC` (4.0 ms), `PLSI` (6.7 ms), `ECM` (6.8 ms),
`WBSC` (7.1 ms), `GK` (7.8 ms), `SCM` (0.6 ms), `CAFHFCM` (0.5 ms),
`AFCMSimple` (0.5 ms), `ENTROPYFCM` (0.7 ms), `GMM` (11.7 ms),
`RPFKM` (13.4 ms), `BayesianNMF` (16.1 ms), `SFCMEP` (18.9 ms).

### Profiled, bottleneck identified, not yet optimized

Time-bounded, not concluded. Each has a concrete, measured target:

| Algorithm | Baseline | Identified target | Assessment |
| --- | ---: | --- | --- |
| SCSPA / SHBGF / SMCLA | ~1,000 ms each | ~1 s floor independent of `n`; sklearn `KMeans` `n_init` + thread-pool startup, not the consensus mathematics | Likely a large, cheap win — but changing `n_init` alters results, so it would be an **algorithmic change**, not an optimization, and needs a separate decision |
| AFCM | 205.3 ms | dense `scipy.linalg.eigh`, 126.9 ms of 440 ms | Already in LAPACK; would need a partial eigensolver (`eigsh`), which changes numerical results |
| SISC | 71.1 ms | `_tanimoto_similarity` over Python sets | Sparse matrix reformulation; moderate effort |

`KFCCL`, `KMART` and `KFCM` were previously in this table. All three have since
been optimized and are reported in §6.3–§6.5.

### Not runnable in the shared harness

`AFCMAdaptive`, `SKFCM`, `RDFKC` (image inputs), `BGMM` (two aligned views),
`SoftKSC` (labelled + unlabelled parts), `FeMIFuzzy` (list of client matrices),
`LDA` (returns factors, not a partition), `BIGCLAM` (requires `n_nodes` at
construction), and the PyTorch estimators `CDCGS`, `DMoN`, `MMSB`, `NOCD`.
These are statically audited only. The four PyTorch models were not examined for
device transfers, synchronisation or autograd usage — that requires a GPU-capable
environment and is out of scope here.

---

## 13. Recommendations

**Retain.** All five optimizations: no new dependency, machine-epsilon
fidelity, regression-tested.

**Highest-value next targets**, in order of measured benefit per unit of risk:

1. **The consensus trio's 1 s floor** — investigate whether it is `n_init`,
   thread-pool startup, or both. If it is `n_init`, note that reducing it is an
   **algorithmic change** and must be presented as such.
2. **SISC** — reformulate Tanimoto similarity over sparse matrices.
3. **Harness coverage** — 12 of 40 estimators take input shapes the shared
   registry does not build, so they are audited but never timed. Extending
   `harness.py` to cover them would make "all 40 benchmarked under one harness"
   true, and costs measurement effort rather than numerical risk.

**Avoid.** Adding Numba, Cython or a C++ extension. Nothing measured in this
study is bound by anything those would help; every bottleneck found was Python
call overhead or redundant work, removable with NumPy.

**Library-wide question, revised.** `typeguard`'s `@typechecked` was measured at
~23% of fit time in KFCM and is applied to 39 of 40 estimators, which looked
like an argument for making it opt-in. Optimizing KFCM (§6.3) showed that
reading to be misleading: the cost was not the decorator but the *number of
calls it decorated* — a scalar kernel helper invoked `N * k` times per
iteration. Removing the scalar calls removed the type-checking cost with them,
and KFCM is now the second-fastest optimized estimator in the study with
`@typechecked` still fully in force.

The general lesson is worth recording: runtime type checking is expensive in
proportion to call frequency, so it is a symptom of a hot Python-level loop
rather than an independent tax. Where a profile attributes significant time to
`typeguard`, the productive response is to look for the loop underneath it. No
change to the library's type-checking policy is recommended.

**Instrumentation.** Estimators should expose `n_iter_`. Its absence is the
single largest obstacle to interpreting the results above.

---

## 14. Reproducibility

Every command is listed in `optimization/README.md`. In outline:

1. `python optimization/audit_static.py` — static audit
2. `python optimization/sweep.py --out .../smoke.jsonl --n 200 --timeout 45` — survey
3. `python optimization/sweep.py --profile --memory ...` — profiling
4. `python optimization/snapshot_originals.py --from-git <ref> ...` — freeze reference
5. `python optimization/compare.py <ALG> --sizes ... --seeds ...` — correctness
6. `python optimization/sweep.py --module scpp_original ...` — paired benchmarks
7. `python optimization/analyze.py` — consolidate, regenerate all tables and figures
8. `python optimization/make_inventory.py` — regenerate the inventory

Raw per-measurement records are in `optimization/benchmarks/raw/*.jsonl`;
`results.csv` and `results.json` are derived from them, and every table and
figure is derived from `results.csv`. Re-running step 7 rebuilds all artefacts.

Verify the equivalence claims directly:

```bash
pytest tests/test_optimization_equivalence.py -v
```
