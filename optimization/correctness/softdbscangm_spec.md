# Correctness specification — SoftDBSCANGM

The formulation the implementation must preserve. Written from the reference
implementation (`optimization/original/scpp_original/_softdbscangm.py`) before
any optimization, and used as the acceptance contract.

## Optimization variables

| Symbol | Shape | Meaning |
| --- | --- | --- |
| `U` | `(N, k)` | membership degrees |
| `V` | `(k, D)` | component centres (`self.centers`) |
| `S_j^-1` | `(D, D)` | inverse covariance of component `j` (`self.cov_inv[j]`) |

`k = (number of DBSCAN clusters) + (number of DBSCAN noise points)`. This is
not a hyperparameter: it is discovered, and grows with `N` on data where DBSCAN
labels many points as noise.

## Initialisation

1. `DBSCAN(eps, min_samples).fit(X)` gives `raw_labels`.
2. `cluster_ids = sorted unique labels excluding -1`; `noise_ids = indices where
   raw_labels == -1`, ascending.
3. `U` is one-hot: sample `i` in DBSCAN cluster `c` takes column
   `index_of(c) in cluster_ids`; noise sample `i` takes column
   `len(cluster_ids) + rank_of(i) in noise_ids`.
4. `V = 0`, `S_j^-1 = I` for all `j`.

The column assignment is part of the contract: an optimization may compute it
differently but must produce identical columns.

## Update equations

Per iteration, in this order:

1. `U_m = U ** m`
2. `w_j = sum_i U_m[i, j]`
3. **Centres** `V_j = (sum_i U_m[i,j] * X_i) / (w_j + 1e-10)`
4. **Covariance** `S_j = (sum_i U_m[i,j] * (X_i - V_j)(X_i - V_j)^T) / (w_j + 1e-10)`,
   then `S_j^-1 = inv(S_j + 1e-6 I)`
5. **Distances** `d(i,j) = sqrt((X_i - V_j)^T S_j^-1 (X_i - V_j))`, clamped
   below at `1e-10`
6. **Memberships** `U[i,j] = 1 / sum_t ( d(i,j) / d(i,t) ) ** (2/(m-1))`

## Constraints and semantics

- `U >= 0`, finite.
- Each row of `U` sums to 1 **by construction** of step 6, even though the class
  declares `_partition_constrained = False` (that flag reflects the density
  interpretation of the degrees, not the arithmetic).
- `labels_ = argmax(U, axis=1)`.
- The `1e-10` distance clamp and the `1e-10` / `1e-6` regularisers are part of
  the definition, not implementation detail: changing them changes results.

## Stopping criterion

`||V - V_prev||_F < tol`, checked after the membership update, at most
`max_iter` iterations. Convergence is tested on centres only.

## Randomisation

None beyond DBSCAN, which is deterministic for fixed input and parameters. Two
builds on the same input must therefore agree deterministically — no seed
alignment is required for this estimator.

## Expected output

`self.U` `(N, k)`, `self.centers` `(k, D)`, `self.cov_inv` a length-`k` list of
`(D, D)` arrays, `self.labels_` `(N,)`.

Note: `centers_` (the protocol attribute) is `None`, because this class stores
prototypes under `centers`, which is not among the names
`BaseSoftClusterer._centers_attrs` searches. Pre-existing; see report §11.

## Permitted transformations

Algebraic identities and changes of evaluation order only. Specifically
permitted and used:

- `1 / sum_t (d_j/d_t)^p == d_j^-p / sum_t d_t^-p`
- multiplying numerator and denominator by `min_t(d_t)^p`
- batching the per-component distance and inverse computations

## Prohibited

Changing `m`, `tol`, `max_iter`, the clamp constants, the convergence test, the
DBSCAN parameters or the column assignment; approximating the distance;
truncating iterations; or reducing `k` by merging noise components.

## Acceptance criteria

| Criterion | Threshold | Measured |
| --- | --- | --- |
| max abs membership difference | < 1e-9 | **2.22e-16** |
| label agreement | = 1.0 | **1.0000** |
| cluster count identical | yes | **yes** |
| invariants hold | yes | **yes** |

Evidence: `optimization/correctness/softdbscangm.json` (9 comparisons),
`tests/test_optimization_equivalence.py::TestSoftDBSCANGM`.
