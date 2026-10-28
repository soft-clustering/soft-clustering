# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

import numpy as np
from scipy.optimize import linear_sum_assignment
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._fcm import _update_U_from_centers, fcm_fit
from ._validation import check_array, check_generator


@typechecked
class FeMIFuzzy(BaseSoftClusterer):
    r"""Federated fuzzy clustering with multiple imputation.

    Clusters data held by several clients that cannot pool their samples and
    may contain missing values, following the structure of Ngo et al. (2024):
    every client works on its own data and only prototypes and cluster counts
    are exchanged.

    1. Clients are restricted to the features they all share.
    2. **Local multiple imputation.** Every client replaces missing values by
       its own column means plus small Gaussian noise, ``n_imputations``
       times, and runs fuzzy c-means on every imputed copy; the copies'
       prototypes are aligned by Hungarian matching of cluster signatures and
       averaged.
    3. **Local model selection.** Each client picks the cluster count in
       ``2 .. max_clusters`` minimising the Xie--Beni index (averaged over
       imputations).
    4. **Server.** The global count is the sample-weighted mean of the local
       choices. Clients refit with it, and the server aligns their prototype
       sets to the largest client's by Hungarian matching on prototype
       distances and averages them, weighted by client size.
    5. Every client's (mean-imputed) samples receive fuzzy c-means
       memberships with respect to the global prototypes.

    Parameters
    ----------
    random_state : int, numpy Generator or None, default=None
    max_iter : int, default=100
        Fuzzy c-means iterations per fit.
    fuzzifier : float, default=2.0
        Fuzzifier ``m > 1``.
    n_imputations : int, default=5
    max_clusters : int, default=10
        Upper end of each client's model-selection sweep.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_total_samples, K)
        Clients stacked in the order given.
    centers_ : ndarray of shape (K, n_common_features)
        Global prototypes (also ``global_centroids_``).
    client_n_clusters_ : list of int
        Cluster count selected by each client.
    client_centroids_ : list of ndarray
        Each client's prototypes for the global cluster count.
    client_sizes_ : list of int

    References
    ----------
    H. Ngo, H. Fang, J. Rumbut and H. Wang. Federated fuzzy clustering for
    decentralized incomplete longitudinal behavioral data. *IEEE Internet of
    Things Journal*, 11(8):14657--14670, 2024. doi:10.1109/JIOT.2023.3343719

    Notes
    -----
    This is an implementation of the federated, multiple-imputation scheme
    described above; it has not been compared with the authors' code.
    Earlier releases Sammon-projected each client to two dimensions
    independently and averaged prototypes across those unaligned projections,
    matched global clusters by index without alignment, and failed on any
    missing value. The Sammon mapping is still available as
    :meth:`_sammon_mapping` but is no longer part of the pipeline.
    """

    def __init__(
        self,
        random_state: int | np.random.Generator | None = None,
        max_iter: int = 100,
        fuzzifier: float = 2.0,
        n_imputations: int = 5,
        max_clusters: int = 10,
    ):
        if fuzzifier <= 1.0:
            raise ValueError(f"FeMIFuzzy: fuzzifier must be > 1; got {fuzzifier}.")
        if n_imputations < 1:
            raise ValueError(
                f"FeMIFuzzy: n_imputations must be >= 1; got {n_imputations}."
            )
        if max_clusters < 2:
            raise ValueError(
                f"FeMIFuzzy: max_clusters must be >= 2; got {max_clusters}."
            )

        self.random_state = random_state
        self.max_iter = max_iter
        self.fuzzifier = fuzzifier
        self.m = fuzzifier
        self.n_imputations = n_imputations
        self.max_clusters = max_clusters

    def _align_clients_features(self, clients, features):
        # Find intersection of features across all clients
        # Ordered as in the first client. Building the list from a set made
        # the column order depend on string hash randomisation, so two runs
        # with the same seed could cluster differently.
        shared = set(features[0])
        for fnames in features[1:]:
            shared &= set(fnames)
        common_features = [f for f in features[0] if f in shared]

        if not common_features:
            raise ValueError("FeMIFuzzy: the clients share no common feature.")

        aligned_clients = []
        for X, fnames in zip(clients, features):
            # Find indices of common features in this client
            indices = [list(fnames).index(f) for f in common_features]
            aligned_clients.append(X[:, indices])

        return aligned_clients

    def _sammon_mapping(self, X, n_components=2, tol=1e-9, magic_factor=0.3):
        """Sammon mapping: a projection that preserves pairwise distances.

        Minimises Sammon's stress

        .. math::

            E = \\frac{1}{c} \\sum_{i<j}
                \\frac{(d^{*}_{ij} - d_{ij})^2}{d^{*}_{ij}},
            \\qquad c = \\sum_{i<j} d^{*}_{ij},

        with the diagonal-Newton step of Sammon (1969),
        :math:`y \\leftarrow y - \\alpha \\,(\\partial E/\\partial y) /
        \\lvert \\partial^2 E/\\partial y^2 \\rvert`.

        Two properties matter for the estimator built on top of this. The
        initialisation is the leading principal components rather than a
        random draw, which makes the projection deterministic and starts it in
        a configuration whose distances are already roughly right; and the
        step is scaled by the curvature, so it does not depend on the
        magnitude of the data. A plain gradient step of fixed size divided by
        the sum of all pairwise distances --- what this routine used
        previously --- is smaller than the required displacement by orders of
        magnitude on well-separated data, and returns the initialisation
        essentially unchanged.
        """
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        if N < 2:
            return np.zeros((N, n_components))

        d_high = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        off = ~np.eye(N, dtype=bool)
        # A zero original distance would divide by zero in the stress; floor
        # it at a small fraction of the smallest non-zero distance.
        positive = d_high[off & (d_high > 0)]
        floor = 1e-9 if positive.size == 0 else max(positive.min() * 1e-6, 1e-12)
        d_high = np.where(off, np.maximum(d_high, floor), 1.0)
        c = float(d_high[off].sum()) / 2.0
        if c <= 0:
            return np.zeros((N, n_components))

        # Deterministic PCA initialisation.
        centred = X - X.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            Y = centred @ vt[:n_components].T
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            rng = np.random.default_rng(self.random_state)
            Y = rng.normal(size=(N, n_components))
        if Y.shape[1] < n_components:
            Y = np.hstack([Y, np.zeros((N, n_components - Y.shape[1]))])

        def stress(Z):
            d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
            return float(np.sum(((d_high - d)[off] ** 2) / d_high[off]) / 2.0 / c)

        previous = stress(Y)
        for _ in range(max(self.max_iter, 1)):
            d_low = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
            d_low = np.where(off, np.maximum(d_low, floor), 1.0)

            delta = d_high - d_low
            inv = np.where(off, 1.0 / (d_high * d_low), 0.0)
            diff = Y[:, None, :] - Y[None, :, :]

            first = -(2.0 / c) * np.einsum("ij,ijp->ip", inv * delta, diff)
            second = -(2.0 / c) * np.einsum(
                "ijp->ip",
                inv[:, :, None]
                * (
                    delta[:, :, None]
                    - (diff**2 / d_low[:, :, None])
                    * (1.0 + delta[:, :, None] / d_low[:, :, None])
                ),
            )

            step = first / np.maximum(np.abs(second), 1e-12)
            candidate = Y - magic_factor * step

            # Sammon's rule: halve the step while it fails to reduce stress.
            current = stress(candidate)
            factor = magic_factor
            for _ in range(10):
                if current <= previous:
                    break
                factor /= 2.0
                candidate = Y - factor * step
                current = stress(candidate)

            if previous - current < tol:
                Y = candidate if current < previous else Y
                break
            Y, previous = candidate, current

        return Y

    def _xie_beni(self, X, U, V, N, m=2.0):
        """Xie--Beni validity index. Defined only for two or more prototypes.

        With a single prototype the minimum pairwise separation is empty, the
        index evaluates to zero, and a model-selection sweep that included
        ``C = 1`` would always select it. Callers must therefore start the
        sweep at ``C = 2``; this returns ``inf`` for ``C = 1`` so that a
        mistake there is loud rather than silent.
        """
        C = U.shape[1]
        if C < 2:
            return np.inf

        d2 = np.sum((X[:, None, :] - V[None, :, :]) ** 2, axis=2)
        num = float(np.sum((U**m) * d2))

        pairwise = np.sum((V[:, None, :] - V[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(pairwise, np.inf)
        min_sep = pairwise.min()
        if not np.isfinite(min_sep) or min_sep <= 0:
            return np.inf

        return num / (N * min_sep)

    def _cluster_signature(self, X, U, j):
        """
        Create a signature vector for cluster j.
        """
        members = X[np.argmax(U, axis=1) == j]  # hard members
        if members.shape[0] == 0:
            return np.zeros(X.shape[1] * 6 + 1)  # empty cluster

        signature = [
            members.shape[0],  # number of observations
            *np.mean(members, axis=0),
            *np.min(members, axis=0),
            *np.max(members, axis=0),
            *np.std(members, axis=0),
            *np.median(members, axis=0),
        ]
        return np.array(signature)

    def _match_centroids(self, X_ref, U_ref, X_new, U_new):
        C = U_ref.shape[1]
        sig_ref = np.array([self._cluster_signature(X_ref, U_ref, j) for j in range(C)])
        sig_new = np.array([self._cluster_signature(X_new, U_new, j) for j in range(C)])

        dist_matrix = np.linalg.norm(sig_ref[:, None, :] - sig_new[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        return col_ind

    def _impute(self, X, rng):
        """Multiple imputation: column mean plus small Gaussian noise."""
        mask = np.isnan(X)
        if not mask.any():
            return [X] * self.n_imputations, X
        col_means = np.nanmean(np.where(mask.all(axis=0), 0.0, X), axis=0)
        mean_imputed = np.where(mask, col_means[None, :], X)
        scale = np.nanstd(X, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        imputations = []
        for _ in range(self.n_imputations):
            noise = rng.normal(0.0, 0.01, size=X.shape) * scale[None, :]
            imputations.append(np.where(mask, col_means[None, :] + noise, X))
        return imputations, mean_imputed

    def _mi_fcm(self, imputations, C, rng):
        """FCM on every imputed copy, aligned to the first and averaged."""
        results = [
            fcm_fit(X, C, self.m, rng, max_iter=self.max_iter, tol=1e-6)[:2]
            for X in imputations
        ]
        U_ref, V_ref = results[0]
        X_ref = imputations[0]
        aligned = [V_ref]
        xb = [self._xie_beni(X_ref, U_ref, V_ref, X_ref.shape[0], m=self.m)]
        for X, (U, V) in zip(imputations[1:], results[1:]):
            mapping = self._match_centroids(X_ref, U_ref, X, U)
            aligned.append(V[mapping])
            xb.append(self._xie_beni(X, U, V, X.shape[0], m=self.m))
        return np.mean(aligned, axis=0), float(np.mean(xb))

    def fit_predict(self, clients: list[np.ndarray], features) -> np.ndarray:
        """Fit the federated model and return the global membership matrix.

        Parameters
        ----------
        clients : list of ndarray
            One ``(n_samples_k, n_features_k)`` matrix per client. Missing
            values may be given as NaN; they are imputed locally.
        features : list of list of str
            Feature names of every client's columns. Only the features common
            to all clients are used, in the order of the first client.

        Returns
        -------
        ndarray of shape (sum of client sample counts, K)
            Memberships of every client's samples in the global clusters,
            clients stacked in the order given.
        """
        name = type(self).__name__
        if len(clients) != len(features):
            raise ValueError(
                f"{name}: got {len(clients)} clients but {len(features)} feature lists."
            )
        clients = [
            check_array(c, estimator=name, name=f"clients[{i}]", allow_nan=True)
            for i, c in enumerate(clients)
        ]
        clients = self._align_clients_features(clients, features)
        for i, X in enumerate(clients):
            if X.shape[0] < 2:
                raise ValueError(
                    f"{name}: each client needs at least 2 samples for the "
                    f"Xie-Beni model-selection sweep; client {i} has {X.shape[0]}."
                )
            empty = np.isnan(X).all(axis=0)
            if empty.any():
                raise ValueError(
                    f"{name}: client {i} has no observed value for "
                    f"{int(empty.sum())} common feature(s); they cannot be "
                    "imputed locally."
                )

        rng = check_generator(self.random_state)
        imputed = [self._impute(X, rng) for X in clients]
        sizes = [X.shape[0] for X in clients]

        # Round 1: every client selects its own cluster count by Xie-Beni.
        # The sweep starts at 2: the index is undefined for one prototype.
        chosen = []
        for imputations, _ in imputed:
            n = imputations[0].shape[0]
            candidates = range(2, min(self.max_clusters, n) + 1)
            scores = [self._mi_fcm(imputations, C, rng)[1] for C in candidates]
            chosen.append(list(candidates)[int(np.argmin(scores))])
        C_global = int(round(np.dot(sizes, chosen) / sum(sizes)))
        C_global = max(2, min(C_global, min(sizes)))

        # Round 2: every client fits the consensus count; the server aligns
        # the prototype sets (Hungarian matching on prototype distances,
        # against the largest client) and averages them, weighted by size.
        local = [self._mi_fcm(imp, C_global, rng)[0] for imp, _ in imputed]
        ref = int(np.argmax(sizes))
        V_global = np.zeros_like(local[ref])
        for V, n in zip(local, sizes):
            cost = np.linalg.norm(local[ref][:, None, :] - V[None, :, :], axis=2)
            _, col = linear_sum_assignment(cost)
            V_global += n * V[col]
        V_global /= sum(sizes)

        blocks = [
            _update_U_from_centers(mean_imputed, V_global, self.m)
            for _, mean_imputed in imputed
        ]
        self.client_n_clusters_ = chosen
        self.client_centroids_ = local
        self.memberships_ = np.vstack(blocks)
        self.centers_ = V_global
        self.global_centroids_ = V_global
        self.client_sizes_ = sizes
        self._n_samples_hint = int(sum(sizes))
        return self.memberships_
