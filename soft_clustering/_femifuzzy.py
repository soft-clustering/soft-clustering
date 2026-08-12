import numpy as np
from scipy.optimize import linear_sum_assignment
from typeguard import typechecked

from ._base import BaseSoftClusterer


class _MIFuzzy:
    """
    Multiple Imputation step for FeMIFuzzy.
    Generates multiple imputed datasets by filling missing values
    with column means plus small Gaussian noise.
    """

    def __init__(
        self,
        c_clusters: int,
        n_imputations: int = 5,
        n_samples: int = 0,
        fuzzifier: float = 2.0,
        random_state: int | None = None,
        max_iter: int = 0,
        tol: float = 1e-5,
    ):
        self.C = c_clusters
        self.n_imputations = n_imputations
        self.N = n_samples
        self.m = fuzzifier
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def multiple_imputer(self):
        # Precompute column means
        self.col_means = np.nanmean(self.X, axis=0)

        # Create imputed datasets
        rng_master = np.random.default_rng(self.random_state)
        imputed_datasets = []

        for _ in range(self.n_imputations):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            X_imp = self.X.copy()
            mask = np.isnan(X_imp)

            # Fill NaNs with mean + small Gaussian noise
            noise = rng.normal(scale=0.01, size=mask.sum())
            X_imp[mask] = np.take(self.col_means, np.where(mask)[1]) + noise

            imputed_datasets.append(X_imp)
        return imputed_datasets

    def generate_membership(self, V):
        """Fuzzy c-means memberships of ``self.X`` against prototypes ``V``.

        .. math::

            u_{ij} = \\Big[\\sum_{l} (d_{ij}^2 / d_{il}^2)^{1/(m-1)}\\Big]^{-1}

        Evaluated in log space with the per-sample minimum factored out, so a
        sample coincident with a prototype yields a one-hot row instead of a
        division by zero.
        """
        d2 = np.sum((self.X[:, None, :] - V[None, :, :]) ** 2, axis=2)
        log_d2 = np.log(np.maximum(d2, np.finfo(float).tiny))
        scores = -(log_d2 - log_d2.min(axis=1, keepdims=True)) / (self.m - 1.0)
        np.exp(scores, out=scores)
        return scores / scores.sum(axis=1, keepdims=True)

    def update_cluster_centers(self, U):
        """Membership-weighted prototypes, :math:`v_j = \\sum_i u_{ij}^m x_i /
        \\sum_i u_{ij}^m`."""
        U_m = U**self.m
        weights = U_m.sum(axis=0)
        V = (U_m.T @ self.X) / np.maximum(weights, np.finfo(float).tiny)[:, None]
        # A prototype that lost all its mass is left at the origin, matching
        # the previous behaviour of the zero-denominator branch.
        V[weights <= 0] = 0.0
        return V

    def fit(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=float)

        # Drop fully-missing columns
        self.full_missing_mask = np.all(np.isnan(self.X), axis=0)
        if np.any(self.full_missing_mask):
            self.X = self.X[:, ~self.full_missing_mask]

        # The sample count is a property of the data, not a hyperparameter.
        # Deriving it here keeps the estimator usable without the caller having
        # to pass n_samples, which otherwise defaults to 0 and makes the
        # centroid initialisation below fail.
        self.N = self.X.shape[0]
        if self.C > self.N:
            raise ValueError(
                f"c_clusters={self.C} exceeds the number of samples ({self.N})."
            )

        imputed_datasets = self.multiple_imputer()

        rng = np.random.default_rng(self.random_state)
        self.cluster_centers = []
        self.membership_matrices = []

        for X in imputed_datasets:
            # Initialize cluster centers randomly
            V = X[rng.choice(self.N, self.C, replace=False)]
            # Initialize membership matrix
            U = self.generate_membership(V)

            for _ in range(self.max_iter - 1):
                V = self.update_cluster_centers(U)
                U_new = self.generate_membership(V)
                # Stop once the partition settles. The federated sweep fits
                # this model once per candidate cluster count per imputation,
                # so running the full iteration budget every time dominates
                # the runtime for no change in the result.
                converged = np.abs(U_new - U).max() < self.tol
                U = U_new
                if converged:
                    break

            self.cluster_centers.append(V)
            self.membership_matrices.append(U)

    def transform(self, X):
        """Impute new data using learned means (single dataset)."""
        X = np.asarray(X, dtype=float)

        # Drop same fully-missing columns
        X = X[:, ~self.full_missing_mask]

        # Fill NaNs with learned means
        mask = np.isnan(X)
        X[mask] = np.take(self.col_means, np.where(mask)[1])

        return X


@typechecked
class FeMIFuzzy(BaseSoftClusterer):
    def __init__(
        self,
        random_state: int | None = None,
        max_iter: int = 100,
        fuzzifier: float = 2.0,
        n_imputations: int = 5,
    ):
        """
        Parameters
        ----------
        random_state : int or None
            Seed for the imputation noise and centroid initialisation.
        max_iter : int
            Iterations of the per-client fuzzy c-means and of Sammon mapping.
        fuzzifier : float
            Fuzzifier ``m > 1`` used by every client's local model and by the
            final assignment to the global prototypes.
        n_imputations : int
            Number of imputed datasets drawn per client.
        """
        if fuzzifier <= 1.0:
            raise ValueError(f"fuzzifier must be > 1, got {fuzzifier}")
        if n_imputations < 1:
            raise ValueError(f"n_imputations must be >= 1, got {n_imputations}")

        self.random_state = random_state
        self.max_iter = max_iter
        self.m = fuzzifier
        self.n_imputations = n_imputations

        self.rng = np.random.default_rng(random_state)

    def _align_clients_features(self, clients, features):
        # Find intersection of features across all clients
        common_features = set(features[0])
        for fnames in features[1:]:
            common_features &= set(fnames)
        common_features = list(common_features)

        if not common_features:
            raise ValueError("No common features across clients!")

        aligned_clients = []
        for X, fnames in zip(clients, features):
            # Find indices of common features in this client
            indices = [fnames.index(f) for f in common_features]
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
            return float(
                np.sum(((d_high - d)[off] ** 2) / d_high[off]) / 2.0 / c
            )

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

    def fit_predict(self, clients: list[np.ndarray], features) -> np.ndarray:
        """Fit the federated model and return the global membership matrix.

        Returns
        -------
        memberships : ndarray of shape (sum of client sample counts, C_global)
            Fuzzy memberships of every client's samples with respect to the
            aggregated global centroids, clients stacked in the order given.
            The global centroids are also stored as ``centers_``.

        Earlier releases returned only the list of global centroids and
        discarded the memberships, which left the estimator without the soft
        output the rest of the library is built on.
        """
        clients = self._align_clients_features(clients, features)
        C_global = 0.0
        N = []
        centroids = []
        projections = []

        for client in clients:
            X = self._sammon_mapping(client)
            projections.append(X)
            n = X.shape[0]
            # The model-selection sweep cannot request more clusters than the
            # client holds samples; centroid initialisation draws k distinct
            # points without replacement.
            max_k = min(10, n)
            if max_k < 2:
                raise ValueError(
                    "each client needs at least 2 samples for the Xie-Beni "
                    f"model-selection sweep; got {n}."
                )
            N.append(n)
            xb_set = []
            V_set = []
            U_set = []

            # The sweep starts at 2: Xie--Beni is undefined for a single
            # prototype and evaluates to zero there, so including C = 1 would
            # make the selection always collapse to one cluster.
            for k in range(2, max_k + 1):
                self.C = k
                mifuzzy = _MIFuzzy(
                    c_clusters=self.C,
                    n_imputations=self.n_imputations,
                    n_samples=n,
                    fuzzifier=self.m,
                    random_state=self.random_state,
                    max_iter=self.max_iter,
                )
                mifuzzy.fit(X)
                U_set.append(mifuzzy.membership_matrices)
                V_set.append(mifuzzy.cluster_centers)

            for V_n_list, U_n_list in zip(V_set, U_set):
                for V_n, U_n in zip(V_n_list, U_n_list):
                    xb_set.append(self._xie_beni(X, U_n, V_n, n, m=self.m))

            n_imp = len(V_set[0])
            best_flat = np.argmin(xb_set)
            best_k_idx = best_flat // n_imp
            best_imp_idx = best_flat % n_imp

            V_1 = V_set[best_k_idx][best_imp_idx]
            U_1 = U_set[best_k_idx][best_imp_idx]

            aligned_centers = [V_1]
            aligned_memberships = [U_1]

            for V_n, U_n in zip(
                V_set[best_k_idx][:best_imp_idx]
                + V_set[best_k_idx][best_imp_idx + 1 :],
                U_set[best_k_idx][:best_imp_idx]
                + U_set[best_k_idx][best_imp_idx + 1 :],
            ):
                mapping = self._match_centroids(X, U_1, X, U_n)
                aligned_centers.append(V_n[mapping])
                aligned_memberships.append(U_n[:, mapping])

            V_final = np.mean(aligned_centers, axis=0)
            centroids.append(V_final)

            # best_k_idx indexes the sweep, which starts at C = 2.
            C_global += n * (best_k_idx + 2)

        # Sample-weighted consensus on the number of clusters, then on the
        # prototypes. A client that selected fewer clusters than the consensus
        # contributes nothing to the surplus prototypes.
        C_global = max(1, int(round(C_global / sum(N))))
        n_features = centroids[0].shape[1]
        V_global = np.zeros((C_global, n_features))
        for j in range(C_global):
            weight = 0.0
            for k in range(len(clients)):
                if j < centroids[k].shape[0]:
                    V_global[j] += N[k] * centroids[k][j]
                    weight += N[k]
            if weight > 0:
                V_global[j] /= weight

        # Every client's samples are now scored against the shared prototypes
        # with the standard fuzzy c-means rule, evaluated in log space so a
        # sample coincident with a prototype cannot overflow.
        exponent = 2.0 / (self.m - 1.0) if getattr(self, "m", 2.0) > 1 else 2.0
        blocks = []
        for X in projections:
            d = np.linalg.norm(X[:, None, :] - V_global[None, :, :], axis=2)
            log_d = np.log(np.maximum(d, np.finfo(float).tiny))
            scores = -exponent * (log_d - log_d.min(axis=1, keepdims=True))
            np.exp(scores, out=scores)
            blocks.append(scores / scores.sum(axis=1, keepdims=True))

        self.memberships_ = np.vstack(blocks)
        self.centers_ = V_global
        self.global_centroids_ = V_global
        self.client_sizes_ = N
        self._n_samples_hint = int(sum(N))
        return self.memberships_
