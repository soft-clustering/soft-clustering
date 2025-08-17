import numpy as np
from typeguard import typechecked
from typing import Optional
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment


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
        fuzzifier: int = 0,
        random_state: Optional[int] = None,
        max_iter: int = 0,
    ):  
        self.C = c_clusters
        self.n_imputations = n_imputations
        self.N = n_samples
        self.m = fuzzifier
        self.random_state = random_state
        self.max_iter = max_iter

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
        U = np.zeros((self.N, self.C))

        for i in range(self.N):
            for j in range(self.C):
                denominator = 0.0
                for _ in range(self.C):
                    dist = (self.X[i] - V[j]) ** 2
                    denominator += dist ** (2 / self.m - 1) / dist
                U[i, j] = 1 / denominator
        return U
    
    def update_cluster_centers(self, U):
        V = np.zeros((self.C, self.X.shape[1]))
        for j in range(self.C):
            for n in range(self.X.shape[1]):
                numerator = 0.0
                denominator = 0.0
                for i in range(self.N):
                    numerator += U[i, j] ** self.m * self.X[i, n]
                    denominator += U[i, j] ** self.m
                V[j, n] = numerator / denominator if denominator != 0 else 0.0
        return V

    def fit(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=float)
        
        # Drop fully-missing columns
        self.full_missing_mask = np.all(np.isnan(self.X), axis=0)
        if np.any(self.full_missing_mask):
            self.X = self.X[:, ~self.full_missing_mask]
        
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
                U = self.generate_membership(V)
            
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
            

class FeMIFuzzy:
    @typechecked
    def __init__(self,
                 random_state: Optional[int] = None,
                 max_iter: int = 100,
                 ):
        self.random_state = random_state
        self.max_iter = max_iter

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

    def _sammon_mapping(self, X, n_components=2, tol=1e-9):
        """
        Sammon mapping: projects data into lower-dimensional space while preserving pairwise distances.
        """
        max_iter = self.max_iter

        N, D = X.shape
        # Original pairwise distances in high-D
        d_high = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(d_high, 1e-9)  # avoid divide by zero
        scale = np.sum(d_high)

        # Initialize low-dim representation randomly
        rng = np.random.default_rng()
        Y = rng.normal(size=(N, n_components))

        for it in range(max_iter):
            d_low = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
            np.fill_diagonal(d_low, 1e-9)

            # Compute gradient
            delta = d_high - d_low
            ratio = delta / d_high

            # Update rule (gradient descent)
            grad = np.zeros_like(Y)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        grad[i] += ratio[i, j] * (Y[i] - Y[j]) / d_low[i, j]

            Y_new = Y + 0.3 * grad / scale  # step size 0.3 (tunable)

            # Convergence check (stress improvement small)
            stress_old = np.sum((delta**2) / d_high) / scale
            d_low_new = np.linalg.norm(Y_new[:, None, :] - Y_new[None, :, :], axis=2)
            np.fill_diagonal(d_low_new, 1e-9)
            delta_new = d_high - d_low_new
            stress_new = np.sum((delta_new**2) / d_high) / scale

            if abs(stress_old - stress_new) < tol:
                break

            Y = Y_new

        return Y

    def _xie_beni(self, X, U, V, N, m=2.0):
        num = 0.0
        pairwise = np.zeros((self.C, self.C))

        for j in range(self.C):
            for i in range(N):
                num += U[i, j] ** m * np.sum((X[i] - V[j]) ** 2)
            for k in range(self.C):
                    pairwise[k, j] = np.sum((V[k] - V[j]) ** 2)
        np.fill_diagonal(pairwise, np.inf)
        min_sep = pairwise.min()

        return num / (N * min_sep)
    
    def _cluster_signature(self, X, U, j):
        """
        Create a signature vector for cluster j.
        """
        members = X[np.argmax(U, axis=1) == j]  # hard members
        if members.shape[0] == 0:
            return np.zeros(X.shape[1] * 6 + 1)  # empty cluster

        signature = [
            members.shape[0],                # number of observations
            *np.mean(members, axis=0),
            *np.min(members, axis=0),
            *np.max(members, axis=0),
            *np.std(members, axis=0),
            *np.median(members, axis=0)
        ]
        return np.array(signature)

    def _match_centroids(self, X_ref, U_ref, X_new, U_new):
        """
        Match clusters between two imputations using Hungarian algorithm on
        signature vectors.
        """
        sig_ref = np.array([self._cluster_signature(X_ref, U_ref, j) for j in range(self.C)])
        sig_new = np.array([self._cluster_signature(X_new, U_new, j) for j in range(self.C)])

        dist_matrix = np.linalg.norm(sig_ref[:, None, :] - sig_new[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        return col_ind

    def fit(self, clients: np.ndarray, features) -> np.ndarray:
        clients = self._align_clients_features(clients, features)
        C_global = 0.0
        N = []
        centroids = []

        for client in clients:
            X = self._sammon_mapping(client)
            max_k = 10
            n = X.shape[0]
            N.append(n)
            xb_set = []
            V_set = []
            U_set = []

            for k in range(1, max_k + 1):
                self.C = k
                mifuzzy = _MIFuzzy(c_clusters=self.C, n_samples=n, random_state=self.random_state, max_iter=self.max_iter)
                mifuzzy.fit(X)
                U_set.append(mifuzzy.membership_matrices)
                V_set.append(mifuzzy.cluster_centers)

                for V_n, U_n in zip(V_set, U_set):
                    xb_set.append(self._xie_beni(X, U_n, V_n, n))
                    #sil = silhouette_score(X, np.argmax(U_final, axis=1))

            best_k = np.argmin(xb_set)

            V_1 = V_set[best_k - 1][0]
            U_1 = U_set[best_k - 1][0]

            aligned_centers = [V_1]
            aligned_memberships = [U_1]

            for V_n, U_n in zip(V_set[best_k - 1][1:], U_set[best_k - 1][1:]):
                mapping = self._match_centroids(X, U_1, X, U_n)
                aligned_centers.append(V_n[mapping])
                aligned_memberships.append(U_n[:, mapping])

            V_final = np.mean(aligned_centers, axis=0)
            U_final = np.mean(aligned_memberships, axis=0)
            centroids.append(V_final)

            C_global += n * best_k

        C_global = int(round(C_global / sum(N)))
        V_global = [np.zeros_like(centroids[0][j]) for j in range(int(C_global))]

        for j in range(C_global):
            V_global[j] = sum(N[k] * centroids[k][j] for k in range(len(clients))) / sum(N)

        return V_global
