import numpy as np
from typeguard import typechecked
from typing import Optional, Dict


@typechecked
class SFCMEP:
    def __init__(self,
                 K: int,
                 random_state: Optional[int] = None,
                 max_iter: int = 200,
                 rho: float = 0.5,
                 lam: float = 1.0,
                 tol: float = 1e-6,
                ):
        """
        Semi-supervised Fuzzy Clustering with Membership Prior (SFCMEP).

        Parameters
        ----------
        K : int
            Number of clusters (C).
        random_state : int, optional
            Random seed for reproducibility.
        max_iter : int, default=200
            Maximum number of iterations.
        rho : float, default=0.5
            Expert preference
        lam : float, default=1.0
            Scaling parameter used in exponential distance weighting.
        tol : float, default=1e-6
            Tolerance for convergence (used as stopping criterion).
        """
        self.C = K
        self.random_state = random_state
        self.max_iter = max_iter
        self.rho = rho
        self.lam = lam
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
    
    def _init_centroids(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers so that cluster i corresponds to class i.
        If y is None (unsupervised), choose random samples instead.
        """
        V_0 = np.zeros((self.C, X.shape[1]))

        if y is None or np.all(y == None):  # fully unsupervised
            idx = self.rng.choice(len(X), size=self.C, replace=False)
            V_0 = X[idx]
        else:
            for i in range(self.C):
                # samples of class i
                class_idx = [n for n in range(self.N) if i == y[n]]
                if len(class_idx) > 0:
                    # mean of samples as init center
                    V_0[i] = X[class_idx].mean(axis=0)
                else:
                    # if no sample for this class → random fallback
                    V_0[i] = X[self.rng.integers(len(X))]

        return V_0

    def _init_prior_membership(self, X: np.ndarray, y: np.ndarray, V):
        """
        Simulate the prior membership matrix based on ground-truth labels (y),
        initial cluster centers (V0), and expert preference rho.
        """
        u_org = np.full((self.C, self.N), np.nan)
        labeled_idx = []

        for j in range(self.N):
            if y[j] is None:    #unlabeled
                continue
            labeled_idx.append(j)
            dists = np.linalg.norm(V - X[j], axis=1)
            d_min = dists.min()
            for i in range(self.C):
                dist = dists[i]
                value = d_min / dist
                if y[j] == i:
                    u_org[i, j] = 1e-6 + self.rho * value
                else:
                    u_org[i, j] = (1 - self.rho) * value
                if u_org[i, j] > 1:
                    u_org[i, j] = 1

        M = np.nanmean(np.nanmax(u_org, axis=0))
        m = np.nanmean(np.nanmin(u_org, axis=0))
        rho_hat = 0.5 * (M + (1 - (self.C - 1) * m))
        u_org[np.isnan(u_org)] = rho_hat

        if labeled_idx:
            n_subset = max(1, len(labeled_idx) // 8)
            chosen_idx = self.rng.choice(labeled_idx, size=n_subset, replace=False)

            for j in chosen_idx:
                i_max = np.argmax(u_org[:, j])   # strongest cluster
                for i in range(self.C):
                    if i != i_max:
                        u_org[i, j] = rho_hat    # replace other clusters with baseline
        return u_org
    
    def _update_centroids(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Update cluster centers V given membership matrix U.
        """
        V = np.zeros((self.C, X.shape[1]))
        for i in range(self.C):
            V[i] = np.sum([U[i, j] * X[j] for j in range(self.N)]) / (U[i].sum() + 1e-8)
        return V
    
    def _update_membership_matrix(self, X: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Update membership matrix U given cluster centers V.
        """
        U_new = np.zeros((self.C, self.N))
        for j in range(self.N):
            if self.lam == 0:
                # HCM: hard assignment
                i_min = np.argmin(np.linalg.norm(V - X[j], axis=1))
                U_new[i_min, j] = 1.
            else:
                num = []
                dists = np.linalg.norm(V - X[j], axis=1)
                for i in range(self.C):
                    dist = dists[i]
                    exp = np.exp(-dist ** 2 / self.lam)
                    if np.isclose(U[i, j], 0.0):
                    # If membership is zero, recompute using exponential distance kernel
                        num.append(exp)
                    else:
                        num.append(U[i, j] * exp) 
                den = float(np.sum(num) + 1e-12)
                U_new[:, j] = np.array(num) / den
        return U_new
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform clustering with prior membership constraints.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Input data.
        y:  ndarray of labeled data.

        Returns
        -------
        U : ndarray of shape (C, N)
            Final membership matrix.
        V : ndarray of shape (C, D)
            Final cluster centers.
        """
        self.N = X.shape[0]
        V_0 = self._init_centroids(X, y)
        U_prior = self._init_prior_membership(X, y, V_0)
        U_prev = self._update_membership_matrix(X, U_prior, V_0)
        V_prev = V_0.copy()

        # Iterative optimization loop
        for t in range(self.max_iter):
            V = self._update_centroids(X, U_prev)
            U = self._update_membership_matrix(X, U_prev, V)
            # Convergence check
            if np.linalg.norm(U - U_prev) <= self.tol and np.linalg.norm(V - V_prev) <= self.tol:
                break
            U_prev = U
            V_prev = V

        return {
            "centroids": V,
            "membership_matrix": U.T
        }
