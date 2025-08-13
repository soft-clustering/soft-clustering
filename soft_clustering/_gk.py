import torch
import random
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from typeguard import typechecked
from typing import Optional


def _kmeanspp_init(X, K, rng):
    """k-means++ style seeding from data points."""
    n, d = X.shape
    centers = np.empty((K, d), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    for k in range(1, K):
        d2_list = [np.sum((X - c) ** 2, axis=1) for c in centers[:k]]
        d2_min = d2_list[0]
        for arr in d2_list[1:]:
            d2_min = np.minimum(d2_min, arr)
        s = float(d2_min.sum())
        if not np.isfinite(s) or s <= 0.0:
            idx = rng.integers(0, n)
        else:
            probs = d2_min / s
            idx = rng.choice(n, p=probs)
        centers[k] = X[idx]
    return centers


def _compute_covariances(X, U, centers, m, reg_covar=1e-6):
    """Fuzzy covariance per cluster."""
    n, d = X.shape
    K = centers.shape[0]
    C = np.empty((K, d, d), dtype=np.float64)
    Um = U ** m
    for k in range(K):
        w = Um[:, k:k+1]
        diff = X - centers[k]
        Nk = float(np.sum(w)) + 1e-12
        S = (diff * w).T @ diff / Nk
        S.flat[:: d + 1] += reg_covar
        C[k] = S
    return C


def _estimate_gk_dist2(X, centers, covariances):
    """GK distance: det(C_k)^{1/d} * (x - c_k)^T C_k^{-1} (x - c_k)."""
    n, d = X.shape
    K = centers.shape[0]
    D2 = np.empty((n, K), dtype=np.float64)
    I = None
    for k in range(K):
        S = covariances[k]
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            if I is None:
                I = np.eye(S.shape[0], dtype=np.float64)
            S = S + 10.0 * np.finfo(np.float64).eps * I
            L = np.linalg.cholesky(S)
        diff = X - centers[k]
        y = np.linalg.solve(L, diff.T)
        sq = np.sum(y * y, axis=0)  # Mahalanobis^2
        log_det = 2.0 * np.sum(np.log(np.diag(L)))  # log|C_k|
        scale = np.exp(log_det / d)  # det(C_k)^{+1/d}  ✅ fixed sign
        D2[:, k] = scale * sq
    np.maximum(D2, 1e-18, out=D2)
    return D2


def _update_U_from_d2(d2, m, eps=1e-12):
    """Update memberships given GK distances."""
    inv = 1.0 / (d2 + eps)
    power = 1.0 / (m - 1.0)
    inv_pow = inv ** power
    denom = np.sum(inv_pow, axis=1, keepdims=True)
    U = inv_pow / (denom + eps)
    U = np.maximum(U, eps)
    U /= (np.sum(U, axis=1, keepdims=True) + eps)
    return U


def _objective(U, d2, m):
    return float(np.sum((U ** m) * d2))


@typechecked
class GustafsonKessel:
    def __init__(self,
                 random_state: Optional[int] = None,
                 m: float = 2.0,
                 max_iter: int = 300,
                 tol: float = 1e-5,
                 init: str = 'kmeans++',
                 reg_covar: float = 1e-6):
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            np.random.seed(random_state)
            random.seed(random_state)

        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.reg_covar = reg_covar

        self.centers_ = None
        self.memberships_ = None
        self.covariances_ = None
        self.metrics_A_ = None
        self.objective_trajectory_ = None

    def fit_predict(self, X, K):
        """Fit the GK model and compute fuzzy memberships.

        Args:
            X: Data matrix (numpy.ndarray or scipy.sparse), shape (n_samples, n_features).
            K: Number of clusters (int).

        Returns:
            memberships: Membership matrix (numpy.ndarray, shape (n_samples, K)).
        """
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if K <= 0 or K > n:
            raise ValueError(f"Invalid K: {K}. Must be in [1, n_samples].")
        if self.m <= 1.0:
            raise ValueError(f"m must be > 1. Got m={self.m}.")

        rng = np.random.default_rng(self.random_state)

        if self.init == 'kmeans++':
            centers = _kmeanspp_init(X, K, rng)
            U = np.full((n, K), 1.0 / K, dtype=np.float64)
        elif self.init == 'random':
            U = rng.random((n, K))
            U = U / (np.sum(U, axis=1, keepdims=True) + 1e-12)
            Um = U ** self.m
            denom = np.sum(Um, axis=0, keepdims=True).T
            centers = (Um.T @ X) / (denom + 1e-12)
        else:
            raise ValueError(f"Unknown init='{self.init}'.")

        obj_prev = np.inf
        traj = []
        for _ in range(self.max_iter):
            C = _compute_covariances(X, U, centers, self.m, reg_covar=self.reg_covar)
            d2 = _estimate_gk_dist2(X, centers, C)
            U = _update_U_from_d2(d2, self.m)
            Um = U ** self.m
            denom = np.sum(Um, axis=0, keepdims=True).T
            centers = (Um.T @ X) / (denom + 1e-12)
            obj = _objective(U, d2, self.m)
            traj.append(obj)
            if abs(obj_prev - obj) <= self.tol:
                break
            obj_prev = obj

        self.centers_ = centers
        self.memberships_ = U
        self.covariances_ = C

        # A_k = det(C_k)^{1/d} * C_k^{-1}  (so det(A_k) = 1)  ✅ fixed sign
        Kc, d = C.shape[0], C.shape[1]
        A = np.empty_like(C)
        for k in range(Kc):
            S = C[k]
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                S = S + 10.0 * np.finfo(np.float64).eps * np.eye(d, dtype=np.float64)
                L = np.linalg.cholesky(S)
            invS = np.linalg.inv(S)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            scale = np.exp(log_det / d)  # det(C_k)^{+1/d}
            A[k] = scale * invS

        self.metrics_A_ = A
        self.objective_trajectory_ = np.array(traj, dtype=np.float64)
        return self.memberships_
