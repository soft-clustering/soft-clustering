import torch
import random
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from typeguard import typechecked
from typing import Optional


def _euclidean_dist2(X, centers):
    """Compute squared Euclidean distances between samples and centers.

    Args:
        X: data matrix, shape (n_samples, n_features)
        centers: cluster centers, shape (K, n_features)

    Returns:
        dist2: squared distances, shape (n_samples, K)
    """
    x_norm = np.sum(X * X, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    dist2 = x_norm + c_norm - 2.0 * (X @ centers.T)
    np.maximum(dist2, 0.0, out=dist2)
    return dist2


def _normalize_memberships(U, eps=1e-12):
    U = np.maximum(U, eps)
    U_sum = U.sum(axis=1, keepdims=True)
    U /= (U_sum + eps)
    return U


def _init_centers_kpp(X, K, rng):
    """k-means++ style seeding from data points."""
    n, _ = X.shape
    centers = np.empty((K, X.shape[1]), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    for k in range(1, K):
        dist2 = _euclidean_dist2(X, centers[:k])
        d_min = np.min(dist2, axis=1)
        s = np.sum(d_min)
        if not np.isfinite(s) or s <= 0.0:
            idx = rng.integers(0, n)
        else:
            probs = d_min / s
            idx = rng.choice(n, p=probs)
        centers[k] = X[idx]
    return centers


def _update_U_from_centers(X, centers, m, eps=1e-12):
    """Update membership matrix given centers."""
    dist2 = _euclidean_dist2(X, centers) + eps
    inv_dist = 1.0 / dist2
    power = 1.0 / (m - 1.0)
    inv_dist_pow = inv_dist ** power
    denom = np.sum(inv_dist_pow, axis=1, keepdims=True)
    U = inv_dist_pow / (denom + eps)
    return _normalize_memberships(U, eps=eps)


def _update_centers_from_U(X, U, m, eps=1e-12):
    """Update centers given membership matrix."""
    Um = U ** m
    num = Um.T @ X
    den = np.sum(Um, axis=0, keepdims=True).T
    centers = num / (den + eps)
    return centers


def _objective(X, U, centers, m):
    dist2 = _euclidean_dist2(X, centers)
    return float(np.sum((U ** m) * dist2))


@typechecked
class FuzzyCMeans:
    def __init__(self,
                 random_state: Optional[int] = None,
                 m: float = 2.0,
                 max_iter: int = 300,
                 tol: float = 1e-5,
                 init: str = 'kmeans++'):
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

        self.centers_ = None
        self.memberships_ = None
        self.objective_trajectory_ = None

    def fit_predict(self, X, K):
        """Fit the model and compute fuzzy memberships.

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
            centers = _init_centers_kpp(X, K, rng)
        elif self.init == 'random':
            U = _normalize_memberships(rng.random((n, K)))
            centers = _update_centers_from_U(X, U, self.m)
        else:
            raise ValueError(f"Unknown init='{self.init}'.")

        obj_prev = np.inf
        trajectory = []
        for _ in range(self.max_iter):
            U = _update_U_from_centers(X, centers, self.m)
            centers = _update_centers_from_U(X, U, self.m)
            obj = _objective(X, U, centers, self.m)
            trajectory.append(obj)
            if abs(obj_prev - obj) <= self.tol:
                break
            obj_prev = obj

        self.centers_ = centers
        self.memberships_ = _normalize_memberships(U)
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        return self.memberships_
