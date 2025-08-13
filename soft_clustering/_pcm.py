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


def _init_centers_kpp(X, K, rng):
    """k-means++ style seeding from data points."""
    n, _ = X.shape
    centers = np.empty((K, X.shape[1]), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    for k in range(1, K):
        d2 = _euclidean_dist2(X, centers[:k])
        dmin = np.min(d2, axis=1)
        s = dmin.sum()
        if not np.isfinite(s) or s <= 0.0:
            idx = rng.integers(0, n)
        else:
            probs = dmin / s
            idx = rng.choice(n, p=probs)
        centers[k] = X[idx]
    return centers


def _update_T_from_centers(X, centers, etas, m, eps=1e-12):
    """Update typicality matrix given centers and eta (per-cluster scale)."""
    d2 = _euclidean_dist2(X, centers) + eps
    power = 1.0 / (m - 1.0)
    # t_ik = 1 / (1 + (d2_ik / eta_k)^{power})
    T = 1.0 / (1.0 + (d2 / (etas[None, :] + eps)) ** power)
    return np.clip(T, eps, 1.0)


def _update_centers_from_T(X, T, m, eps=1e-12):
    """Update centers given typicalities."""
    Tm = T ** m
    num = Tm.T @ X
    den = np.sum(Tm, axis=0, keepdims=True).T
    centers = num / (den + eps)
    return centers


def _update_etas(X, T, centers, m, alpha=1.0, eps=1e-12):
    """Update per-cluster eta as weighted intra-cluster variance."""
    d2 = _euclidean_dist2(X, centers)
    Tm = T ** m
    num = np.sum(Tm * d2, axis=0)  # (K,)
    den = np.sum(Tm, axis=0) + eps  # (K,)
    etas = alpha * (num / den + eps)
    return np.maximum(etas, eps)


def _objective_pcm(X, T, centers, etas, m):
    d2 = _euclidean_dist2(X, centers)
    term1 = np.sum((T ** m) * d2)
    term2 = np.sum(etas[None, :] * ((1.0 - T) ** m))
    return float(term1 + term2)


@typechecked
class PossibilisticCMeans:
    def __init__(self,
                 random_state: Optional[int] = None,
                 m: float = 2.0,
                 alpha: float = 1.0,
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
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.init = init

        self.centers_ = None
        self.typicalities_ = None
        self.etas_ = None
        self.objective_trajectory_ = None

    def fit_predict(self, X, K):
        """Fit the PCM model and compute typicalities.

        Args:
            X: Data matrix (numpy.ndarray or scipy.sparse), shape (n_samples, n_features).
            K: Number of clusters (int).

        Returns:
            typicalities: Typicality matrix (numpy.ndarray, shape (n_samples, K)).
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

        # initialize centers
        if self.init == 'kmeans++':
            centers = _init_centers_kpp(X, K, rng)
        elif self.init == 'random':
            idx = rng.integers(0, n, size=K)
            centers = X[idx].copy()
        else:
            raise ValueError(f"Unknown init='{self.init}'.")

        # initialize etas by average within-cluster squared distance (hard assign)
        d2 = _euclidean_dist2(X, centers)
        labels = np.argmin(d2, axis=1)
        etas = np.empty(K, dtype=np.float64)
        global_mean = np.mean(d2, axis=0)
        for k in range(K):
            mask = (labels == k)
            if np.any(mask):
                etas[k] = np.mean(d2[mask, k])
            else:
                etas[k] = global_mean[k]
        etas = np.maximum(self.alpha * (etas + 1e-12), 1e-12)

        # initial typicalities
        T = _update_T_from_centers(X, centers, etas, self.m)

        obj_prev = np.inf
        trajectory = []
        for _ in range(self.max_iter):
            centers = _update_centers_from_T(X, T, self.m)
            etas = _update_etas(X, T, centers, self.m, alpha=self.alpha)
            T = _update_T_from_centers(X, centers, etas, self.m)
            obj = _objective_pcm(X, T, centers, etas, self.m)
            trajectory.append(obj)
            if abs(obj_prev - obj) <= self.tol:
                break
            obj_prev = obj

        self.centers_ = centers
        self.typicalities_ = np.clip(T, 1e-12, 1.0)
        self.etas_ = etas
        self.objective_trajectory_ = np.array(trajectory, dtype=np.float64)
        return self.typicalities_
