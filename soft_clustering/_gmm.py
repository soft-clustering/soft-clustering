import torch
import random
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from typeguard import typechecked
from typing import Optional


def _logsumexp(a, axis=1):
    """Stable log-sum-exp."""
    m = np.max(a, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))


def _kmeanspp_init(X, K, rng):
    """k-means++ initialization for means."""
    n, d = X.shape
    means = np.empty((K, d), dtype=X.dtype)
    idx0 = rng.integers(0, n)
    means[0] = X[idx0]
    for k in range(1, K):
        # distances to nearest current center
        d2_list = [np.sum((X - m) ** 2, axis=1) for m in means[:k]]
        d2_min = d2_list[0]
        for arr in d2_list[1:]:
            d2_min = np.minimum(d2_min, arr)
        s = d2_min.sum()
        if not np.isfinite(s) or s <= 0.0:
            idx = rng.integers(0, n)
        else:
            probs = d2_min / s
            idx = rng.choice(n, p=probs)
        means[k] = X[idx]
    return means


def _estimate_log_gaussian_prob_full(X, means, covariances, eps=1e-6):
    """Log N(x|mu,Sigma) for 'full' covariances."""
    n, d = X.shape
    K = means.shape[0]
    log_prob = np.empty((n, K), dtype=np.float64)
    I = np.eye(d, dtype=np.float64)
    for k in range(K):
        S = covariances[k] + eps * I
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            S = S + (10 * eps) * I
            L = np.linalg.cholesky(S)
        diff = X - means[k]
        y = np.linalg.solve(L, diff.T)
        sq = np.sum(y * y, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_prob[:, k] = -0.5 * (d * np.log(2.0 * np.pi) + log_det + sq)
    return log_prob


def _estimate_log_gaussian_prob_diag(X, means, covariances, eps=1e-6):
    """Log N(x|mu,diag(var)) for 'diag' covariances."""
    n, d = X.shape
    K = means.shape[0]
    log_prob = np.empty((n, K), dtype=np.float64)
    for k in range(K):
        var = covariances[k] + eps
        diff = X - means[k]
        log_prob[:, k] = -0.5 * (np.sum(np.log(2.0 * np.pi * var)) +
                                 np.sum((diff * diff) / var, axis=1))
    return log_prob


def _estimate_log_gaussian_prob_spherical(X, means, covariances, eps=1e-6):
    """Log N(x|mu, s^2 I) for 'spherical' covariances."""
    n, d = X.shape
    K = means.shape[0]
    log_prob = np.empty((n, K), dtype=np.float64)
    for k in range(K):
        var = covariances[k] + eps
        diff = X - means[k]
        log_prob[:, k] = -0.5 * (d * np.log(2.0 * np.pi * var) +
                                 np.sum(diff * diff, axis=1) / var)
    return log_prob


def _estimate_log_gaussian_prob(X, means, covariances, covariance_type, eps=1e-6):
    if covariance_type == 'full':
        return _estimate_log_gaussian_prob_full(X, means, covariances, eps)
    if covariance_type == 'diag':
        return _estimate_log_gaussian_prob_diag(X, means, covariances, eps)
    if covariance_type == 'spherical':
        return _estimate_log_gaussian_prob_spherical(X, means, covariances, eps)
    raise ValueError(f"Unknown covariance_type='{covariance_type}'.")


def _m_step(X, resp, covariance_type, reg_covar):
    """M-step: update weights, means, and covariances."""
    n, d = X.shape
    K = resp.shape[1]
    Nk = resp.sum(axis=0) + 1e-12
    weights = Nk / n
    means = (resp.T @ X) / Nk[:, None]

    if covariance_type == 'full':
        covariances = np.empty((K, d, d), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            covariances[k] = (diff * resp[:, [k]]).T @ diff / Nk[k]
            covariances[k].flat[:: d + 1] += reg_covar
    elif covariance_type == 'diag':
        covariances = np.empty((K, d), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            covariances[k] = (resp[:, [k]] * (diff * diff)).sum(axis=0) / Nk[k] + reg_covar
    elif covariance_type == 'spherical':
        covariances = np.empty((K,), dtype=np.float64)
        for k in range(K):
            diff = X - means[k]
            covariances[k] = (resp[:, k] * np.sum(diff * diff, axis=1)).sum() / (Nk[k] * d) + reg_covar
    else:
        raise ValueError(f"Unknown covariance_type='{covariance_type}'.")
    return weights, means, covariances


@typechecked
class GaussianMixtureEM:
    def __init__(self,
                 covariance_type: str = 'full',
                 reg_covar: float = 1e-6,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 init_params: str = 'kmeans++',
                 random_state: Optional[int] = None):
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.init_params = init_params
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            np.random.seed(random_state)
            random.seed(random_state)

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.responsibilities_ = None
        self.lower_bound_ = None
        self.log_likelihood_trajectory_ = None

    def fit_predict(self, X, K):
        """Fit the model with EM and return responsibilities.

        Args:
            X: Data matrix (numpy.ndarray or scipy.sparse), shape (n_samples, n_features).
            K: Number of mixture components (int).

        Returns:
            responsibilities: Posterior probabilities (numpy.ndarray, shape (n_samples, K)).
        """
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if K <= 0 or K > n:
            raise ValueError(f"Invalid K: {K}. Must be in [1, n_samples].")

        rng = np.random.default_rng(self.random_state)

        # initialization
        if self.init_params == 'kmeans++':
            means = _kmeanspp_init(X, K, rng)
        elif self.init_params == 'random':
            idx = rng.integers(0, n, size=K)
            means = X[idx].copy()
        else:
            raise ValueError(f"Unknown init_params='{self.init_params}'.")

        weights = np.full(K, 1.0 / K, dtype=np.float64)

        # initial covariances from global variance
        Xc = X - np.mean(X, axis=0, keepdims=True)
        S = (Xc.T @ Xc) / max(n - 1, 1)
        if self.covariance_type == 'full':
            covariances = np.array([S + self.reg_covar * np.eye(d) for _ in range(K)], dtype=np.float64)
        elif self.covariance_type == 'diag':
            diag = np.diag(S) + self.reg_covar
            covariances = np.tile(diag, (K, 1))
        elif self.covariance_type == 'spherical':
            var = np.mean(np.diag(S)) + self.reg_covar
            covariances = np.full(K, var, dtype=np.float64)
        else:
            raise ValueError(f"Unknown covariance_type='{self.covariance_type}'.")

        ll_old = -np.inf
        ll_trace = []
        for _ in range(self.max_iter):
            # E-step
            log_prob = _estimate_log_gaussian_prob(
                X, means, covariances, self.covariance_type, eps=self.reg_covar
            )
            log_prob += np.log(weights + 1e-32)
            log_norm = _logsumexp(log_prob, axis=1)
            ll = float(np.sum(log_norm))
            ll_trace.append(ll)
            log_resp = log_prob - log_norm
            resp = np.exp(log_resp)
            resp /= (resp.sum(axis=1, keepdims=True) + 1e-12)

            # M-step
            weights, means, covariances = _m_step(X, resp, self.covariance_type, self.reg_covar)

            # convergence
            if abs(ll - ll_old) <= self.tol:
                break
            ll_old = ll

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.responsibilities_ = resp
        self.lower_bound_ = ll
        self.log_likelihood_trajectory_ = np.array(ll_trace, dtype=np.float64)
        return resp
