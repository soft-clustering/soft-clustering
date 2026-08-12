"""Multivariate Beta Mixture Model.

Implementation note (optimization study)
----------------------------------------
Vectorised implementation of the same model. The likelihood, the E and M
updates, the initialisation draws, the clipping order and the convergence test
are unchanged. The reference implementation is preserved at
``optimization/original/scpp_original/_mbmm.py``.

Profiling attributed the runtime to ``scipy.stats.beta.logpdf``, called
``n_components * n_features`` times per E-step and again per log-likelihood
evaluation — 4,800 calls for a 200x8 input. Almost all of that is
``rv_continuous`` machinery (argument checking, support masks, broadcasting)
rather than the density itself: the profile showed ``_argcheck`` and
``_support_mask`` costing as much as ``_logpdf``.

The frozen distribution is therefore replaced by the identity SciPy evaluates
internally,

    log Beta(x; a, b) = xlogy(a - 1, x) + xlog1py(b - 1, -x) - betaln(a, b)

applied to all features at once. This is the same expression SciPy's
``beta._logpdf`` computes, using the same ``scipy.special`` primitives, so the
values agree to floating point; it simply evaluates one array per component
instead of one scalar call per (component, feature). Samples are required to
lie in (0, 1), which the model already assumes, so the support mask SciPy
applies is not needed.

The M-step loop over features is likewise replaced by array operations, keeping
the original order of computation: the variance uses the *unclipped* mean, and
the mean is clipped only afterwards.
"""

import numpy as np
from scipy.special import betaln, xlog1py, xlogy
from typeguard import typechecked

from ._base import BaseSoftClusterer


@typechecked
class MBMM(BaseSoftClusterer):
    def __init__(self, n_components: int = 3, max_iter: int = 100, tol: float = 1e-5):
        """
        Parameters:
        - n_components (int): Number of mixture components
        - max_iter (int): Maximum number of EM iterations
        - tol (float): Convergence threshold on log-likelihood
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.alpha = None  # (K, D)
        self.beta = None  # (K, D)
        self.resp = None  # (N, K)

    def _initialize_params(self, X: np.ndarray):
        N, D = X.shape
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.alpha = np.random.uniform(1.0, 3.0, size=(self.n_components, D))
        self.beta = np.random.uniform(1.0, 3.0, size=(self.n_components, D))
        self.resp = np.full((N, self.n_components), 1 / self.n_components)

    def _log_prob(self, X: np.ndarray) -> np.ndarray:
        """Per-sample, per-component log joint probability, shape (N, K).

        ``log w_k + sum_d log Beta(x_nd; alpha_kd, beta_kd)`` evaluated with the
        same ``scipy.special`` expression the frozen distribution uses.
        """
        log_prob = np.empty((X.shape[0], self.n_components))
        for k in range(self.n_components):
            a = self.alpha[k]
            b = self.beta[k]
            per_feature = xlogy(a - 1.0, X) + xlog1py(b - 1.0, -X) - betaln(a, b)
            log_prob[:, k] = np.log(self.weights[k] + 1e-10) + per_feature.sum(axis=1)
        return log_prob

    def _e_step(self, X: np.ndarray):
        log_resp = self._log_prob(X)
        log_resp -= log_resp.max(axis=1, keepdims=True)  # Stability
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        self.resp = resp

    def _m_step(self, X: np.ndarray):
        N, D = X.shape
        Nk = self.resp.sum(axis=0) + 1e-10
        self.weights = Nk / N

        for k in range(self.n_components):
            r = self.resp[:, k]
            # Unclipped mean, exactly as the reference: the variance below is
            # taken about this value, and only then is the mean clipped.
            mean = (r @ X) / Nk[k]
            diff = X - mean
            var = (r @ (diff * diff)) / Nk[k]

            mean = np.clip(mean, 1e-3, 1 - 1e-3)
            var = np.maximum(var, 1e-5)

            common = (mean * (1 - mean)) / var - 1
            self.alpha[k] = np.maximum(mean * common, 1e-2)
            self.beta[k] = np.maximum((1 - mean) * common, 1e-2)

    def fit(self, X: np.ndarray):
        """
        Fit MBMM to multivariate data in (0,1)

        Parameters:
        - X (np.ndarray): Input data (N x D), values ∈ (0, 1)
        """
        self._initialize_params(X)
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)

            # Compute log-likelihood under the updated parameters.
            ll = float(np.sum(self.resp * self._log_prob(X)))

            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

    # predict() and predict_proba() come from BaseSoftClusterer; ``resp``
    # remains available for the raw responsibilities.
