# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Multivariate Beta Mixture Model.

Implementation note (optimization study)
----------------------------------------
Vectorised implementation of the same model. The likelihood, the E and M
updates, the initialisation draws, the clipping order and the convergence test
are unchanged. The reference implementation is preserved at
``optimization/original/scpp_original/_mbmm.py``.

After the unchanged parameter-update loop, a final E-step refreshes the
published responsibilities under the fitted parameters. The preserved
reference instead publishes responsibilities from before its final M-step.

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
from ._validation import check_array, check_n_clusters, check_random_state


@typechecked
class MBMM(BaseSoftClusterer):
    r"""Mixture of Beta distributions for data on the open unit hypercube.

    Each component is a product of independent univariate Beta densities,

    .. math::

        p(x \mid k) = \prod_{d} \mathrm{Beta}(x_d;\, \alpha_{kd}, \beta_{kd}),

    and the soft output is the posterior responsibility of each component.

    **Simplifications relative to the reference.** Hsu and Chen (2024) use a
    multivariate Beta distribution that can model dependence between
    features; this implementation assumes conditional independence of the
    features within a component. The M-step estimates
    :math:`(\alpha, \beta)` by the method of moments from the
    responsibility-weighted mean and variance rather than by maximum
    likelihood, so the likelihood is not guaranteed to increase at every
    iteration. Convergence is declared when the expected complete-data
    log-likelihood changes by less than ``tol``.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of mixture components.
    max_iter : int, default=100
    tol : float, default=1e-5
    random_state : int, numpy RandomState or None, default=None
        Seed for the initial Beta parameters, drawn uniformly from
        ``[1, 3]``. ``None`` draws from NumPy's global random state, as
        earlier releases did.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Posterior responsibilities under the final fitted parameters (also ``resp``).
    weights, alpha, beta : ndarray
        Mixing weights and per-component, per-feature Beta parameters.

    References
    ----------
    Y.-P. Hsu and H.-H. Chen. Multivariate beta mixture model: probabilistic
    clustering with flexible cluster shapes. In *Advances in Knowledge
    Discovery and Data Mining (PAKDD 2024)*, pp. 233--245, 2024.
    doi:10.1007/978-981-97-2242-6_19
    """

    # Fitted attributes, set by ``fit`` (not before).
    weights: np.ndarray  # (K,)
    alpha: np.ndarray  # (K, D)
    beta: np.ndarray  # (K, D)
    resp: np.ndarray  # (N, K)

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_components = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def _initialize_params(self, X: np.ndarray, rng=None):
        N, D = X.shape
        rng = check_random_state(None) if rng is None else rng
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.alpha = rng.uniform(1.0, 3.0, size=(self.n_components, D))
        self.beta = rng.uniform(1.0, 3.0, size=(self.n_components, D))
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
        name = type(self).__name__
        X = check_array(X, estimator=name)
        if np.any(X <= 0.0) or np.any(X >= 1.0):
            raise ValueError(
                f"{name}: every value of X must lie in the open interval (0, 1), "
                f"where the Beta density is defined; got values in "
                f"[{X.min():.6g}, {X.max():.6g}]. Rescale, e.g. with a "
                "min-max transform followed by clipping to [eps, 1 - eps]."
            )
        self.n_components = check_n_clusters(
            self.n_clusters, X.shape[0], estimator=name
        )
        self._initialize_params(X, check_random_state(self.random_state))
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)

            # Compute log-likelihood under the updated parameters.
            ll = float(np.sum(self.resp * self._log_prob(X)))

            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        # Publish the posterior for the final parameters without another M-step.
        self._e_step(X)

    # predict() and predict_proba() come from BaseSoftClusterer; ``resp``
    # remains available for the raw responsibilities.
