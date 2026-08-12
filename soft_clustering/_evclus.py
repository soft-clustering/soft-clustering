"""EVCLUS: evidential clustering of proximity data.

Reference
---------
T. Denoeux and M.-H. Masson. *EVCLUS: evidential clustering of proximity
data.* IEEE Transactions on Systems, Man, and Cybernetics, Part B
(Cybernetics), 34(1):95--109, 2004.

EVCLUS assigns each object a mass function over a restricted frame of focal
sets --- the ``K`` singletons plus the whole frame :math:`\\Omega`, which
carries the ignorance --- and fits those mass functions so that the *degree of
conflict* between two objects tracks their dissimilarity:

.. math::

    J(M) \\;=\\; \\frac{\\sum_{i<j} (\\kappa_{ij} - \\delta_{ij})^2}
                       {\\sum_{i<j} \\delta_{ij}^2},
    \\qquad
    \\kappa_{ij} = \\!\\!\\sum_{A \\cap B = \\emptyset}\\!\\! m_i(A)\\, m_j(B).

With singleton-plus-:math:`\\Omega` focal sets the conflict has the closed form
:math:`\\kappa_{ij} = s_i s_j - \\langle S_i, S_j\\rangle`, where :math:`S` holds
the singleton masses and :math:`s_i = \\sum_k S_{ik} = 1 - m_i(\\Omega)`.
Dissimilarities are rescaled to :math:`[0, 1]` because a conflict is a
probability-like quantity.

Implementation note
-------------------
The simplex constraint is handled by a softmax reparameterisation rather than
by projection, and the objective is minimised with L-BFGS-B using the analytic
gradient

.. math::

    \\frac{\\partial J}{\\partial S}
      = \\frac{2}{C}\\Big[ (G s)\\mathbf{1}^\\top - G S \\Big],
    \\qquad G_{ij} = \\kappa_{ij} - \\delta_{ij}\\;(i \\neq j),

which makes the fit deterministic given the initial parameters and avoids the
finite-difference cost of a gradient-free optimiser.

Outputs
-------
``masses_``
    The credal partition, shape ``(n_samples, n_clusters + 1)``. The last
    column is the mass assigned to :math:`\\Omega`.
``memberships_``
    The pignistic probabilities :math:`\\mathrm{BetP}_{ik} = S_{ik} +
    m_i(\\Omega)/K`, which sum to one and are what the shared evaluation
    metrics consume. The credal partition remains available in ``masses_``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typeguard import typechecked

from ._base import BaseSoftClusterer


def _softmax(alpha: np.ndarray) -> np.ndarray:
    shifted = alpha - alpha.max(axis=1, keepdims=True)
    np.exp(shifted, out=shifted)
    return shifted / shifted.sum(axis=1, keepdims=True)


@typechecked
class EVCLUS(BaseSoftClusterer):
    """Evidential clustering of proximity data (EVCLUS)."""

    # BetP is a probability distribution, so the reported memberships do sum
    # to one; the underlying credal partition in ``masses_`` does not have to.
    _partition_constrained = True
    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 200,
        n_init: int = 3,
        tol: float = 1e-6,
        metric: str = "euclidean",
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        n_clusters : int
            Number of singleton focal sets (clusters).
        max_iter : int
            Maximum number of L-BFGS-B iterations per restart.
        n_init : int
            Number of random restarts; the lowest stress is kept. The default
            is 3 rather than 1 because the stress surface is very flat near
            its optimum: with a single restart L-BFGS-B stops at points whose
            stress differs by a factor of two between runs that differ only in
            the memory layout of the input, while three restarts agree to
            1e-11. The recovered partition is stable either way.
        tol : float
            Convergence tolerance passed to the optimiser.
        metric : {"euclidean", "precomputed"}
            ``"euclidean"`` treats ``fit``'s argument as a feature matrix and
            derives pairwise distances from it. ``"precomputed"`` treats it as
            an ``(n, n)`` dissimilarity matrix, which is the input EVCLUS was
            originally defined for.
        random_state : int or None
            Seed for the restarts.
        """
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")
        if metric not in ("euclidean", "precomputed"):
            raise ValueError(
                f"metric must be 'euclidean' or 'precomputed', got {metric!r}"
            )

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.metric = metric
        self.random_state = random_state

        self.masses_: np.ndarray | None = None
        self.memberships_: np.ndarray | None = None
        self.stress_: float | None = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dissimilarities(self, data: np.ndarray) -> np.ndarray:
        if self.metric == "precomputed":
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                raise ValueError(
                    "metric='precomputed' requires a square (n, n) "
                    f"dissimilarity matrix, got shape {data.shape}"
                )
            delta = np.array(data, dtype=np.float64, copy=True)
        else:
            if data.ndim != 2:
                raise ValueError(f"X must be 2-D, got shape {data.shape}")
            sq = np.sum(data**2, axis=1)
            delta = sq[:, None] + sq[None, :] - 2.0 * (data @ data.T)
            np.maximum(delta, 0.0, out=delta)
            np.sqrt(delta, out=delta)

        np.fill_diagonal(delta, 0.0)
        # Conflict lives in [0, 1]; the dissimilarities must share that range
        # for the stress to be meaningful.
        largest = delta.max()
        if largest > 0:
            delta /= largest
        return delta

    def _objective(self, flat: np.ndarray, delta: np.ndarray, norm: float):
        n_samples = delta.shape[0]
        alpha = flat.reshape(n_samples, self.n_clusters + 1)
        M = _softmax(alpha)
        S = M[:, : self.n_clusters]
        s = S.sum(axis=1)

        kappa = np.outer(s, s) - S @ S.T
        G = kappa - delta
        np.fill_diagonal(G, 0.0)

        stress = 0.5 * float(np.sum(G**2)) / norm

        # dJ/dS, then push through the softmax.
        grad_S = (2.0 / norm) * ((G @ s)[:, None] - G @ S)
        grad_M = np.zeros_like(M)
        grad_M[:, : self.n_clusters] = grad_S
        grad_alpha = M * (grad_M - (grad_M * M).sum(axis=1, keepdims=True))
        return stress, grad_alpha.ravel()

    # ------------------------------------------------------------------
    # Estimator interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> EVCLUS:
        """Fit on a feature matrix, or on a dissimilarity matrix when
        ``metric='precomputed'``."""
        X = np.asarray(X, dtype=np.float64)
        delta = self._dissimilarities(X)
        n_samples = delta.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} is smaller than n_clusters={self.n_clusters}"
            )

        norm = 0.5 * float(np.sum(delta**2))
        if norm <= 0:
            # Every object is identical: total ignorance is the only sensible
            # credal partition, and its pignistic transform is uniform.
            masses = np.zeros((n_samples, self.n_clusters + 1))
            masses[:, -1] = 1.0
            self.masses_ = masses
            self.memberships_ = np.full(
                (n_samples, self.n_clusters), 1.0 / self.n_clusters
            )
            self.stress_ = 0.0
            return self

        rng = np.random.default_rng(self.random_state)
        best_stress = np.inf
        best_alpha = None

        for _ in range(self.n_init):
            alpha0 = rng.normal(scale=0.1, size=(n_samples, self.n_clusters + 1))
            result = minimize(
                self._objective,
                alpha0.ravel(),
                args=(delta, norm),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )
            if result.fun < best_stress:
                best_stress = float(result.fun)
                best_alpha = result.x

        masses = _softmax(best_alpha.reshape(n_samples, self.n_clusters + 1))
        self.masses_ = masses
        # Pignistic transform: ignorance is shared equally among the singletons.
        self.memberships_ = (
            masses[:, : self.n_clusters] + masses[:, [-1]] / self.n_clusters
        )
        self.stress_ = best_stress
        return self

    # ------------------------------------------------------------------
    # Evidential summaries
    # ------------------------------------------------------------------

    def betp(self) -> np.ndarray:
        """Pignistic probabilities, identical to ``memberships_``."""
        self._check_fitted()
        return self.memberships_

    def ignorance(self) -> np.ndarray:
        """Mass assigned to :math:`\\Omega` for each object, shape ``(n,)``."""
        self._check_fitted()
        return self.masses_[:, -1]
