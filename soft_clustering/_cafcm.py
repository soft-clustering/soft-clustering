# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Collaborative annealing fuzzy c-means (after Li and Wang, 2024)."""

from __future__ import annotations

import numpy as np
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._fcm import (
    _euclidean_dist2,
    _init_centers_kpp,
    _update_centers_from_U,
    _update_U_from_centers,
)
from ._validation import (
    check_array,
    check_generator,
    check_n_clusters,
    check_scalar,
    data_scale,
)


def _fcm_from(X, centers, m, max_iter, tol):
    """Fuzzy c-means started from ``centers``; returns ``(U, centers)``."""
    U = _update_U_from_centers(X, centers, m)
    for _ in range(max_iter):
        new_centers = _update_centers_from_U(X, U, m)
        U = _update_U_from_centers(X, new_centers, m)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol * data_scale(X):
            break
    return U, centers


def _sse(X, centers):
    """Crisp within-cluster sum of squares; comparable across fuzzifiers."""
    return float(np.min(_euclidean_dist2(X, centers), axis=1).sum())


@typechecked
class CAFCM(BaseSoftClusterer):
    r"""Collaborative annealing fuzzy c-means (CAFCM).

    Runs a population of ``n_modules`` fuzzy c-means modules while lowering
    the fuzzifier along a geometric cooling schedule,

    .. math::

        m_t = 1 + (m_{\mathrm{start}} - 1)\, r^t, \qquad t = 0, 1, \dots,

    until :math:`m_t` reaches ``m_end``. At every temperature each module is
    run to convergence from its current prototypes, so memberships are
    phased from soft towards crisp (annealing). After every temperature the
    modules collaborate: the module with the lowest crisp within-cluster sum
    of squares is kept, and every other module is re-initialised towards it.
    The returned partition is that of the best module at the final
    temperature.

    **Relation to the reference.** Li and Wang (2024) combine exactly these
    two ingredients --- an annealing schedule on the fuzzifier and multiple
    FCM modules "repeatedly re-initialized using a meta-heuristic rule". The
    specific meta-heuristic of the reference is not reproduced; SCPP uses a
    particle-swarm-style rule in which a non-best module's prototypes move to
    :math:`v \leftarrow v + c\,\xi\,(v^{\mathrm{best}} - v) + \epsilon`,
    with :math:`\xi \sim U(0, 1)` per prototype, :math:`c = 1.5`, and
    Gaussian jitter :math:`\epsilon` scaled by the data's standard deviation
    and the current temperature. Prototypes are matched to the best module's
    by a nearest-prototype assignment before the move.

    Parameters
    ----------
    n_clusters : int, default=3
    m_start : float, default=2.0
        Initial fuzzifier, ``> m_end``.
    m_end : float, default=1.01
        Final fuzzifier, ``> 1``.
    cooling_rate : float, default=0.8
        Factor :math:`r \in (0, 1)` applied to :math:`m - 1` per temperature.
    n_modules : int, default=5
        Number of collaborating FCM modules (``1`` disables collaboration).
    max_iter : int, default=100
        Maximum FCM sweeps per module per temperature.
    tol : float, default=1e-5
        Prototype-shift tolerance of each FCM run, relative to the data
        spread (root mean per-feature standard deviation).
    random_state : int, numpy Generator or None, default=None

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
        Memberships of the best module at fuzzifier ``m_end``.
    centers_ : ndarray of shape (n_clusters, n_features)
    fuzzifier_schedule_ : ndarray
        The fuzzifiers visited.
    sse_trajectory_ : ndarray
        Best crisp within-cluster sum of squares after each temperature.

    References
    ----------
    H. Li and J. Wang. From soft clustering to hard clustering: a
    collaborative annealing fuzzy c-means algorithm. *IEEE Transactions on
    Fuzzy Systems*, 32(3):1181--1194, 2024. doi:10.1109/TFUZZ.2023.3319663
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ("centers_",)

    def __init__(
        self,
        n_clusters: int = 3,
        m_start: float = 2.0,
        m_end: float = 1.01,
        cooling_rate: float = 0.8,
        n_modules: int = 5,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | np.random.Generator | None = None,
    ):
        self.n_clusters = n_clusters
        self.m_start = m_start
        self.m_end = m_end
        self.cooling_rate = cooling_rate
        self.n_modules = n_modules
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X) -> CAFCM:
        """Fit on a feature matrix of shape ``(n_samples, n_features)``."""
        name = type(self).__name__
        X = check_array(X, estimator=name)
        K = check_n_clusters(self.n_clusters, X.shape[0], estimator=name)
        m_end = check_scalar(
            self.m_end, "m_end", estimator=name, lower=1.0, lower_inclusive=False
        )
        m_start = check_scalar(self.m_start, "m_start", estimator=name, lower=m_end)
        rate = check_scalar(
            self.cooling_rate,
            "cooling_rate",
            estimator=name,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
            upper_inclusive=False,
        )
        if self.n_modules < 1:
            raise ValueError(f"{name}: n_modules must be >= 1; got {self.n_modules}.")

        rng = check_generator(self.random_state)
        scale = float(np.std(X)) or 1.0

        steps: list[float] = []
        m = m_start
        while True:
            steps.append(max(m, m_end))
            if m <= m_end:
                break
            next_m = 1.0 + (m - 1.0) * rate
            if next_m >= m:
                raise ValueError(
                    "CAFCM: cooling schedule stalled at floating-point precision"
                )
            m = next_m
        schedule = np.array(steps)

        modules = [_init_centers_kpp(X, K, rng) for _ in range(self.n_modules)]
        sse_trace = []
        best = None
        for t, m in enumerate(schedule):
            results = [_fcm_from(X, c, m, self.max_iter, self.tol) for c in modules]
            scores = [_sse(X, c) for _, c in results]
            b = int(np.argmin(scores))
            best = (results[b][0], results[b][1], m)
            sse_trace.append(scores[b])

            # Collaboration: move every other module towards the best one.
            v_best = results[b][1]
            temperature = (m - 1.0) / max(m_start - 1.0, 1e-12)
            modules = []
            for i, (_, v) in enumerate(results):
                if i == b or t == len(schedule) - 1:
                    modules.append(v)
                    continue
                order = np.argmin(_euclidean_dist2(v_best, v), axis=1)
                target = v[order]
                xi = rng.random((K, 1))
                moved = target + 1.5 * xi * (v_best - target)
                moved += rng.normal(0.0, 0.1 * scale * temperature, size=moved.shape)
                modules.append(moved)

        assert best is not None  # the schedule has at least one step
        U, centers, m = best
        self.centers_ = centers
        self.memberships_ = _update_U_from_centers(X, centers, m)
        self.fuzzifier_schedule_ = schedule
        self.sse_trajectory_ = np.array(sse_trace)
        return self

    def fit_predict(self, X) -> np.ndarray:
        """Fit and return ``memberships_``."""
        return self.fit(X).memberships_
