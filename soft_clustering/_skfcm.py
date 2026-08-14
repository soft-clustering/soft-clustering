# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics.pairwise import rbf_kernel
from typeguard import typechecked

from ._base import BaseSoftClusterer, ratio_memberships


@typechecked
class SKFCM(BaseSoftClusterer):
    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        gamma: float = 1.0,
        lambda_: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-5,
    ):
        """
        Parameters:
        - n_clusters (int): Number of clusters
        - m (float): Fuzziness degree
        - gamma (float): Kernel RBF parameter
        - lambda_ (float): Spatial constraint weight
        - max_iter (int): Max number of iterations
        - tol (float): Convergence threshold
        """
        self.n_clusters = n_clusters
        self.m = m
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.U = None
        self.K = None
        self.labels_ = None
        self.N = None

    def _initialize_U(self):
        self.U = np.random.dirichlet(np.ones(self.n_clusters), size=self.N)

    def _compute_kernel(self, X: np.ndarray):
        return rbf_kernel(X, gamma=self.gamma)

    def _spatial_term(self, U: np.ndarray, shape: tuple[int, int]):
        spatial_U = U.copy()
        for k in range(self.n_clusters):
            u_k = U[:, k].reshape(shape)
            spatial_U[:, k] = uniform_filter(u_k, size=3).reshape(-1)
        return spatial_U

    def _update_U(self, spatial_U: np.ndarray):
        Um = self.U**self.m
        d = np.zeros((self.N, self.n_clusters))

        for k in range(self.n_clusters):
            num = (
                np.diag(self.K)
                - (2 / np.sum(Um[:, k])) * (self.K @ Um[:, k])
                + (1 / np.sum(Um[:, k]) ** 2) * (Um[:, k].T @ self.K @ Um[:, k])
            )
            d[:, k] = num + self.lambda_ * (1 - spatial_U[:, k])

        d = np.clip(d, 1e-10, None)
        self.U = ratio_memberships(d, 1.0 / (self.m - 1.0))

    def fit(self, X: np.ndarray, shape: tuple[int, int]):
        self.N = X.shape[0]
        self.K = self._compute_kernel(X)
        self._initialize_U()

        for _ in range(self.max_iter):
            U_old = self.U.copy()
            spatial_U = self._spatial_term(self.U, shape)
            self._update_U(spatial_U)
            if np.linalg.norm(self.U - U_old) < self.tol:
                break

        self.labels_ = np.argmax(self.U, axis=1)

    # predict() and predict_proba() come from BaseSoftClusterer, which also
    # raises before a fit instead of returning None.
