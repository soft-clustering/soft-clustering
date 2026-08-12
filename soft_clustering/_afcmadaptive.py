import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from typeguard import typechecked

from ._base import BaseSoftClusterer, ratio_memberships


@typechecked
class AFCMAdaptive(BaseSoftClusterer):
    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        k1: float = 0.1,
        k2: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        """
        Parameters:
        - n_clusters (int): Number of clusters
        - m (float): Fuzziness degree
        - k1 (float): First-order regularization weight
        - k2 (float): Second-order regularization weight
        - max_iter (int): Maximum number of iterations
        - tol (float): Convergence threshold
        """
        self.n_clusters = n_clusters
        self.m = m
        self.k1 = k1
        self.k2 = k2
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.membership = None
        self.multiplier = None

    def _initialize(self, image: np.ndarray):
        H, W = image.shape
        self.membership = np.random.dirichlet(
            np.ones(self.n_clusters), size=H * W
        ).reshape(H, W, self.n_clusters)
        self.multiplier = np.ones((H, W))
        self.centers = np.linspace(np.min(image), np.max(image), self.n_clusters)

    def fit(self, image: np.ndarray):
        """
        Fit AFCMAdaptive on 2D image.

        Parameters:
        - image (np.ndarray): 2D grayscale image
        """
        H, W = image.shape
        self._initialize(image)

        for iteration in range(self.max_iter):
            u_m = self.membership**self.m
            prev_centers = self.centers.copy()

            # update centers
            for k in range(self.n_clusters):
                numerator = np.sum(
                    u_m[:, :, k] * (image**2) / (self.multiplier**2 + 1e-8)
                )
                denominator = np.sum(u_m[:, :, k] * image / (self.multiplier + 1e-8))
                self.centers[k] = (denominator / (np.sum(u_m[:, :, k]) + 1e-8)) / (
                    numerator / (np.sum(u_m[:, :, k]) + 1e-8) + 1e-8
                )

            # update multiplier m(i,j)
            v = self.centers
            nom = np.sum(
                u_m
                * (image[:, :, None] - self.multiplier[:, :, None] * v[None, None, :])
                * (-v[None, None, :]),
                axis=2,
            )
            denom = np.sum(u_m * (v[None, None, :] ** 2), axis=2) + 1e-8

            # regularization
            grad = gaussian_filter(self.multiplier, sigma=1)
            lap = laplace(self.multiplier)

            self.multiplier = nom / denom - self.k1 * grad + self.k2 * lap

            # update membership
            dist = np.zeros((H, W, self.n_clusters))
            for k in range(self.n_clusters):
                diff = image - self.multiplier * v[k]
                dist[:, :, k] = diff**2

            dist = np.clip(dist, 1e-8, None)
            # Flatten to (pixels, clusters) so the shared, overflow-free ratio
            # rule applies, then restore the image layout.
            flat = ratio_memberships(
                dist.reshape(H * W, self.n_clusters), 1.0 / (self.m - 1.0)
            )
            self.membership = flat.reshape(H, W, self.n_clusters)

            # check convergence
            if np.linalg.norm(self.centers - prev_centers) < self.tol:
                break

        # The algorithm segments an image, so its natural membership array is
        # (H, W, K). Publish the pixel-major (H*W, K) view as well, so the
        # estimator satisfies the shared protocol and can be evaluated by the
        # same metrics and conformance checks as every other model. The image
        # shape is retained for :meth:`label_map`.
        self.image_shape_ = (H, W)
        self._n_samples_hint = H * W
        self.memberships_ = self.membership.reshape(H * W, self.n_clusters)
        self.centers_ = np.asarray(self.centers, dtype=np.float64).reshape(-1, 1)

    def label_map(self) -> np.ndarray:
        """Hard segmentation in image layout, shape ``(H, W)``.

        ``predict()`` returns the flat ``(H*W,)`` labelling required by the
        estimator protocol; this is the same assignment reshaped back to the
        image grid.
        """
        self._check_fitted()
        return self.labels_.reshape(self.image_shape_)

    def get_membership(self) -> np.ndarray:
        """Membership array in image layout, shape ``(H, W, K)``."""
        return self.membership
