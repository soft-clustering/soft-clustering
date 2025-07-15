import numpy as np


class SCM:
    def __init__(self, ra: float = 0.5, ea: float = 0.5, er: float = 0.15):
        """
        Initialize the Subtractive Clustering parameters.

        Parameters
        ----------
        ra : float, optional
            Neighborhood radius (r_a) for potential calculation. Default is 0.5 based on original paper.
        ea : float, optional
            Acceptance ratio (epsilon_a) for stopping criterion, default 0.5.
        er : float, optional
            Rejection ratio (epsilon_r), currently unused, default 0.15.
        """
        self.ra = ra
        self.ea = ea
        self.er = er
        self.centers_ = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Subtractive Clustering to input data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data points.

        Returns
        -------
        centers : np.ndarray, shape (n_clusters, n_features)
            Coordinates of cluster centers.
        """
        # Normalize data to [0,1]
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        diff = X_max - X_min
        diff[diff == 0] = 1.0  # avoid division by zero
        X_norm = (X - X_min) / diff

        n_samples, n_features = X_norm.shape

        # Compute squared distances matrix
        dist_sq = (
            np.sum(X_norm**2, axis=1, keepdims=True)
            + np.sum(X_norm**2, axis=1)
            - 2 * X_norm.dot(X_norm.T)
        )

        # Compute potential for each point
        alpha = 4.0 / (self.ra**2)
        potentials = np.sum(np.exp(-alpha * dist_sq), axis=1)

        # Initial maximum potential
        p1 = potentials.max()

        centers = []
        rb = 1.5 * self.ra
        beta = 4.0 / (rb**2)

        while True:
            idx = np.argmax(potentials)
            p = potentials[idx]
            if p < self.ea * p1:
                break
            centers.append(X[idx].copy())
            diff_to_center = dist_sq[idx]
            suppression = p * np.exp(-beta * diff_to_center)
            potentials = potentials - suppression

        self.centers_ = np.array(centers)
        return self.centers_
