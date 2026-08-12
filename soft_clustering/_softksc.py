import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from typeguard import typechecked

from ._base import BaseSoftClusterer


@typechecked
class SoftKSC(BaseSoftClusterer):
    """Soft kernel spectral clustering (Langone et al., 2013).

    One of the few genuinely inductive estimators in SCPP: the model solves a
    kernel system on the training points and can score an unseen point through
    the same kernel, so ``predict(X)`` and ``predict_proba(X)`` accept new
    data. Called with no argument they return the fitted partition, matching
    every other estimator.
    """

    _supports_out_of_sample = True

    #: The two soft assignment scores are normalised, so the rows of
    #: ``memberships_`` sum to one up to the 1e-10 guard in
    #: :meth:`predict_proba`.
    _partition_constrained = True

    def __init__(self, gamma: float = 1.0, C: float = 1.0):
        """
        Parameters:
        - gamma (float): Kernel coefficient for RBF kernel
        - C (float): Regularization term
        """
        self.gamma = gamma
        self.C = C
        self.alpha = None
        self.beta = None
        self.X_train = None
        self.y_train = None

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return rbf_kernel(X, Y, gamma=self.gamma)

    def fit(
        self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: np.ndarray
    ):
        """
        Semi-supervised training on labeled and unlabeled data.

        Parameters:
        - X_labeled (np.ndarray): Labeled input data (n_labeled x d)
        - y_labeled (np.ndarray): Labels for labeled data (n_labeled,)
        - X_unlabeled (np.ndarray): Unlabeled data (n_unlabeled x d)
        """
        self.X_train = np.vstack((X_labeled, X_unlabeled))
        self.y_train = np.hstack((y_labeled, np.zeros(len(X_unlabeled))))
        K = self._compute_kernel(self.X_train, self.X_train)
        n = len(self.X_train)

        y1 = (self.y_train == 1).astype(float)
        y2 = (self.y_train == -1).astype(float)

        A = K + self.C * np.eye(n)
        self.alpha = np.linalg.solve(A, y1)
        self.beta = np.linalg.solve(A, y2)

        # Publish the training partition so the estimator satisfies the shared
        # protocol; the fit wrapper in _base picks these up. The partition
        # covers labelled and unlabelled samples together, which is more than
        # the first fit argument holds, so the count is stated explicitly.
        self._n_samples_hint = self.X_train.shape[0]
        self.memberships_ = self._soft_assign(self.X_train)

    def _soft_assign(self, X: np.ndarray) -> np.ndarray:
        K_test = self._compute_kernel(X, self.X_train)
        f1 = K_test @ self.alpha
        f2 = K_test @ self.beta

        d1 = np.abs(f1)
        d2 = np.abs(f2)
        d_sum = d1 + d2 + 1e-10

        prob1 = 1 - (d1 / d_sum)
        prob2 = 1 - (d2 / d_sum)

        scores = np.vstack([prob1, prob2]).T
        # The two scores sum to 1 only in the limit: the 1e-10 guard in d_sum
        # leaves a residual of order 1e-10/(d1+d2), which is visible whenever a
        # point sits almost exactly on the decision boundary. Normalise so the
        # output is an exact distribution, as the partition-constraint
        # declaration promises.
        row_sums = scores.sum(axis=1, keepdims=True)
        return scores / np.where(row_sums > 0, row_sums, 1.0)

    def predict_proba(self, X: np.ndarray | None = None) -> np.ndarray:
        """Soft assignment scores, shape ``(n, 2)``.

        Parameters
        ----------
        X : ndarray of shape (n_test, n_features), optional
            Unseen data to score. When omitted, the fitted memberships of the
            training set are returned.
        """
        self._check_fitted()
        if X is None:
            return self.memberships_
        return self._soft_assign(X)

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        """Cluster indices in ``{0, 1}``, matching ``argmax(memberships_)``.

        Earlier releases returned the signed encoding ``{-1, +1}`` here, which
        contradicted the library-wide rule ``labels_ == argmax(U)``. Use
        :meth:`signed_labels` for the signed convention.
        """
        self._check_fitted()
        if X is None:
            return self.labels_
        return np.argmax(self._soft_assign(X), axis=1)

    def signed_labels(self, X: np.ndarray | None = None) -> np.ndarray:
        """Assignments in the signed ``{-1, +1}`` encoding used by the fit labels."""
        return self.predict(X) * 2 - 1
