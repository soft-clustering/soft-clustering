import numpy as np
from typeguard import typechecked
from sklearn.metrics.pairwise import rbf_kernel


@typechecked
class SoftKSC:
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

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: np.ndarray):
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute soft assignment probabilities.

        Parameters:
        - X (np.ndarray): Test data (n_test x d)

        Returns:
        - probs (np.ndarray): (n_test x 2) probability matrix
        """
        K_test = self._compute_kernel(X, self.X_train)
        f1 = K_test @ self.alpha
        f2 = K_test @ self.beta

        d1 = np.abs(f1)
        d2 = np.abs(f2)
        d_sum = d1 + d2 + 1e-10

        prob1 = 1 - (d1 / d_sum)
        prob2 = 1 - (d2 / d_sum)

        return np.vstack([prob1, prob2]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) * 2 - 1  # Map to labels: -1, 1
