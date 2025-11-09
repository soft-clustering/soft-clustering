import numpy as np
from scipy.stats import beta
from typeguard import typechecked


@typechecked
class MBMM:
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
        self.beta = None   # (K, D)
        self.resp = None   # (N, K)

    def _initialize_params(self, X: np.ndarray):
        N, D = X.shape
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.alpha = np.random.uniform(1.0, 3.0, size=(self.n_components, D))
        self.beta = np.random.uniform(1.0, 3.0, size=(self.n_components, D))
        self.resp = np.full((N, self.n_components), 1 / self.n_components)

    def _e_step(self, X: np.ndarray):
        N, D = X.shape
        log_resp = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            log_prob = np.log(self.weights[k] + 1e-10)
            for d in range(D):
                a, b = self.alpha[k, d], self.beta[k, d]
                log_prob += beta.logpdf(X[:, d], a, b)
            log_resp[:, k] = log_prob

        log_resp -= log_resp.max(axis=1, keepdims=True)  # Stability
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        self.resp = resp

    def _m_step(self, X: np.ndarray):
        N, D = X.shape
        Nk = self.resp.sum(axis=0) + 1e-10
        self.weights = Nk / N

        for k in range(self.n_components):
            for d in range(D):
                x = X[:, d]
                r = self.resp[:, k]

                mean = np.sum(r * x) / Nk[k]
                var = np.sum(r * (x - mean) ** 2) / Nk[k]
                mean = np.clip(mean, 1e-3, 1 - 1e-3)
                var = max(var, 1e-5)

                alpha = mean * ((mean * (1 - mean)) / var - 1)
                beta_val = (1 - mean) * ((mean * (1 - mean)) / var - 1)

                self.alpha[k, d] = max(alpha, 1e-2)
                self.beta[k, d] = max(beta_val, 1e-2)

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

            # Compute log-likelihood
            ll = 0
            for k in range(self.n_components):
                log_prob = np.log(self.weights[k] + 1e-10)
                for d in range(X.shape[1]):
                    log_prob += beta.logpdf(X[:, d],
                                            self.alpha[k, d], self.beta[k, d])
                ll += np.sum(self.resp[:, k] * log_prob)

            if np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

    def predict_proba(self) -> np.ndarray:
        return self.resp

    def predict(self) -> np.ndarray:
        return np.argmax(self.resp, axis=1)
