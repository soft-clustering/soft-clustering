import numpy as np
from scipy.stats import norm, beta
from typeguard import typechecked


@typechecked
class BGMM:
    def __init__(self, n_components: int = 3, max_iter: int = 100, tol: float = 1e-5):
        """
        Parameters:
        - n_components (int): Number of clusters
        - max_iter (int): Maximum number of EM iterations
        - tol (float): Convergence threshold
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.gaussian_params = None  # List of (mean, std)
        self.beta_params = None      # List of (alpha, beta)
        self.resp = None             # Responsibility matrix

    def _initialize_params(self, Xg: np.ndarray, Xb: np.ndarray):
        N = Xg.shape[0]
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.gaussian_params = [(np.mean(Xg), np.std(Xg) + 1e-3)
                                for _ in range(self.n_components)]
        self.beta_params = [(2.0, 2.0) for _ in range(self.n_components)]
        self.resp = np.full((N, self.n_components), 1 / self.n_components)

    def _e_step(self, Xg: np.ndarray, Xb: np.ndarray):
        N = Xg.shape[0]
        log_resp = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            mu, sigma = self.gaussian_params[k]
            a, b = self.beta_params[k]
            log_prob = (
                np.log(self.weights[k] + 1e-10) +
                norm.logpdf(Xg, mu, sigma) +
                beta.logpdf(Xb, a, b)
            )
            log_resp[:, k] = log_prob

        # For numerical stability
        log_resp -= log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        self.resp = resp

    def _m_step(self, Xg: np.ndarray, Xb: np.ndarray):
        N = Xg.shape[0]
        Nk = self.resp.sum(axis=0) + 1e-10
        self.weights = Nk / N

        for k in range(self.n_components):
            # Gaussian update
            r = self.resp[:, k]
            mu = np.sum(r * Xg) / Nk[k]
            sigma = np.sqrt(np.sum(r * (Xg - mu) ** 2) / Nk[k]) + 1e-3
            self.gaussian_params[k] = (mu, sigma)

            # Beta update (method of moments)
            x = Xb
            mean = np.sum(r * x) / Nk[k]
            var = np.sum(r * (x - mean) ** 2) / Nk[k]
            mean = np.clip(mean, 1e-3, 1 - 1e-3)
            var = max(var, 1e-5)
            alpha = mean * ((mean * (1 - mean)) / var - 1)
            beta_val = (1 - mean) * ((mean * (1 - mean)) / var - 1)
            alpha = max(alpha, 1e-2)
            beta_val = max(beta_val, 1e-2)
            self.beta_params[k] = (alpha, beta_val)

    def fit(self, Xg: np.ndarray, Xb: np.ndarray):
        """
        Fit the BGMM to the data.

        Parameters:
        - Xg: Gaussian-type features (N,)
        - Xb: Beta-type features (N,) – values in (0,1)
        """
        self._initialize_params(Xg, Xb)
        prev_ll = -np.inf

        for _ in range(self.max_iter):
            self._e_step(Xg, Xb)
            self._m_step(Xg, Xb)

            log_likelihood = 0
            for k in range(self.n_components):
                mu, sigma = self.gaussian_params[k]
                a, b = self.beta_params[k]
                log_likelihood += np.sum(
                    self.resp[:, k] * (
                        np.log(self.weights[k] + 1e-10) +
                        norm.logpdf(Xg, mu, sigma) +
                        beta.logpdf(Xb, a, b)
                    )
                )
            if np.abs(log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = log_likelihood

    def predict_proba(self) -> np.ndarray:
        return self.resp

    def predict(self) -> np.ndarray:
        return np.argmax(self.resp, axis=1)
