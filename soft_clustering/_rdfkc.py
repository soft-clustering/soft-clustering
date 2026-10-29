# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._torch_utils import isolated_torch_rng
from ._validation import check_n_clusters


class COIL20Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, 6, kernel_size=13, stride=1, padding=0
        )  # → (6, 20, 20)
        self.conv2 = nn.Conv2d(
            6, 12, kernel_size=11, stride=1, padding=0
        )  # → (12, 10, 10)
        self.conv3 = nn.Conv2d(
            12, 16, kernel_size=5, stride=1, padding=0
        )  # → (16, 6, 6)
        self.fc = nn.Linear(16 * 6 * 6, 70)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class COIL20Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(70, 16 * 6 * 6)
        self.deconv1 = nn.ConvTranspose2d(16, 12, kernel_size=5)  # → (12, 10, 10)
        self.deconv2 = nn.ConvTranspose2d(12, 6, kernel_size=11)  # → (6, 20, 20)
        self.deconv3 = nn.ConvTranspose2d(6, 1, kernel_size=13)  # → (1, 32, 32)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 16, 6, 6)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class FashionMNISTEncoder(nn.Module):
    """Convolutional encoder for single-channel images.

    Every convolution is padded to preserve the spatial extent, so the flattened
    width of the bottleneck is ``16 * image_size ** 2``. ``image_size`` defaults
    to 28, the actual resolution of Fashion-MNIST; pass the matching value for
    other inputs.
    """

    def __init__(self, image_size: int = 28, latent_dim: int = 20):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * image_size * image_size, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class FashionMNISTDecoder(nn.Module):
    """Mirror of :class:`FashionMNISTEncoder`; ``image_size`` must match it."""

    def __init__(self, image_size: int = 28, latent_dim: int = 20):
        super().__init__()
        self.image_size = image_size
        self.fc = nn.Linear(latent_dim, 16 * image_size * image_size)
        self.deconv1 = nn.ConvTranspose2d(16, 12, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(12, 6, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(6, 1, kernel_size=5, stride=1, padding=2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 16, self.image_size, self.image_size)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


def _simplex_quadratic(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Row-wise ``argmin_u sum_c q_c u_c^2 - p_c u_c`` over the probability simplex.

    The KKT conditions give ``u_c = max(0, (p_c + lam) / (2 q_c))`` with the
    multiplier ``lam`` chosen so that the row sums to one; the row sum is
    non-decreasing in ``lam``, so it is found by bisection (60 halvings give
    double-precision accuracy). ``q`` must be positive.
    """
    # Bracket: at lam = lo every (p_c + lam) <= 0, so the row sums to 0 < 1;
    # at lam = hi the largest u_c is already >= 1, so the row sums to >= 1.
    lo = (-p - 2.0 * q).min(dim=1, keepdim=True).values
    hi = (2.0 * q - p).max(dim=1, keepdim=True).values
    lo = torch.minimum(lo, hi - 1.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        total = torch.clamp((p + mid) / (2.0 * q), min=0.0).sum(dim=1, keepdim=True)
        lo = torch.where(total < 1.0, mid, lo)
        hi = torch.where(total < 1.0, hi, mid)
    U = torch.clamp((p + hi) / (2.0 * q), min=0.0)
    return U / U.sum(dim=1, keepdim=True)


@typechecked
class RDFKC(BaseSoftClusterer):
    r"""Robust deep fuzzy k-means clustering (RD-FKC) for image data.

    Jointly trains an autoencoder and a fuzzy k-means model in its latent
    space (Wu et al., 2024). With latent codes :math:`z_n`, prototypes
    :math:`v_c`, memberships :math:`U` and the adaptive robust loss
    :math:`\ell_\tau(d) = (1 + \tau) d^2 / (d + \tau)`, the method alternates

    1. one gradient step on the network weights for the reconstruction
       error (summed over each sample's elements, averaged over samples),
       plus the re-weighted clustering loss
       :math:`\sum_{n,c} u_{nc}^2 k_{nc} \lVert z_n - v_c\rVert^2`, with
       :math:`k_{nc} = (1+\tau)(d_{nc} + 2\tau) / (2 (d_{nc} + \tau)^2)`, plus
       an :math:`L_2` penalty ``gamma`` on the weights. The robust weights
       :math:`k_{nc}` are held fixed during this network step;
    2. the closed-form prototype update
       :math:`v_c = \sum_n u_{nc}^2 k_{nc} z_n / \sum_n u_{nc}^2 k_{nc}`,
       using refreshed evaluation-mode codes from the updated encoder;
    3. the membership update, which minimises, row by row,
       :math:`\sum_c \ell_\tau(d_{nc})\, u_{nc}^2 + \mu \sum_j S_{nj}
       (u_{nc} - u_{jc})^2` over the probability simplex, where :math:`S`
       is a Gaussian 10-nearest-neighbour graph built on the initial latent
       codes.

    Step 3 is solved exactly, non-negativity included (see
    :func:`_simplex_quadratic`); the unconstrained Lagrange solution used by
    earlier releases can produce negative memberships.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters (``K`` is accepted as an alias).
    encoder, decoder : torch.nn.Module, optional
        The autoencoder. Copied, never trained in place. Required unless
        ``dataset`` names a bundled architecture.
    dataset : {"coil20", "fashion"} or None
        Use the bundled convolutional autoencoder for that image format.
    random_state : int or None, default=None
        Seed for the network initialisation, the initial memberships and the
        prototypes. The global torch stream is not modified.
    max_iter : int, default=100
        Number of alternating rounds.
    batch_size : int or None, default=None
        Mini-batch size of the forward passes; ``None`` picks 64, 256 or 1000
        by data size.
    lr : float, default=1e-4
        Adam learning rate.
    mu : float, default=1.0
        Weight of the graph smoothness term.
    gamma : float, default=1e-4
        Weight decay on the network parameters.
    tau : float, default=0.1
        Robustness parameter of the adaptive loss.

    Attributes
    ----------
    memberships_ : ndarray of shape (n_samples, n_clusters)
    latent_ : ndarray of shape (n_samples, latent_dim)
        Final evaluation-mode latent codes, also used for the last V/U update.
    latent_centers_ : ndarray of shape (n_clusters, latent_dim)
    encoder_, decoder_ : torch.nn.Module
        The trained copies of the autoencoder, left in evaluation mode.

    References
    ----------
    X. Wu, Y.-F. Yu, L. Chen, W. Ding and Y. Wang. Robust deep fuzzy
    k-means clustering for image data. *Pattern Recognition*, 153:110504,
    2024. doi:10.1016/j.patcog.2024.110504

    Notes
    -----
    Earlier releases left the memberships in a torch tensor that the
    estimator protocol did not read, so ``memberships_`` silently fell back
    to the one-hot encoding of the hard labels. The update rules follow the
    reference as implemented by the original contributors and have not been
    checked against an independent implementation.
    """

    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        n_clusters: int = 3,
        encoder: torch.nn.Module | None = None,
        decoder: torch.nn.Module | None = None,
        dataset: str | None = None,
        random_state: int | None = None,
        max_iter: int = 100,
        batch_size: int | None = None,
        lr: float = 1e-4,
        mu: float = 1.0,
        gamma: float = 1e-4,
        tau: float = 0.1,
    ):
        self.n_clusters = n_clusters
        self.K = n_clusters
        self.mu = mu
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state
        self.dataset = dataset

        if (encoder is None or decoder is None) and dataset not in (
            "coil20",
            "fashion",
        ):
            raise ValueError(
                "RDFKC: provide both encoder and decoder, or dataset='coil20' "
                "or dataset='fashion' for a bundled architecture."
            )
        # The caller's modules are parameters and are never trained: fit
        # trains copies, published as ``encoder_`` and ``decoder_``.
        self.encoder = encoder
        self.decoder = decoder

    def _build_networks(self):
        if self.encoder is not None and self.decoder is not None:
            self.encoder_ = copy.deepcopy(self.encoder)
            self.decoder_ = copy.deepcopy(self.decoder)
            for module in list(self.encoder_.modules()) + list(self.decoder_.modules()):
                if module is not self.encoder_ and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        elif self.dataset == "coil20":
            self.encoder_, self.decoder_ = COIL20Encoder(), COIL20Decoder()
        else:
            self.encoder_, self.decoder_ = FashionMNISTEncoder(), FashionMNISTDecoder()

    def _robust_weight(self, dist):
        return (1 + self.tau) * (dist + 2 * self.tau) / (2 * (dist + self.tau) ** 2)

    def _encode_decode(self, X, batch_size):
        Z, R = [], []
        for start in range(0, X.shape[0], batch_size):
            z = self.encoder_(X[start : start + batch_size])
            Z.append(z)
            R.append(self.decoder_(z))
        return torch.cat(Z), torch.cat(R)

    def _encode(self, X, batch_size):
        """Batched latent codes without an unused decoder pass."""
        return torch.cat(
            [
                self.encoder_(X[start : start + batch_size])
                for start in range(0, X.shape[0], batch_size)
            ]
        )

    def _similarity(self, Z, k=10):
        n = Z.shape[0]
        k = min(k, n - 1)
        D = torch.cdist(Z, Z)
        idx = torch.argsort(D, dim=1)[:, 1 : k + 1]
        S = torch.zeros(n, n)
        S.scatter_(1, idx, torch.exp(-torch.gather(D, 1, idx) / 0.5))
        return 0.5 * (S + S.T)

    def fit_predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Train RD-FKC on ``X`` and return the membership matrix."""
        X = (
            X.float()
            if isinstance(X, torch.Tensor)
            else torch.as_tensor(np.asarray(X), dtype=torch.float32)
        )
        if not torch.isfinite(X).all():
            raise ValueError("RDFKC: X contains NaN or infinite values.")
        N = X.shape[0]
        K = check_n_clusters(self.n_clusters, N, estimator="RDFKC")
        if N < 2:
            raise ValueError("RDFKC: at least 2 samples are required.")
        batch_size = self.batch_size or (64 if N < 1000 else 256 if N < 10000 else 1000)

        with isolated_torch_rng(self.random_state):
            self._build_networks()
            params = list(self.encoder_.parameters()) + list(self.decoder_.parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr)

            self.encoder_.eval()
            self.decoder_.eval()
            with torch.no_grad():
                Z = self._encode(X, batch_size)
            U = torch.rand(N, K)
            U = U / U.sum(dim=1, keepdim=True)
            V = torch.randn(K, Z.shape[1])
            S = self._similarity(Z)
            S_rowsum = S.sum(dim=1, keepdim=True)

            for _ in range(self.max_iter):
                # 1. network step
                self.encoder_.train()
                self.decoder_.train()
                optimizer.zero_grad()
                Z, R = self._encode_decode(X, batch_size)
                dist = torch.cdist(Z, V)
                weights = self._robust_weight(dist).detach()
                cluster = (U**2 * weights * dist**2).sum()
                reg = sum((w**2).sum() for w in params)
                reconstruction = (R - X).square().sum() / N
                loss = reconstruction + cluster + self.gamma * reg
                loss.backward()
                optimizer.step()
                self.encoder_.eval()
                self.decoder_.eval()
                with torch.no_grad():
                    Z = self._encode(X, batch_size)

                # 2. prototypes
                dist = torch.cdist(Z, V)
                w = U**2 * self._robust_weight(dist)
                V = (w.T @ Z) / (w.sum(dim=0)[:, None] + 1e-8)

                # 3. memberships, exactly on the simplex
                dist = torch.cdist(Z, V) + 1e-8
                q = (1 + self.tau) * dist**2 / (dist + self.tau) + self.mu * S_rowsum
                p = 2 * self.mu * (S @ U)
                U = _simplex_quadratic(p, torch.clamp(q, min=1e-12))

        self.encoder_.eval()
        self.decoder_.eval()
        self.U = U.detach().numpy().astype(np.float64)
        self.memberships_ = self.U
        self.latent_ = Z.numpy().astype(np.float64)
        self.latent_centers_ = V.detach().numpy().astype(np.float64)
        return self.memberships_
