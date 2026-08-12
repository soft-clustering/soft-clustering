"""Mixed Membership Stochastic Blockmodel (MMSB).

Reference
---------
E. M. Airoldi, D. M. Blei, S. E. Fienberg and E. P. Xing. *Mixed membership
stochastic blockmodels.* Advances in Neural Information Processing Systems 21,
2008.

Generative model. Each node :math:`p` draws a membership vector
:math:`\\pi_p \\sim \\mathrm{Dirichlet}(\\alpha)`. For an ordered pair
:math:`(p, q)` the sender draws :math:`z_{p \\to q} \\sim \\pi_p`, the receiver
draws :math:`z_{p \\leftarrow q} \\sim \\pi_q`, and the edge is
:math:`Y_{pq} \\sim \\mathrm{Bernoulli}(B_{z_{p\\to q},\\, z_{p\\leftarrow q}})`.

Inference. This module implements the naive mean-field variational EM of
Airoldi et al. (Section 4), with variational factors
:math:`q(\\pi_p) = \\mathrm{Dirichlet}(\\gamma_p)` and
:math:`q(z_{p\\to q}) = \\mathrm{Mult}(\\phi_{p\\to q})`. The updates are

.. math::

    \\phi_{p\\to q, g} &\\propto \\exp\\!\\big(\\mathbb{E}[\\log \\pi_{pg}]\\big)
        \\prod_h f(Y_{pq}; B_{gh})^{\\phi_{p\\leftarrow q, h}}, \\\\
    \\gamma_{pg} &= \\alpha_g + \\sum_{q} \\phi_{p \\to q, g}
                                + \\sum_{q} \\phi_{q \\leftarrow p, g}, \\\\
    B_{gh} &= \\frac{\\sum_{p \\neq q} \\phi_{p\\to q,g}\\,
                     \\phi_{p\\leftarrow q,h}\\, Y_{pq}}
                    {\\sum_{p \\neq q} \\phi_{p\\to q,g}\\,
                     \\phi_{p\\leftarrow q,h}},

evaluated in log space and vectorised over all :math:`n^2` pairs.

Complexity. Memory is :math:`O(n^2 K)` for the two per-pair variational
factors and time is :math:`O(n^2 K^2)` per sweep; this is inherent to the
model, which places a latent variable on every ordered pair. A 500-node graph
with :math:`K = 10` needs roughly 40 MB.

Known bias. Because the pair-level factors :math:`\\phi` are themselves
informed by :math:`Y_{pq}`, the mean-field estimate of :math:`B` is the edge
density *conditional on a pair being assigned to that block pair*, and its
diagonal is biased upward relative to the generating block probabilities. On
a planted 3-block graph with :math:`p_{\\mathrm{in}} = 0.6` this implementation
recovers the partition exactly but reports :math:`\\hat{B}_{gg} \\approx 0.93`.
Airoldi et al. introduce a sparsity parameter :math:`\\rho` to counteract this;
it is not implemented here. Node memberships, which are what
``memberships_`` exposes and what ``tests/test_mmsb.py`` checks against a
planted partition, are unaffected.

The previous release exposed only the generative sampler
(:meth:`sample_graph`) and returned prior draws from :meth:`get_memberships`.
That behaviour was not inference and has been replaced; the sampler is
retained because it is useful for generating synthetic benchmarks, but
:meth:`get_memberships` now requires a fit and returns the variational
posterior mean.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma
from typeguard import typechecked

from ._base import BaseSoftClusterer

_EPS = 1e-10


@typechecked
class MMSB(BaseSoftClusterer):
    """Mixed Membership Stochastic Blockmodel fitted by variational EM."""

    _k_param = "n_blocks"
    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        n_blocks: int = 3,
        alpha: float = 0.5,
        max_iter: int = 100,
        n_inner: int = 5,
        tol: float = 1e-4,
        init: str = "spectral",
        random_state: int | None = None,
        n_nodes: int | None = None,
    ):
        """
        Parameters
        ----------
        n_blocks : int
            Number of latent blocks.
        alpha : float
            Symmetric Dirichlet prior on the node memberships.
        max_iter : int
            Maximum number of variational EM sweeps.
        n_inner : int
            Mean-field passes over the per-pair factors within each sweep.
        tol : float
            Convergence threshold on the mean absolute change in the
            normalised memberships.
        init : {"spectral", "random"}
            Naive mean field has a symmetric fixed point at which every node
            is uniformly mixed and every block probability equals the graph
            density; a uniform start never leaves it. ``"spectral"`` breaks
            the symmetry by seeding the variational factors from a k-means
            partition of the leading adjacency eigenvectors, which is the
            standard practical initialiser for blockmodel variational
            inference. ``"random"`` seeds from a random hard partition and is
            provided for ablation.
        random_state : int or None
            Seed for the variational initialisation and for
            :meth:`sample_graph`.
        n_nodes : int or None
            Number of nodes, used only by :meth:`sample_graph` when the model
            is used generatively without a fit. Ignored by :meth:`fit`, which
            takes the node count from the adjacency matrix.
        """
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if n_inner < 1:
            raise ValueError(f"n_inner must be >= 1, got {n_inner}")
        if init not in ("spectral", "random"):
            raise ValueError(f"init must be 'spectral' or 'random', got {init!r}")

        self.n_blocks = n_blocks
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_inner = n_inner
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.n_nodes = n_nodes

        self.memberships_: np.ndarray | None = None
        self.gamma_: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.pi: np.ndarray | None = None
        self.n_iter_: int | None = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _initial_partition(self, Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Hard partition used to break the mean-field symmetry."""
        n, K = Y.shape[0], self.n_blocks
        if self.init == "random" or K == 1 or n <= K:
            return rng.integers(0, K, size=n)

        # Leading eigenvectors of the symmetrised adjacency, then k-means.
        symmetric = 0.5 * (Y + Y.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return rng.integers(0, K, size=n)

        order = np.argsort(np.abs(eigenvalues))[::-1][:K]
        embedding = eigenvectors[:, order]
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.maximum(norms, _EPS)

        from sklearn.cluster import KMeans

        seed = self.random_state if self.random_state is not None else 0
        labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(
            embedding
        )
        return np.asarray(labels)

    def fit(self, adjacency_matrix: np.ndarray) -> MMSB:
        """Fit the model to a binary ``(n_nodes, n_nodes)`` adjacency matrix."""
        Y = np.asarray(adjacency_matrix, dtype=np.float64)
        if Y.ndim != 2 or Y.shape[0] != Y.shape[1]:
            raise ValueError(
                f"adjacency_matrix must be square (n, n), got shape {Y.shape}"
            )
        n = Y.shape[0]
        K = self.n_blocks
        if n < K:
            raise ValueError(f"n_nodes={n} is smaller than n_blocks={K}")

        rng = np.random.default_rng(self.random_state)

        # Self-pairs carry no information about the block structure.
        off_diagonal = ~np.eye(n, dtype=bool)
        mask = off_diagonal[:, :, None]

        # Symmetry-breaking initialisation: a hard partition, softened into
        # the per-pair factors and used to seed the block matrix.
        assignment = self._initial_partition(Y, rng)
        soft = np.full((n, K), 0.1 / K)
        soft[np.arange(n), assignment] += 0.9
        soft /= soft.sum(axis=1, keepdims=True)
        phi_out = np.broadcast_to(soft[:, None, :], (n, n, K)).copy()
        phi_in = np.broadcast_to(soft[None, :, :], (n, n, K)).copy()

        pair_mass = np.einsum("pqg,pqh->gh", phi_out * mask, phi_in, optimize=True)
        edge_mass = np.einsum(
            "pqg,pqh,pq->gh", phi_out * mask, phi_in, Y, optimize=True
        )
        B = np.clip(edge_mass / (pair_mass + _EPS), _EPS, 1.0 - _EPS)

        gamma = (
            self.alpha
            + (phi_out * mask).sum(axis=1)
            + (phi_in * mask).transpose(1, 0, 2).sum(axis=1)
        )
        memberships = gamma / gamma.sum(axis=1, keepdims=True)
        self.n_iter_ = self.max_iter

        for sweep in range(self.max_iter):
            previous = memberships

            e_log_pi = digamma(gamma) - digamma(gamma.sum(axis=1, keepdims=True))
            log_B = np.log(B)
            log_1mB = np.log1p(-B)

            for _ in range(self.n_inner):
                # phi_{p->q}: expectation of the edge likelihood under phi_{p<-q}
                lik = Y[:, :, None] * (phi_in @ log_B.T) + (1.0 - Y)[:, :, None] * (
                    phi_in @ log_1mB.T
                )
                phi_out = _row_softmax(e_log_pi[:, None, :] + lik)

                # phi_{p<-q}: symmetric update, contracting over the sender.
                lik = Y[:, :, None] * (phi_out @ log_B) + (1.0 - Y)[:, :, None] * (
                    phi_out @ log_1mB
                )
                phi_in = _row_softmax(e_log_pi[None, :, :] + lik)

            # Self-pairs contribute no evidence and are dropped from every
            # sufficient statistic.
            phi_out_m = phi_out * mask
            phi_in_m = phi_in * mask

            # gamma_p collects the sender mass of p over all pairs (p, q) and
            # the receiver mass of p over all pairs (q, p).
            gamma = (
                self.alpha
                + phi_out_m.sum(axis=1)
                + phi_in_m.transpose(1, 0, 2).sum(axis=1)
            )

            # Block matrix.
            pair_mass = np.einsum("pqg,pqh->gh", phi_out_m, phi_in_m, optimize=True)
            edge_mass = np.einsum(
                "pqg,pqh,pq->gh", phi_out_m, phi_in_m, Y, optimize=True
            )
            B = np.clip(edge_mass / (pair_mass + _EPS), _EPS, 1.0 - _EPS)

            memberships = gamma / gamma.sum(axis=1, keepdims=True)
            if np.abs(memberships - previous).mean() < self.tol:
                self.n_iter_ = sweep + 1
                break

        self.gamma_ = gamma
        self.B = B
        self.memberships_ = memberships
        self.pi = memberships
        return self

    # ------------------------------------------------------------------
    # Accessors and generative use
    # ------------------------------------------------------------------

    def get_memberships(self) -> np.ndarray:
        """Posterior mean membership of every node, shape ``(n_nodes, n_blocks)``."""
        self._check_fitted()
        return self.memberships_

    def get_block_matrix(self) -> np.ndarray:
        """Estimated block interaction probabilities, shape ``(n_blocks, n_blocks)``."""
        self._check_fitted()
        return self.B

    def sample_graph(self, n_nodes: int | None = None) -> np.ndarray:
        """Draw an adjacency matrix from the MMSB generative model.

        Uses the fitted parameters when the model has been fitted, and the
        prior otherwise. This is a data generator, not an inference routine;
        :meth:`fit` is what estimates memberships from an observed graph.
        """
        rng = np.random.default_rng(self.random_state)
        K = self.n_blocks

        if self.memberships_ is not None:
            pi = self.memberships_
            B = self.B
            n = pi.shape[0] if n_nodes is None else n_nodes
            if n != pi.shape[0]:
                pi = rng.dirichlet(np.full(K, self.alpha), size=n)
        else:
            n = n_nodes if n_nodes is not None else self.n_nodes
            if n is None:
                raise ValueError(
                    "sample_graph needs a node count: pass n_nodes, set it in "
                    "the constructor, or fit the model first."
                )
            pi = rng.dirichlet(np.full(K, self.alpha), size=n)
            B = rng.random((K, K))

        # z_send[p, q] ~ pi_p and z_recv[p, q] ~ pi_q, drawn for all pairs at once.
        z_send = _categorical(rng, np.broadcast_to(pi[:, None, :], (n, n, K)))
        z_recv = _categorical(rng, np.broadcast_to(pi[None, :, :], (n, n, K)))
        probabilities = B[z_send, z_recv]
        return (rng.random((n, n)) < probabilities).astype(np.float64)


def _row_softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, shifted for numerical stability."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    np.exp(shifted, out=shifted)
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _categorical(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    """Vectorised categorical draw over the last axis via inverse CDF."""
    cumulative = np.cumsum(probabilities, axis=-1)
    cumulative /= cumulative[..., -1:]
    uniform = rng.random(probabilities.shape[:-1] + (1,))
    return (uniform > cumulative).sum(axis=-1)
