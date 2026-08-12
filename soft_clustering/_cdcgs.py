"""Community detection clustering via Gumbel-Softmax (CDCGS).

Reference
---------
D. B. Acharya and H. Zhang. *Community detection clustering via Gumbel
softmax.* SN Computer Science, 1(5):262, 2020.

The method holds one free logit vector per node, relaxes the hard community
assignment with the Gumbel-Softmax reparameterisation so that the assignment
stays differentiable, and optimises a community-structure objective by
gradient descent. Sampling noise during training lets the assignment escape
the local optima that a plain softmax relaxation settles into, while the
learned logits give a deterministic soft partition at inference time.

Objectives
----------
``objective="modularity"`` (default)
    Maximise the soft modularity :math:`\\frac{1}{2m}\\,\\mathrm{tr}(C^\\top B C)`
    with :math:`B = A - dd^\\top/2m`, which is the community criterion the
    reference method targets.

``objective="block_diagonal"``
    Minimise :math:`\\lVert \\mathrm{softmax}(C^\\top A C) - I \\rVert^2`,
    a block-diagonality surrogate. This was the only objective exposed by
    earlier releases of SCPP and is kept so that existing code reproduces.

Temperature annealing. The Gumbel-Softmax temperature decays geometrically
from ``tau`` to ``tau_min`` over training, as Jang et al. prescribe and the
reference method uses: a high temperature early on keeps the relaxation smooth
enough to explore, and a low temperature late on drives the assignment towards
a decisive partition. Set ``tau_min = tau`` for a fixed temperature.

Deviation from the reference. Acharya and Zhang additionally describe a
graph-preprocessing stage that is not reproduced here. The optimisation
target, the Gumbel-Softmax relaxation with annealing, and the per-node logit
parameterisation follow the paper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from ._base import BaseSoftClusterer


@typechecked
class CDCGS(nn.Module, BaseSoftClusterer):
    """Gumbel-Softmax community detection, trainable through :meth:`fit`."""

    # Soft community assignments are row-stochastic by construction.
    _partition_constrained = True
    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        n_clusters: int = 3,
        num_nodes: int | None = None,
        tau: float = 1.0,
        tau_min: float = 0.1,
        objective: str = "modularity",
        max_epochs: int = 500,
        n_init: int = 5,
        lr: float = 0.05,
        tol: float = 1e-6,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        n_clusters : int
            Number of communities.
        num_nodes : int or None
            Node count. Optional: :meth:`fit` infers it from the adjacency
            matrix and builds the logits then. It is required only when the
            module is used directly through :meth:`forward` without fitting.
        tau : float
            Initial Gumbel-Softmax temperature.
        tau_min : float
            Final temperature. ``fit`` anneals ``tau`` down to this value
            geometrically; set it equal to ``tau`` to disable annealing.
        objective : {"modularity", "block_diagonal"}
            Training criterion; see the module docstring.
        max_epochs : int
            Number of gradient steps per restart.
        n_init : int
            Number of restarts; the run with the best training objective is
            kept. Free per-node logits under Gumbel noise land in a local
            optimum often enough that a single run is unreliable: on a planted
            3-block graph one restart recovers modularity 0.30 against the
            planted partition's 0.46, while five restarts recover 0.46 exactly.
        lr : float
            Adam learning rate.
        tol : float
            Stop when the loss improves by less than this over a step.
        random_state : int or None
            Seed for the logit initialisation and the Gumbel noise.
        """
        nn.Module.__init__(self)

        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if tau_min <= 0:
            raise ValueError(f"tau_min must be > 0, got {tau_min}")
        if objective not in ("modularity", "block_diagonal"):
            raise ValueError(
                "objective must be 'modularity' or 'block_diagonal', "
                f"got {objective!r}"
            )
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")

        self.n_init = n_init
        self.n_clusters = n_clusters
        self.num_nodes = num_nodes
        self.tau = tau
        self.tau_min = tau_min
        self._tau = tau  # current temperature, annealed by fit()
        self.objective = objective
        self.max_epochs = max_epochs
        self.lr = lr
        self.tol = tol
        self.random_state = random_state

        self.memberships_: np.ndarray | None = None
        self.loss_curve_: list[float] = []

        if random_state is not None:
            torch.manual_seed(random_state)
        self.W_C = (
            nn.Parameter(torch.randn(num_nodes, n_clusters)) if num_nodes else None
        )

    # ------------------------------------------------------------------
    # Module interface
    # ------------------------------------------------------------------

    def _ensure_parameters(self, num_nodes: int, seed: int | None = None) -> None:
        if seed is not None or self.W_C is None or self.W_C.shape[0] != num_nodes:
            if seed is not None:
                torch.manual_seed(seed)
            elif self.random_state is not None:
                torch.manual_seed(self.random_state)
            self.W_C = nn.Parameter(torch.randn(num_nodes, self.n_clusters))
            self.num_nodes = num_nodes

    def forward(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(community interaction matrix, soft assignment)``.

        The soft assignment is a Gumbel-Softmax sample while the module is in
        training mode and a plain softmax in evaluation mode, so that a fitted
        model produces deterministic memberships.
        """
        self._ensure_parameters(adj.shape[0])
        if self.training:
            soft_assign = F.gumbel_softmax(self.W_C, tau=self._tau, hard=False, dim=1)
        else:
            soft_assign = F.softmax(self.W_C / self._tau, dim=1)

        R = soft_assign.T @ adj @ soft_assign  # (k, k)
        output = F.softmax(R, dim=1)
        return output, soft_assign

    def loss(
        self,
        output: torch.Tensor,
        soft_assign: torch.Tensor | None = None,
        adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training criterion.

        ``loss(output)`` evaluates the block-diagonality surrogate, which is
        the signature earlier releases exposed. Passing ``soft_assign`` and
        ``adj`` evaluates the negative soft modularity instead.
        """
        if self.objective == "modularity" and soft_assign is not None:
            if adj is None:
                raise ValueError("the modularity objective needs `adj`")
            return -_soft_modularity(soft_assign, adj)

        identity = torch.eye(self.n_clusters, device=output.device)
        return F.mse_loss(output, identity)

    # ------------------------------------------------------------------
    # Estimator interface
    # ------------------------------------------------------------------

    def fit(self, adjacency_matrix) -> CDCGS:
        """Fit on a binary ``(n_nodes, n_nodes)`` adjacency matrix."""
        adj = _as_tensor(adjacency_matrix)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(
                f"adjacency_matrix must be square (n, n), got shape {tuple(adj.shape)}"
            )
        if adj.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_nodes={adj.shape[0]} is smaller than n_clusters={self.n_clusters}"
            )

        # Geometric anneal from tau to tau_min across each restart.
        steps = max(self.max_epochs - 1, 1)
        decay = (self.tau_min / self.tau) ** (1.0 / steps)
        base_seed = self.random_state if self.random_state is not None else 0

        best_loss = float("inf")
        best_logits = None
        best_curve: list[float] = []

        for restart in range(self.n_init):
            self._ensure_parameters(adj.shape[0], seed=base_seed + restart)
            optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)
            curve: list[float] = []
            previous = float("inf")

            self.train()
            for epoch in range(self.max_epochs):
                self._tau = max(self.tau * decay**epoch, self.tau_min)
                optimiser.zero_grad()
                output, soft_assign = self(adj)
                loss = self.loss(output, soft_assign, adj)
                loss.backward()
                optimiser.step()

                value = float(loss.detach())
                curve.append(value)
                # Only stop early once the temperature schedule has finished;
                # a plateau at high temperature is noise, not convergence.
                if self._tau <= self.tau_min and abs(previous - value) < self.tol:
                    break
                previous = value

            # Score the restart at zero temperature noise, so restarts are
            # compared on the partition they actually produce.
            self._tau = self.tau_min
            self.eval()
            with torch.no_grad():
                output, soft_assign = self(adj)
                final = float(self.loss(output, soft_assign, adj))
            if final < best_loss:
                best_loss = final
                best_logits = self.W_C.detach().clone()
                best_curve = curve

        self.W_C = nn.Parameter(best_logits)
        self._tau = self.tau_min
        self.loss_curve_ = best_curve
        self.best_loss_ = best_loss
        self.eval()
        with torch.no_grad():
            _, soft_assign = self(adj)
        self.memberships_ = soft_assign.detach().cpu().numpy().astype(np.float64)
        return self


def _soft_modularity(soft_assign: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """``tr(C^T B C) / 2m`` with ``B = A - d d^T / 2m``."""
    degrees = adj.sum(dim=1, keepdim=True)
    two_m = adj.sum()
    if float(two_m) <= 0:
        # An edgeless graph has no modularity to maximise. Return a zero that
        # is still attached to the autograd graph, so the caller's backward
        # pass stays valid instead of raising on a detached constant.
        return soft_assign.sum() * 0.0
    left = soft_assign.T @ adj @ soft_assign
    projected = soft_assign.T @ degrees
    null = (projected @ projected.T) / two_m
    return torch.trace(left - null) / two_m


def _as_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(torch.float32)
    return torch.as_tensor(np.asarray(data, dtype=np.float32))
