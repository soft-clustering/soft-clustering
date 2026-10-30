# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Deep Modularity Networks (DMoN).

Reference
---------
A. Tsitsulin, J. Palowitch, B. Perozzi and E. Mueller. *Graph clustering with
graph neural networks.* Journal of Machine Learning Research, 24(127):1--21,
2023.

A GCN encoder maps node features to a soft cluster assignment
:math:`C \\in [0,1]^{n \\times k}` and is trained to maximise the spectral
modularity of that assignment while a collapse regulariser keeps the clusters
from merging:

.. math::

    \\mathcal{L}_{\\mathrm{DMoN}}
      = -\\frac{1}{2m}\\,\\mathrm{tr}\\!\\big(C^\\top B\\, C\\big)
        \\;+\\; \\frac{\\sqrt{k}}{n}
        \\Big\\lVert \\textstyle\\sum_i C_i^\\top \\Big\\rVert_F - 1,
    \\qquad B = A - \\frac{d\\,d^\\top}{2m}.

Both terms follow Equation (7) of the reference. Earlier releases of SCPP
exposed the encoder and the loss but no training procedure, so the module was
not usable as an estimator; :meth:`fit` now runs the optimisation and
populates the standard fitted attributes.

Deviation from the reference. The published architecture applies a
SeLU-activated GCN with skip connections and dropout; this implementation uses
a two-layer GCN with ReLU and no dropout. The objective, the collapse
regulariser and the soft-assignment semantics are those of the paper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._torch_utils import isolated_torch_rng


class _GCNLayer(nn.Module):
    r"""Graph convolution of Kipf and Welling (2017), in plain PyTorch.

    Computes :math:`\hat D^{-1/2} \hat A \hat D^{-1/2} H W + b`, where
    :math:`\hat A` is the adjacency matrix with a self-loop added to every node
    that lacks one and :math:`\hat D` its degree matrix; messages flow along
    ``edge_index`` from ``edge_index[0]`` (source) to ``edge_index[1]``
    (target). This is exactly what ``torch_geometric.nn.GCNConv`` computes
    with its defaults (``tests/test_dmon.py`` checks the two against each
    other when torch_geometric is installed), without the dependency.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    @staticmethod
    def propagation_matrix(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Dense :math:`\\hat D^{-1/2} \\hat A \\hat D^{-1/2}` with ``A[target, source]``."""
        M = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
        M.index_put_(
            (edge_index[1], edge_index[0]),
            torch.ones(edge_index.shape[1], dtype=torch.float32),
            accumulate=True,
        )
        diag = torch.arange(n_nodes)
        M[diag, diag] = torch.where(M[diag, diag] > 0, M[diag, diag], 1.0)
        deg = M.sum(dim=1)
        inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        return inv_sqrt[:, None] * M * inv_sqrt[None, :]

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        return propagation @ (x @ self.weight) + self.bias


@typechecked
class DMoN(nn.Module, BaseSoftClusterer):
    """Deep Modularity Networks, trainable through :meth:`fit`."""

    # The encoder ends in a softmax, so assignments are row-stochastic.
    _partition_constrained = True
    _membership_attrs = ("memberships_",)
    _centers_attrs = ()

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        n_clusters: int = 3,
        collapse_weight: float = 1.0,
        max_epochs: int = 300,
        lr: float = 1e-2,
        tol: float = 1e-6,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        in_channels : int
            Node-feature dimension.
        hidden_channels : int
            Width of the hidden GCN layer.
        n_clusters : int
            Number of clusters.
        collapse_weight : float
            Weight on the collapse regulariser.
        max_epochs : int
            Maximum number of Adam updates taken by :meth:`fit`.
        lr : float
            Adam learning rate.
        tol : float
            Stop after two consecutive absolute recorded-loss changes are
            strictly below this threshold, including small increases.
            Zero disables tolerance stopping.
        random_state : int or None
            Seed for the encoder initialisation.
        """
        nn.Module.__init__(self)

        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        with isolated_torch_rng(random_state):
            self.conv1 = _GCNLayer(in_channels, hidden_channels)
            self.conv2 = _GCNLayer(hidden_channels, n_clusters)
        self.n_clusters = n_clusters
        self.collapse_weight = collapse_weight
        self.max_epochs = max_epochs
        self.lr = lr
        self.tol = tol
        self.random_state = random_state

        self.loss_curve_: list[float] = []

    # ------------------------------------------------------------------
    # Module interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the soft cluster assignment, shape ``(n_nodes, n_clusters)``."""
        propagation = _GCNLayer.propagation_matrix(edge_index, int(x.shape[0]))
        h = F.relu(self.conv1(x, propagation))
        return F.softmax(self.conv2(h, propagation), dim=1)

    def loss(self, soft_assign: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Negative soft modularity plus the collapse regulariser."""
        n, k = soft_assign.shape
        two_m = adj.sum()
        if float(two_m) <= 0:
            # Keep the zero attached to the autograd graph; see _cdcgs.
            modularity = soft_assign.sum() * 0.0
        else:
            degrees = adj.sum(dim=1, keepdim=True)
            left = soft_assign.T @ adj @ soft_assign
            projected = soft_assign.T @ degrees
            null = (projected @ projected.T) / two_m
            modularity = torch.trace(left - null) / two_m

        # Equation (7): (sqrt(k)/n) * ||sum_i C_i||_F - 1.
        cluster_size = soft_assign.sum(dim=0)
        collapse = (k**0.5 / n) * torch.norm(cluster_size, p=2) - 1.0

        return -modularity + self.collapse_weight * collapse

    # ------------------------------------------------------------------
    # Estimator interface
    # ------------------------------------------------------------------

    def fit(self, x, edge_index=None, adj=None) -> DMoN:
        """Fit the encoder on a graph.

        Parameters
        ----------
        x : array-like of shape ``(n_nodes, in_channels)``
            Node features.
        edge_index : array-like of shape ``(2, n_edges)``, optional
            Edge list. Derived from ``adj`` when omitted.
        adj : array-like of shape ``(n_nodes, n_nodes)``, optional
            Dense adjacency matrix. Derived from ``edge_index`` when omitted.

        Notes
        -----
        Each recorded loss is evaluated before its Adam update; tolerance
        stopping is checked after that update. Memberships are recomputed
        from the final updated parameters, whose stationarity is not certified
        by the recorded-loss comparisons. Each fit resets the loss history
        and consecutive-small-change count.
        """
        x = _as_tensor(x)
        if edge_index is None and adj is None:
            raise ValueError("provide at least one of `edge_index` or `adj`")

        if adj is None:
            edge_index = torch.as_tensor(np.asarray(edge_index), dtype=torch.long)
            n = int(x.shape[0])
            adj = torch.zeros((n, n), dtype=torch.float32)
            adj[edge_index[0], edge_index[1]] = 1.0
            adj = torch.maximum(adj, adj.T)
            adj.fill_diagonal_(0.0)
        else:
            adj = _as_tensor(adj)
            if edge_index is None:
                edge_index = adj.nonzero(as_tuple=False).T.contiguous()
            else:
                edge_index = torch.as_tensor(np.asarray(edge_index), dtype=torch.long)

        if adj.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_nodes={adj.shape[0]} is smaller than n_clusters={self.n_clusters}"
            )

        self.loss_curve_ = []
        previous = float("inf")
        small_changes = 0

        # Every fit starts from freshly initialised weights drawn from a
        # private, seeded stream, so refitting an instance reproduces the
        # first fit instead of continuing from its trained weights.
        with isolated_torch_rng(self.random_state):
            self.conv1.reset_parameters()
            self.conv2.reset_parameters()
            optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.train()
            for _ in range(self.max_epochs):
                optimiser.zero_grad()
                soft_assign = self(x, edge_index, adj)
                loss = self.loss(soft_assign, adj)
                loss.backward()
                optimiser.step()

                value = float(loss.detach())
                self.loss_curve_.append(value)
                if abs(previous - value) < self.tol:
                    small_changes += 1
                else:
                    small_changes = 0
                if small_changes >= 2:
                    break
                previous = value

        self.eval()
        with torch.no_grad():
            soft_assign = self(x, edge_index, adj)
        self.memberships_ = soft_assign.detach().cpu().numpy().astype(np.float64)
        return self


def _as_tensor(data) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(torch.float32)
    return torch.as_tensor(np.asarray(data, dtype=np.float32))
