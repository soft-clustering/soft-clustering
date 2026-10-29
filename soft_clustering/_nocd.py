# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.
#
# Portions of this module (the edge sampler, the GCN and Bernoulli-Poisson
# decoder, the model saver and the early-stopping monitor) are adapted from
# the reference implementation of NOCD,
# https://github.com/shchur/overlapping-community-detection, which carries
# the following notice:
#
# Copyright (c) 2019 Oleksandr Shchur
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
from typeguard import typechecked

from ._base import BaseSoftClusterer
from ._torch_utils import isolated_torch_rng
from ._validation import check_n_clusters, int_seed


def _to_sparse_tensor(
    matrix: sp.spmatrix | torch.Tensor,
    cuda: bool = False,
) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse tensor.
    Args:
    matrix: Sparse matrix to convert.
    cuda: Whether to move the resulting tensor to GPU.

    Returns:
    sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

    """
    if sp.issparse(matrix):
        coo = sp.coo_matrix(matrix)
        indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        values = torch.as_tensor(coo.data, dtype=torch.float32)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, coo.shape)
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        sparse_tensor = torch.sparse_coo_tensor(indices, values, matrix.shape)
    else:
        raise ValueError(
            f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead)."
        )
    if cuda:
        sparse_tensor = sparse_tensor.cuda()
    return sparse_tensor.coalesce()


def _sparse_or_dense_dropout(x, p=0.5, training=True):
    if x.is_sparse:
        x = x.coalesce()
        new_values = torch.nn.functional.dropout(x.values(), p=p, training=training)
        return torch.sparse_coo_tensor(x.indices(), new_values, x.size())
    else:
        return torch.nn.functional.dropout(x, p=p, training=training)


def _l2_reg_loss(model, scale=1e-5):
    """Get L2 loss for model weights."""
    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.0).sum()
    return loss * scale


def _collate_fn(batch):
    edges, nonedges = batch[0]
    return (edges, nonedges)


def _check_edge_counts(num_nodes, num_edges):
    if num_nodes < 2 or not 0 < num_edges < num_nodes * (num_nodes - 1):
        raise ValueError(
            "NOCD: the graph must contain at least one off-diagonal edge "
            "and one off-diagonal nonedge."
        )


def _prepare_adjacency(A):
    """Copy the weighted topology, coalesce entries, and ignore self-loops."""
    A = sp.csr_matrix(A, dtype=np.float32, copy=True)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"NOCD: adjacency_matrix must be square; got {A.shape}.")
    A.sum_duplicates()
    A.setdiag(0)
    A.eliminate_zeros()
    _check_edge_counts(n, A.nnz)
    return A


def _get_edge_sampler(
    A, num_pos=1000, num_neg=1000, num_workers=0, random_seed=None, device="cpu"
):
    """Build a DataLoader over sampled edges and non-edges.

    ``num_workers`` defaults to 0 (in-process loading). Worker processes are
    started with the ``spawn`` method on macOS and Windows, which re-imports
    the caller's ``__main__`` module; without an ``if __name__ == "__main__"``
    guard — as when running under pytest or in a notebook — the workers fail
    and the parent blocks on an empty queue. Sampling a batch is a few NumPy
    operations, so in-process loading is also faster here.
    """
    base_seed = (
        int(random_seed)
        if random_seed is not None
        else int(np.random.default_rng().integers(0, 2**31 - 1))
    )
    data_source = _EdgeSampler(A, num_pos, num_neg, base_seed=base_seed)
    return torch.utils.data.DataLoader(
        data_source, num_workers=num_workers, collate_fn=_collate_fn
    )


class _EdgeSampler(torch.utils.data.Dataset):
    """Sample ordered edges and nonedges uniformly with replacement.

    A finite rank map uses O(nnz + n) storage, without a dense complement or
    rejection loop. Self-loops are ignored, and both edge classes must exist.

    Args:
        A: adjacency matrix.
        num_pos: number of edges per batch.
        num_neg: number of non-edges per batch.
    """

    def __init__(self, A, num_pos=1000, num_neg=1000, base_seed=0):
        self.base_seed = base_seed
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.A = A = _prepare_adjacency(A)
        self.edges = np.transpose(A.nonzero()).astype(np.int64, copy=False)
        self.num_nodes = A.shape[0]
        self.num_edges = self.edges.shape[0]
        forbidden = np.unique(
            np.concatenate(
                [
                    self.edges[:, 0] * self.num_nodes + self.edges[:, 1],
                    np.arange(self.num_nodes, dtype=np.int64) * (self.num_nodes + 1),
                ]
            )
        )
        self._nonedge_offsets = forbidden - np.arange(len(forbidden), dtype=np.int64)
        self.num_nonedges = self.num_nodes**2 - len(forbidden)

    def __getitem__(self, key):
        # A private generator per batch, derived from the fit's seed and the
        # batch index. The reference reseeded NumPy's global generator here.
        rng = np.random.default_rng([self.base_seed, key])
        edges_idx = rng.integers(0, self.num_edges, size=self.num_pos, dtype=np.int64)
        next_edges = self.edges[edges_idx, :]

        ranks = rng.integers(0, self.num_nonedges, size=self.num_neg, dtype=np.int64)
        flat = ranks + np.searchsorted(self._nonedge_offsets, ranks, side="right")
        next_nonedges = np.column_stack((flat // self.num_nodes, flat % self.num_nodes))
        return torch.LongTensor(next_edges), torch.LongTensor(next_nonedges)

    def __len__(self):
        return 2**32


class _GraphConvolution(torch.nn.Module):
    """Graph convolution layer.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.

    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        return adj @ (x @ self.weight) + self.bias


class _GCN(torch.nn.Module):
    """Graph convolution network.

    References:
        "Semi-superivsed learning with graph convolutional networks",
        Kipf and Welling, ICLR 2017
    """

    def __init__(
        self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False
    ):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = torch.nn.ModuleList([_GraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(_GraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if batch_norm:
            self.batch_norm = [
                torch.nn.BatchNorm1d(dim, affine=False, track_running_stats=False)
                for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    @staticmethod
    def normalize_adj(adj: sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(
                deg_sqrt_inv[None, :]
            )
        elif torch.is_tensor(adj):
            deg_t = adj.sum(1)
            deg_t_sqrt_inv = 1 / torch.sqrt(deg_t)
            adj_norm = adj * deg_t_sqrt_inv[:, None] * deg_t_sqrt_inv[None, :]
        return _to_sparse_tensor(adj_norm)

    def forward(self, x, adj):
        for idx, gcn in enumerate(self.layers):
            if self.dropout != 0:
                x = _sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gcn(x, adj)
            if idx != len(self.layers) - 1:
                x = torch.nn.functional.relu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if "bias" not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if "bias" in n]


class _BernoulliDecoder(torch.nn.Module):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        """Base class for Bernoulli decoder.

        Args:
            num_nodes: Number of nodes in a graph.
            num_edges: Number of edges in a graph.
            balance_loss: Whether to balance contribution from edges and non-edges.
        """
        super().__init__()
        _check_edge_counts(num_nodes, num_edges)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        self.balance_loss = balance_loss

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        raise NotImplementedError

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        raise NotImplementedError

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute loss for given edges and non-edges."""
        raise NotImplementedError

    def loss_full(self, emb, adj):
        """Compute loss for all edges and non-edges."""
        raise NotImplementedError


class _BerpoDecoder(_BernoulliDecoder):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        super().__init__(num_nodes, num_edges, balance_loss)
        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        self.eps = -np.log(1 - edge_proba)

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        e1, e2 = idx.t()
        logits = torch.sum(emb[e1] * emb[e2], dim=1)
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return torch.distributions.Bernoulli(probs=probs)

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        logits = emb @ emb.t()
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return torch.distributions.Bernoulli(probs=probs)

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute BerPo loss for a batch of edges and non-edges."""
        # Loss for edges
        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Loss for non-edges
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))
        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale)

    def loss_full(self, emb, adj):
        """Compute BerPo loss for all edges & non-edges in a graph."""
        e1, e2 = adj.nonzero()
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.sum(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Correct for overcounting F_u * F_v for edges and nodes with themselves
        self_dots_sum = torch.sum(emb * emb)
        correction = self_dots_sum + torch.sum(edge_dots)
        sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
        loss_nonedges = torch.sum(emb @ sum_emb) - correction

        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (
            loss_edges / self.num_edges + neg_scale * loss_nonedges / self.num_nonedges
        ) / (1 + neg_scale)


class _ModelSaver:
    """In-memory saver for model parameters.

    Storing weights in memory is faster than saving to disk with torch.save.
    """

    def __init__(self, model):
        self.model = model

    def save(self):
        self.state_dict = deepcopy(self.model.state_dict())

    def restore(self):
        self.model.load_state_dict(self.state_dict)


class _EarlyStopping:
    """Base class for an early stopping monitor that says when it's time to stop training.

    Examples
    --------
    early_stopping = _EarlyStopping()
    for epoch in range(max_epochs):
        sess.run(train_op)  # perform training operation
        early_stopping.next_step()
        if early_stopping.should_stop():
            break
        if early_stopping.should_save():
            model_saver.save()  # save model weights

    """

    def __init__(self):
        pass

    def reset(self):
        """Reset the internal state."""
        raise NotImplementedError

    def next_step(self):
        """Should be called at every iteration."""
        raise NotImplementedError

    def should_save(self):
        """Says if it's time to save model weights."""
        raise NotImplementedError

    def should_stop(self):
        """Says if it's time to stop training."""
        raise NotImplementedError


class _NoImprovementStopping(_EarlyStopping):
    """Stop training when the validation metric stops improving.

    Parameters
    ----------
    validation_fn : function
        Calling this function returns the current value of the validation metric.
    mode : {'min', 'max'}
        Should the validation metric be minimized or maximized?
    patience : int
        Number of iterations without improvement before stopping.
    tolerance : float
        Minimal improvement in validation metric to not trigger patience.
    relative : bool
        Is tolerance measured in absolute units or relatively?

    Attributes
    ----------
    _best_value : float
        Best value of the validation loss.
    _num_bad_epochs : int
        Number of epochs since last significant improvement in validation metric.
    _time_to_save : bool
        Is it time to save the model weights?
    _is_better : function
        Tells if new validation metric value is better than the best one so far.
        Signature self._is_better(new_value, best_value).

    """

    def __init__(
        self, validation_fn, mode="min", patience=10, tolerance=0.0, relative=False
    ):
        super().__init__()
        self.validation_fn = validation_fn
        self.mode = mode
        self.patience = patience
        self.tolerance = tolerance
        self.relative = relative
        self.reset()

        if mode not in ["min", "max"]:
            raise ValueError(
                f"Mode should be either 'min' or 'max' (got {mode} instead)."
            )

        # Create the comparison function
        if relative:
            if mode == "min":
                self._is_better = lambda new, best: new < best - (best * tolerance)
            if mode == "max":
                self._is_better = lambda new, best: new > best + (best * tolerance)
        else:
            if mode == "min":
                self._is_better = lambda new, best: new < best - tolerance
            if mode == "max":
                self._is_better = lambda new, best: new > best + tolerance

    def reset(self):
        """Reset the internal state."""
        self._best_value = self.validation_fn()
        self._num_bad_epochs = 0
        self._time_to_save = False

    def next_step(self):
        """Should be called at every iteration."""
        last_value = self.validation_fn()
        if self._is_better(last_value, self._best_value):
            self._time_to_save = True
            self._best_value = last_value
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1

    def should_save(self):
        """Says if it's time to save model weights."""
        if self._time_to_save:
            self._time_to_save = False
            return True
        else:
            return False

    def should_stop(self):
        """Says if it's time to stop training."""
        return self._num_bad_epochs > self.patience


@typechecked
class NOCD(BaseSoftClusterer):
    r"""Neural overlapping community detection (NOCD).

    A graph convolutional network maps every node to a non-negative
    affiliation vector :math:`F_u`, trained so that the Bernoulli--Poisson
    model :math:`P(A_{uv} = 1) = 1 - \exp(-F_u^\top F_v)` explains the observed
    edges, with a loss balanced between edges and non-edges, mini-batches of
    sampled edges and non-edges, weight decay, and early stopping on the full
    loss (Shchur and Gunnemann, 2019). The weights with the best loss are
    restored and the affiliations are computed in evaluation mode.

    ``memberships_`` holds the affiliations divided by their global maximum,
    so they lie in ``[0, 1]`` and keep their relative magnitudes; the raw
    strengths are in ``affiliations_``. The reference obtains hard,
    overlapping communities by thresholding the raw strengths.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of communities.
    random_state : int or None, default=None
        Seed for the network initialisation, dropout and edge sampling. A
        seeded fit is reproducible and leaves the global NumPy and PyTorch
        random states untouched.
    hidden_sizes : sequence of int, default=(128,)
        Widths of the hidden GCN layers.
    weight_decay : float, default=1e-2
    dropout : float, default=0.5
    batch_norm : bool, default=True
    lr : float, default=1e-3
    max_epochs : int, default=500
        Maximum optimizer updates. Zero returns the evaluated initial model.
        Evaluate after every 25 updates and the final update, keeping the best
        full-loss checkpoint (including initialization).
    balance_loss : bool, default=True
        Weight edges and non-edges equally in the loss.
    stochastic_loss : bool, default=True
        Train on sampled mini-batches rather than the full graph.
    batch_size : int, default=20000
        Edges (and non-edges) per mini-batch.
    num_workers : int, default=0
        DataLoader worker processes; 0 samples in-process.

    References
    ----------
    O. Shchur and S. Gunnemann. Overlapping community detection with graph
    neural networks. arXiv:1909.12201, 2019.

    Notes
    -----
    Earlier releases seeded torch, CUDA and NumPy globally and enabled
    ``torch.use_deterministic_algorithms`` for the whole process in the
    constructor, set the global default device in ``fit``, never restored
    the best weights they saved, and computed the output with dropout active.
    """

    # Bernoulli-Poisson affiliations for overlapping communities are independent
    # per community (Shchur and Guennemann, 2019)
    _partition_constrained = False

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int | None = None,
        hidden_sizes: list[int] | tuple[int, ...] = (128,),
        weight_decay: float = 1e-2,
        dropout: float = 0.5,
        batch_norm: bool = True,
        lr: float = 1e-3,
        max_epochs: int = 500,
        balance_loss: bool = True,
        stochastic_loss: bool = True,
        batch_size: int = 20000,
        num_workers: int = 0,
    ):

        # Earlier releases seeded torch, CUDA and NumPy globally here and
        # switched on torch.use_deterministic_algorithms for the whole
        # process. Randomness is now confined to a private stream inside fit.
        self.n_clusters = n_clusters
        self.random_state = random_state

        # hyperparameters
        self.hidden_sizes = hidden_sizes  # hidden sizes of the GNN
        self.weight_decay = weight_decay  # strength of L2 regularization on GNN weights
        self.dropout = dropout  # whether to use dropout
        self.batch_norm = batch_norm  # whether to use batch norm
        self.lr = lr  # learning rate
        self.max_epochs = max_epochs  # number of epochs to train
        self.balance_loss = balance_loss  # whether to use balanced loss
        # whether to use stochastic or full-batch training
        self.stochastic_loss = stochastic_loss
        # batch size (only for stochastic training)
        self.batch_size = batch_size
        # DataLoader worker processes; 0 loads in-process (see _get_edge_sampler)
        self.num_workers = num_workers

    def fit_predict(self, adjacency_matrix, feature_matrix=None, K: int | None = None):
        """Train the model and return community affiliations.

        Parameters
        ----------
        adjacency_matrix : scipy sparse matrix or ndarray of shape (n_nodes, n_nodes)
            Must contain an off-diagonal edge and an off-diagonal nonedge.
            Self-loops are ignored on a private copy; duplicate coordinates are
            summed and stored zeros removed. Off-diagonal weights and direction
            are preserved, with nonzero entries defining edges.
        feature_matrix : scipy sparse matrix or ndarray of shape (n_nodes, n_features), optional
            Node attributes. ``None`` uses the identity (no attributes); the
            adjacency rows are always appended, as in the reference.
        K : int, optional
            Deprecated fit-time community count; prefer ``n_clusters``.

        Returns
        -------
        ndarray of shape (n_nodes, n_clusters)
        """
        if self.max_epochs < 0:
            raise ValueError("NOCD: max_epochs must be nonnegative.")
        adjacency_matrix = _prepare_adjacency(adjacency_matrix)
        n = adjacency_matrix.shape[0]
        K = check_n_clusters(self.n_clusters if K is None else K, n, estimator="NOCD")
        if feature_matrix is None:
            feature_matrix = sp.identity(n, format="csr", dtype=np.float32)
        feature_matrix = sp.csr_matrix(feature_matrix, dtype=np.float32)
        with isolated_torch_rng(self.random_state):
            return self._fit(adjacency_matrix, feature_matrix, K)

    def _fit(self, adjacency_matrix, feature_matrix, K):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_norm = sp.hstack([feature_matrix, adjacency_matrix])
        x_norm = _to_sparse_tensor(x_norm).to(device)
        gnn = _GCN(
            x_norm.shape[1],
            self.hidden_sizes,
            K,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(device)
        adj_norm = gnn.normalize_adj(adjacency_matrix).to(device)
        decoder = _BerpoDecoder(
            adjacency_matrix.shape[0],
            adjacency_matrix.nnz,
            balance_loss=self.balance_loss,
        )
        opt = torch.optim.Adam(gnn.parameters(), lr=self.lr)

        val_loss = np.inf

        def validation_fn():
            return val_loss

        early_stopping = _NoImprovementStopping(validation_fn, patience=10)
        model_saver = _ModelSaver(gnn)

        def validate():
            nonlocal val_loss
            with torch.no_grad():
                gnn.eval()
                Z = torch.nn.functional.relu(gnn(x_norm, adj_norm))
                val_loss = decoder.loss_full(Z, adjacency_matrix)
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
            return early_stopping.should_stop()

        validate()
        if not hasattr(model_saver, "state_dict"):
            model_saver.save()
        batches = None
        if self.stochastic_loss and self.max_epochs > 0:
            batches = iter(
                _get_edge_sampler(
                    adjacency_matrix,
                    self.batch_size,
                    self.batch_size,
                    num_workers=self.num_workers,
                    random_seed=int_seed(self.random_state),
                    device=device,
                )
            )

        for completed in range(1, self.max_epochs + 1):

            # Training step
            gnn.train()
            opt.zero_grad()
            Z = torch.nn.functional.relu(gnn(x_norm, adj_norm))
            if batches is not None:
                ones_idx, zeros_idx = next(batches)
                loss = decoder.loss_batch(Z, ones_idx, zeros_idx)
            else:
                loss = decoder.loss_full(Z, adjacency_matrix)
            loss += _l2_reg_loss(gnn, scale=self.weight_decay)
            loss.backward()
            opt.step()
            if completed % 25 == 0 or completed == self.max_epochs:
                if validate():
                    break

        # Restore the weights with the best validation loss and evaluate
        # without dropout. Earlier releases saved those weights but never
        # restored them, and computed the output in training mode, so the
        # memberships carried dropout noise.
        if hasattr(model_saver, "state_dict"):
            model_saver.restore()
        gnn.eval()
        with torch.no_grad():
            Z = torch.nn.functional.relu(gnn(x_norm, adj_norm))
        F = Z.cpu().numpy().astype(np.float64)
        self.affiliations_ = F
        # Affiliation strengths are rescaled to [0, 1] by the global maximum,
        # which keeps their relative magnitudes; the reference thresholds the
        # raw strengths to obtain hard, overlapping communities.
        top = float(F.max())
        self.memberships_ = F / top if top > 0 else F
        return self.memberships_
