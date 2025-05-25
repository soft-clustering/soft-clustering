import torch
import random
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
from typeguard import typechecked
from typing import Union, Optional, List


def _to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
					cuda: bool = torch.cuda.is_available(),
					) -> Union[torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor]:
	"""Convert a scipy sparse matrix to a torch sparse tensor.
	Args:
	matrix: Sparse matrix to convert.
	cuda: Whether to move the resulting tensor to GPU.

	Returns:
	sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

	"""
	if sp.issparse(matrix):
		coo = matrix.tocoo()
		indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
		values = torch.FloatTensor(coo.data)
		shape = torch.Size(coo.shape)
		sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
	elif torch.is_tensor(matrix):
		row, col = matrix.nonzero().t()
		indices = torch.stack([row, col])
		values = matrix[row, col]
		shape = torch.Size(matrix.shape)
		sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
	else:
		raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
	if cuda:
		sparse_tensor = sparse_tensor.cuda()
	return sparse_tensor.coalesce() 


def _sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
        new_values = torch.nn.functional.dropout(x.values(), p=p, training=training)
        if torch.cuda.is_available():
            return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
        return torch.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return torch.nn.functional.dropout(x, p=p, training=training)


def _l2_reg_loss(model, scale=1e-5):
    """Get L2 loss for model weights."""
    loss = 0.0
    for w in model.get_weights():
        loss += w.pow(2.).sum()
    return loss * scale


def _collate_fn(batch):
	edges, nonedges = batch[0]
	return (edges, nonedges)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _get_edge_sampler(A, num_pos=1000, num_neg=1000, num_workers=2, random_seed=None):
    data_source = _EdgeSampler(A, num_pos, num_neg)
    if random_seed:
        return torch.utils.data.DataLoader(data_source, num_workers=num_workers, collate_fn=_collate_fn, worker_init_fn=_seed_worker, generator=torch.Generator().manual_seed(random_seed))
    return torch.utils.data.DataLoader(data_source, num_workers=num_workers, collate_fn=_collate_fn)


class _EdgeSampler(torch.utils.data.Dataset):
    """Sample edges and non-edges uniformly from a graph.

    Args:
        A: adjacency matrix.
        num_pos: number of edges per batch.
        num_neg: number of non-edges per batch.
    """
    def __init__(self, A, num_pos=1000, num_neg=1000):
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.A = A
        self.edges = np.transpose(A.nonzero())
        self.num_nodes = A.shape[0]
        self.num_edges = self.edges.shape[0]

    def __getitem__(self, key):
        np.random.seed(key)
        edges_idx = np.random.randint(0, self.num_edges, size=self.num_pos, dtype=np.int64)
        next_edges = self.edges[edges_idx, :]

        # Select num_neg non-edges
        generated = False
        while not generated:
            candidate_ne = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)
            cne1, cne2 = candidate_ne[:, 0], candidate_ne[:, 1]
            to_keep = (1 - self.A[cne1, cne2]).astype(bool).A1 * (cne1 != cne2)
            next_nonedges = candidate_ne[to_keep][:self.num_neg]
            generated = to_keep.sum() >= self.num_neg
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
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = torch.nn.ModuleList([_GraphConvolution(input_dim, layer_dims[0])])
        for idx in range(len(layer_dims) - 1):
            self.layers.append(_GraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        if batch_norm:
            self.batch_norm = [
                torch.nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
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
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]


class _BernoulliDecoder(torch.nn.Module):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        """Base class for Bernoulli decoder.

        Args:
            num_nodes: Number of nodes in a graph.
            num_edges: Number of edges in a graph.
            balance_loss: Whether to balance contribution from edges and non-edges.
        """
        super().__init__()
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
        return (loss_edges / self.num_edges + neg_scale * loss_nonedges / self.num_nonedges) / (1 + neg_scale)


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
    def __init__(self, validation_fn, mode='min', patience=10, tolerance=0.0, relative=False):
        super().__init__()
        self.validation_fn = validation_fn
        self.mode = mode
        self.patience = patience
        self.tolerance = tolerance
        self.relative = relative
        self.reset()

        if mode not in ['min', 'max']:
            raise ValueError(f"Mode should be either 'min' or 'max' (got {mode} instead).")

        # Create the comparison function
        if relative:
            if mode == 'min':
                self._is_better = lambda new, best: new < best - (best * tolerance)
            if mode == 'max':
                self._is_better = lambda new, best: new > best + (best * tolerance)
        else:
            if mode == 'min':
                self._is_better = lambda new, best: new < best - tolerance
            if mode == 'max':
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


class NOCD:
	@typechecked
	def __init__(self, random_state: Optional[int] = None,
				hidden_sizes: List[int] = [128],
				weight_decay: float = 1e-2,
				dropout: float = 0.5,
				batch_norm: bool = True,
				lr: float = 1e-3,
				max_epochs: int = 500,
				balance_loss: bool = True,
				stochastic_loss: bool = True,
				batch_size: int = 20000):

		self.random_state = random_state
		if random_state:
			torch.manual_seed(random_state)
			torch.cuda.manual_seed(random_state)
			torch.cuda.manual_seed_all(random_state)
			torch.backends.cudnn.deterministic = True
			torch.use_deterministic_algorithms(True)
			np.random.seed(random_state)

		# hyperparameters
		self.hidden_sizes = hidden_sizes  # hidden sizes of the GNN
		self.weight_decay = weight_decay  # strength of L2 regularization on GNN weights
		self.dropout = dropout  # whether to use dropout
		self.batch_norm = batch_norm  # whether to use batch norm
		self.lr = lr  # learning rate
		self.max_epochs = max_epochs  # number of epochs to train
		self.balance_loss = balance_loss  # whether to use balanced loss
		self.stochastic_loss = stochastic_loss  # whether to use stochastic or full-batch training
		self.batch_size = batch_size  # batch size (only for stochastic training)


	def fit_predict(self, adjacency_matrix, feature_matrix, K):
		"""Train the model and compute community memberships.
            Args:
                adjacency_matrix: Adjacency matrix of the graph (scipy.sparse.csr_matrix).
                feature_matrix: Feature matrix of the graph nodes (scipy.sparse.csr_matrix).
                K: Number of communities (int).

            Returns:
                memberships: Community membership probabilities (numpy.ndarray, shape (num_nodes, K)).
        """
        # set torch device: cuda vs cpu
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		torch.set_default_device(device)
		x_norm = sp.hstack([feature_matrix, adjacency_matrix])
		x_norm = _to_sparse_tensor(x_norm).to(device)
		sampler = _get_edge_sampler(adjacency_matrix, self.batch_size, self.batch_size, num_workers=2, random_seed=self.random_state)
		gnn = _GCN(x_norm.shape[1], self.hidden_sizes, K, dropout=self.dropout, batch_norm=self.batch_norm).to(device)
		adj_norm = gnn.normalize_adj(adjacency_matrix)
		decoder = _BerpoDecoder(adjacency_matrix.shape[0], adjacency_matrix.nnz, balance_loss=self.balance_loss)
		opt = torch.optim.Adam(gnn.parameters(), lr=self.lr)

		val_loss = np.inf
		validation_fn = lambda: val_loss
		early_stopping = _NoImprovementStopping(validation_fn, patience=10)
		model_saver = _ModelSaver(gnn)

		for epoch, batch in enumerate(sampler):
			if epoch > self.max_epochs:
				break
			if epoch % 25 == 0:
				with torch.no_grad():
					gnn.eval()
					# Compute validation loss
					Z = torch.nn.functional.relu(gnn(x_norm, adj_norm))
					val_loss = decoder.loss_full(Z, adjacency_matrix)
					# Check if it's time for early stopping / to save the model
					early_stopping.next_step()
					if early_stopping.should_save():
						model_saver.save()
					if early_stopping.should_stop():
						break

			# Training step
			gnn.train()
			opt.zero_grad()
			Z = torch.nn.functional.relu(gnn(x_norm, adj_norm))
			ones_idx, zeros_idx = batch
			if self.stochastic_loss:
				loss = decoder.loss_batch(Z, ones_idx, zeros_idx)
			else:
				loss = decoder.loss_full(Z, adjacency_matrix)
			loss += _l2_reg_loss(gnn, scale=self.weight_decay)
			loss.backward()
			opt.step()

		Z = torch.nn.functional.relu(gnn(x_norm, adj_norm))
		Z_min = torch.min(Z)
		Z_max = torch.max(Z)
		denominator = Z_max - Z_min + 1e-8
		Z = (Z - Z_min) / denominator
		memberships = Z.cpu().detach().numpy()
		return memberships
