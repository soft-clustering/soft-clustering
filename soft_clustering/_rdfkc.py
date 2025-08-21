import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typeguard import typechecked
from typing import Optional, Union


class COIL20Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=13, stride=1, padding=0)  # → (6, 20, 20)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=11, stride=1, padding=0) # → (12, 10, 10)
        self.conv3 = nn.Conv2d(12, 16, kernel_size=5, stride=1, padding=0) # → (16, 6, 6)
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
        self.deconv3 = nn.ConvTranspose2d(6, 1, kernel_size=13)   # → (1, 32, 32)
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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)   # → (6, 32, 32)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)  # → (12, 32, 32)
        self.conv3 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1) # → (16, 32, 32)
        self.fc = nn.Linear(16 * 32 * 32, 20)
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
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 16 * 32 * 32)
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
        x = x.view(-1, 16, 32, 32)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


@typechecked
class RDFKC:
    def __init__(self, 
                K: int,
                encoder: Optional[torch.nn.Module] = None,
                decoder: Optional[torch.nn.Module] = None,
                dataset: Optional[str] = None,
                random_state: Optional[int] = None,
                max_iter: int = 100,
                batch_size: Optional[int] = None,
                lr: float = 1e-4,
                mu: float = 1.0,
                gamma: float = 1e-4,
                tau: float = 0.1):
        """
        Initialize the Robust Deep Fuzzy K-Means Clustering model.

        Parameters:
            K: Number of clusters.
            encoder: Encoder network (optional).
            decoder: Decoder network (optional).
            dataset: Dataset name/path (optional).
            random_state: Random seed (optional).
            max_iter: Max iterations (default 100).
            batch_size: Batch size (optional).
            lr: Learning rate (default 1e-4).
            mu: Laplacian regularization weight.
            gamma: Weight regularization coefficient.
            tau: Robustness coefficient.
        """
        self.K = K
        self.mu = mu
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr = lr
        self.max_iter = max_iter

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        # Set encoder/decoder defaults if not provided
        if encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder
        elif dataset == "coil20":
            self.encoder = COIL20Encoder()
            self.decoder = COIL20Decoder()
        elif dataset == "fashion":
            self.encoder = FashionMNISTEncoder()
            self.decoder = FashionMNISTDecoder()
        else:
            raise ValueError("Either provide encoder/decoder or specify a known dataset.")
        
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)


    def _k_nearest_to_nth_sample(self, n: int, k: int = 10) -> list:
        distances = torch.norm(self.Z[n] - self.Z, dim=1)
        neighbors = torch.argsort(distances)[1:k+1]
        return neighbors.tolist()

    def _initialize_membership_matrix(self) -> torch.Tensor:
        """Randomly initialize and normalize the membership matrix."""
        U = torch.rand(self.N, self.K)
        return U / U.sum(dim=1, keepdim=True)

    def _initialize_cluster_centers(self) -> torch.Tensor:
        """Initialize cluster centers with standard normal distribution."""
        return torch.randn(self.K, self.Z.shape[1])

    def _initialize_similarity_matrix(self) -> torch.Tensor:
        """Construct the similarity matrix using k-nearest neighbors"""
        S = torch.zeros(self.N, self.N)
        for i in range(self.N):
            distances = torch.tensor([torch.norm(self.Z[i] - self.Z[j]) for j in range(self.N)])
            neighbors = self._k_nearest_to_nth_sample(i)
            for j in neighbors:
                S[i, j] = torch.exp(-distances[j] / 0.5)
        return (S + S.T) / 2

    def _compute_ki(self, dist: float) -> float:
        """Compute the robustness coefficient k_i for adaptive loss."""
        numerator = (1 + self.tau) * (dist + 2 * self.tau)
        denominator = 2 * (dist + self.tau) ** 2
        return numerator / denominator

    def _update_network_parameters(self) -> None:
        """Step 1: Update encoder/decoder parameters using backpropagation."""
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        loader = DataLoader(TensorDataset(self.X), batch_size=self.batch_size or 256, shuffle=False)
        all_Z, all_X_recon = [], []
        for (x_batch,) in loader:
            z_batch = self.encoder(x_batch)
            x_recon_batch = self.decoder(z_batch)
            all_Z.append(z_batch)
            all_X_recon.append(x_recon_batch)

        self.Z = torch.cat(all_Z, dim=0)
        self.X_recon = torch.cat(all_X_recon, dim=0)

        recon_loss = F.mse_loss(self.X_recon, self.X)

        cluster_loss = 0.0
        for n in range(self.N):
            for c in range(self.K):
                dist = torch.norm(self.Z[n] - self.V[c])
                k_i = self._compute_ki(dist)
                cluster_loss += self.U[n, c] ** 2 * k_i * dist ** 2

        reg_loss = 0.0
        for module in list(self.encoder.modules()) + list(self.decoder.modules()):
            if hasattr(module, "weight") and module.weight is not None:
                reg_loss += torch.norm(module.weight, p='fro') ** 2
            if hasattr(module, "bias") and module.bias is not None:
                reg_loss += torch.norm(module.bias, p=2) ** 2
        reg_loss *= self.gamma

        total_loss = recon_loss + cluster_loss + reg_loss
        total_loss.backward()
        self.optimizer.step()

        self.Z = self.Z.detach()
        self.X_recon = self.X_recon.detach()

    def _update_cluster_center_matrix(self) -> None:
        """Step 2: Update cluster centers V using the current membership and latent features."""
        d = self.Z.shape[1]
        V_new = torch.zeros((self.K, d))
        for c in range(self.K):
            numerator = torch.zeros(d)
            denominator = 0.0
            for n in range(self.N):
                dist = torch.norm(self.Z[n] - self.V[c])
                k_i = self._compute_ki(dist)
                w = self.U[n, c] ** 2 * k_i
                numerator += w * self.Z[n]
                denominator += w
            V_new[c] = numerator / (denominator + 1e-8)

        self.V = V_new

    def _update_membership_matrix(self) -> None:
        """Step 3: Update membership matrix U using closed-form solution with Lagrange multipliers."""
        U_new = torch.zeros(self.N, self.K)
        for n in range(self.N):
            neighbors = self._k_nearest_to_nth_sample(n)
            p = torch.zeros(self.K)
            q = torch.zeros(self.K)
            for c in range(self.K):
                p[c] = 2 * self.mu * sum(self.S[n, j] * self.U[j, c] for j in neighbors)
                dist = torch.norm(self.Z[n] - self.V[c]) + 1e-8
                q[c] = ((1 + self.tau) * dist ** 2) / (dist + self.tau)
                q[c] += self.mu * sum(self.S[n, j] for j in neighbors)

            lambda_n = (2 - torch.sum(p / q)) / (torch.sum(1 / q) + 1e-8)
            for c in range(self.K):
                U_new[n, c] = (p[c] + lambda_n) / (2 * q[c] + 1e-8)

        self.U = U_new

    def fit_predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray: 
        """Train the RD-FKC model and return soft cluster assignments."""
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = X.float()

        self.N = len(X)
        # Auto batch size selection
        if self.batch_size is None:
            if self.N < 1000:
                self.batch_size = 64
            elif self.N < 10000:
                self.batch_size = 256
            else:
                self.batch_size = 1000

        self.Z = self.encoder(self.X)
        self.X_recon = self.decoder(self.Z)

        self.U = self._initialize_membership_matrix()
        self.V = self._initialize_cluster_centers()
        self.S = self._initialize_similarity_matrix()

        for _ in range(self.max_iter):
            self._update_network_parameters()
            self._update_cluster_center_matrix()
            self._update_membership_matrix()

        return torch.argmax(self.U, dim=1).numpy()