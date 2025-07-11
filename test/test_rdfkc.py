import torch
from os import path
import sys


if __name__ == '__main__':
    base_dir = path.dirname(path.realpath(__file__))
    sys.path.append(base_dir[:-4])
    from soft_clustering import RDFKC
    """
    Test the RDFKC clustering on synthetic grayscale image-like data (e.g., 32x32).
    The goal is to verify the end-to-end clustering pipeline executes without errors.
    """
    torch.manual_seed(42)

    # Simulate a batch of grayscale images: 100 samples of size 1x32x32
    N = 100
    images = torch.rand((N, 1, 32, 32))  # shape: (N, C, H, W)

    # Instantiate the RDFKC clustering model
    model = RDFKC(K=5, dataset="coil20", max_iter=5)

    # Run clustering and get the predicted cluster labels
    cluster_labels = model.fit_predict(images)

    # Optional: print basic cluster stats
    unique_clusters = torch.unique(cluster_labels)
    print(f"Test passed. Found {len(unique_clusters)} unique clusters.")





