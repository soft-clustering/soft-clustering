"""
Dataset utilities for SCPP benchmarking.

This module provides access to commonly used clustering
benchmark datasets from:

1. scikit-learn
2. OpenML
3. Synthetic generators

All datasets are returned as:

    X, y

where:

    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.datasets import (
        load_iris,
        load_wine,
        load_digits,
        load_breast_cancer,
        load_diabetes,
        fetch_olivetti_faces,
        fetch_openml,
        make_blobs,
        make_moons,
        make_circles,
    )
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

    def _sklearn_missing(*args, **kwargs):
        raise ImportError(
            "scikit-learn is required for dataset loading. "
            "Install it with: pip install scikit-learn"
        )

    load_iris = load_wine = load_digits = load_breast_cancer = _sklearn_missing
    load_diabetes = fetch_olivetti_faces = fetch_openml = _sklearn_missing
    make_blobs = make_moons = make_circles = _sklearn_missing


# ============================================================
# Dataset Registry
# ============================================================

DATASET_GROUPS: Dict[str, List[str]] = {
    "real": [
        "iris",
        "wine",
        "digits",
        "breast_cancer",
        "olivetti_faces",
    ],
    "synthetic": [
        "blobs",
        "moons",
        "circles",
        "anisotropic_blobs",
        "varied_blobs",
        "high_dimensional_blobs",
    ],
    "openml": [
        "glass",
        "vehicle",
        "ecoli",
        "yeast",
        "segment",
        "satimage",
        "letter",
        "pendigits",
        "optdigits",
    ],
}


OPENML_DATASETS = {
    "glass": "glass",
    "vehicle": "vehicle",
    "ecoli": "ecoli",
    "yeast": "yeast",
    "segment": "segment",
    "satimage": "satimage",
    "letter": "letter",
    "pendigits": "pendigits",
    "optdigits": "optdigits",
}


# ============================================================
# Public API
# ============================================================

def available_datasets() -> List[str]:
    """
    Return all available datasets.
    """
    names = []

    for datasets in DATASET_GROUPS.values():
        names.extend(datasets)

    return sorted(set(names))


def available_groups() -> List[str]:
    """
    Return available dataset groups.
    """
    return sorted(DATASET_GROUPS.keys())


def datasets_in_group(group: str) -> List[str]:
    """
    Return datasets belonging to a group.
    """
    if group not in DATASET_GROUPS:
        raise ValueError(
            f"Unknown group '{group}'. "
            f"Available groups: {available_groups()}"
        )

    return DATASET_GROUPS[group]


# ============================================================
# Main Loader
# ============================================================

def get_dataset(
    name: str,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a benchmark dataset.

    Parameters
    ----------
    name : str
        Dataset name.

    random_state : int, default=42
        Random seed used for synthetic datasets.

    Returns
    -------
    X : ndarray

    y : ndarray
    """

    name = name.lower()

    # ========================================================
    # Real datasets
    # ========================================================

    if name == "iris":
        return load_iris(return_X_y=True)

    if name == "wine":
        return load_wine(return_X_y=True)

    if name == "digits":
        return load_digits(return_X_y=True)

    if name == "breast_cancer":
        return load_breast_cancer(return_X_y=True)

    if name == "olivetti_faces":

        dataset = fetch_olivetti_faces()

        return (
            dataset.data.astype(np.float64),
            dataset.target,
        )

    # ========================================================
    # Synthetic datasets
    # ========================================================

    if name == "blobs":

        return make_blobs(
            n_samples=1000,
            centers=5,
            cluster_std=1.0,
            random_state=random_state,
        )

    if name == "moons":

        return make_moons(
            n_samples=1000,
            noise=0.05,
            random_state=random_state,
        )

    if name == "circles":

        return make_circles(
            n_samples=1000,
            factor=0.5,
            noise=0.05,
            random_state=random_state,
        )

    if name == "varied_blobs":

        return make_blobs(
            n_samples=1500,
            centers=4,
            cluster_std=[
                1.0,
                2.5,
                0.5,
                3.0,
            ],
            random_state=random_state,
        )

    if name == "high_dimensional_blobs":

        return make_blobs(
            n_samples=3000,
            centers=10,
            n_features=100,
            random_state=random_state,
        )

    if name == "anisotropic_blobs":

        X, y = make_blobs(
            n_samples=1500,
            centers=4,
            random_state=random_state,
        )

        transformation = np.array(
            [
                [0.6, -0.6],
                [-0.4, 0.8],
            ]
        )

        X = X @ transformation

        return X, y

    # ========================================================
    # OpenML datasets
    # ========================================================

    if name in OPENML_DATASETS:

        dataset = fetch_openml(
            OPENML_DATASETS[name],
            version=1,
            as_frame=False,
        )

        X = dataset.data
        y = dataset.target

        if y is not None:
            y = np.asarray(y)

        return X, y

    raise ValueError(
        f"Unknown dataset '{name}'. "
        f"Available datasets: {available_datasets()}"
    )


# ============================================================
# Dataset Metadata
# ============================================================

def dataset_info(name: str) -> Dict:
    """
    Return metadata for a dataset.
    """

    X, y = get_dataset(name)

    return {
        "name": name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": (
            len(np.unique(y))
            if y is not None
            else None
        ),
    }


def benchmark_suite(
    group: str = "real",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a full benchmark suite.

    Example
    -------
    >>> datasets = benchmark_suite("real")
    >>> for name, (X, y) in datasets.items():
    >>>     ...
    """

    datasets = {}

    for name in datasets_in_group(group):
        datasets[name] = get_dataset(name)

    return datasets


# ============================================================
# CLI Helper
# ============================================================

if __name__ == "__main__":

    print("Available datasets:")
    print()

    for dataset in available_datasets():

        info = dataset_info(dataset)

        print(
            f"{dataset:25s}"
            f" samples={info['n_samples']:6d}"
            f" features={info['n_features']:4d}"
            f" classes={info['n_classes']}"
        )
