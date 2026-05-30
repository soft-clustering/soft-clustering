from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from ..base import BaseBenchmark


class ClusteringQualityBenchmark(
    BaseBenchmark
):
    """
    Compute clustering quality metrics.
    """

    name = "quality"

    def evaluate(
        self,
        model: Any,
        X,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        model.fit(X)

        if hasattr(model, "predict"):
            labels = model.predict(X)

        elif hasattr(model, "labels_"):
            labels = model.labels_

        elif hasattr(model, "membership_"):

            labels = np.argmax(
                model.membership_,
                axis=1,
            )

        else:
            raise ValueError(
                "Cannot obtain cluster labels."
            )

        results = {}

        n_clusters = len(
            np.unique(labels)
        )

        if n_clusters > 1:

            results[
                "silhouette"
            ] = silhouette_score(
                X,
                labels,
            )

            results[
                "calinski_harabasz"
            ] = calinski_harabasz_score(
                X,
                labels,
            )

            results[
                "davies_bouldin"
            ] = davies_bouldin_score(
                X,
                labels,
            )

        if y is not None:

            results[
                "ari"
            ] = adjusted_rand_score(
                y,
                labels,
            )

            results[
                "nmi"
            ] = normalized_mutual_info_score(
                y,
                labels,
            )

        if hasattr(
            model,
            "membership_",
        ):

            U = model.membership_

            results[
                "partition_coefficient"
            ] = np.mean(
                np.sum(U**2, axis=1)
            )

            entropy = -np.sum(
                U * np.log(
                    U + 1e-12
                ),
                axis=1,
            )

            results[
                "partition_entropy"
            ] = np.mean(entropy)

        return results