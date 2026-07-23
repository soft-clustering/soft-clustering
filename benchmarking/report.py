from __future__ import annotations

from pathlib import Path

import pandas as pd


class BenchmarkReport:
    """
    Export benchmark results.
    """

    def __init__(
        self,
        results: pd.DataFrame,
    ):
        self.results = results

    def to_csv(
        self,
        path: str,
    ) -> None:

        self.results.to_csv(path, index=False)

    def to_markdown(
        self,
        path: str,
    ) -> None:

        Path(path).write_text(self.results.to_markdown(index=False))

    def summary(self):

        return self.results.describe()

    def leaderboard(
        self,
        metric: str,
        ascending: bool = False,
    ):

        return self.results.sort_values(
            metric,
            ascending=ascending,
        )
