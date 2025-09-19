from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math
import pandas as pd


@dataclass
class LineupMetrics:
    """Aggregated ownership metrics for a single lineup."""

    lineup_id: str
    players: int
    cumulative_ownership: float
    ownership_product: float
    duplicate_interval: float
    log10_ownership_product: float
    total_salary: Optional[float] = None


class LineupOwnershipAnalyzer:
    """Calculate cumulative and multiplicative ownership metrics for lineups.

    The implementation mirrors the discussion in the "The Fallacy of Cumulative
    Ownership" article, emphasising that the ownership product is a better proxy for
    lineup duplication risk than the cumulative sum of ownership percentages.
    """

    def __init__(
        self,
        lineup_col: str = "lineup_id",
        ownership_col: str = "ownership",
        salary_col: Optional[str] = None,
    ) -> None:
        self.lineup_col = lineup_col
        self.ownership_col = ownership_col
        self.salary_col = salary_col

    def analyze(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.lineup_col not in frame.columns:
            raise ValueError(f"Missing lineup identifier column `{self.lineup_col}`")
        if self.ownership_col not in frame.columns:
            raise ValueError(f"Missing ownership column `{self.ownership_col}`")

        grouped = frame.groupby(self.lineup_col)
        metrics = [self._compute_metrics(name, group) for name, group in grouped]
        result = pd.DataFrame([metric.__dict__ for metric in metrics])
        return result

    def analyze_file(self, path: Path | str) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Lineup CSV not found: {path}")
        frame = pd.read_csv(path)
        return self.analyze(frame)

    def _compute_metrics(self, lineup_id: str, group: pd.DataFrame) -> LineupMetrics:
        ownership_raw = pd.to_numeric(group[self.ownership_col], errors="coerce")
        ownership_prob = self._to_probability(ownership_raw)

        log_product = ownership_prob.map(lambda x: math.log10(max(x, 1e-12))).sum()
        product = 10 ** log_product
        duplicate_interval = math.inf if product <= 0 else 1.0 / product

        cumulative = float(ownership_raw.sum(skipna=True))
        total_salary = None
        if self.salary_col and self.salary_col in group.columns:
            total_salary = float(pd.to_numeric(group[self.salary_col], errors="coerce").sum(skipna=True))

        return LineupMetrics(
            lineup_id=str(lineup_id),
            players=len(group),
            cumulative_ownership=cumulative,
            ownership_product=product,
            duplicate_interval=duplicate_interval,
            log10_ownership_product=log_product,
            total_salary=total_salary,
        )

    @staticmethod
    def _to_probability(series: pd.Series) -> pd.Series:
        cleaned = series.fillna(0).clip(lower=0)
        if cleaned.empty:
            return cleaned
        max_val = cleaned.max()
        # If values look like percentages (>1), normalise to probabilities.
        if max_val > 1:
            return cleaned / 100
        return cleaned
