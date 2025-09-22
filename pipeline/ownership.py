from __future__ import annotations

import numpy as np
import pandas as pd


class OwnershipEstimator:
    """Heuristic ownership estimator driven by projection value, lineup spot, and betting context."""

    def __init__(self, hitter_slots: int = 8, pitcher_slots: int = 2) -> None:
        self.hitter_slots = hitter_slots
        self.pitcher_slots = pitcher_slots

    def estimate(self, projections: pd.DataFrame) -> pd.DataFrame:
        if projections.empty:
            projections["ownership"] = []
            return projections

        hitters = projections[projections["model_type"] == "hitter"].copy()
        pitchers = projections[projections["model_type"] == "pitcher"].copy()

        hitters = self._estimate_hitters(hitters)
        pitchers = self._estimate_pitchers(pitchers)

        combined = pd.concat([hitters, pitchers], axis=0, ignore_index=True)
        return combined.sort_values("player_name")

    def _estimate_hitters(self, hitters: pd.DataFrame) -> pd.DataFrame:
        if hitters.empty:
            hitters["ownership"] = []
            return hitters

        hitters = hitters.copy()
        hitters["salary"] = hitters["salary"].replace(0, np.nan)
        hitters["value_per_dollar"] = hitters["projection"] / hitters["salary"]

        hitters["value_norm"] = self._normalize(hitters["value_per_dollar"].fillna(hitters["value_per_dollar"].median()))
        hitters["projection_norm"] = self._normalize(hitters["projection"])
        hitters["vegas_norm"] = self._normalize(hitters["vegas_total"].fillna(hitters["vegas_total"].median()))

        order = hitters.get("batting_order", pd.Series(np.nan, index=hitters.index))
        hitters["order_score"] = (9 - order.fillna(6)) / 9
        hitters["order_score"] = hitters["order_score"].clip(lower=0.2, upper=1.0)

        hitters["raw_score"] = (
            0.45 * hitters["value_norm"]
            + 0.35 * hitters["projection_norm"]
            + 0.10 * hitters["order_score"]
            + 0.10 * hitters["vegas_norm"]
        )

        top_projection_mask = hitters["projection"].rank(ascending=False, method="min") <= 5
        hitters.loc[top_projection_mask, "raw_score"] += 0.15

        hitters["raw_score"] = hitters["raw_score"].clip(lower=0.05)

        total_score = hitters["raw_score"].sum()
        target_total = self.hitter_slots * 100
        if total_score == 0:
            hitters["ownership"] = target_total / len(hitters)
        else:
            hitters["ownership"] = hitters["raw_score"] / total_score * target_total
        return hitters

    def _estimate_pitchers(self, pitchers: pd.DataFrame) -> pd.DataFrame:
        if pitchers.empty:
            pitchers["ownership"] = []
            return pitchers
        pitchers = pitchers.copy()
        pitchers["salary"] = pitchers["salary"].replace(0, np.nan)
        pitchers["value_per_dollar"] = pitchers["projection"] / pitchers["salary"]

        pitchers["value_norm"] = self._normalize(pitchers["value_per_dollar"].fillna(pitchers["value_per_dollar"].median()))
        pitchers["projection_norm"] = self._normalize(pitchers["projection"])
        pitchers["win_prob"] = pitchers["vegas_line"].map(self._moneyline_to_prob).fillna(0.5)

        pitchers["raw_score"] = 0.5 * pitchers["projection_norm"] + 0.3 * pitchers["value_norm"] + 0.2 * pitchers["win_prob"]
        top_proj = pitchers["projection"].rank(ascending=False, method="min") <= 2
        pitchers.loc[top_proj, "raw_score"] += 0.2
        pitchers["raw_score"] = pitchers["raw_score"].clip(lower=0.1)

        total_score = pitchers["raw_score"].sum()
        target_total = self.pitcher_slots * 100
        if total_score == 0:
            pitchers["ownership"] = target_total / len(pitchers)
        else:
            pitchers["ownership"] = pitchers["raw_score"] / total_score * target_total
        return pitchers

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        series = series.replace([np.inf, -np.inf], np.nan)
        valid = series.dropna()
        if valid.empty:
            return pd.Series(0.5, index=series.index)
        min_val, max_val = valid.min(), valid.max()
        if np.isclose(min_val, max_val):
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    @staticmethod
    def _moneyline_to_prob(value: Optional[float]) -> Optional[float]:
        if value is None or np.isnan(value):
            return None
        line = float(value)
        if line < 0:
            return (-line) / ((-line) + 100)
        return 100 / (line + 100)
