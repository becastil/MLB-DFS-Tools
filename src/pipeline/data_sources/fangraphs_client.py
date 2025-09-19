from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    from pybaseball import fg_batting_data, fg_pitching_data
except ImportError as exc:  # pragma: no cover - guards optional dependency
    raise RuntimeError(
        "pybaseball is required for FanGraphs data collection. Install it with `pip install pybaseball`."
    ) from exc


class FanGraphsDataFetcher:
    """Pulls FanGraphs leaderboards for hitters and pitchers across multiple seasons."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, prefix: str, seasons: Iterable[int]) -> Path:
        season_tag = f"{min(seasons)}_{max(seasons)}"
        return self.cache_dir / f"{prefix}_{season_tag}.parquet"

    def load_batting(self, seasons: Iterable[int], force_refresh: bool = False) -> pd.DataFrame:
        seasons = list(seasons)
        cache_path = self._cache_path("fangraphs_batting", seasons)
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        start, end = min(seasons), max(seasons)
        data = fg_batting_data(start, end, stat_columns="ALL", split_seasons=True)
        data = data.rename(columns={"Name": "player_name", "Team": "team", "Season": "season"})
        data.to_parquet(cache_path, index=False)
        return data

    def load_pitching(self, seasons: Iterable[int], force_refresh: bool = False) -> pd.DataFrame:
        seasons = list(seasons)
        cache_path = self._cache_path("fangraphs_pitching", seasons)
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        start, end = min(seasons), max(seasons)
        data = fg_pitching_data(start, end, stat_columns="ALL", split_seasons=True)
        data = data.rename(columns={"Name": "player_name", "Team": "team", "Season": "season"})
        data.to_parquet(cache_path, index=False)
        return data

    def filter_active_roster(self, data: pd.DataFrame, roster: pd.Index | Optional[pd.Series]) -> pd.DataFrame:
        if roster is None:
            return data
        roster_index = roster if isinstance(roster, pd.Index) else roster.index
        roster_index = roster_index.str.lower()
        return data[data["player_name"].str.lower().isin(roster_index)]
