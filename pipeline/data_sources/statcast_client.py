from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    from pybaseball import statcast
except ImportError as exc:  # pragma: no cover - guards optional dependency
    raise RuntimeError(
        "pybaseball is required for Statcast data collection. Install it with `pip install pybaseball`."
    ) from exc


class StatcastDataFetcher:
    """Download and cache Statcast pitch-level data for subsequent feature engineering."""

    def __init__(self, cache_dir: Path | str, chunk_size: int = 7) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.chunk_size = chunk_size

    def _cache_path(self, start_date: str, end_date: str, team: Optional[str]) -> Path:
        team_tag = team or "all"
        return self.cache_dir / f"statcast_{start_date}_{end_date}_{team_tag}.parquet"

    def _iter_chunks(self, start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
        cursor = start
        delta = timedelta(days=self.chunk_size - 1)
        while cursor <= end:
            chunk_end = min(cursor + delta, end)
            yield cursor, chunk_end
            cursor = chunk_end + timedelta(days=1)

    def load(self, start_date: str, end_date: str, team: Optional[str] = None, force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self._cache_path(start_date, end_date, team)
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        frames: list[pd.DataFrame] = []

        for chunk_start, chunk_end in self._iter_chunks(start_dt, end_dt):
            chunk = statcast(
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                team=team,
                verbose=False,
                parallel=False,
            )
            if not chunk.empty:
                frames.append(chunk)

        if not frames:
            data = pd.DataFrame()
        else:
            data = pd.concat(frames, ignore_index=True)
            data = data.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

        data.to_parquet(cache_path, index=False)
        return data
