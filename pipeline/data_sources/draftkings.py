from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

_NAME_ID_PATTERN = re.compile(r"^(?P<name>.*)\s+\((?P<id>\d+)\)$")


class DraftKingsSlateLoader:
    """Parse DraftKings salaried player CSVs exported from the lineup screen."""

    def __init__(self, csv_path: Path | str) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"DraftKings CSV not found at {self.csv_path}")

    @staticmethod
    def _parse_name_id(value: str) -> tuple[str, Optional[int]]:
        match = _NAME_ID_PATTERN.match(value.strip())
        if not match:
            return value.strip(), None
        return match.group("name"), int(match.group("id"))

    @staticmethod
    def _normalize_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9\s]", "", value).replace(" Jr", "").replace(" Sr", "").strip()

    def load(self) -> pd.DataFrame:
        data = pd.read_csv(self.csv_path)
        data = data.rename(columns=str.lower)

        if "name" not in data.columns:
            raise ValueError("DraftKings CSV missing `Name` column")

        parsed = data["name"].apply(self._parse_name_id)
        data["player_name"] = parsed.apply(lambda x: x[0])
        data["dk_id"] = parsed.apply(lambda x: x[1])
        data["normalized_name"] = data["player_name"].map(self._normalize_name)

        column_map = {
            "teamabbrev": "team",
            "avgpointspergame": "avg_fpts",
            "salary": "salary",
            "position": "positions",
            "gameinfo": "game_info",
            "id": "row_id",
            "opponent": "opponent",
            "notes": "notes",
        }

        for raw, friendly in column_map.items():
            if raw in data.columns:
                data.rename(columns={raw: friendly}, inplace=True)

        numeric_cols = ["salary", "avg_fpts"]
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        return data
