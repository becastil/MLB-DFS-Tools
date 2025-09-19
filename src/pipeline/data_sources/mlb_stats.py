from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
import requests


class MLBStatsClient:
    """Minimal client for MLB Stats API schedule and boxscores with filesystem caching."""

    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self, cache_dir: Path | str, request_pause: float = 0.2) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_pause = request_pause

    def _cache_path(self, kind: str, identifier: str) -> Path:
        return self.cache_dir / f"{kind}_{identifier}.json"

    def _get(self, endpoint: str, params: Optional[dict[str, object]] = None) -> dict:
        response = requests.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    def get_schedule(self, start_date: date, end_date: date) -> dict:
        cache_key = f"schedule_{start_date.isoformat()}_{end_date.isoformat()}"
        cache_path = self._cache_path("schedule", cache_key)
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        params = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "sportId": 1,
            "gameType": "R",  # regular season
        }
        payload = self._get("schedule", params=params)
        cache_path.write_text(json.dumps(payload))
        time.sleep(self.request_pause)
        return payload

    def iter_game_pks(self, start_date: date, end_date: date) -> Generator[int, None, None]:
        schedule = self.get_schedule(start_date, end_date)
        for date_block in schedule.get("dates", []):
            for game in date_block.get("games", []):
                game_pk = game.get("gamePk")
                if game_pk is not None:
                    yield int(game_pk)

    def get_boxscore(self, game_pk: int, force_refresh: bool = False) -> dict:
        cache_path = self._cache_path("boxscore", str(game_pk))
        if cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text())

        payload = self._get(f"game/{game_pk}/boxscore")
        cache_path.write_text(json.dumps(payload))
        time.sleep(self.request_pause)
        return payload

    def get_game_metadata(self, game_pk: int) -> dict:
        cache_path = self._cache_path("game", str(game_pk))
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        payload = self._get(f"game/{game_pk}")
        cache_path.write_text(json.dumps(payload))
        time.sleep(self.request_pause)
        return payload

    def collect_player_game_logs(
        self,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for game_pk in self.iter_game_pks(start_date, end_date):
            meta = self.get_game_metadata(game_pk)
            game_date_str = meta.get("gameData", {}).get("datetime", {}).get("dateTime")
            if game_date_str:
                game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
            else:
                game_dt = datetime.combine(start_date, datetime.min.time())

            boxscore = self.get_boxscore(game_pk, force_refresh=force_refresh)
            for home_away in ("away", "home"):
                team_block = boxscore["teams"].get(home_away, {})
                team = team_block.get("team", {}).get("abbreviation")
                opponent_block = boxscore["teams"].get("home" if home_away == "away" else "away", {})
                opponent = opponent_block.get("team", {}).get("abbreviation")
                for player_code, player in (team_block.get("players") or {}).items():
                    person = player.get("person", {})
                    stats = player.get("stats", {})
                    batting = stats.get("batting") or {}
                    pitching = stats.get("pitching") or {}
                    if not batting and not pitching:
                        continue

                    records.append(
                        {
                            "game_pk": game_pk,
                            "game_datetime": game_dt,
                            "team": team,
                            "opponent": opponent,
                            "is_home": home_away == "home",
                            "player_id": person.get("id"),
                            "player_name": person.get("fullName"),
                            "primary_position": (player.get("position") or {}).get("code"),
                            "batting_order": player.get("battingOrder"),
                            "batting_stats": batting,
                            "pitching_stats": pitching,
                            "fielding_stats": stats.get("fielding") or {},
                        }
                    )
        return pd.DataFrame(records)
