from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .data_sources.draftkings import DraftKingsSlateLoader

try:  # Optional Firecrawl dependency
    from .data_sources.firecrawl_client import get_firecrawl_client
except Exception:  # pragma: no cover - Firecrawl may not be installed locally
    get_firecrawl_client = None  # type: ignore

from .sample_data import data_path


LOGGER = logging.getLogger(__name__)


def _normalise_name(value: str) -> str:
    """Return a normalised, accent-free variant of a player's name."""

    stripped = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = stripped.replace(".", " ").replace("'", "").replace(" Jr", "").replace(" Sr", "")
    cleaned = cleaned.replace(" III", "").replace(" II", "")
    return " ".join(cleaned.lower().split())


def _load_json_fallback(filename: str) -> Dict[str, Any]:
    """Load static sample payloads when Firecrawl is unavailable."""

    with data_path(filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class FirecrawlWorkflowConfig:
    fangraphs_url: str = (
        "https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&qual=0&"
        "type=8&season={season}&month=0&season1={season}&ind=0"
    )
    fangraphs_pitching_url: str = (
        "https://www.fangraphs.com/leaders/major-league?pos=all&stats=pit&lg=all&qual=0&"
        "type=8&season={season}&month=0&season1={season}&ind=0"
    )
    rotowire_url: str = "https://www.rotowire.com/baseball/injuries.php"
    vegas_url: str = "https://www.scoresandodds.com/mlb"


class FirecrawlWorkflow:
    """Orchestrates the Firecrawl-powered DFS data collection described in the article."""

    def __init__(self, base_dir: Path | str, config: FirecrawlWorkflowConfig | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.base_dir / "firecrawl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or FirecrawlWorkflowConfig()

        self._client = None
        if get_firecrawl_client is not None:
            try:
                self._client = get_firecrawl_client()
                LOGGER.info("Initialised Firecrawl client for workflow pipeline")
            except Exception as exc:  # pragma: no cover - depends on local env
                LOGGER.warning("Firecrawl unavailable (%s). Sample payloads will be used.", exc)

    # ------------------------------------------------------------------
    # Firecrawl-backed fetchers (with graceful fallback)
    # ------------------------------------------------------------------
    def fetch_fangraphs_leaders(
        self,
        season: int,
        limit: int = 30,
        use_sample: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return hitter and pitcher leaderboards."""

        if self._client and not use_sample:
            schema = {
                "type": "object",
                "properties": {
                    "hitters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "team": {"type": "string"},
                                "position": {"type": "string"},
                                "woba": {"type": "number"},
                                "pa": {"type": "number"},
                                "iso": {"type": "number"},
                                "k_rate": {"type": "number"}
                            },
                            "required": ["name", "team"]
                        }
                    },
                    "pitchers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "team": {"type": "string"},
                                "k_rate": {"type": "number"},
                                "ip": {"type": "number"},
                                "era": {"type": "number"},
                                "whip": {"type": "number"}
                            },
                            "required": ["name", "team"]
                        }
                    }
                }
            }

            prompt = (
                "Return the league leaders as JSON. Include the first {limit} hitters with wOBA, PA, ISO, "
                "and strikeout rate, plus {limit} starting pitchers with K%, innings pitched, ERA, and WHIP."
            ).format(limit=limit)

            try:  # pragma: no cover - relies on external service
                urls = [
                    self.config.fangraphs_url.format(season=season),
                    self.config.fangraphs_pitching_url.format(season=season),
                ]
                result = self._client.extract(
                    urls=urls,
                    prompt=prompt,
                    schema=schema,
                    enable_web_search=False,
                )

                hitters = (result.get("hitters") or result.get("data", {}).get("hitters") or [])[:limit]
                pitchers = (result.get("pitchers") or result.get("data", {}).get("pitchers") or [])[:limit]

                if hitters or pitchers:
                    LOGGER.info("Fetched %d hitters and %d pitchers from FanGraphs via Firecrawl", len(hitters), len(pitchers))
                    return {"hitters": hitters[:limit], "pitchers": pitchers[:limit]}
            except Exception as exc:
                LOGGER.warning("FanGraphs extraction failed (%s); using sample payload", exc)

        payload = _load_json_fallback("fangraphs_leaders.json")
        return {
            "hitters": payload.get("hitters", [])[:limit],
            "pitchers": payload.get("pitchers", [])[:limit],
        }

    def fetch_rotowire_injuries(self, use_sample: bool = False) -> List[Dict[str, Any]]:
        if self._client and not use_sample:
            schema = {
                "type": "object",
                "properties": {
                    "injuries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "player_name": {"type": "string"},
                                "team": {"type": "string"},
                                "position": {"type": "string"},
                                "injury_type": {"type": "string"},
                                "status": {"type": "string"},
                                "injury_note": {"type": "string"}
                            },
                            "required": ["player_name", "team", "status"]
                        }
                    }
                }
            }

            prompt = (
                "Extract every injured MLB player with their team, position, injury type, injury list status, "
                "and a concise status note suitable for DFS updates."
            )

            try:  # pragma: no cover - remote call
                result = self._client.extract(
                    urls=[self.config.rotowire_url],
                    prompt=prompt,
                    schema=schema,
                    enable_web_search=False,
                )
                injuries = result.get("injuries") or result.get("data", {}).get("injuries") or []
                if injuries:
                    LOGGER.info("Retrieved %d injury records via Firecrawl", len(injuries))
                    return injuries
            except Exception as exc:
                LOGGER.warning("Injury extraction failed (%s); using sample payload", exc)

        payload = _load_json_fallback("rotowire_injuries.json")
        return payload.get("injuries", [])

    def fetch_vegas_totals(self, use_sample: bool = False) -> List[Dict[str, Any]]:
        if self._client and not use_sample:
            schema = {
                "type": "object",
                "properties": {
                    "games": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "team": {"type": "string"},
                                "opponent": {"type": "string"},
                                "implied_runs": {"type": "number"},
                                "total": {"type": "number"}
                            },
                            "required": ["team", "opponent", "implied_runs"]
                        }
                    }
                }
            }

            prompt = (
                "For each MLB matchup list the implied run total for both teams and the overall game total."
            )

            try:  # pragma: no cover - remote call
                result = self._client.extract(
                    urls=[self.config.vegas_url],
                    prompt=prompt,
                    schema=schema,
                    enable_web_search=False,
                )
                games = result.get("games") or result.get("data", {}).get("games") or []
                if games:
                    LOGGER.info("Retrieved implied totals for %d teams", len(games))
                    return games
            except Exception as exc:
                LOGGER.warning("Vegas extraction failed (%s); using sample payload", exc)

        payload = _load_json_fallback("vegas_totals.json")
        return payload.get("games", [])

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_lookup(records: Iterable[Dict[str, Any]], name_field: str = "name") -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for record in records:
            name = record.get(name_field)
            if not isinstance(name, str):
                continue
            mapping[_normalise_name(name)] = record
        return mapping

    @staticmethod
    def _injury_lookup(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for record in records:
            name = record.get("player_name")
            if not isinstance(name, str):
                continue
            mapping[_normalise_name(name)] = record
        return mapping

    @staticmethod
    def _team_totals_lookup(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for record in records:
            team = record.get("team")
            if isinstance(team, str):
                mapping[team.upper()] = record
        return mapping

    def build_player_dataset(
        self,
        dk_csv: Path | str,
        slate_date: date,
        leaders: Dict[str, List[Dict[str, Any]]],
        injuries: List[Dict[str, Any]],
        vegas_totals: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Merge DraftKings, FanGraphs, injury, and odds data into a single table."""

        slate = DraftKingsSlateLoader(dk_csv).load()
        slate["normalized_name"] = slate["player_name"].map(_normalise_name)

        hitter_lookup = self._build_lookup(leaders.get("hitters", []))
        pitcher_lookup = self._build_lookup(leaders.get("pitchers", []))
        injury_lookup = self._injury_lookup(injuries)
        totals_lookup = self._team_totals_lookup(vegas_totals)

        records: List[Dict[str, Any]] = []

        for _, row in slate.iterrows():
            normalized = row["normalized_name"]
            positions = str(row.get("positions", ""))
            is_pitcher = "P" in positions.split("/")

            hitter_stats = hitter_lookup.get(normalized, {})
            pitcher_stats = pitcher_lookup.get(normalized, {}) if is_pitcher else {}
            injury = injury_lookup.get(normalized, {})

            team = str(row.get("team", "")).upper() if row.get("team") else hitter_stats.get("team")
            opponent = None

            if isinstance(row.get("game_info"), str) and row["game_info"]:
                parts = row["game_info"].split()
                if len(parts) >= 1 and "@" in parts[0]:
                    away, home = parts[0].split("@")
                    if team == away.upper():
                        opponent = home.upper()
                    elif team == home.upper():
                        opponent = away.upper()

            team_total = totals_lookup.get(team or "", {}).get("implied_runs") if team else None
            opponent_total = None
            if opponent:
                opponent_total = totals_lookup.get(opponent, {}).get("implied_runs")

            record = {
                "slate_date": slate_date,
                "player_name": row.get("player_name"),
                "team": team,
                "opponent": opponent,
                "positions": positions,
                "salary": float(row.get("salary")) if not pd.isna(row.get("salary")) else None,
                "dk_id": int(row.get("dk_id")) if not pd.isna(row.get("dk_id")) else None,
                "row_id": row.get("row_id"),
                "avg_fpts": float(row.get("avg_fpts")) if not pd.isna(row.get("avg_fpts")) else None,
                "woba": hitter_stats.get("woba"),
                "plate_appearances": hitter_stats.get("pa"),
                "iso": hitter_stats.get("iso"),
                "hitter_k_rate": hitter_stats.get("k_rate"),
                "pitcher_k_rate": pitcher_stats.get("k_rate"),
                "innings_pitched": pitcher_stats.get("ip"),
                "era": pitcher_stats.get("era"),
                "whip": pitcher_stats.get("whip"),
                "injury_status": injury.get("status", "ACT"),
                "injury_type": injury.get("injury_type"),
                "injury_note": injury.get("injury_note"),
                "team_total": team_total,
                "opponent_total": opponent_total,
                "is_pitcher": is_pitcher,
            }

            records.append(record)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Projection heuristics
    # ------------------------------------------------------------------
    def compute_projections(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Attach toy projections and ownership estimates to the merged dataset."""

        players = dataset.copy()
        players["projection"] = 0.0
        players["ownership_pct"] = 0.0

        # Hitters
        hitter_mask = ~players["is_pitcher"]
        hitters = players[hitter_mask].copy()

        for idx, hitter in hitters.iterrows():
            status = hitter.get("injury_status")
            if status and status not in {"ACT", "probable", "questionable", "DTD"}:
                players.loc[idx, "projection"] = 0.0
                continue

            woba = hitter.get("woba") or 0.320
            pa = hitter.get("plate_appearances") or 4.2
            pa = max(pa, 3.5)

            team_total = hitter.get("team_total") or 4.5
            pa *= min(1.25, max(0.75, team_total / 4.5))

            projection = woba * 10 * (pa / 4.2)

            if hitter.get("avg_fpts"):
                projection = 0.7 * projection + 0.3 * float(hitter["avg_fpts"])

            players.loc[idx, "projection"] = float(projection)

        hitters = players[hitter_mask].sort_values("projection", ascending=False)
        for rank, (idx, row) in enumerate(hitters.iterrows()):
            proj = row["projection"]
            if proj <= 0:
                ownership = 0.0
            elif rank < 3:
                ownership = 20.0
            elif rank < 8:
                ownership = 12.0
            elif rank < 15:
                ownership = 8.0
            else:
                ownership = 4.0
            players.loc[idx, "ownership_pct"] = ownership

        # Pitchers
        pitcher_mask = players["is_pitcher"]
        pitchers = players[pitcher_mask].copy()

        for idx, pitcher in pitchers.iterrows():
            status = pitcher.get("injury_status")
            if status and status not in {"ACT", "probable"}:
                players.loc[idx, "projection"] = 0.0
                continue

            k_rate = pitcher.get("pitcher_k_rate") or 0.22
            innings = pitcher.get("innings_pitched") or 5.5
            innings = max(5.0, float(innings))

            strikeouts = k_rate * (innings * 3.0)
            projection = strikeouts * 2.0 + innings * 5.0

            opponent_total = pitcher.get("opponent_total")
            if opponent_total:
                projection -= float(opponent_total) * 2.0

            team_total = pitcher.get("team_total")
            if team_total and (not opponent_total or team_total > opponent_total):
                projection += 4.0

            players.loc[idx, "projection"] = float(projection)

        pitchers = players[pitcher_mask].sort_values("projection", ascending=False)
        for rank, (idx, row) in enumerate(pitchers.iterrows()):
            proj = row["projection"]
            if proj <= 0:
                ownership = 0.0
            elif rank == 0:
                ownership = 50.0
            elif rank == 1:
                ownership = 35.0
            else:
                ownership = 15.0
            players.loc[idx, "ownership_pct"] = ownership

        return players

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        slate_csv: Path | str,
        slate_date: date,
        season: int,
        limit: int = 30,
        use_sample: bool = False,
        output_path: Optional[Path | str] = None,
    ) -> pd.DataFrame:
        """Execute the full workflow and optionally serialise results to JSON."""

        leaders = self.fetch_fangraphs_leaders(season=season, limit=limit, use_sample=use_sample)
        injuries = self.fetch_rotowire_injuries(use_sample=use_sample)
        totals = self.fetch_vegas_totals(use_sample=use_sample)

        dataset = self.build_player_dataset(
            dk_csv=slate_csv,
            slate_date=slate_date,
            leaders=leaders,
            injuries=injuries,
            vegas_totals=totals,
        )
        enriched = self.compute_projections(dataset)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = enriched.to_dict(orient="records")
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
            LOGGER.info("Wrote Firecrawl workflow output for %d players to %s", len(payload), output_path)

        return enriched

