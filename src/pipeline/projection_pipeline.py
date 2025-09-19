from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .data_sources.draftkings import DraftKingsSlateLoader
from .data_sources.fangraphs_client import FanGraphsDataFetcher
from .data_sources.mlb_stats import MLBStatsClient
from .data_sources.rotowire import RotoWireLineupFetcher
from .data_sources.statcast_client import StatcastDataFetcher  # retained for optional use
from .features import FeatureEngineer, FeatureSets
from .modeling import ProjectionModeler
from .ownership import OwnershipEstimator


@dataclass
class ProjectionResult:
    player_name: str
    team: str
    opponent: str
    positions: str
    salary: Optional[float]
    projection: float
    model_type: str
    batting_order: Optional[float]
    is_home: bool
    vegas_total: Optional[float]
    vegas_line: Optional[float]


class ProjectionPipeline:
    """End-to-end orchestration for training DFS projection models and scoring a slate."""

    def __init__(
        self,
        base_dir: Path | str,
        seasons: Iterable[int],
        chunk_size: int = 7,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.base_dir / "raw"
        self.feature_dir = self.base_dir / "features"
        self.model_dir = self.base_dir / "models"
        self.output_dir = self.base_dir / "output"
        for path in [self.raw_dir, self.feature_dir, self.model_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.seasons = sorted(set(seasons))

        self.stats_client = MLBStatsClient(self.raw_dir / "mlb_stats")
        self.fangraphs_fetcher = FanGraphsDataFetcher(self.raw_dir / "fangraphs")
        self.rotowire_fetcher = RotoWireLineupFetcher(self.raw_dir / "rotowire")
        self.statcast_fetcher = StatcastDataFetcher(self.raw_dir / "statcast", chunk_size=chunk_size)
        self.modeler = ProjectionModeler(self.model_dir)
        self.ownership_estimator = OwnershipEstimator()

        self._feature_sets: Optional[FeatureSets] = None

    # ----------------------------
    # Training data preparation
    # ----------------------------
    def build_training_corpus(self, force_refresh: bool = False) -> FeatureSets:
        if self._feature_sets and not force_refresh:
            return self._feature_sets

        start_date = date(min(self.seasons), 3, 1)
        end_date = date(max(self.seasons), 11, 30)

        game_log_path = self.raw_dir / "player_game_logs.parquet"
        if game_log_path.exists() and not force_refresh:
            player_logs = pd.read_parquet(game_log_path)
        else:
            player_logs = self.stats_client.collect_player_game_logs(start_date, end_date, force_refresh=force_refresh)
            player_logs.to_parquet(game_log_path, index=False)

        fg_hitters = self.fangraphs_fetcher.load_batting(self.seasons, force_refresh=force_refresh)
        fg_pitchers = self.fangraphs_fetcher.load_pitching(self.seasons, force_refresh=force_refresh)

        engineer = FeatureEngineer(fg_hitters, fg_pitchers)
        feature_sets = engineer.build(player_logs)

        feature_sets.hitters.to_parquet(self.feature_dir / "hitters.parquet", index=False)
        feature_sets.pitchers.to_parquet(self.feature_dir / "pitchers.parquet", index=False)

        self._feature_sets = feature_sets
        return feature_sets

    def train(self, force_refresh: bool = False) -> Dict[str, dict]:
        feature_sets = self.build_training_corpus(force_refresh=force_refresh)
        model_payload = self.modeler.train_models(feature_sets)

        metrics = {}
        for name, (pipeline, model_metrics, feature_list) in model_payload.items():
            self.modeler.save_model(name, pipeline, feature_list)
            metrics[name] = {
                "r2": model_metrics.r2,
                "rmse": model_metrics.rmse,
                "mae": model_metrics.mae,
                "features": feature_list,
            }

        (self.model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    # ----------------------------
    # Projection generation
    # ----------------------------
    def generate_projections(
        self,
        slate_csv: Path | str,
        slate_date: date,
        output_csv: Optional[Path | str] = None,
    ) -> pd.DataFrame:
        features_hitters, features_pitchers = self._load_feature_tables()
        hitter_store = self._latest_feature_store(features_hitters)
        pitcher_store = self._latest_feature_store(features_pitchers)

        models = {
            "hitters": self.modeler.load_model("hitters"),
            "pitchers": self.modeler.load_model("pitchers"),
        }

        slate = DraftKingsSlateLoader(slate_csv).load()
        lineups = self.rotowire_fetcher.fetch(slate_date)
        lineups["normalized_name"] = lineups["player_name"].map(self._normalize_name)
        lineup_lookup = lineups.set_index("normalized_name") if not lineups.empty else pd.DataFrame()

        vegas_by_team = self._build_team_context(lineups)

        projections: list[ProjectionResult] = []

        for _, player in slate.iterrows():
            normalized = player["normalized_name"]
            lineup_row = lineup_lookup.loc[normalized] if normalized in lineup_lookup.index else None
            context = self._context_from_sources(player, lineup_row, vegas_by_team, slate_date)

            if self._is_pitcher(player.get("positions")):
                feature_row = self._prepare_pitcher_features(player, context, pitcher_store, features_pitchers)
                if feature_row is None:
                    continue
                projection = self.modeler.predict(models["pitchers"], feature_row)[0]
                model_type = "pitcher"
            else:
                feature_row = self._prepare_hitter_features(player, context, hitter_store, features_hitters)
                if feature_row is None:
                    continue
                projection = self.modeler.predict(models["hitters"], feature_row)[0]
                model_type = "hitter"

            projections.append(
                ProjectionResult(
                    player_name=player["player_name"],
                    team=context["team"],
                    opponent=context["opponent"],
                    positions=player.get("positions", ""),
                    salary=float(player.get("salary")) if not pd.isna(player.get("salary")) else None,
                    projection=float(projection),
                    model_type=model_type,
                    batting_order=context.get("batting_order"),
                    is_home=context.get("is_home", False),
                    vegas_total=context.get("vegas_total"),
                    vegas_line=context.get("vegas_line"),
                )
            )

        output = pd.DataFrame([pr.__dict__ for pr in projections])
        output = self.ownership_estimator.estimate(output)
        if output_csv:
            path = Path(output_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(path, index=False)
        return output

    # ----------------------------
    # Helpers
    # ----------------------------
    def _load_feature_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        hitters_path = self.feature_dir / "hitters.parquet"
        pitchers_path = self.feature_dir / "pitchers.parquet"
        if not hitters_path.exists() or not pitchers_path.exists():
            raise FileNotFoundError("Feature tables not found. Run training first.")
        return pd.read_parquet(hitters_path), pd.read_parquet(pitchers_path)

    @staticmethod
    def _latest_feature_store(features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return features
        ordered = features.sort_values("game_datetime")
        return ordered.groupby("player_name").tail(1).set_index("player_name")

    def _prepare_hitter_features(
        self,
        dk_row: pd.Series,
        context: dict,
        feature_store: pd.DataFrame,
        feature_table: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        base_row = self._fetch_base_row(dk_row["player_name"], feature_store, feature_table)
        if base_row is None:
            return None
        updated = base_row.copy()
        updated["game_datetime"] = pd.Timestamp(context["game_datetime"])
        updated["is_home"] = int(context.get("is_home", False))
        updated["team"] = context.get("team")
        updated["opponent"] = context.get("opponent")
        updated["batting_order"] = context.get("batting_order")
        updated["vegas_total"] = context.get("vegas_total")
        updated["vegas_line"] = context.get("vegas_line")
        updated["days_since_last"] = self._days_since(base_row["game_datetime"], context["game_datetime"])
        return updated.to_frame().T

    def _prepare_pitcher_features(
        self,
        dk_row: pd.Series,
        context: dict,
        feature_store: pd.DataFrame,
        feature_table: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        base_row = self._fetch_base_row(dk_row["player_name"], feature_store, feature_table)
        if base_row is None:
            return None
        updated = base_row.copy()
        updated["game_datetime"] = pd.Timestamp(context["game_datetime"])
        updated["is_home"] = int(context.get("is_home", False))
        updated["team"] = context.get("team")
        updated["opponent"] = context.get("opponent")
        updated["vegas_total"] = context.get("vegas_total")
        updated["vegas_line"] = context.get("vegas_line")
        updated["days_since_last"] = self._days_since(base_row["game_datetime"], context["game_datetime"])
        return updated.to_frame().T

    def _fetch_base_row(
        self,
        player_name: str,
        feature_store: pd.DataFrame,
        feature_table: pd.DataFrame,
    ) -> Optional[pd.Series]:
        if player_name in feature_store.index:
            row = feature_store.loc[player_name]
            return row

        historical = feature_table[feature_table["player_name"] == player_name]
        if not historical.empty:
            return historical.sort_values("game_datetime").tail(1).iloc[0]

        columns = feature_table.columns
        empty = pd.Series({col: 0 for col in columns}, dtype="float64")
        empty["player_name"] = player_name
        empty["game_datetime"] = pd.Timestamp(self.seasons[-1], 3, 1)
        return empty

    def _context_from_sources(
        self,
        dk_row: pd.Series,
        lineup_row: Optional[pd.Series],
        vegas_by_team: dict,
        slate_date: date,
    ) -> dict:
        team = dk_row.get("team")
        game_info = dk_row.get("game_info")
        opponent, is_home = self._opponent_from_game_info(team, game_info)

        batting_order = None
        if lineup_row is not None:
            opponent = lineup_row.get("opponent", opponent)
            is_home = bool(lineup_row.get("is_home", is_home))
            batting_order = lineup_row.get("batting_order", np.nan)

        vegas_total = vegas_by_team.get((team, "total"))
        vegas_line = vegas_by_team.get((team, "line"))

        lineup_dt = datetime.combine(slate_date, datetime.min.time())
        context = {
            "team": team,
            "opponent": opponent,
            "is_home": is_home,
            "batting_order": batting_order,
            "vegas_total": vegas_total,
            "vegas_line": vegas_line,
            "game_datetime": lineup_dt,
        }
        return context

    @staticmethod
    def _opponent_from_game_info(team: Optional[str], game_info: Optional[str]) -> tuple[Optional[str], bool]:
        if not team or not isinstance(game_info, str):
            return None, False
        parts = game_info.split()
        if not parts:
            return None, False
        matchup = parts[0]
        if "@" not in matchup:
            return None, False
        away, home = matchup.split("@")
        if team == home:
            return away, True
        if team == away:
            return home, False
        return None, False

    def _build_team_context(self, lineups: pd.DataFrame) -> dict:
        context = {}
        if lineups.empty:
            return context
        for _, row in lineups.drop_duplicates(subset=["team"]).iterrows():
            context[(row["team"], "total")] = self._parse_run_total(row.get("vegas_total"))
            context[(row["team"], "line")] = self._parse_money_line(row.get("vegas_line"))
        return context

    @staticmethod
    def _parse_run_total(value: Optional[str]) -> Optional[float]:
        if not value or not isinstance(value, str):
            return None
        match = re.search(r"(-?\d+(?:\.\d+)?)", value)
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_money_line(value: Optional[str]) -> Optional[float]:
        if not value or not isinstance(value, str):
            return None
        match = re.search(r"(-?\d+(?:\.\d+)?)", value)
        return float(match.group(1)) if match else None

    @staticmethod
    def _normalize_name(value: str) -> str:
        if not isinstance(value, str):
            return ""
        cleaned = re.sub(r"[^A-Za-z0-9\s]", "", value)
        return cleaned.replace(" Jr", "").replace(" Sr", "").upper().strip()

    @staticmethod
    def _days_since(last_game: object, upcoming: datetime) -> float:
        try:
            last_dt = pd.to_datetime(last_game)
        except Exception:  # noqa: BLE001
            return 5.0
        delta = (upcoming - last_dt).days
        return float(max(0, min(delta, 30)))

    @staticmethod
    def _is_pitcher(positions: Optional[str]) -> bool:
        if not positions:
            return False
        pos_list = [p.strip() for p in str(positions).split("/")]
        return all(pos in {"P", "SP", "RP"} for pos in pos_list)
