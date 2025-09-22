from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .features import FeatureSets


HITTER_FEATURES = [
    "dk_avg_3",
    "dk_avg_7",
    "dk_avg_15",
    "pa_avg_5",
    "hr_sum_15",
    "sb_sum_15",
    "days_since_last",
    "batting_order",
    "is_home",
    "bat_plateAppearances",
    "bat_totalBases",
    "bat_runs",
    "bat_rbi",
    "wOBA",
    "wRC+",
    "ISO",
    "BB%",
    "K%",
    "PA",
    "vegas_total",
    "vegas_line",
]

PITCHER_FEATURES = [
    "dk_avg_3",
    "dk_avg_10",
    "outs_avg_5",
    "k_avg_5",
    "days_since_last",
    "pit_inningsPitched_outs",
    "pit_strikeOuts",
    "pit_earnedRuns",
    "pit_hits",
    "pit_homeRuns",
    "pit_baseOnBalls",
    "pit_wins",
    "pit_losses",
    "is_home",
    "ERA",
    "WHIP",
    "K/9",
    "BB/9",
    "HR/9",
    "FIP",
    "xFIP",
    "SIERA",
    "vegas_total",
    "vegas_line",
]


@dataclass
class ModelMetrics:
    r2: float
    rmse: float
    mae: float


class ProjectionModeler:
    def __init__(self, model_dir: Path | str, alpha: float = 3.0) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha

    def _build_pipeline(self, feature_names: List[str]) -> Pipeline:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_pipeline, feature_names)]
        )
        model = Ridge(alpha=self.alpha)
        return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    def _fit_model(self, df: pd.DataFrame, feature_names: List[str], target_col: str) -> Tuple[Pipeline, ModelMetrics]:
        model = self._build_pipeline(feature_names)
        X = df[feature_names]
        y = df[target_col]
        model.fit(X, y)
        preds = model.predict(X)
        metrics = ModelMetrics(
            r2=float(r2_score(y, preds)),
            rmse=float(np.sqrt(mean_squared_error(y, preds))),
            mae=float(mean_absolute_error(y, preds)),
        )
        return model, metrics

    def train_models(self, feature_sets: FeatureSets) -> Dict[str, Tuple[Pipeline, ModelMetrics, List[str]]]:
        hitters = feature_sets.hitters.copy()
        pitchers = feature_sets.pitchers.copy()

        hitters["is_home"] = hitters["is_home"].astype(int)
        pitchers["is_home"] = pitchers["is_home"].astype(int)

        hitters = hitters.dropna(subset=["dk_points_hitter"])
        pitchers = pitchers.dropna(subset=["dk_points_pitcher"])

        hitter_features = [col for col in HITTER_FEATURES if col in hitters.columns]
        pitcher_features = [col for col in PITCHER_FEATURES if col in pitchers.columns]

        hitter_model, hitter_metrics = self._fit_model(hitters, hitter_features, "dk_points_hitter")
        pitcher_model, pitcher_metrics = self._fit_model(pitchers, pitcher_features, "dk_points_pitcher")

        return {
            "hitters": (hitter_model, hitter_metrics, hitter_features),
            "pitchers": (pitcher_model, pitcher_metrics, pitcher_features),
        }

    def save_model(self, name: str, pipeline: Pipeline, feature_names: Iterable[str]) -> None:
        payload = {
            "model": pipeline,
            "features": list(feature_names),
        }
        joblib.dump(payload, self.model_dir / f"{name}.joblib")

    def load_model(self, name: str) -> dict:
        path = self.model_dir / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return joblib.load(path)

    @staticmethod
    def predict(model_payload: dict, frame: pd.DataFrame) -> pd.Series:
        pipeline: Pipeline = model_payload["model"]
        features: List[str] = model_payload["features"]
        missing = [feature for feature in features if feature not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return pd.Series(pipeline.predict(frame[features]), index=frame.index)
