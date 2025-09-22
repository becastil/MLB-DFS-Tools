from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .scoring import assign_points


def _flatten_stats(frame: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "game_pk",
        "game_datetime",
        "team",
        "opponent",
        "is_home",
        "player_id",
        "player_name",
        "primary_position",
        "batting_order",
    ]

    others = frame[base_cols].copy()

    batting = pd.json_normalize(frame["batting_stats"])
    pitching = pd.json_normalize(frame["pitching_stats"])

    if not batting.empty:
        batting = batting.add_prefix("bat_")
    else:
        batting = pd.DataFrame(index=frame.index)

    if not pitching.empty:
        pitching = pitching.add_prefix("pit_")
    else:
        pitching = pd.DataFrame(index=frame.index)

    fielding = pd.json_normalize(frame["fielding_stats"]).add_prefix("fld_")

    combined = pd.concat([others, batting, pitching, fielding], axis=1)
    combined["batting_order"] = combined["batting_order"].apply(_parse_batting_order)
    combined["season"] = combined["game_datetime"].dt.year
    combined["player_key"] = combined["player_id"].fillna(0).astype("Int64").astype(str)
    combined["vegas_total"] = np.nan
    combined["vegas_line"] = np.nan
    combined.loc[combined["player_key"] == "0", "player_key"] = combined["player_name"].fillna("")
    combined = assign_points(combined)
    numeric_cols = combined.select_dtypes(include=["float", "int", "Int64"]).columns
    combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return combined


def _parse_batting_order(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        order = int(str(value))
        if order >= 100:
            return order // 100
        return order
    except ValueError:
        return np.nan


def _rolling_mean(df: pd.DataFrame, group_key: str, value_col: str, window: int, new_col: str) -> None:
    df.sort_values([group_key, "game_datetime"], inplace=True)
    df[new_col] = (
        df.groupby(group_key)[value_col]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )


def _rolling_sum(df: pd.DataFrame, group_key: str, value_col: str, window: int, new_col: str) -> None:
    df.sort_values([group_key, "game_datetime"], inplace=True)
    df[new_col] = (
        df.groupby(group_key)[value_col]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )


def _days_since_last(df: pd.DataFrame, group_key: str, new_col: str = "days_since_last") -> None:
    df.sort_values([group_key, "game_datetime"], inplace=True)
    df[new_col] = (
        df.groupby(group_key)["game_datetime"]
        .diff()
        .dt.days
        .fillna(7)
        .clip(lower=0, upper=30)
    )


@dataclass
class FeatureSets:
    hitters: pd.DataFrame
    pitchers: pd.DataFrame


class FeatureEngineer:
    """Transforms raw player game logs into model-ready hitter and pitcher feature tables."""

    def __init__(self, fangraphs_hitters: pd.DataFrame, fangraphs_pitchers: pd.DataFrame) -> None:
        self.fg_hitters = fangraphs_hitters.copy()
        self.fg_pitchers = fangraphs_pitchers.copy()

    def build(self, game_logs: pd.DataFrame) -> FeatureSets:
        flat = _flatten_stats(game_logs)

        hitter_df = flat[flat.get("bat_plateAppearances", 0) > 0].copy()
        pitcher_df = flat[flat.get("pit_outs", 0) > 0].copy()

        _rolling_mean(hitter_df, "player_key", "dk_points_hitter", 3, "dk_avg_3")
        _rolling_mean(hitter_df, "player_key", "dk_points_hitter", 7, "dk_avg_7")
        _rolling_mean(hitter_df, "player_key", "dk_points_hitter", 15, "dk_avg_15")
        _rolling_mean(hitter_df, "player_key", "bat_plateAppearances", 5, "pa_avg_5")
        _rolling_sum(hitter_df, "player_key", "bat_homeRuns", 15, "hr_sum_15")
        _rolling_sum(hitter_df, "player_key", "bat_stolenBases", 15, "sb_sum_15")
        _days_since_last(hitter_df, "player_key")

        pitcher_df["pit_inningsPitched_outs"] = pitcher_df["pit_inningsPitched"].apply(_string_ip_to_outs)
        _rolling_mean(pitcher_df, "player_key", "dk_points_pitcher", 3, "dk_avg_3")
        _rolling_mean(pitcher_df, "player_key", "dk_points_pitcher", 10, "dk_avg_10")
        _rolling_mean(pitcher_df, "player_key", "pit_inningsPitched_outs", 5, "outs_avg_5")
        _rolling_mean(pitcher_df, "player_key", "pit_strikeOuts", 5, "k_avg_5")
        _days_since_last(pitcher_df, "player_key")

        hitter_final = self._merge_fangraphs_hitters(hitter_df)
        pitcher_final = self._merge_fangraphs_pitchers(pitcher_df)

        return FeatureSets(hitters=hitter_final, pitchers=pitcher_final)

    def _merge_fangraphs_hitters(self, hitters: pd.DataFrame) -> pd.DataFrame:
        fg_cols = [
            "player_name",
            "season",
            "PA",
            "HR",
            "SB",
            "wOBA",
            "wRC+",
            "ISO",
            "BB%",
            "K%",
        ]
        fg = self.fg_hitters[fg_cols].drop_duplicates(subset=["player_name", "season"])
        merged = hitters.merge(fg, on=["player_name", "season"], how="left", suffixes=("", "_fg"))
        return merged

    def _merge_fangraphs_pitchers(self, pitchers: pd.DataFrame) -> pd.DataFrame:
        fg_cols = [
            "player_name",
            "season",
            "ERA",
            "WHIP",
            "K/9",
            "BB/9",
            "HR/9",
            "FIP",
            "xFIP",
            "SIERA",
        ]
        fg = self.fg_pitchers[fg_cols].drop_duplicates(subset=["player_name", "season"])
        merged = pitchers.merge(fg, on=["player_name", "season"], how="left", suffixes=("", "_fg"))
        return merged


def _string_ip_to_outs(value: object) -> float:
    from .scoring import innings_to_outs

    return innings_to_outs(value)
