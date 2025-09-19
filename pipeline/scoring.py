from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


HITTER_EVENT_POINTS = {
    "singles": 3.0,
    "doubles": 5.0,
    "triples": 8.0,
    "homeRuns": 10.0,
    "rbi": 2.0,
    "runs": 2.0,
    "baseOnBalls": 2.0,
    "hitByPitch": 2.0,
    "stolenBases": 5.0,
    "caughtStealing": -2.0,
}

PITCHER_POINTS = {
    "strikeOuts": 2.0,
    "wins": 4.0,
    "earnedRuns": -2.0,
    "hits": -0.6,
    "baseOnBalls": -0.6,
    "hitBatsmen": -0.6,
    "completeGames": 2.5,
    "shutouts": 2.5,
    "saves": 5.0,
}


def innings_to_outs(innings_pitched: str | float | int | None) -> float:
    if innings_pitched is None:
        return 0.0
    if isinstance(innings_pitched, (int, float)):
        return float(innings_pitched) * 3
    if not innings_pitched:
        return 0.0
    try:
        whole, _, fraction = innings_pitched.partition(".")
        outs = int(whole) * 3
        outs += int(fraction or 0)
        return float(outs)
    except ValueError:
        return 0.0


def calculate_hitter_points(row: Mapping[str, float]) -> float:
    hits = row.get("bat_hits", 0) or 0
    doubles = row.get("bat_doubles", 0) or 0
    triples = row.get("bat_triples", 0) or 0
    home_runs = row.get("bat_homeRuns", 0) or 0
    singles = max(hits - doubles - triples - home_runs, 0)

    score = 0.0
    score += singles * HITTER_EVENT_POINTS["singles"]
    score += doubles * HITTER_EVENT_POINTS["doubles"]
    score += triples * HITTER_EVENT_POINTS["triples"]
    score += home_runs * HITTER_EVENT_POINTS["homeRuns"]
    score += (row.get("bat_rbi", 0) or 0) * HITTER_EVENT_POINTS["rbi"]
    score += (row.get("bat_runs", 0) or 0) * HITTER_EVENT_POINTS["runs"]
    score += (row.get("bat_baseOnBalls", 0) or 0) * HITTER_EVENT_POINTS["baseOnBalls"]
    score += (row.get("bat_hitByPitch", 0) or 0) * HITTER_EVENT_POINTS["hitByPitch"]
    score += (row.get("bat_stolenBases", 0) or 0) * HITTER_EVENT_POINTS["stolenBases"]
    score += (row.get("bat_caughtStealing", 0) or 0) * HITTER_EVENT_POINTS["caughtStealing"]
    return float(score)


def calculate_pitcher_points(row: Mapping[str, float]) -> float:
    outs = innings_to_outs(row.get("pit_inningsPitched"))
    score = (outs / 3.0) * 2.25

    for key, weight in PITCHER_POINTS.items():
        value = row.get(f"pit_{key}")
        if value is None:
            continue
        score += float(value) * weight

    # DraftKings quality start bonus
    outs_int = int(outs)
    earned_runs = row.get("pit_earnedRuns") or 0
    games_started = row.get("pit_gamesStarted") or 0
    if games_started >= 1 and outs_int >= 18 and earned_runs <= 3:
        score += 2.0

    # No-hitter bonus if complete game, zero hits, and zero runs
    hits_allowed = row.get("pit_hits") or 0
    if (row.get("pit_completeGames") or 0) >= 1 and hits_allowed == 0:
        score += 5.0

    return float(score)


def assign_points(frame: pd.DataFrame) -> pd.DataFrame:
    if "bat_hits" in frame.columns:
        frame["dk_points_hitter"] = frame.apply(calculate_hitter_points, axis=1)
    else:
        frame["dk_points_hitter"] = 0.0

    if "pit_inningsPitched" in frame.columns:
        frame["dk_points_pitcher"] = frame.apply(calculate_pitcher_points, axis=1)
    else:
        frame["dk_points_pitcher"] = 0.0

    return frame
