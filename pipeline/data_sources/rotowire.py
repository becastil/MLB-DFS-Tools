from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

_MATCHUP_ID_PATTERN = re.compile(r"-(\d+)$")


class RotoWireLineupFetcher:
    """Scrape RotoWire daily lineups, capturing starters, batting order, and context."""

    BASE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, slate_date: date) -> Path:
        return self.cache_dir / f"rotowire_lineups_{slate_date.isoformat()}.parquet"

    @staticmethod
    def _extract_game_id(href: Optional[str]) -> Optional[int]:
        if not href:
            return None
        match = _MATCHUP_ID_PATTERN.search(href)
        return int(match.group(1)) if match else None

    @staticmethod
    def _clean_text(value: Optional[str]) -> Optional[str]:
        return value.strip() if value else None

    @staticmethod
    def _select_odds_text(container: Optional[BeautifulSoup], selector: str) -> Optional[str]:
        if not container:
            return None
        span = container.find("span", class_="composite")
        if span and span.get_text(strip=True) not in {"-", "–"}:
            return span.get_text(strip=True)
        fallback = container.find("span", class_=selector)
        return fallback.get_text(strip=True) if fallback else None

    def fetch(self, slate_date: date, force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self._cache_path(slate_date)
        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        params = {"date": slate_date.isoformat()}
        response = requests.get(self.BASE_URL, params=params, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        rows: list[dict[str, object]] = []

        for box in soup.select("div.lineup__box"):
            team_abbrevs = [div.get_text(strip=True) for div in box.select("div.lineup__abbr")]
            if len(team_abbrevs) != 2:
                continue

            matchup_link = box.find("a", class_="lineup__matchup")
            game_id = self._extract_game_id(matchup_link["href"]) if matchup_link and matchup_link.has_attr("href") else None
            odds_container = box.find("div", class_="lineup__odds")
            line_text = None
            total_text = None
            if odds_container:
                odds_items = odds_container.find_all("div", class_="lineup__odds-item")
                if len(odds_items) >= 1:
                    line_text = self._select_odds_text(odds_items[0], "fanduel")
                if len(odds_items) >= 2:
                    total_text = self._select_odds_text(odds_items[1], "fanduel")

            umpire_el = box.find("div", class_="lineup__umpire")
            umpire_text = self._clean_text(umpire_el.get_text(" ", strip=True)) if umpire_el else None
            weather_el = box.find("div", class_="lineup__weather-text")
            weather_text = self._clean_text(weather_el.get_text(" ", strip=True)) if weather_el else None

            lineup_lists = box.select("ul.lineup__list")
            if len(lineup_lists) != 2:
                continue

            for team_index, lineup_list in enumerate(lineup_lists):
                team = team_abbrevs[team_index]
                opponent = team_abbrevs[1 - team_index]
                is_home = team_index == 1

                status_el = lineup_list.find("li", class_="lineup__status")
                status = "confirmed" if status_el and "is-confirmed" in status_el.get("class", []) else "projected"

                pitcher_el = lineup_list.find("li", class_="lineup__player-highlight")
                pitcher_anchor = pitcher_el.find("a") if pitcher_el else None
                pitcher_name = self._clean_text(pitcher_anchor.get_text()) if pitcher_anchor else None
                pitcher_throws_el = pitcher_el.find("span", class_="lineup__throws") if pitcher_el else None
                pitcher_throws = self._clean_text(pitcher_throws_el.get_text()) if pitcher_throws_el else None

                for order, player_el in enumerate(lineup_list.find_all("li", class_="lineup__player"), start=1):
                    name_anchor = player_el.find("a")
                    display_name = self._clean_text(name_anchor.get_text()) if name_anchor else None
                    full_name = self._clean_text(name_anchor.get("title")) if name_anchor and name_anchor.has_attr("title") else display_name
                    pos_el = player_el.find("div", class_="lineup__pos")
                    bat_el = player_el.find("span", class_="lineup__bats")

                    rows.append(
                        {
                            "game_date": slate_date,
                            "game_id": game_id,
                            "team": team,
                            "opponent": opponent,
                            "is_home": is_home,
                            "status": status,
                            "batting_order": order,
                            "player_name": full_name,
                            "display_name": display_name,
                            "position": self._clean_text(pos_el.get_text()) if pos_el else None,
                            "bat_hand": self._clean_text(bat_el.get_text()) if bat_el else None,
                            "starting_pitcher": pitcher_name,
                            "starting_pitcher_throws": pitcher_throws,
                            "vegas_line": line_text,
                            "vegas_total": total_text,
                            "umpire": umpire_text,
                            "weather": weather_text,
                        }
                    )

        data = pd.DataFrame(rows)
        data.to_parquet(cache_path, index=False)
        return data
