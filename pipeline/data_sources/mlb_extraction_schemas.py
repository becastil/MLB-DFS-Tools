"""
Pre-built MLB extraction schemas and templates for Firecrawl.
Ready-to-use schemas for common MLB data extraction scenarios.
"""

from typing import Dict, Any, List

class MLBExtractionSchemas:
    """Collection of pre-built schemas for MLB data extraction."""
    
    @staticmethod
    def player_stats_schema() -> Dict[str, Any]:
        """Schema for extracting individual player statistics."""
        return {
            "type": "object",
            "properties": {
                "season": {"type": "string"},
                "players": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "team": {"type": "string"},
                            "position": {"type": "string"},
                            "jersey_number": {"type": "string"},
                            "age": {"type": "number"},
                            "batting_stats": {
                                "type": "object",
                                "properties": {
                                    "games": {"type": "number"},
                                    "at_bats": {"type": "number"},
                                    "runs": {"type": "number"},
                                    "hits": {"type": "number"},
                                    "doubles": {"type": "number"},
                                    "triples": {"type": "number"},
                                    "home_runs": {"type": "number"},
                                    "rbi": {"type": "number"},
                                    "stolen_bases": {"type": "number"},
                                    "walks": {"type": "number"},
                                    "strikeouts": {"type": "number"},
                                    "batting_avg": {"type": "number"},
                                    "on_base_pct": {"type": "number"},
                                    "slugging_pct": {"type": "number"},
                                    "ops": {"type": "number"},
                                    "wrc_plus": {"type": "number"}
                                }
                            },
                            "pitching_stats": {
                                "type": "object",
                                "properties": {
                                    "games": {"type": "number"},
                                    "games_started": {"type": "number"},
                                    "innings_pitched": {"type": "number"},
                                    "wins": {"type": "number"},
                                    "losses": {"type": "number"},
                                    "saves": {"type": "number"},
                                    "era": {"type": "number"},
                                    "whip": {"type": "number"},
                                    "strikeouts": {"type": "number"},
                                    "walks": {"type": "number"},
                                    "k_per_9": {"type": "number"},
                                    "bb_per_9": {"type": "number"},
                                    "hr_per_9": {"type": "number"},
                                    "fip": {"type": "number"}
                                }
                            }
                        },
                        "required": ["name", "team", "position"]
                    }
                }
            },
            "required": ["players"]
        }
    
    @staticmethod
    def team_standings_schema() -> Dict[str, Any]:
        """Schema for extracting team standings and records."""
        return {
            "type": "object",
            "properties": {
                "season": {"type": "string"},
                "last_updated": {"type": "string"},
                "divisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "division": {"type": "string"},
                            "league": {"type": "string"},
                            "teams": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "rank": {"type": "number"},
                                        "team": {"type": "string"},
                                        "wins": {"type": "number"},
                                        "losses": {"type": "number"},
                                        "win_pct": {"type": "number"},
                                        "games_back": {"type": "string"},
                                        "wildcard_games_back": {"type": "string"},
                                        "home_record": {"type": "string"},
                                        "away_record": {"type": "string"},
                                        "last_10": {"type": "string"},
                                        "streak": {"type": "string"},
                                        "runs_scored": {"type": "number"},
                                        "runs_allowed": {"type": "number"},
                                        "run_differential": {"type": "number"},
                                        "playoff_odds": {"type": "number"}
                                    },
                                    "required": ["team", "wins", "losses", "win_pct"]
                                }
                            }
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def injury_report_schema() -> Dict[str, Any]:
        """Schema for extracting comprehensive injury reports."""
        return {
            "type": "object",
            "properties": {
                "report_date": {"type": "string"},
                "last_updated": {"type": "string"},
                "injuries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "team": {"type": "string"},
                            "position": {"type": "string"},
                            "injury_type": {"type": "string"},
                            "body_part": {"type": "string"},
                            "severity": {"type": "string"},
                            "status": {"type": "string"},
                            "il_designation": {"type": "string"},
                            "date_injured": {"type": "string"},
                            "expected_return": {"type": "string"},
                            "days_out": {"type": "number"},
                            "games_missed": {"type": "number"},
                            "replacement_player": {"type": "string"},
                            "fantasy_impact": {"type": "string"},
                            "recent_updates": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["player_name", "team", "injury_type", "status"]
                    }
                }
            }
        }
    
    @staticmethod
    def weather_conditions_schema() -> Dict[str, Any]:
        """Schema for extracting game weather and park conditions."""
        return {
            "type": "object",
            "properties": {
                "report_date": {"type": "string"},
                "games": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "game_id": {"type": "string"},
                            "away_team": {"type": "string"},
                            "home_team": {"type": "string"},
                            "stadium": {"type": "string"},
                            "game_time": {"type": "string"},
                            "weather": {
                                "type": "object",
                                "properties": {
                                    "temperature": {"type": "number"},
                                    "feels_like": {"type": "number"},
                                    "humidity": {"type": "number"},
                                    "wind_speed": {"type": "number"},
                                    "wind_direction": {"type": "string"},
                                    "wind_gust": {"type": "number"},
                                    "precipitation": {"type": "number"},
                                    "cloud_cover": {"type": "string"},
                                    "visibility": {"type": "number"},
                                    "pressure": {"type": "number"},
                                    "uv_index": {"type": "number"}
                                }
                            },
                            "park_factors": {
                                "type": "object",
                                "properties": {
                                    "elevation": {"type": "number"},
                                    "dimensions": {
                                        "type": "object",
                                        "properties": {
                                            "left_field": {"type": "number"},
                                            "center_field": {"type": "number"},
                                            "right_field": {"type": "number"},
                                            "foul_territory": {"type": "string"}
                                        }
                                    },
                                    "surface": {"type": "string"},
                                    "roof_type": {"type": "string"}
                                }
                            },
                            "impact_analysis": {
                                "type": "object",
                                "properties": {
                                    "run_environment": {"type": "string"},
                                    "wind_impact": {"type": "string"},
                                    "temperature_impact": {"type": "string"},
                                    "humidity_impact": {"type": "string"},
                                    "overall_scoring": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def dfs_slate_schema() -> Dict[str, Any]:
        """Schema for extracting DFS slate information."""
        return {
            "type": "object",
            "properties": {
                "slate_date": {"type": "string"},
                "slate_type": {"type": "string"},
                "total_games": {"type": "number"},
                "games": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "game_id": {"type": "string"},
                            "away_team": {"type": "string"},
                            "home_team": {"type": "string"},
                            "game_time": {"type": "string"},
                            "vegas_total": {"type": "number"},
                            "spread": {"type": "number"},
                            "implied_runs": {
                                "type": "object",
                                "properties": {
                                    "away_team": {"type": "number"},
                                    "home_team": {"type": "number"}
                                }
                            },
                            "weather_impact": {"type": "string"},
                            "park_factor": {"type": "number"},
                            "pitching_matchup": {
                                "type": "object",
                                "properties": {
                                    "away_pitcher": {"type": "string"},
                                    "home_pitcher": {"type": "string"},
                                    "away_handedness": {"type": "string"},
                                    "home_handedness": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "player_projections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "team": {"type": "string"},
                            "position": {"type": "string"},
                            "salary": {"type": "number"},
                            "projected_points": {"type": "number"},
                            "projected_ownership": {"type": "number"},
                            "value_score": {"type": "number"},
                            "ceiling": {"type": "number"},
                            "floor": {"type": "number"},
                            "recent_form": {"type": "string"},
                            "matchup_rating": {"type": "string"},
                            "stack_eligibility": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["name", "team", "position", "salary"]
                    }
                }
            }
        }
    
    @staticmethod
    def betting_lines_schema() -> Dict[str, Any]:
        """Schema for extracting betting lines and odds."""
        return {
            "type": "object",
            "properties": {
                "update_time": {"type": "string"},
                "games": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "game_id": {"type": "string"},
                            "matchup": {"type": "string"},
                            "game_time": {"type": "string"},
                            "status": {"type": "string"},
                            "moneyline": {
                                "type": "object",
                                "properties": {
                                    "away": {"type": "number"},
                                    "home": {"type": "number"},
                                    "movement": {"type": "string"}
                                }
                            },
                            "total": {
                                "type": "object",
                                "properties": {
                                    "line": {"type": "number"},
                                    "over_odds": {"type": "number"},
                                    "under_odds": {"type": "number"},
                                    "movement": {"type": "string"}
                                }
                            },
                            "run_line": {
                                "type": "object",
                                "properties": {
                                    "spread": {"type": "number"},
                                    "favorite_odds": {"type": "number"},
                                    "underdog_odds": {"type": "number"},
                                    "movement": {"type": "string"}
                                }
                            },
                            "first_5_innings": {
                                "type": "object",
                                "properties": {
                                    "total": {"type": "number"},
                                    "moneyline_away": {"type": "number"},
                                    "moneyline_home": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                "player_props": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player": {"type": "string"},
                            "team": {"type": "string"},
                            "prop_type": {"type": "string"},
                            "line": {"type": "number"},
                            "over_odds": {"type": "number"},
                            "under_odds": {"type": "number"}
                        }
                    }
                },
                "futures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "market": {"type": "string"},
                            "selection": {"type": "string"},
                            "odds": {"type": "number"},
                            "movement": {"type": "string"}
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def roster_moves_schema() -> Dict[str, Any]:
        """Schema for extracting roster moves and transactions."""
        return {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "transactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "team": {"type": "string"},
                            "transaction_type": {"type": "string"},
                            "player": {"type": "string"},
                            "position": {"type": "string"},
                            "details": {"type": "string"},
                            "from_level": {"type": "string"},
                            "to_level": {"type": "string"},
                            "effective_date": {"type": "string"},
                            "fantasy_impact": {"type": "string"}
                        }
                    }
                }
            }
        }

class MLBExtractionPrompts:
    """Collection of pre-built prompts for MLB data extraction."""
    
    @staticmethod
    def player_stats_prompt() -> str:
        return """Extract comprehensive MLB player statistics including:
        1. Basic player information (name, team, position, age)
        2. Complete batting statistics (avg, HR, RBI, SB, OPS, wRC+)
        3. Complete pitching statistics (ERA, WHIP, K/9, FIP)
        4. Recent performance trends and hot/cold streaks
        5. Advanced metrics and sabermetric stats"""
    
    @staticmethod
    def team_standings_prompt() -> str:
        return """Extract MLB team standings and records including:
        1. Division standings with wins, losses, win percentage
        2. Games back from division lead and wild card
        3. Home/away records and recent form (last 10 games)
        4. Run differential and pythagorean win percentage
        5. Playoff odds and postseason positioning"""
    
    @staticmethod
    def injury_report_prompt() -> str:
        return """Extract detailed MLB injury reports including:
        1. Player injury details with specific diagnosis
        2. Injury severity and expected recovery timeline
        3. IL designation (10-day, 15-day, 60-day)
        4. Likely replacement players and their impact
        5. Fantasy baseball implications and recommendations
        6. Recent updates and progress reports"""
    
    @staticmethod
    def weather_conditions_prompt() -> str:
        return """Extract comprehensive game weather and park conditions including:
        1. Temperature, humidity, wind speed and direction
        2. Precipitation chances and cloud cover
        3. Stadium characteristics (elevation, dimensions, surface)
        4. Weather impact on scoring and offensive environment
        5. Wind effects on home run potential
        6. Historical weather patterns for the venue"""
    
    @staticmethod
    def dfs_slate_prompt() -> str:
        return """Extract DFS slate information including:
        1. Game schedule with Vegas totals and implied runs
        2. Player salaries and projected points
        3. Ownership projections and leverage opportunities
        4. Weather and park factors affecting scoring
        5. Pitching matchups and handedness advantages
        6. Stack opportunities and correlation plays
        7. Value plays and tournament strategy recommendations"""
    
    @staticmethod
    def betting_lines_prompt() -> str:
        return """Extract MLB betting information including:
        1. Game lines (moneyline, run line, totals)
        2. Current odds and line movement indicators
        3. First 5 innings betting options
        4. Player prop bets and alternate lines
        5. Futures markets (World Series, awards, season wins)
        6. Sharp money movement and betting trends
        7. Best available odds across sportsbooks"""
    
    @staticmethod
    def roster_moves_prompt() -> str:
        return """Extract MLB roster moves and transactions including:
        1. Recent call-ups, send-downs, and trades
        2. IL placements and activations
        3. DFA and waiver claims
        4. Minor league assignments and promotions
        5. Fantasy impact of each transaction
        6. Expected playing time changes
        7. Ripple effects on team depth charts"""

# Common URL patterns for MLB data extraction
MLB_COMMON_URLS = {
    "baseball_reference": {
        "base": "https://www.baseball-reference.com",
        "patterns": [
            "/leagues/MLB/{year}-standard-batting.shtml",
            "/leagues/MLB/{year}-standard-pitching.shtml",
            "/teams/{team}/{year}.shtml",
            "/players/{letter}/{player_id}.shtml"
        ]
    },
    "fangraphs": {
        "base": "https://www.fangraphs.com",
        "patterns": [
            "/leaders.aspx?pos=all&stats=bat&lg=all",
            "/leaders.aspx?pos=all&stats=pit&lg=all",
            "/depthcharts.aspx?position=Team",
            "/players/{player_name}/{player_id}"
        ]
    },
    "mlb_com": {
        "base": "https://www.mlb.com",
        "patterns": [
            "/stats/",
            "/standings/",
            "/news/injuries",
            "/team/{team}/roster"
        ]
    },
    "espn": {
        "base": "https://www.espn.com/mlb",
        "patterns": [
            "/stats/",
            "/standings/",
            "/injuries/",
            "/schedule/",
            "/team/{team}"
        ]
    }
}