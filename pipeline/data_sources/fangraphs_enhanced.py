"""
Enhanced FanGraphs client using Firecrawl for comprehensive MLB DFS data extraction.
Extends the basic FanGraphs functionality with real-time contextual data.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from .firecrawl_client import get_firecrawl_client
from .mlb_extraction_schemas import MLBExtractionSchemas, MLBExtractionPrompts

logger = logging.getLogger(__name__)

@dataclass 
class DFSContextualData:
    """Container for contextual data that impacts DFS projections."""
    hitters: pd.DataFrame
    pitchers: pd.DataFrame
    team_stats: pd.DataFrame
    weather_conditions: pd.DataFrame
    betting_lines: pd.DataFrame
    injury_updates: pd.DataFrame
    narrative_factors: pd.DataFrame
    last_updated: datetime

class EnhancedFanGraphsClient:
    """Enhanced FanGraphs client that uses Firecrawl to extract comprehensive DFS-relevant data."""
    
    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.firecrawl = get_firecrawl_client()
        self.schemas = MLBExtractionSchemas()
        self.prompts = MLBExtractionPrompts()
        
    def extract_dfs_contextual_data(self, slate_date: Optional[date] = None) -> DFSContextualData:
        """
        Extract comprehensive contextual data for DFS projections using Firecrawl.
        
        Args:
            slate_date: Date for the slate (defaults to today)
            
        Returns:
            DFSContextualData containing all relevant factors
        """
        if slate_date is None:
            slate_date = date.today()
            
        logger.info(f"Extracting DFS contextual data for {slate_date}")
        
        # Define the comprehensive extraction schema
        schema = self._build_comprehensive_dfs_schema()
        
        # Create extraction prompt
        prompt = self._build_dfs_extraction_prompt()
        
        # Define URLs to extract from (using wildcards for comprehensive coverage)
        urls = [
            "https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat*",
            "https://www.fangraphs.com/leaders.aspx?pos=all&stats=pit*", 
            "https://www.fangraphs.com/depthcharts.aspx*",
            "https://www.fangraphs.com/projections*",
            "https://www.baseball-reference.com/leagues/majors/*",
            "https://www.mlb.com/news/injuries*",
            "https://www.espn.com/mlb/stats/*"
        ]
        
        try:
            # Perform the extraction with caching
            raw_data = self.firecrawl.extract_with_cache(
                urls=urls,
                prompt=prompt,
                schema=schema,
                cache_key='dfs_contextual',
                enable_web_search=True
            )
            
            # Process and structure the extracted data
            contextual_data = self._process_extracted_data(raw_data, slate_date)
            
            # Cache the results
            self._cache_contextual_data(contextual_data, slate_date)
            
            return contextual_data
            
        except Exception as e:
            logger.error(f"Error extracting contextual data: {e}")
            # Return cached data if available
            cached_data = self._load_cached_data(slate_date)
            if cached_data:
                logger.warning("Using cached data due to extraction error")
                return cached_data
            raise
    
    def extract_ownership_factors(self, slate_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Extract factors that influence DFS ownership using Firecrawl.
        
        Args:
            slate_date: Date for the slate
            
        Returns:
            Dictionary containing ownership-influencing factors
        """
        if slate_date is None:
            slate_date = date.today()
            
        ownership_schema = {
            "type": "object",
            "properties": {
                "slate_date": {"type": "string"},
                "popular_narratives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player": {"type": "string"},
                            "team": {"type": "string"},
                            "narrative": {"type": "string"},
                            "media_mentions": {"type": "number"},
                            "sentiment": {"type": "string"},
                            "ownership_impact": {"type": "string"}
                        }
                    }
                },
                "betting_sharp_money": {
                    "type": "array", 
                    "items": {
                        "type": "object",
                        "properties": {
                            "game": {"type": "string"},
                            "sharp_side": {"type": "string"},
                            "public_percentage": {"type": "number"},
                            "line_movement": {"type": "string"},
                            "dfs_impact": {"type": "string"}
                        }
                    }
                },
                "leverage_opportunities": {
                    "type": "array",
                    "items": {
                        "type": "object", 
                        "properties": {
                            "player": {"type": "string"},
                            "position": {"type": "string"},
                            "leverage_type": {"type": "string"},
                            "reason": {"type": "string"},
                            "tournament_value": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        ownership_prompt = """Extract DFS ownership factors including:
        1. Popular media narratives around specific players
        2. Sharp money betting indicators and public vs sharp sides
        3. Leverage opportunities in tournament play
        4. Contrarian plays based on recency bias or pricing errors
        5. Weather or park factors that public may overlook
        6. Injury return situations and their ownership implications
        7. Stack popularity and correlation play opportunities"""
        
        urls = [
            "https://www.rotogrinders.com/*",
            "https://www.fantasylabs.com/*",
            "https://www.rotowire.com/daily/mlb/*",
            "https://www.actionnetwork.com/mlb/*",
            "https://www.vegasinsider.com/mlb/*"
        ]
        
        try:
            ownership_data = self.firecrawl.extract_with_cache(
                urls=urls,
                prompt=ownership_prompt,
                schema=ownership_schema,
                cache_key='ownership_factors',
                enable_web_search=True
            )
            
            return ownership_data
            
        except Exception as e:
            logger.error(f"Error extracting ownership factors: {e}")
            return {}
    
    def extract_bullpen_strength(self, teams: List[str]) -> pd.DataFrame:
        """
        Extract bullpen strength metrics for specified teams.
        
        Args:
            teams: List of team abbreviations
            
        Returns:
            DataFrame with bullpen metrics by team
        """
        bullpen_schema = {
            "type": "object",
            "properties": {
                "teams": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "team": {"type": "string"},
                            "bullpen_era": {"type": "number"},
                            "bullpen_whip": {"type": "number"},
                            "bullpen_k_per_9": {"type": "number"},
                            "bullpen_fip": {"type": "number"},
                            "late_inning_era": {"type": "number"},
                            "high_leverage_era": {"type": "number"},
                            "closer_situation": {"type": "string"},
                            "setup_depth": {"type": "string"},
                            "recent_form": {"type": "string"},
                            "injury_concerns": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        }
        
        prompt = """Extract bullpen strength and depth information including:
        1. Team bullpen ERA, WHIP, K/9, and FIP 
        2. Late-inning and high-leverage performance
        3. Closer situations and committee situations
        4. Setup man depth and reliability
        5. Recent form over last 15 games
        6. Key injuries affecting bullpen depth"""
        
        urls = [
            "https://www.fangraphs.com/leaders.aspx?pos=all&stats=rel*",
            "https://www.baseball-reference.com/teams/*",
            "https://www.mlb.com/team/*/roster*"
        ]
        
        try:
            bullpen_data = self.firecrawl.extract_with_cache(
                urls=urls,
                prompt=prompt,
                schema=bullpen_schema,
                cache_key='bullpen_analysis'
            )
            
            if 'teams' in bullpen_data:
                return pd.DataFrame(bullpen_data['teams'])
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting bullpen data: {e}")
            return pd.DataFrame()
    
    def _build_comprehensive_dfs_schema(self) -> Dict[str, Any]:
        """Build a comprehensive schema for DFS-relevant data extraction."""
        return {
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
                            "batting_order": {"type": "number"},
                            "recent_form": {"type": "string"},
                            "vs_pitcher_handedness": {"type": "string"},
                            "ballpark_factor": {"type": "number"},
                            "weather_impact": {"type": "string"},
                            "rest_days": {"type": "number"},
                            "injury_status": {"type": "string"},
                            "lineup_protection": {"type": "string"},
                            "home_away_splits": {"type": "object"},
                            "advanced_metrics": {
                                "type": "object",
                                "properties": {
                                    "wrc_plus": {"type": "number"},
                                    "xwoba": {"type": "number"}, 
                                    "hard_hit_rate": {"type": "number"},
                                    "barrel_rate": {"type": "number"},
                                    "chase_rate": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                "pitchers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "team": {"type": "string"},
                            "recent_form": {"type": "string"},
                            "opponent_strength": {"type": "string"},
                            "ballpark_factor": {"type": "number"},
                            "weather_impact": {"type": "string"},
                            "rest_days": {"type": "number"},
                            "pitch_count_expectations": {"type": "number"},
                            "bullpen_support": {"type": "string"},
                            "advanced_metrics": {
                                "type": "object",
                                "properties": {
                                    "fip": {"type": "number"},
                                    "xfip": {"type": "number"},
                                    "k_minus_bb": {"type": "number"},
                                    "csi": {"type": "number"},
                                    "whiff_rate": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                "contextual_data": {
                    "type": "object",
                    "properties": {
                        "weather_conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "stadium": {"type": "string"},
                                    "temperature": {"type": "number"},
                                    "wind_speed": {"type": "number"},
                                    "wind_direction": {"type": "string"},
                                    "humidity": {"type": "number"},
                                    "scoring_impact": {"type": "string"}
                                }
                            }
                        },
                        "betting_context": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "game": {"type": "string"},
                                    "total": {"type": "number"},
                                    "implied_runs": {"type": "object"},
                                    "line_movement": {"type": "string"},
                                    "sharp_action": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "team_level_stats": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "team": {"type": "string"},
                            "implied_runs": {"type": "number"},
                            "win_probability": {"type": "number"},
                            "stack_correlation": {"type": "number"},
                            "recent_offensive_form": {"type": "string"},
                            "bullpen_usage": {"type": "string"}
                        }
                    }
                }
            }
        }
    
    def _build_dfs_extraction_prompt(self) -> str:
        """Build a comprehensive prompt for DFS data extraction."""
        return """Extract comprehensive MLB daily fantasy sports data including:
        
        HITTERS:
        1. Current batting order positions and lineup status
        2. Recent form over last 15 games (hot/cold streaks)
        3. Performance vs LHP/RHP and home/away splits
        4. Ballpark factors and weather impact on offensive environment
        5. Advanced metrics: wRC+, xwOBA, hard-hit rate, barrel rate
        6. Lineup protection and RBI opportunities
        7. Rest/fatigue factors and injury status
        
        PITCHERS: 
        1. Recent form and pitch efficiency trends
        2. Opponent offensive strength and matchup difficulty
        3. Ballpark and weather factors affecting pitching
        4. Expected pitch count and innings projection
        5. Bullpen support quality and win probability
        6. Advanced metrics: FIP, xFIP, K-BB%, CSW, whiff rate
        
        CONTEXTUAL FACTORS:
        1. Weather conditions affecting scoring environment
        2. Betting lines, totals, and implied run environments  
        3. Line movement and sharp money indicators
        4. Team-level offensive and defensive form
        5. Bullpen usage patterns and availability
        6. Park factors and historical scoring rates
        7. Injury updates and roster changes
        
        Structure data for daily fantasy projections with focus on:
        - Ceiling/floor outcome probability
        - Stack correlation opportunities  
        - Tournament leverage and ownership factors
        - Value plays based on pricing inefficiencies
        """
    
    def _process_extracted_data(self, raw_data: Dict[str, Any], slate_date: date) -> DFSContextualData:
        """Process raw Firecrawl extraction into structured DataFrames."""
        
        # Process hitters data
        hitters_data = raw_data.get('hitters', [])
        hitters_df = pd.DataFrame(hitters_data) if hitters_data else pd.DataFrame()
        
        # Process pitchers data
        pitchers_data = raw_data.get('pitchers', [])
        pitchers_df = pd.DataFrame(pitchers_data) if pitchers_data else pd.DataFrame()
        
        # Process contextual data
        contextual = raw_data.get('contextual_data', {})
        weather_df = pd.DataFrame(contextual.get('weather_conditions', []))
        betting_df = pd.DataFrame(contextual.get('betting_context', []))
        
        # Process team-level stats
        team_stats_data = raw_data.get('team_level_stats', [])
        team_stats_df = pd.DataFrame(team_stats_data) if team_stats_data else pd.DataFrame()
        
        # Create placeholder DataFrames for other components
        injury_df = pd.DataFrame()
        narrative_df = pd.DataFrame()
        
        return DFSContextualData(
            hitters=hitters_df,
            pitchers=pitchers_df, 
            team_stats=team_stats_df,
            weather_conditions=weather_df,
            betting_lines=betting_df,
            injury_updates=injury_df,
            narrative_factors=narrative_df,
            last_updated=datetime.now()
        )
    
    def _cache_contextual_data(self, data: DFSContextualData, slate_date: date) -> None:
        """Cache contextual data to disk with expiration."""
        cache_file = self.cache_dir / f"contextual_data_{slate_date.isoformat()}.json"
        
        cache_data = {
            'hitters': data.hitters.to_dict('records') if not data.hitters.empty else [],
            'pitchers': data.pitchers.to_dict('records') if not data.pitchers.empty else [],
            'team_stats': data.team_stats.to_dict('records') if not data.team_stats.empty else [],
            'weather_conditions': data.weather_conditions.to_dict('records') if not data.weather_conditions.empty else [],
            'betting_lines': data.betting_lines.to_dict('records') if not data.betting_lines.empty else [],
            'injury_updates': data.injury_updates.to_dict('records') if not data.injury_updates.empty else [],
            'narrative_factors': data.narrative_factors.to_dict('records') if not data.narrative_factors.empty else [],
            'last_updated': data.last_updated.isoformat()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
            
        logger.info(f"Cached contextual data to {cache_file}")
    
    def _load_cached_data(self, slate_date: date) -> Optional[DFSContextualData]:
        """Load cached contextual data if available and recent."""
        cache_file = self.cache_dir / f"contextual_data_{slate_date.isoformat()}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is recent (within 4 hours)
            last_updated = datetime.fromisoformat(cache_data['last_updated'])
            if (datetime.now() - last_updated).seconds > 4 * 3600:
                return None
            
            return DFSContextualData(
                hitters=pd.DataFrame(cache_data['hitters']),
                pitchers=pd.DataFrame(cache_data['pitchers']),
                team_stats=pd.DataFrame(cache_data['team_stats']),
                weather_conditions=pd.DataFrame(cache_data['weather_conditions']),
                betting_lines=pd.DataFrame(cache_data['betting_lines']),
                injury_updates=pd.DataFrame(cache_data['injury_updates']),
                narrative_factors=pd.DataFrame(cache_data['narrative_factors']),
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return None