"""
Contextual Data Aggregator for MLB DFS projections.

Combines data from multiple Firecrawl extractions and provides a unified interface
for the projection pipeline with intelligent weighting and missing data handling.
"""

import logging
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .data_sources.fangraphs_enhanced import EnhancedFanGraphsClient, DFSContextualData
from .data_sources.firecrawl_client import get_firecrawl_client

logger = logging.getLogger(__name__)

@dataclass
class AggregatedFactors:
    """Container for aggregated contextual factors."""
    player_adjustments: pd.DataFrame  # Player-level projection adjustments
    ownership_modifiers: pd.DataFrame  # Ownership multipliers/adjustments
    team_environments: pd.DataFrame   # Team-level scoring environments
    market_factors: pd.DataFrame      # Betting market insights
    weather_impacts: pd.DataFrame     # Weather-based adjustments
    narrative_weights: pd.DataFrame   # Media/narrative influence factors
    reliability_scores: pd.DataFrame  # Data quality/confidence scores

class ContextualDataAggregator:
    """
    Aggregates contextual data from multiple sources to enhance DFS projections.
    
    Handles data quality assessment, factor weighting, and missing data imputation
    to provide reliable contextual adjustments for projection models.
    """
    
    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enhanced_client = EnhancedFanGraphsClient(cache_dir)
        self.firecrawl = get_firecrawl_client()
        
        # Historical factor weights (can be tuned based on backtesting)
        self.factor_weights = {
            'weather_impact': 0.15,
            'recent_form': 0.25, 
            'matchup_quality': 0.20,
            'betting_context': 0.15,
            'ballpark_factor': 0.10,
            'rest_fatigue': 0.10,
            'narrative_factor': 0.05
        }
        
        # Confidence thresholds for data quality
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
    
    def aggregate_contextual_factors(self, slate_date: Optional[date] = None) -> AggregatedFactors:
        """
        Aggregate all contextual factors for a slate into unified adjustments.
        
        Args:
            slate_date: Date for the slate (defaults to today)
            
        Returns:
            AggregatedFactors containing all contextual adjustments
        """
        if slate_date is None:
            slate_date = date.today()
        
        logger.info(f"Aggregating contextual factors for {slate_date}")
        
        # Extract base contextual data
        contextual_data = self.enhanced_client.extract_dfs_contextual_data(slate_date)
        
        # Extract ownership factors
        ownership_factors = self.enhanced_client.extract_ownership_factors(slate_date)
        
        # Extract additional market data
        market_data = self._extract_market_intelligence(slate_date)
        
        # Aggregate and weight all factors
        aggregated = self._combine_all_factors(
            contextual_data, ownership_factors, market_data, slate_date
        )
        
        return aggregated
    
    def calculate_projection_adjustments(self, base_projections: pd.DataFrame, 
                                       slate_date: Optional[date] = None) -> pd.DataFrame:
        """
        Calculate projection adjustments based on contextual factors.
        
        Args:
            base_projections: DataFrame with base projections
            slate_date: Date for contextual data
            
        Returns:
            DataFrame with adjusted projections and adjustment reasons
        """
        factors = self.aggregate_contextual_factors(slate_date)
        
        adjusted_projections = base_projections.copy()
        
        # Apply player-level adjustments
        if not factors.player_adjustments.empty:
            adjusted_projections = self._apply_player_adjustments(
                adjusted_projections, factors.player_adjustments
            )
        
        # Apply team-environment adjustments
        if not factors.team_environments.empty:
            adjusted_projections = self._apply_team_environment_adjustments(
                adjusted_projections, factors.team_environments
            )
        
        # Apply weather adjustments
        if not factors.weather_impacts.empty:
            adjusted_projections = self._apply_weather_adjustments(
                adjusted_projections, factors.weather_impacts
            )
        
        # Add reliability scores
        if not factors.reliability_scores.empty:
            adjusted_projections = self._add_reliability_scores(
                adjusted_projections, factors.reliability_scores
            )
        
        return adjusted_projections
    
    def calculate_ownership_adjustments(self, base_ownership: pd.DataFrame,
                                      slate_date: Optional[date] = None) -> pd.DataFrame:
        """
        Calculate ownership adjustments based on narrative and market factors.
        
        Args:
            base_ownership: DataFrame with base ownership estimates
            slate_date: Date for contextual data
            
        Returns:
            DataFrame with adjusted ownership estimates
        """
        factors = self.aggregate_contextual_factors(slate_date)
        
        adjusted_ownership = base_ownership.copy()
        
        # Apply ownership modifiers
        if not factors.ownership_modifiers.empty:
            adjusted_ownership = self._apply_ownership_modifiers(
                adjusted_ownership, factors.ownership_modifiers
            )
        
        # Apply narrative factors
        if not factors.narrative_weights.empty:
            adjusted_ownership = self._apply_narrative_adjustments(
                adjusted_ownership, factors.narrative_weights
            )
        
        return adjusted_ownership
    
    def _extract_market_intelligence(self, slate_date: date) -> Dict[str, Any]:
        """Extract additional market intelligence beyond standard DFS data."""
        
        market_schema = {
            "type": "object",
            "properties": {
                "sharp_money_indicators": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "game": {"type": "string"},
                            "metric": {"type": "string"},
                            "sharp_side": {"type": "string"},
                            "public_percentage": {"type": "number"},
                            "line_movement": {"type": "string"},
                            "confidence": {"type": "string"}
                        }
                    }
                },
                "injury_impact_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player": {"type": "string"},
                            "team": {"type": "string"},
                            "injury_status": {"type": "string"},
                            "replacement_impact": {"type": "string"},
                            "lineup_shuffle": {"type": "string"},
                            "dfs_opportunity": {"type": "string"}
                        }
                    }
                },
                "contrarian_opportunities": {
                    "type": "array",
                    "items": {
                        "type": "object", 
                        "properties": {
                            "player": {"type": "string"},
                            "opportunity_type": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "leverage_score": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        market_prompt = """Extract market intelligence for DFS strategy including:
        1. Sharp money movement on totals and game lines
        2. Public betting percentages vs line movement
        3. Injury situations creating value opportunities
        4. Lineup changes and their ripple effects
        5. Contrarian play opportunities based on recent bias
        6. Tournament leverage spots based on ownership expectations
        7. Weather-related market overreactions or underreactions"""
        
        urls = [
            "https://www.actionnetwork.com/mlb/sharp-report*",
            "https://www.vegasinsider.com/mlb/odds/*",
            "https://www.covers.com/mlb/*",
            "https://www.rotowire.com/daily/mlb/optimizer*"
        ]
        
        try:
            market_data = self.firecrawl.extract(
                urls=urls,
                prompt=market_prompt,
                schema=market_schema,
                enable_web_search=True
            )
            return market_data
        except Exception as e:
            logger.error(f"Error extracting market intelligence: {e}")
            return {}
    
    def _combine_all_factors(self, contextual_data: DFSContextualData,
                            ownership_factors: Dict[str, Any],
                            market_data: Dict[str, Any],
                            slate_date: date) -> AggregatedFactors:
        """Combine all factor sources into unified aggregated factors."""
        
        # Process player-level adjustments
        player_adjustments = self._process_player_adjustments(contextual_data)
        
        # Process ownership modifiers
        ownership_modifiers = self._process_ownership_modifiers(ownership_factors)
        
        # Process team environments
        team_environments = self._process_team_environments(contextual_data)
        
        # Process market factors
        market_factors = self._process_market_factors(market_data)
        
        # Process weather impacts
        weather_impacts = self._process_weather_impacts(contextual_data)
        
        # Process narrative weights
        narrative_weights = self._process_narrative_factors(ownership_factors)
        
        # Calculate reliability scores
        reliability_scores = self._calculate_reliability_scores(
            contextual_data, ownership_factors, market_data
        )
        
        return AggregatedFactors(
            player_adjustments=player_adjustments,
            ownership_modifiers=ownership_modifiers,
            team_environments=team_environments,
            market_factors=market_factors,
            weather_impacts=weather_impacts,
            narrative_weights=narrative_weights,
            reliability_scores=reliability_scores
        )
    
    def _process_player_adjustments(self, contextual_data: DFSContextualData) -> pd.DataFrame:
        """Process player-level contextual factors into projection adjustments."""
        adjustments = []
        
        # Process hitter adjustments
        for _, hitter in contextual_data.hitters.iterrows():
            if not hitter.empty:
                adjustment = self._calculate_hitter_adjustment(hitter)
                adjustments.append({
                    'player_name': hitter.get('name', ''),
                    'team': hitter.get('team', ''),
                    'position': hitter.get('position', ''),
                    'adjustment_factor': adjustment,
                    'adjustment_reasons': self._get_adjustment_reasons(hitter),
                    'confidence': self._assess_data_confidence(hitter)
                })
        
        # Process pitcher adjustments  
        for _, pitcher in contextual_data.pitchers.iterrows():
            if not pitcher.empty:
                adjustment = self._calculate_pitcher_adjustment(pitcher)
                adjustments.append({
                    'player_name': pitcher.get('name', ''),
                    'team': pitcher.get('team', ''),
                    'position': 'P',
                    'adjustment_factor': adjustment,
                    'adjustment_reasons': self._get_adjustment_reasons(pitcher),
                    'confidence': self._assess_data_confidence(pitcher)
                })
        
        return pd.DataFrame(adjustments)
    
    def _calculate_hitter_adjustment(self, hitter: pd.Series) -> float:
        """Calculate projection adjustment factor for a hitter."""
        adjustment = 1.0  # Base multiplier
        
        # Recent form adjustment
        recent_form = hitter.get('recent_form', 'average')
        if recent_form == 'hot':
            adjustment += 0.15 * self.factor_weights['recent_form']
        elif recent_form == 'cold':
            adjustment -= 0.10 * self.factor_weights['recent_form']
        
        # Matchup adjustment  
        vs_handedness = hitter.get('vs_pitcher_handedness', 'neutral')
        if vs_handedness == 'favorable':
            adjustment += 0.12 * self.factor_weights['matchup_quality']
        elif vs_handedness == 'unfavorable':
            adjustment -= 0.08 * self.factor_weights['matchup_quality']
        
        # Ballpark factor
        ballpark_factor = hitter.get('ballpark_factor', 1.0)
        if isinstance(ballpark_factor, (int, float)):
            adjustment += (ballpark_factor - 1.0) * self.factor_weights['ballpark_factor']
        
        # Weather impact
        weather_impact = hitter.get('weather_impact', 'neutral')
        if weather_impact == 'positive':
            adjustment += 0.08 * self.factor_weights['weather_impact']
        elif weather_impact == 'negative':
            adjustment -= 0.08 * self.factor_weights['weather_impact']
        
        return max(0.5, min(1.8, adjustment))  # Cap adjustments
    
    def _calculate_pitcher_adjustment(self, pitcher: pd.Series) -> float:
        """Calculate projection adjustment factor for a pitcher."""
        adjustment = 1.0  # Base multiplier
        
        # Recent form adjustment
        recent_form = pitcher.get('recent_form', 'average')
        if recent_form == 'hot':
            adjustment += 0.18 * self.factor_weights['recent_form']
        elif recent_form == 'cold':
            adjustment -= 0.15 * self.factor_weights['recent_form']
        
        # Opponent strength
        opponent_strength = pitcher.get('opponent_strength', 'average')
        if opponent_strength == 'weak':
            adjustment += 0.15 * self.factor_weights['matchup_quality']
        elif opponent_strength == 'strong':
            adjustment -= 0.12 * self.factor_weights['matchup_quality']
        
        # Bullpen support
        bullpen_support = pitcher.get('bullpen_support', 'average')
        if bullpen_support == 'strong':
            adjustment += 0.10 * self.factor_weights['matchup_quality']
        elif bullpen_support == 'weak':
            adjustment -= 0.08 * self.factor_weights['matchup_quality']
        
        # Weather impact
        weather_impact = pitcher.get('weather_impact', 'neutral')
        if weather_impact == 'positive':
            adjustment += 0.10 * self.factor_weights['weather_impact']
        elif weather_impact == 'negative':
            adjustment -= 0.10 * self.factor_weights['weather_impact']
        
        return max(0.4, min(2.0, adjustment))  # Cap adjustments
    
    def _get_adjustment_reasons(self, player: pd.Series) -> List[str]:
        """Get human-readable reasons for projection adjustments."""
        reasons = []
        
        recent_form = player.get('recent_form', '')
        if recent_form == 'hot':
            reasons.append("Hot recent form (+15 games)")
        elif recent_form == 'cold':
            reasons.append("Cold recent form (+15 games)")
        
        weather = player.get('weather_impact', '')
        if weather in ['positive', 'negative']:
            reasons.append(f"Weather impact: {weather}")
        
        ballpark = player.get('ballpark_factor', 1.0)
        if isinstance(ballpark, (int, float)) and ballpark != 1.0:
            direction = "favorable" if ballpark > 1.0 else "unfavorable"
            reasons.append(f"Ballpark factor: {direction}")
        
        return reasons
    
    def _assess_data_confidence(self, player: pd.Series) -> float:
        """Assess confidence level in the contextual data."""
        confidence_score = 1.0
        
        # Reduce confidence for missing key data
        key_fields = ['recent_form', 'ballpark_factor', 'weather_impact']
        missing_count = sum(1 for field in key_fields if pd.isna(player.get(field)))
        confidence_score -= (missing_count * 0.15)
        
        # Boost confidence for advanced metrics
        if player.get('advanced_metrics') and isinstance(player['advanced_metrics'], dict):
            confidence_score += 0.10
        
        return max(0.2, min(1.0, confidence_score))
    
    def _process_ownership_modifiers(self, ownership_factors: Dict[str, Any]) -> pd.DataFrame:
        """Process ownership factors into ownership multipliers."""
        modifiers = []
        
        # Process popular narratives
        narratives = ownership_factors.get('popular_narratives', [])
        for narrative in narratives:
            ownership_impact = narrative.get('ownership_impact', 'neutral')
            multiplier = 1.0
            
            if ownership_impact == 'increase':
                multiplier = 1.3
            elif ownership_impact == 'decrease':
                multiplier = 0.7
            elif ownership_impact == 'high_increase':
                multiplier = 1.6
            
            modifiers.append({
                'player_name': narrative.get('player', ''),
                'team': narrative.get('team', ''),
                'modifier_type': 'narrative',
                'ownership_multiplier': multiplier,
                'reason': narrative.get('narrative', '')
            })
        
        # Process leverage opportunities
        leverage_opps = ownership_factors.get('leverage_opportunities', [])
        for opp in leverage_opps:
            modifiers.append({
                'player_name': opp.get('player', ''),
                'team': '',
                'modifier_type': 'leverage',
                'ownership_multiplier': 0.8,  # Leverage plays typically lower owned
                'reason': opp.get('reason', '')
            })
        
        return pd.DataFrame(modifiers)
    
    def _process_team_environments(self, contextual_data: DFSContextualData) -> pd.DataFrame:
        """Process team-level environmental factors."""
        return contextual_data.team_stats.copy() if not contextual_data.team_stats.empty else pd.DataFrame()
    
    def _process_market_factors(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Process market intelligence factors."""
        factors = []
        
        sharp_indicators = market_data.get('sharp_money_indicators', [])
        for indicator in sharp_indicators:
            factors.append({
                'game': indicator.get('game', ''),
                'metric': indicator.get('metric', ''),
                'sharp_side': indicator.get('sharp_side', ''),
                'confidence': indicator.get('confidence', 'medium')
            })
        
        return pd.DataFrame(factors)
    
    def _process_weather_impacts(self, contextual_data: DFSContextualData) -> pd.DataFrame:
        """Process weather impact factors."""
        return contextual_data.weather_conditions.copy() if not contextual_data.weather_conditions.empty else pd.DataFrame()
    
    def _process_narrative_factors(self, ownership_factors: Dict[str, Any]) -> pd.DataFrame:
        """Process media narrative factors."""
        narratives = ownership_factors.get('popular_narratives', [])
        return pd.DataFrame(narratives) if narratives else pd.DataFrame()
    
    def _calculate_reliability_scores(self, contextual_data: DFSContextualData,
                                    ownership_factors: Dict[str, Any], 
                                    market_data: Dict[str, Any]) -> pd.DataFrame:
        """Calculate reliability scores for the extracted data."""
        scores = []
        
        # Base reliability on data completeness and freshness
        data_age_hours = (datetime.now() - contextual_data.last_updated).seconds / 3600
        freshness_score = max(0.2, 1.0 - (data_age_hours / 12))  # Decay over 12 hours
        
        # Calculate completeness scores
        hitter_completeness = self._calculate_completeness_score(contextual_data.hitters)
        pitcher_completeness = self._calculate_completeness_score(contextual_data.pitchers)
        
        scores.append({
            'data_type': 'hitters',
            'reliability_score': hitter_completeness * freshness_score,
            'freshness_score': freshness_score,
            'completeness_score': hitter_completeness
        })
        
        scores.append({
            'data_type': 'pitchers', 
            'reliability_score': pitcher_completeness * freshness_score,
            'freshness_score': freshness_score,
            'completeness_score': pitcher_completeness
        })
        
        return pd.DataFrame(scores)
    
    def _calculate_completeness_score(self, df: pd.DataFrame) -> float:
        """Calculate completeness score for a dataframe."""
        if df.empty:
            return 0.0
        
        # Key fields that should be present
        key_fields = ['name', 'team', 'recent_form']
        present_fields = sum(1 for field in key_fields if field in df.columns and not df[field].isna().all())
        
        return present_fields / len(key_fields)
    
    def _apply_player_adjustments(self, projections: pd.DataFrame, 
                                adjustments: pd.DataFrame) -> pd.DataFrame:
        """Apply player-level contextual adjustments to projections."""
        if adjustments.empty:
            return projections
        
        adjusted = projections.copy()
        
        for _, adj in adjustments.iterrows():
            mask = (adjusted['player_name'] == adj['player_name']) & (adjusted['team'] == adj['team'])
            if mask.any():
                adjusted.loc[mask, 'projection'] *= adj['adjustment_factor']
                
                # Add adjustment metadata
                reasons = ', '.join(adj.get('adjustment_reasons', []))
                adjusted.loc[mask, 'contextual_adjustments'] = reasons
                adjusted.loc[mask, 'adjustment_confidence'] = adj.get('confidence', 0.5)
        
        return adjusted
    
    def _apply_team_environment_adjustments(self, projections: pd.DataFrame,
                                          environments: pd.DataFrame) -> pd.DataFrame:
        """Apply team-level environmental adjustments."""
        if environments.empty:
            return projections
        
        adjusted = projections.copy()
        
        for _, env in environments.iterrows():
            team_mask = adjusted['team'] == env['team']
            if team_mask.any() and 'implied_runs' in env:
                # Adjust based on implied run environment
                run_multiplier = min(1.3, max(0.7, env['implied_runs'] / 5.0))  # Normalize around 5 runs
                adjusted.loc[team_mask, 'projection'] *= run_multiplier
        
        return adjusted
    
    def _apply_weather_adjustments(self, projections: pd.DataFrame,
                                 weather: pd.DataFrame) -> pd.DataFrame:
        """Apply weather-based adjustments.""" 
        # Implementation would map stadiums to teams and apply weather factors
        return projections  # Placeholder
    
    def _add_reliability_scores(self, projections: pd.DataFrame,
                              reliability: pd.DataFrame) -> pd.DataFrame:
        """Add data reliability scores to projections."""
        adjusted = projections.copy()
        
        # Add overall reliability score
        if not reliability.empty:
            avg_reliability = reliability['reliability_score'].mean()
            adjusted['data_reliability'] = avg_reliability
        
        return adjusted
    
    def _apply_ownership_modifiers(self, ownership: pd.DataFrame,
                                 modifiers: pd.DataFrame) -> pd.DataFrame:
        """Apply ownership modifiers based on contextual factors."""
        if modifiers.empty:
            return ownership
        
        adjusted = ownership.copy()
        
        for _, mod in modifiers.iterrows():
            player_mask = adjusted['player_name'] == mod['player_name']
            if player_mask.any():
                adjusted.loc[player_mask, 'ownership'] *= mod['ownership_multiplier']
                adjusted.loc[player_mask, 'ownership_reason'] = mod['reason']
        
        return adjusted
    
    def _apply_narrative_adjustments(self, ownership: pd.DataFrame,
                                   narratives: pd.DataFrame) -> pd.DataFrame:
        """Apply narrative-based ownership adjustments."""
        # Implementation would adjust ownership based on media narratives
        return ownership  # Placeholder