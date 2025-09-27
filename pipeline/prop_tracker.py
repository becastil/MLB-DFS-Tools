"""
Real-time prop tracking and line movement monitoring.
Monitors Underdog, PrizePicks, and sportsbook lines for changes and +EV opportunities.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from .data_sources.firecrawl_client import get_firecrawl_client
except Exception:
    get_firecrawl_client = None

logger = logging.getLogger(__name__)


class Platform(Enum):
    UNDERDOG = "underdog"
    PRIZEPICKS = "prizepicks"
    SPORTSBOOK = "sportsbook"


class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PropLine:
    """Represents a single prop bet line."""
    player_name: str
    team: str
    stat_type: str
    line: float
    platform: Platform
    odds: Optional[float] = None
    multiplier: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class LineMovement:
    """Represents line movement for a prop."""
    player_name: str
    stat_type: str
    platform: Platform
    old_line: float
    new_line: float
    old_odds: Optional[float]
    new_odds: Optional[float]
    movement_size: float
    timestamp: datetime
    
    @property
    def movement_direction(self) -> str:
        if self.new_line > self.old_line:
            return "up"
        elif self.new_line < self.old_line:
            return "down"
        return "unchanged"


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity across platforms."""
    player_name: str
    stat_type: str
    platform_a: Platform
    platform_b: Platform
    line_a: float
    line_b: float
    odds_a: float
    odds_b: float
    profit_margin: float
    confidence_level: str
    timestamp: datetime


@dataclass
class PropAlert:
    """Represents an alert for prop betting opportunities."""
    alert_id: str
    alert_type: str
    level: AlertLevel
    player_name: str
    stat_type: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    processed: bool = False


class PropTracker:
    """
    Real-time prop tracking and monitoring system.
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.cwd() / '.prop_tracker_cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        self.firecrawl_client = None
        if get_firecrawl_client:
            try:
                self.firecrawl_client = get_firecrawl_client()
            except Exception as e:
                logger.warning(f"Firecrawl client not available: {e}")
        
        # Configuration
        self.tracking_config = {
            'update_interval_minutes': 5,
            'line_movement_threshold': 0.5,  # Minimum line movement to trigger alert
            'arbitrage_profit_threshold': 2.0,  # Minimum profit % for arbitrage alert
            'max_prop_age_hours': 24,  # Remove props older than this
        }
        
        # Storage
        self.current_props: Dict[str, PropLine] = {}
        self.line_history: List[LineMovement] = []
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.alerts: List[PropAlert] = []
        
        # Platform URLs - these would be configurable
        self.platform_urls = {
            Platform.UNDERDOG: [
                "https://underdogfantasy.com/picks/mlb",
                "https://underdogfantasy.com/pick-em/mlb"
            ],
            Platform.PRIZEPICKS: [
                "https://app.prizepicks.com/board/MLB",
                "https://app.prizepicks.com/board"
            ]
        }
    
    def _generate_prop_key(self, player_name: str, stat_type: str, platform: Platform) -> str:
        """Generate unique key for a prop."""
        return f"{player_name.lower()}_{stat_type.lower()}_{platform.value}"
    
    async def fetch_platform_props(self, platform: Platform) -> List[PropLine]:
        """Fetch current props from a platform."""
        if not self.firecrawl_client:
            logger.warning(f"Firecrawl client not available for {platform.value}")
            return []
        
        urls = self.platform_urls.get(platform, [])
        if not urls:
            return []
        
        try:
            if platform == Platform.UNDERDOG:
                result = self.firecrawl_client.extract_underdog_props(urls)
                return self._parse_underdog_props(result)
            elif platform == Platform.PRIZEPICKS:
                result = self.firecrawl_client.extract_prizepicks_props(urls)
                return self._parse_prizepicks_props(result)
            else:
                logger.warning(f"Unsupported platform: {platform}")
                return []
        except Exception as e:
            logger.error(f"Failed to fetch props from {platform.value}: {e}")
            return []
    
    def _parse_underdog_props(self, extraction_result: Dict[str, Any]) -> List[PropLine]:
        """Parse Underdog extraction results into PropLine objects."""
        props = []
        data = extraction_result.get('data', {})
        prop_list = data.get('props', [])
        
        for prop in prop_list:
            try:
                line = PropLine(
                    player_name=prop.get('player_name', ''),
                    team=prop.get('team', ''),
                    stat_type=prop.get('stat_type', ''),
                    line=float(prop.get('line', 0)),
                    platform=Platform.UNDERDOG,
                    multiplier=prop.get('over_multiplier') or prop.get('higher_multiplier'),
                    timestamp=datetime.now()
                )
                props.append(line)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse Underdog prop: {prop}, error: {e}")
        
        return props
    
    def _parse_prizepicks_props(self, extraction_result: Dict[str, Any]) -> List[PropLine]:
        """Parse PrizePicks extraction results into PropLine objects."""
        props = []
        data = extraction_result.get('data', {})
        prop_list = data.get('props', [])
        
        for prop in prop_list:
            try:
                line = PropLine(
                    player_name=prop.get('player_name', ''),
                    team=prop.get('team', ''),
                    stat_type=prop.get('stat_type', ''),
                    line=float(prop.get('line', 0)),
                    platform=Platform.PRIZEPICKS,
                    odds=prop.get('odds'),
                    multiplier=prop.get('payout_multiplier'),
                    timestamp=datetime.now()
                )
                props.append(line)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse PrizePicks prop: {prop}, error: {e}")
        
        return props
    
    def update_props(self, new_props: List[PropLine]) -> List[LineMovement]:
        """Update current props and detect line movements."""
        movements = []
        
        for prop in new_props:
            prop_key = self._generate_prop_key(prop.player_name, prop.stat_type, prop.platform)
            
            if prop_key in self.current_props:
                old_prop = self.current_props[prop_key]
                
                # Check for line movement
                line_diff = abs(prop.line - old_prop.line)
                odds_diff = None
                if prop.odds and old_prop.odds:
                    odds_diff = abs(prop.odds - old_prop.odds)
                
                if line_diff >= self.tracking_config['line_movement_threshold']:
                    movement = LineMovement(
                        player_name=prop.player_name,
                        stat_type=prop.stat_type,
                        platform=prop.platform,
                        old_line=old_prop.line,
                        new_line=prop.line,
                        old_odds=old_prop.odds,
                        new_odds=prop.odds,
                        movement_size=line_diff,
                        timestamp=datetime.now()
                    )
                    movements.append(movement)
                    self.line_history.append(movement)
                    
                    # Generate alert for significant movement
                    self._generate_line_movement_alert(movement)
            
            # Update current prop
            self.current_props[prop_key] = prop
        
        return movements
    
    def detect_arbitrage_opportunities(self) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across platforms."""
        opportunities = []
        
        # Group props by player and stat type
        grouped_props: Dict[Tuple[str, str], List[PropLine]] = {}
        for prop in self.current_props.values():
            key = (prop.player_name.lower(), prop.stat_type.lower())
            if key not in grouped_props:
                grouped_props[key] = []
            grouped_props[key].append(prop)
        
        # Look for arbitrage opportunities
        for (player, stat_type), props in grouped_props.items():
            if len(props) < 2:
                continue
            
            for i, prop_a in enumerate(props):
                for prop_b in props[i+1:]:
                    if prop_a.platform == prop_b.platform:
                        continue
                    
                    # Simple arbitrage detection (this could be more sophisticated)
                    if prop_a.odds and prop_b.odds and prop_a.line != prop_b.line:
                        # Calculate potential profit margin
                        profit_margin = self._calculate_arbitrage_profit(prop_a, prop_b)
                        
                        if profit_margin >= self.tracking_config['arbitrage_profit_threshold']:
                            opportunity = ArbitrageOpportunity(
                                player_name=prop_a.player_name,
                                stat_type=prop_a.stat_type,
                                platform_a=prop_a.platform,
                                platform_b=prop_b.platform,
                                line_a=prop_a.line,
                                line_b=prop_b.line,
                                odds_a=prop_a.odds,
                                odds_b=prop_b.odds,
                                profit_margin=profit_margin,
                                confidence_level="medium",
                                timestamp=datetime.now()
                            )
                            opportunities.append(opportunity)
                            
                            # Generate alert
                            self._generate_arbitrage_alert(opportunity)
        
        self.arbitrage_opportunities.extend(opportunities)
        return opportunities
    
    def _calculate_arbitrage_profit(self, prop_a: PropLine, prop_b: PropLine) -> float:
        """Calculate potential profit margin for arbitrage opportunity."""
        # Simplified calculation - real implementation would be more complex
        if not prop_a.odds or not prop_b.odds:
            return 0.0
        
        # Convert odds to probabilities and calculate profit
        prob_a = 1 / prop_a.odds if prop_a.odds > 0 else 0
        prob_b = 1 / prop_b.odds if prop_b.odds > 0 else 0
        
        total_prob = prob_a + prob_b
        if total_prob < 1.0:
            return (1.0 - total_prob) * 100  # Profit margin as percentage
        
        return 0.0
    
    def _generate_line_movement_alert(self, movement: LineMovement):
        """Generate alert for line movement."""
        level = AlertLevel.LOW
        if movement.movement_size >= 1.0:
            level = AlertLevel.MEDIUM
        if movement.movement_size >= 2.0:
            level = AlertLevel.HIGH
        
        alert = PropAlert(
            alert_id=f"movement_{datetime.now().timestamp()}",
            alert_type="line_movement",
            level=level,
            player_name=movement.player_name,
            stat_type=movement.stat_type,
            message=f"{movement.player_name} {movement.stat_type} moved {movement.movement_direction} by {movement.movement_size} on {movement.platform.value}",
            data=asdict(movement),
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        logger.info(f"Generated line movement alert: {alert.message}")
    
    def _generate_arbitrage_alert(self, opportunity: ArbitrageOpportunity):
        """Generate alert for arbitrage opportunity."""
        alert = PropAlert(
            alert_id=f"arbitrage_{datetime.now().timestamp()}",
            alert_type="arbitrage",
            level=AlertLevel.HIGH,
            player_name=opportunity.player_name,
            stat_type=opportunity.stat_type,
            message=f"Arbitrage opportunity: {opportunity.player_name} {opportunity.stat_type} - {opportunity.profit_margin:.1f}% profit",
            data=asdict(opportunity),
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        logger.info(f"Generated arbitrage alert: {alert.message}")
    
    async def run_tracking_cycle(self):
        """Run a single tracking cycle."""
        logger.info("Starting prop tracking cycle")
        
        # Fetch props from all platforms
        all_new_props = []
        for platform in [Platform.UNDERDOG, Platform.PRIZEPICKS]:
            props = await self.fetch_platform_props(platform)
            all_new_props.extend(props)
            logger.info(f"Fetched {len(props)} props from {platform.value}")
        
        # Update props and detect movements
        movements = self.update_props(all_new_props)
        logger.info(f"Detected {len(movements)} line movements")
        
        # Detect arbitrage opportunities
        opportunities = self.detect_arbitrage_opportunities()
        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        
        # Clean up old data
        self._cleanup_old_data()
        
        # Save state
        self._save_tracking_state()
        
        return {
            'props_fetched': len(all_new_props),
            'movements_detected': len(movements),
            'arbitrage_opportunities': len(opportunities),
            'total_alerts': len([a for a in self.alerts if not a.processed])
        }
    
    async def start_continuous_tracking(self, interval_minutes: int = None):
        """Start continuous prop tracking."""
        interval = interval_minutes or self.tracking_config['update_interval_minutes']
        logger.info(f"Starting continuous prop tracking with {interval} minute intervals")
        
        while True:
            try:
                await self.run_tracking_cycle()
                await asyncio.sleep(interval * 60)
            except KeyboardInterrupt:
                logger.info("Stopping prop tracking")
                break
            except Exception as e:
                logger.error(f"Error in tracking cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def _cleanup_old_data(self):
        """Clean up old props and data."""
        cutoff_time = datetime.now() - timedelta(hours=self.tracking_config['max_prop_age_hours'])
        
        # Remove old props
        old_keys = [
            key for key, prop in self.current_props.items()
            if prop.timestamp < cutoff_time
        ]
        for key in old_keys:
            del self.current_props[key]
        
        # Limit line history size
        if len(self.line_history) > 1000:
            self.line_history = self.line_history[-500:]
        
        # Limit arbitrage opportunities
        if len(self.arbitrage_opportunities) > 500:
            self.arbitrage_opportunities = self.arbitrage_opportunities[-250:]
    
    def _save_tracking_state(self):
        """Save current tracking state to disk."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'props': {key: asdict(prop) for key, prop in self.current_props.items()},
            'recent_movements': [asdict(m) for m in self.line_history[-50:]],
            'recent_opportunities': [asdict(o) for o in self.arbitrage_opportunities[-20:]],
            'unprocessed_alerts': [asdict(a) for a in self.alerts if not a.processed]
        }
        
        state_file = self.cache_dir / 'tracking_state.json'
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of current tracking state."""
        return {
            'active_props': len(self.current_props),
            'platforms_tracked': len(set(prop.platform for prop in self.current_props.values())),
            'recent_movements': len([m for m in self.line_history if m.timestamp > datetime.now() - timedelta(hours=1)]),
            'active_opportunities': len([o for o in self.arbitrage_opportunities if o.timestamp > datetime.now() - timedelta(hours=1)]),
            'unprocessed_alerts': len([a for a in self.alerts if not a.processed]),
            'last_update': max([prop.timestamp for prop in self.current_props.values()]) if self.current_props else None
        }
    
    def get_recent_alerts(self, limit: int = 20) -> List[PropAlert]:
        """Get recent unprocessed alerts."""
        unprocessed = [a for a in self.alerts if not a.processed]
        return sorted(unprocessed, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def mark_alert_processed(self, alert_id: str):
        """Mark an alert as processed."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.processed = True
                break


async def main():
    """Test the prop tracker."""
    tracker = PropTracker()
    
    # Run a single cycle
    result = await tracker.run_tracking_cycle()
    print("Tracking cycle results:", result)
    
    # Print summary
    summary = tracker.get_tracking_summary()
    print("Tracking summary:", summary)
    
    # Print recent alerts
    alerts = tracker.get_recent_alerts()
    print(f"Recent alerts ({len(alerts)}):")
    for alert in alerts:
        print(f"  {alert.level.value}: {alert.message}")


if __name__ == "__main__":
    asyncio.run(main())