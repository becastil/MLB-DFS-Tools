"""
Firecrawl integration for web scraping and crawling functionality.
Provides easy access to web data for MLB DFS analysis.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FirecrawlClient:
    """Client for interacting with Firecrawl API for web scraping."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Firecrawl client.
        
        Args:
            api_key: Firecrawl API key. If None, will try to load from environment.
        """
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key not found. Set FIRECRAWL_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.app = FirecrawlApp(api_key=self.api_key)
        
        # Cache settings for real-time data refresh
        self.cache_ttl_minutes = {
            'dfs_contextual': 60,  # 1 hour for comprehensive contextual data
            'ownership_factors': 30,  # 30 minutes for ownership factors
            'market_intelligence': 15,  # 15 minutes for market data
            'injury_reports': 120,  # 2 hours for injury reports  
            'weather_data': 45,  # 45 minutes for weather
        }
        
        # Cache directory for storing extraction results
        self.cache_dir = Path.cwd() / '.firecrawl_cache'
        self.cache_dir.mkdir(exist_ok=True)
    
    def scrape_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Scrape a single URL and return structured data.
        
        Args:
            url: URL to scrape
            **kwargs: Additional parameters for Firecrawl scrape
        
        Returns:
            Dict containing scraped data with content, metadata, etc.
        """
        try:
            logger.info(f"Scraping URL: {url}")
            response = self.app.scrape_url(url, **kwargs)
            logger.info(f"Successfully scraped {url}")
            return response
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            raise
    
    def crawl_website(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Crawl an entire website starting from the given URL.
        
        Args:
            url: Starting URL for crawling
            **kwargs: Additional parameters for Firecrawl crawl
        
        Returns:
            Dict containing crawl results with multiple pages of data
        """
        try:
            logger.info(f"Starting crawl from URL: {url}")
            response = self.app.crawl_url(url, **kwargs)
            logger.info(f"Successfully completed crawl from {url}")
            return response
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            raise
    
    def scrape_mlb_stats_site(self, url: str) -> Dict[str, Any]:
        """
        Scrape MLB statistics websites with optimized settings.
        
        Args:
            url: URL of MLB stats site to scrape
        
        Returns:
            Dict containing scraped MLB data
        """
        scrape_params = {
            'formats': ['markdown', 'html'],
            'extract': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'player_stats': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'name': {'type': 'string'},
                                    'team': {'type': 'string'},
                                    'position': {'type': 'string'},
                                    'stats': {'type': 'object'}
                                }
                            }
                        },
                        'game_info': {
                            'type': 'object',
                            'properties': {
                                'date': {'type': 'string'},
                                'matchup': {'type': 'string'},
                                'weather': {'type': 'string'}
                            }
                        }
                    }
                }
            }
        }
        
        return self.scrape_url(url, **scrape_params)

    def search_web(self, query: str, limit: int = 5, include_content: bool = False,
                   **kwargs) -> Dict[str, Any]:
        """Leverage Firecrawl's search feature to discover relevant URLs.

        Args:
            query: Natural language search query.
            limit: Maximum number of search hits to return.
            include_content: Whether to ask Firecrawl to fetch the page content for
                each hit (can increase latency and credit usage).
            **kwargs: Additional parameters forwarded to the Firecrawl search API.

        Returns:
            Dict describing the search results, including URLs and optional snippets.
        """

        try:
            logger.info("Searching the web for query: %s", query)
            response = self.app.search(
                query=query,
                limit=limit,
                include_content=include_content,
                **kwargs,
            )
            logger.info("Search returned %d results", len(response.get("results", [])))
            return response
        except Exception as exc:
            logger.error("Firecrawl search failed for '%s': %s", query, exc)
            raise
    
    def scrape_injury_report(self, url: str) -> Dict[str, Any]:
        """
        Scrape injury reports from MLB news sites.
        
        Args:
            url: URL of injury report page
        
        Returns:
            Dict containing injury information
        """
        scrape_params = {
            'formats': ['markdown'],
            'extract': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'injuries': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'player_name': {'type': 'string'},
                                    'team': {'type': 'string'},
                                    'injury_type': {'type': 'string'},
                                    'status': {'type': 'string'},
                                    'expected_return': {'type': 'string'}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return self.scrape_url(url, **scrape_params)
    
    def scrape_weather_data(self, url: str) -> Dict[str, Any]:
        """
        Scrape weather data for MLB games.
        
        Args:
            url: URL of weather information page
        
        Returns:
            Dict containing weather data
        """
        scrape_params = {
            'formats': ['markdown'],
            'extract': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'weather_conditions': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'stadium': {'type': 'string'},
                                    'game_time': {'type': 'string'},
                                    'temperature': {'type': 'string'},
                                    'wind_speed': {'type': 'string'},
                                    'wind_direction': {'type': 'string'},
                                    'humidity': {'type': 'string'},
                                    'precipitation': {'type': 'string'}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return self.scrape_url(url, **scrape_params)
    
    def bulk_scrape_urls(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs in batch.
        
        Args:
            urls: List of URLs to scrape
            **kwargs: Additional parameters for scraping
        
        Returns:
            List of scrape results for each URL
        """
        results = []
        for url in urls:
            try:
                result = self.scrape_url(url, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                results.append({'error': str(e), 'url': url})
        
        return results
    
    # Advanced Extract Methods using Firecrawl v2
    
    def extract(self, urls: List[str], prompt: Optional[str] = None, 
                schema: Optional[Dict[str, Any]] = None, 
                enable_web_search: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Extract structured data using LLMs from one or multiple URLs.
        
        Args:
            urls: List of URLs to extract from (supports wildcards like domain.com/*)
            prompt: Natural language description of data to extract
            schema: JSON schema for structured output
            enable_web_search: Whether to search related pages for more context
            **kwargs: Additional parameters
        
        Returns:
            Dict containing extracted structured data
        """
        try:
            logger.info(f"Starting extract for {len(urls)} URLs")
            
            # Build extraction parameters
            extract_params = {
                'urls': urls,
                'enableWebSearch': enable_web_search,
                **kwargs
            }
            
            if prompt:
                extract_params['prompt'] = prompt
            if schema:
                extract_params['schema'] = schema
            
            response = self.app.extract(**extract_params)
            logger.info(f"Successfully extracted data from {len(urls)} URLs")
            return response
        except Exception as e:
            logger.error(f"Error extracting from {urls}: {str(e)}")
            raise
    
    def start_extract(self, urls: List[str], prompt: Optional[str] = None,
                     schema: Optional[Dict[str, Any]] = None,
                     enable_web_search: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Start an extraction job without waiting for completion.
        
        Args:
            urls: List of URLs to extract from
            prompt: Natural language description of data to extract
            schema: JSON schema for structured output
            enable_web_search: Whether to search related pages
            **kwargs: Additional parameters
        
        Returns:
            Dict containing job ID and initial status
        """
        try:
            logger.info(f"Starting async extract job for {len(urls)} URLs")
            
            extract_params = {
                'urls': urls,
                'enableWebSearch': enable_web_search,
                **kwargs
            }
            
            if prompt:
                extract_params['prompt'] = prompt
            if schema:
                extract_params['schema'] = schema
            
            response = self.app.start_extract(**extract_params)
            logger.info(f"Started extract job with ID: {response.id}")
            return response
        except Exception as e:
            logger.error(f"Error starting extract job: {str(e)}")
            raise
    
    def get_extract_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of an extraction job.
        
        Args:
            job_id: ID of the extraction job
        
        Returns:
            Dict containing job status and data if completed
        """
        try:
            response = self.app.get_extract_status(job_id)
            logger.info(f"Retrieved status for job {job_id}: {response.status}")
            return response
        except Exception as e:
            logger.error(f"Error getting status for job {job_id}: {str(e)}")
            raise
    
    def extract_mlb_season_stats(self, base_url: str, 
                                enable_web_search: bool = True) -> Dict[str, Any]:
        """
        Extract comprehensive MLB season statistics from a site.
        
        Args:
            base_url: Base URL (e.g., "https://www.baseball-reference.com/leagues/MLB/2024-standard-batting.shtml")
            enable_web_search: Whether to search for related stats pages
        
        Returns:
            Dict containing structured season statistics
        """
        schema = {
            "type": "object",
            "properties": {
                "season": {"type": "string"},
                "league": {"type": "string"},
                "players": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "team": {"type": "string"},
                            "position": {"type": "string"},
                            "games": {"type": "number"},
                            "batting_avg": {"type": "number"},
                            "home_runs": {"type": "number"},
                            "rbi": {"type": "number"},
                            "stolen_bases": {"type": "number"},
                            "ops": {"type": "number"},
                            "era": {"type": "number"},
                            "whip": {"type": "number"},
                            "strikeouts": {"type": "number"},
                            "wins": {"type": "number"},
                            "saves": {"type": "number"}
                        }
                    }
                },
                "team_stats": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "team": {"type": "string"},
                            "wins": {"type": "number"},
                            "losses": {"type": "number"},
                            "win_pct": {"type": "number"},
                            "runs_scored": {"type": "number"},
                            "runs_allowed": {"type": "number"},
                            "run_differential": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        prompt = """Extract comprehensive MLB season statistics including:
        1. Individual player batting and pitching stats
        2. Team records and run differentials
        3. Current season leaders in major categories
        4. Playoff standings and wildcard positions"""
        
        return self.extract(
            urls=[base_url],
            prompt=prompt,
            schema=schema,
            enable_web_search=enable_web_search
        )
    
    def extract_dfs_slate_info(self, urls: List[str]) -> Dict[str, Any]:
        """
        Extract daily fantasy sports slate information.
        
        Args:
            urls: URLs containing DFS slate data
        
        Returns:
            Dict containing DFS slate information
        """
        schema = {
            "type": "object",
            "properties": {
                "slate_date": {"type": "string"},
                "games": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "away_team": {"type": "string"},
                            "home_team": {"type": "string"},
                            "game_time": {"type": "string"},
                            "total": {"type": "number"},
                            "spread": {"type": "number"},
                            "weather": {"type": "string"},
                            "stadium": {"type": "string"}
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
                            "ownership_projection": {"type": "number"},
                            "value": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        prompt = """Extract DFS slate information including:
        1. Game schedule with times and totals
        2. Player salaries and projections
        3. Ownership projections
        4. Weather and park factors
        5. Value plays and chalk players"""
        
        return self.extract(
            urls=urls,
            prompt=prompt,
            schema=schema,
            enable_web_search=True
        )
    
    def extract_betting_lines(self, urls: List[str]) -> Dict[str, Any]:
        """
        Extract betting lines and odds information.
        
        Args:
            urls: URLs containing betting information
        
        Returns:
            Dict containing betting lines and odds
        """
        schema = {
            "type": "object",
            "properties": {
                "games": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "matchup": {"type": "string"},
                            "game_time": {"type": "string"},
                            "moneyline_home": {"type": "number"},
                            "moneyline_away": {"type": "number"},
                            "total": {"type": "number"},
                            "total_over_odds": {"type": "number"},
                            "total_under_odds": {"type": "number"},
                            "run_line": {"type": "number"},
                            "run_line_home_odds": {"type": "number"},
                            "run_line_away_odds": {"type": "number"}
                        }
                    }
                },
                "futures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "market": {"type": "string"},
                            "team_or_player": {"type": "string"},
                            "odds": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        prompt = """Extract MLB betting information including:
        1. Game lines (moneyline, total, run line)
        2. Current odds for each bet type
        3. Line movement indicators
        4. Futures markets (World Series, MVP, etc.)
        5. Prop bet opportunities"""
        
        return self.extract(
            urls=urls,
            prompt=prompt,
            schema=schema
        )
    
    def extract_advanced_injury_report(self, urls: List[str]) -> Dict[str, Any]:
        """
        Extract comprehensive injury reports with timelines and impact.
        
        Args:
            urls: URLs containing injury information
        
        Returns:
            Dict containing detailed injury data
        """
        schema = {
            "type": "object",
            "properties": {
                "report_date": {"type": "string"},
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
                            "expected_return": {"type": "string"},
                            "days_out": {"type": "number"},
                            "replacement_player": {"type": "string"},
                            "fantasy_impact": {"type": "string"},
                            "last_update": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        prompt = """Extract detailed injury reports including:
        1. Player injury details and severity
        2. Expected return timelines
        3. Likely replacement players
        4. Fantasy baseball impact assessment
        5. Recent updates and status changes"""
        
        return self.extract(
            urls=urls,
            prompt=prompt,
            schema=schema,
            enable_web_search=True
        )
    
    def extract_with_cache(self, urls: List[str], prompt: Optional[str] = None,
                          schema: Optional[Dict[str, Any]] = None,
                          cache_key: str = 'default',
                          force_refresh: bool = False,
                          **kwargs) -> Dict[str, Any]:
        """
        Extract data with intelligent caching and refresh logic.
        
        Args:
            urls: List of URLs to extract from
            prompt: Extraction prompt
            schema: JSON schema for extraction
            cache_key: Key for cache storage (determines TTL)
            force_refresh: Force refresh even if cache is valid
            **kwargs: Additional extraction parameters
            
        Returns:
            Dict containing extracted data (from cache or fresh extraction)
        """
        import json
        import hashlib
        from datetime import datetime, timedelta
        
        # Generate cache filename based on parameters
        cache_params = {
            'urls': sorted(urls),
            'prompt': prompt,
            'cache_key': cache_key
        }
        cache_hash = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()[:12]
        cache_file = self.cache_dir / f"{cache_key}_{cache_hash}.json"
        
        # Check if cached data is still valid
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                
                cached_time = datetime.fromisoformat(cached['timestamp'])
                ttl_minutes = self.cache_ttl_minutes.get(cache_key, 60)
                expires_at = cached_time + timedelta(minutes=ttl_minutes)
                
                if datetime.now() < expires_at:
                    logger.info(f"Using cached data for {cache_key} (expires in {(expires_at - datetime.now()).total_seconds() / 60:.1f} min)")
                    return cached['data']
                else:
                    logger.info(f"Cache expired for {cache_key}, refreshing...")
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
        
        # Extract fresh data
        logger.info(f"Extracting fresh data for {cache_key}")
        try:
            data = self.extract(urls=urls, prompt=prompt, schema=schema, **kwargs)
            
            # Cache the results
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'cache_key': cache_key,
                'urls': urls,
                'data': data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cached extraction results for {cache_key}")
            return data
            
        except Exception as e:
            # If extraction fails, try to return stale cache data
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    logger.warning(f"Extraction failed, using stale cache for {cache_key}: {e}")
                    return cached['data']
                except:
                    pass
            raise
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            cache_key: Specific cache key to clear. If None, clears all cache.
        """
        if cache_key is None:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared all Firecrawl cache")
        else:
            # Clear specific cache key
            for cache_file in self.cache_dir.glob(f"{cache_key}_*.json"):
                cache_file.unlink()
            logger.info(f"Cleared cache for {cache_key}")
    
    def get_cache_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all cached data.
        
        Returns:
            List of cache status information
        """
        import json
        from datetime import datetime, timedelta
        
        cache_status = []
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                
                cached_time = datetime.fromisoformat(cached['timestamp'])
                cache_key = cached.get('cache_key', 'unknown')
                ttl_minutes = self.cache_ttl_minutes.get(cache_key, 60)
                expires_at = cached_time + timedelta(minutes=ttl_minutes)
                
                status = {
                    'cache_key': cache_key,
                    'cached_at': cached_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'expires_at': expires_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'is_valid': datetime.now() < expires_at,
                    'ttl_minutes': ttl_minutes,
                    'file_size_kb': cache_file.stat().st_size / 1024
                }
                
                cache_status.append(status)
                
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return sorted(cache_status, key=lambda x: x['cached_at'], reverse=True)


def get_firecrawl_client() -> FirecrawlClient:
    """Get a configured Firecrawl client instance."""
    return FirecrawlClient()

# Enhanced extraction method for DFS contextual data
def extract_dfs_contextual_data(urls: list[str], api_key: str = None) -> dict:
    """
    Extract comprehensive DFS contextual data using the user's schema pattern.
    
    This function implements the pattern from the user's example:
    from firecrawl import Firecrawl
    from pydantic import BaseModel, Field
    
    Args:
        urls: List of URLs to extract from (supports wildcards)
        api_key: Firecrawl API key (uses environment variable if not provided)
    
    Returns:
        Dictionary containing extracted DFS data
    """
    try:
        from firecrawl import Firecrawl
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError("firecrawl library not available. Install with: pip install firecrawl-py")
    
    # Use provided API key or fallback to environment/default
    if api_key:
        app = Firecrawl(api_key=api_key)
    else:
        # Try to get from environment, fallback to user's example key
        import os
        key = os.getenv('FIRECRAWL_API_KEY', 'fc-dd26161f85a847298e92cc4064770984')
        app = Firecrawl(api_key=key)
    
    # Define extraction schema based on user's pattern
    class ExtractSchema(BaseModel):
        hitters: list = Field(description="The hitters with current form, matchups, and advanced metrics")
        pitchers: list = Field(description="The pitchers with recent performance and opponent analysis") 
        contextual_data: any = Field(description="Weather, park factors, betting lines, and environmental context")
        team_level_stats: any = Field(description="Team implied runs, bullpen strength, and stack correlation")
    
    # Enhanced prompt for comprehensive DFS data
    prompt = """Extract comprehensive data relevant to daily fantasy sports projections for MLB hitters and pitchers. Include:

    HITTERS:
    - Current batting order and lineup status
    - Recent form over last 15 games (hot/cold streaks)
    - Matchup analysis vs pitcher handedness  
    - Advanced metrics: wRC+, xwOBA, hard-hit rate, barrel rate
    - Park factors and weather impact on offensive environment
    - Lineup protection and RBI opportunities

    PITCHERS:
    - Recent form and pitch efficiency trends
    - Opponent offensive strength and difficulty rating
    - Expected pitch count and innings projection
    - Advanced metrics: FIP, xFIP, K-BB%, CSW rate
    - Bullpen support quality and win probability

    CONTEXTUAL DATA:
    - Weather conditions affecting scoring (wind, temperature, humidity)
    - Betting lines, totals, and implied run environments
    - Line movement and sharp money indicators
    - Park factors and historical scoring rates

    TEAM LEVEL STATS:
    - Implied run totals and win probability
    - Bullpen usage patterns and availability  
    - Stack correlation opportunities
    - Recent offensive and defensive form

    Structure the data to support generating projections with ceiling/floor outcomes and ownership estimates for DFS tournaments and cash games."""
    
    # Perform the extraction using user's pattern
    data = app.extract(
        urls=urls,
        prompt=prompt,
        schema=ExtractSchema.model_json_schema()
    )
    
    return data

    # ------------------------------------------------------------------
    # Underdog Fantasy & PrizePicks Extraction Methods  
    # ------------------------------------------------------------------
    
    def extract_underdog_props(self, urls: List[str]) -> Dict[str, Any]:
        """
        Extract Underdog Fantasy prop bets and multipliers.
        
        Args:
            urls: URLs containing Underdog Fantasy prop data
        
        Returns:
            Dict containing Underdog prop information
        """
        schema = {
            "type": "object",
            "properties": {
                "extraction_date": {"type": "string"},
                "props": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "team": {"type": "string"},
                            "position": {"type": "string"},
                            "stat_type": {"type": "string"},
                            "line": {"type": "number"},
                            "over_multiplier": {"type": "number"},
                            "under_multiplier": {"type": "number"},
                            "higher_multiplier": {"type": "number"},
                            "lower_multiplier": {"type": "number"},
                            "pick_percentage": {"type": "number"},
                            "value_grade": {"type": "string"},
                            "last_10_games": {"type": "string"},
                            "season_average": {"type": "number"},
                            "matchup_rating": {"type": "string"}
                        }
                    }
                },
                "best_ball_adp": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "position": {"type": "string"},
                            "team": {"type": "string"},
                            "adp": {"type": "number"},
                            "draft_percentage": {"type": "number"},
                            "tier": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        prompt = """Extract Underdog Fantasy data including:
        1. All available prop bets with player names, stat types, and lines
        2. Over/Under multipliers and Higher/Lower multipliers
        3. Pick percentages and popularity metrics
        4. Player value grades (A, B, C, D, F ratings)
        5. Recent form data (last 10 games performance vs line)
        6. Season averages for each stat type
        7. Matchup difficulty ratings
        8. Best Ball ADP data if available
        9. Draft percentages and player tiers
        
        Focus on MLB players and props only."""
        
        return self.extract(
            urls=urls,
            prompt=prompt,
            schema=schema,
            enable_web_search=True
        )
    
    def extract_prizepicks_props(self, urls: List[str]) -> Dict[str, Any]:
        """
        Extract PrizePicks prop lines and data.
        
        Args:
            urls: URLs containing PrizePicks prop data
        
        Returns:
            Dict containing PrizePicks prop information
        """
        schema = {
            "type": "object",
            "properties": {
                "extraction_date": {"type": "string"},
                "props": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "team": {"type": "string"},
                            "position": {"type": "string"},
                            "stat_type": {"type": "string"},
                            "line": {"type": "number"},
                            "pick_type": {"type": "string"},
                            "odds": {"type": "number"},
                            "payout_multiplier": {"type": "number"},
                            "popularity": {"type": "number"},
                            "recent_form": {"type": "string"},
                            "injury_status": {"type": "string"},
                            "weather_impact": {"type": "string"},
                            "ballpark_factor": {"type": "string"}
                        }
                    }
                },
                "trending_picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "stat_type": {"type": "string"},
                            "pick_percentage": {"type": "number"},
                            "sharp_action": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        prompt = """Extract PrizePicks MLB prop data including:
        1. All available MLB prop bets with player names and stat types
        2. Prop lines and pick types (Over/Under, More/Less)
        3. Odds and payout multipliers for different bet types
        4. Pick popularity and trending data
        5. Recent player form vs the line (hits, strikeouts over last 5-10 games)
        6. Injury status updates affecting props
        7. Weather conditions and ballpark factors
        8. Sharp vs public action indicators
        9. Line movement data if available
        
        Focus specifically on MLB players and exclude other sports."""
        
        return self.extract(
            urls=urls,
            prompt=prompt,
            schema=schema,
            enable_web_search=True
        )
    
    def extract_prop_comparison(self, underdog_urls: List[str], prizepicks_urls: List[str], 
                               sportsbook_urls: List[str] = None) -> Dict[str, Any]:
        """
        Compare prop lines across Underdog, PrizePicks, and sportsbooks.
        
        Args:
            underdog_urls: Underdog Fantasy URLs
            prizepicks_urls: PrizePicks URLs  
            sportsbook_urls: Optional sportsbook URLs for comparison
        
        Returns:
            Dict containing cross-platform prop comparison
        """
        all_urls = underdog_urls + prizepicks_urls + (sportsbook_urls or [])
        
        schema = {
            "type": "object",
            "properties": {
                "comparison_date": {"type": "string"},
                "prop_arbitrage": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "stat_type": {"type": "string"},
                            "underdog_line": {"type": "number"},
                            "underdog_multiplier": {"type": "number"},
                            "prizepicks_line": {"type": "number"},
                            "prizepicks_odds": {"type": "number"},
                            "sportsbook_line": {"type": "number"},
                            "sportsbook_odds": {"type": "number"},
                            "edge_opportunity": {"type": "string"},
                            "recommended_side": {"type": "string"},
                            "confidence_level": {"type": "string"}
                        }
                    }
                },
                "market_inefficiencies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "stat_type": {"type": "string"},
                            "line_discrepancy": {"type": "number"},
                            "best_platform": {"type": "string"},
                            "expected_value": {"type": "number"}
                        }
                    }
                }
            }
        }
        
        prompt = """Compare MLB prop lines across platforms to identify:
        1. Line discrepancies between Underdog, PrizePicks, and sportsbooks
        2. Arbitrage opportunities where you can bet both sides profitably
        3. Market inefficiencies and soft lines
        4. Best platforms for specific types of props
        5. Expected value calculations for each prop
        6. Sharp vs recreational platform tendencies
        7. Recommended sides based on line shopping
        8. Confidence levels for each edge opportunity
        
        Focus on finding +EV (positive expected value) opportunities for MLB props."""
        
        return self.extract(
            urls=all_urls,
            prompt=prompt,
            schema=schema,
            enable_web_search=True
        )
