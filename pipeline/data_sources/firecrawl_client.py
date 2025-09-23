"""
Firecrawl integration for web scraping and crawling functionality.
Provides easy access to web data for MLB DFS analysis.
"""

import os
import logging
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


def get_firecrawl_client() -> FirecrawlClient:
    """Get a configured Firecrawl client instance."""
    return FirecrawlClient()