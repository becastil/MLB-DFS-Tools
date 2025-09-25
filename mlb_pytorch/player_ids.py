"""
Player ID unification module.

Builds a master player ID mapping using existing data sources:
- DraftKings slate data (dk_id, player_name)
- MLB Stats API (player_id, player_name) 
- FanGraphs data (player_name)

Provides fuzzy matching and join integrity testing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
from fuzzywuzzy import fuzz

# Import existing data sources
import sys
sys.path.append('..')
from pipeline.data_sources.draftkings import DraftKingsSlateLoader
from pipeline.data_sources.mlb_stats import MLBStatsClient


logger = logging.getLogger(__name__)


class PlayerIDMapper:
    """Unifies player IDs across DraftKings, MLB Stats API, and FanGraphs."""
    
    def __init__(self, mapping_file: Path | str = "data/clean/player_id_map.csv"):
        self.mapping_file = Path(mapping_file)
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Master mapping DataFrame
        self.id_map: Optional[pd.DataFrame] = None
        
        # Load existing mapping or create new one
        self.load_or_create_mapping()
    
    def load_or_create_mapping(self) -> None:
        """Load existing mapping file or create new one from data sources."""
        if self.mapping_file.exists():
            logger.info(f"Loading existing player ID mapping from {self.mapping_file}")
            self.id_map = pd.read_csv(self.mapping_file)
        else:
            logger.info("Creating new player ID mapping from existing data sources")
            self.create_mapping_from_sources()
    
    def create_mapping_from_sources(self) -> None:
        """Build player ID mapping from existing data sources."""
        players = []
        
        # Get unique players from existing DK slate data
        dk_players = self._extract_dk_players()
        players.extend(dk_players)
        
        # Get unique players from MLB Stats API data  
        mlb_players = self._extract_mlb_players()
        players.extend(mlb_players)
        
        # Combine and deduplicate
        combined_df = pd.DataFrame(players)
        if combined_df.empty:
            logger.warning("No player data found in existing sources")
            self.id_map = pd.DataFrame(columns=[
                'canonical_id', 'dk_name', 'dk_id', 'mlbam_id', 
                'normalized_name', 'source'
            ])
            return
        
        # Deduplicate and create canonical IDs
        self.id_map = self._deduplicate_players(combined_df)
        
        # Save to file
        self.save_mapping()
        logger.info(f"Created mapping for {len(self.id_map)} unique players")
    
    def _extract_dk_players(self) -> list[dict]:
        """Extract player info from existing DraftKings data."""
        players = []
        dk_data_dir = Path("dk_data")
        
        if not dk_data_dir.exists():
            logger.warning("No DraftKings data directory found")
            return players
        
        # Look for recent slate CSV files
        for csv_file in dk_data_dir.glob("*.csv"):
            try:
                slate_df = pd.read_csv(csv_file)
                required_cols = ["Name", "ID"]
                
                if not all(col in slate_df.columns for col in required_cols):
                    continue
                
                for _, row in slate_df.iterrows():
                    if pd.notna(row["Name"]) and pd.notna(row["ID"]):
                        players.append({
                            'dk_name': str(row["Name"]).strip(),
                            'dk_id': str(row["ID"]).strip(),
                            'normalized_name': self._normalize_name(row["Name"]),
                            'source': 'draftkings'
                        })
                        
                # Only process first few files to avoid duplicates
                if len(players) > 1000:
                    break
                    
            except Exception as e:
                logger.debug(f"Could not read {csv_file}: {e}")
                continue
        
        logger.info(f"Extracted {len(players)} players from DraftKings data")
        return players
    
    def _extract_mlb_players(self) -> list[dict]:
        """Extract player info from MLB Stats cached data."""
        players = []
        cache_dir = Path("pipeline/raw/mlb_stats")
        
        if not cache_dir.exists():
            logger.warning("No MLB Stats cache directory found")
            return players
        
        # Look for boxscore JSON files
        for json_file in list(cache_dir.glob("boxscore_*.json"))[:50]:  # Limit to avoid slowness
            try:
                with open(json_file, 'r') as f:
                    boxscore = json.load(f)
                
                for home_away in ("away", "home"):
                    team_block = boxscore.get("teams", {}).get(home_away, {})
                    for player_code, player in (team_block.get("players") or {}).items():
                        person = player.get("person", {})
                        if person.get("id") and person.get("fullName"):
                            players.append({
                                'mlbam_id': str(person["id"]),
                                'mlb_name': str(person["fullName"]).strip(),
                                'normalized_name': self._normalize_name(person["fullName"]),
                                'source': 'mlb_stats'
                            })
                            
            except Exception as e:
                logger.debug(f"Could not read {json_file}: {e}")
                continue
        
        logger.info(f"Extracted {len(players)} players from MLB Stats data")
        return players
    
    def _deduplicate_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate players and assign canonical IDs."""
        if df.empty:
            return df
        
        # Group by normalized name for fuzzy matching
        unique_players = []
        processed_names: Set[str] = set()
        
        for _, row in df.iterrows():
            norm_name = row.get('normalized_name', '')
            if not norm_name or norm_name in processed_names:
                continue
            
            # Find all similar names
            similar_rows = df[df['normalized_name'] == norm_name]
            
            # Combine information from all similar rows
            canonical_player = self._merge_player_info(similar_rows)
            unique_players.append(canonical_player)
            processed_names.add(norm_name)
        
        result_df = pd.DataFrame(unique_players)
        
        # Assign canonical IDs
        result_df['canonical_id'] = range(1, len(result_df) + 1)
        result_df['canonical_id'] = result_df['canonical_id'].apply(lambda x: f"PLR_{x:06d}")
        
        return result_df
    
    def _merge_player_info(self, similar_rows: pd.DataFrame) -> dict:
        """Merge information from rows representing the same player."""
        merged = {
            'dk_name': None,
            'dk_id': None, 
            'mlbam_id': None,
            'mlb_name': None,
            'normalized_name': similar_rows.iloc[0].get('normalized_name', ''),
            'source': 'merged'
        }
        
        for _, row in similar_rows.iterrows():
            if row.get('dk_name') and not merged['dk_name']:
                merged['dk_name'] = row['dk_name']
            if row.get('dk_id') and not merged['dk_id']:
                merged['dk_id'] = row['dk_id']
            if row.get('mlbam_id') and not merged['mlbam_id']:
                merged['mlbam_id'] = row['mlbam_id']
            if row.get('mlb_name') and not merged['mlb_name']:
                merged['mlb_name'] = row['mlb_name']
        
        return merged
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize player name for matching."""
        if not isinstance(name, str):
            return ""
        
        # Remove punctuation, convert to uppercase, handle Jr/Sr
        clean = name.upper().strip()
        clean = clean.replace(".", "").replace(",", "").replace("-", " ")
        clean = clean.replace(" JR", "").replace(" SR", "").replace(" III", "").replace(" II", "")
        
        # Remove extra whitespace
        clean = " ".join(clean.split())
        
        return clean
    
    def match_player(self, name: str, source: str = 'dk', fuzzy_threshold: float = 85.0) -> Optional[str]:
        """
        Match a player name to a canonical ID using fuzzy matching.
        
        Args:
            name: Player name to match
            source: Source system ('dk', 'mlb', etc.)
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            
        Returns:
            Canonical ID if match found, None otherwise
        """
        if self.id_map is None or self.id_map.empty:
            return None
        
        normalized_input = self._normalize_name(name)
        
        if not normalized_input:
            return None
        
        # First try exact match
        exact_match = self.id_map[self.id_map['normalized_name'] == normalized_input]
        if not exact_match.empty:
            return exact_match.iloc[0]['canonical_id']
        
        # Try fuzzy matching
        best_match_id = None
        best_score = 0.0
        
        for _, row in self.id_map.iterrows():
            if not row['normalized_name']:
                continue
                
            score = fuzz.ratio(normalized_input, row['normalized_name'])
            
            if score > best_score and score >= fuzzy_threshold:
                best_score = score
                best_match_id = row['canonical_id']
        
        if best_match_id:
            logger.debug(f"Fuzzy matched '{name}' to ID {best_match_id} (score: {best_score})")
        
        return best_match_id
    
    def get_canonical_id(self, player_info: dict) -> Optional[str]:
        """
        Get canonical ID from any player info dict.
        
        Args:
            player_info: Dict with 'Name'/'player_name', 'ID'/'dk_id', etc.
            
        Returns:
            Canonical ID if found
        """
        # Try various name fields
        name = (player_info.get('Name') or 
                player_info.get('player_name') or 
                player_info.get('fullName') or 
                player_info.get('name'))
        
        if name:
            return self.match_player(str(name))
        
        return None
    
    def validate_joins(self, df1: pd.DataFrame, df2: pd.DataFrame, on: str = 'canonical_id') -> bool:
        """
        Validate that joining two DataFrames doesn't cause row explosion.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame  
            on: Join column
            
        Returns:
            True if join is safe, False otherwise
        """
        before_count = len(df1)
        
        # Perform join
        merged = df1.merge(df2, on=on, how='left')
        after_count = len(merged)
        
        if after_count != before_count:
            logger.error(f"Join caused row explosion: {before_count} -> {after_count}")
            return False
        
        logger.info(f"Join validation passed: {before_count} rows maintained")
        return True
    
    def add_canonical_ids(self, df: pd.DataFrame, name_col: str = 'Name') -> pd.DataFrame:
        """
        Add canonical IDs to a DataFrame based on player names.
        
        Args:
            df: DataFrame with player names
            name_col: Column containing player names
            
        Returns:
            DataFrame with canonical_id column added
        """
        result = df.copy()
        result['canonical_id'] = None
        
        for idx, row in result.iterrows():
            name = row.get(name_col)
            if name:
                canonical_id = self.match_player(str(name))
                result.at[idx, 'canonical_id'] = canonical_id
        
        matched_count = result['canonical_id'].notna().sum()
        total_count = len(result)
        
        logger.info(f"Added canonical IDs: {matched_count}/{total_count} players matched")
        
        return result
    
    def save_mapping(self) -> None:
        """Save current mapping to CSV file."""
        if self.id_map is not None:
            self.id_map.to_csv(self.mapping_file, index=False)
            logger.info(f"Saved player mapping to {self.mapping_file}")
    
    def get_stats(self) -> dict:
        """Get statistics about the player mapping."""
        if self.id_map is None or self.id_map.empty:
            return {'total_players': 0}
        
        stats = {
            'total_players': len(self.id_map),
            'with_dk_id': self.id_map['dk_id'].notna().sum(),
            'with_mlbam_id': self.id_map['mlbam_id'].notna().sum(),
            'source_breakdown': self.id_map['source'].value_counts().to_dict()
        }
        
        return stats


# Utility functions for testing
def test_join_integrity(df1: pd.DataFrame, df2: pd.DataFrame, on: str) -> None:
    """Test function to ensure joins don't explode row counts."""
    before_count = len(df1)
    merged = df1.merge(df2, on=on, how='left')
    after_count = len(merged)
    
    assert after_count == before_count, f"Join caused row explosion: {before_count} -> {after_count}"
    print(f"✓ Join integrity test passed: {before_count} rows maintained")


def get_firecrawl_client():
    """Get Firecrawl client for data collection."""
    try:
        from pipeline.data_sources.firecrawl_client import FirecrawlClient
        return FirecrawlClient()
    except Exception as e:
        logger.error(f"Could not initialize Firecrawl client: {e}")
        return None


if __name__ == "__main__":
    # Test the player ID mapper
    logging.basicConfig(level=logging.INFO)
    
    mapper = PlayerIDMapper()
    stats = mapper.get_stats()
    
    print(f"\nPlayer ID Mapping Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test a few name matches
    test_names = ["Shohei Ohtani", "Juan Soto", "Aaron Judge"]
    
    print(f"\nTesting name matches:")
    for name in test_names:
        canonical_id = mapper.match_player(name)
        print(f"  {name} -> {canonical_id}")