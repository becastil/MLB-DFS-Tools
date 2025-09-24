"""
Example usage of enhanced MLB DFS projections with Firecrawl integration.

This demonstrates how the new Firecrawl integration enhances projections
and ownership estimates with real-time contextual data.
"""

import sys
from pathlib import Path
from datetime import date

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.projection_pipeline import ProjectionPipeline
from pipeline.data_sources.firecrawl_client import extract_dfs_contextual_data

def example_enhanced_projections():
    """Example of generating projections with Firecrawl contextual data."""
    
    # Initialize the projection pipeline
    base_dir = Path("./data")
    seasons = [2022, 2023, 2024]
    pipeline = ProjectionPipeline(base_dir, seasons)
    
    print("🔥 Enhanced MLB DFS Projections with Firecrawl")
    print("=" * 50)
    
    # Example slate date (replace with actual date)
    slate_date = date(2024, 7, 15)
    
    # Path to DraftKings slate CSV (you would provide actual path)
    slate_csv = base_dir / "slates" / f"dk_slate_{slate_date.isoformat()}.csv"
    
    try:
        print(f"📅 Generating projections for {slate_date}")
        
        # Generate projections with contextual data enabled
        projections, template = pipeline.generate_projections(
            slate_csv=slate_csv,
            slate_date=slate_date,
            use_contextual_data=True  # This enables Firecrawl integration
        )
        
        print(f"✅ Generated projections for {len(projections)} players")
        
        # Show contextual adjustments applied
        if 'contextual_adjustments' in projections.columns:
            adjusted_players = projections[
                projections['contextual_adjustments'].notna() & 
                (projections['contextual_adjustments'] != '')
            ]
            print(f"🎯 Applied contextual adjustments to {len(adjusted_players)} players")
            
            # Show top adjustments
            for _, player in adjusted_players.head(3).iterrows():
                print(f"  • {player['player_name']}: {player['contextual_adjustments']}")
        
        # Show ownership adjustments
        ownership_adjustments = projections[
            projections.get('ownership_reason', '').notna() & 
            (projections.get('ownership_reason', '') != '')
        ]
        
        if not ownership_adjustments.empty:
            print(f"📊 Applied ownership adjustments to {len(ownership_adjustments)} players")
            for _, player in ownership_adjustments.head(3).iterrows():
                print(f"  • {player['player_name']}: {player['ownership_reason']}")
        
        # Show data reliability scores
        if 'data_reliability' in projections.columns:
            avg_reliability = projections['data_reliability'].mean()
            print(f"📈 Average data reliability: {avg_reliability:.2f}")
        
        print("\n🎯 Top projected plays:")
        top_plays = projections.nlargest(5, 'projection')
        for _, player in top_plays.iterrows():
            salary = player.get('salary', 0)
            value = player['projection'] / (salary/1000) if salary > 0 else 0
            print(f"  • {player['player_name']} ({player['team']}): {player['projection']:.2f} pts, ${salary}, {value:.2f} value")
        
    except FileNotFoundError:
        print(f"⚠️  Slate file not found: {slate_csv}")
        print("   This is expected for the example. Provide an actual DraftKings slate CSV.")
    except Exception as e:
        print(f"❌ Error generating projections: {e}")

def example_direct_firecrawl_usage():
    """Example of directly using the enhanced Firecrawl extraction."""
    
    print("\n🔥 Direct Firecrawl Usage Example")
    print("=" * 35)
    
    # URLs to extract from (using the pattern from user's example)
    urls = [
        'https://fangraphs.com/leaders.aspx?pos=all&stats=bat*',
        'https://fangraphs.com/leaders.aspx?pos=all&stats=pit*'
    ]
    
    try:
        print("📡 Extracting contextual DFS data...")
        
        # Use the enhanced extraction function
        data = extract_dfs_contextual_data(urls)
        
        print("✅ Extraction completed!")
        
        # Display extracted data structure
        if 'hitters' in data:
            hitter_count = len(data['hitters']) if isinstance(data['hitters'], list) else 0
            print(f"  📊 Hitters data: {hitter_count} players")
        
        if 'pitchers' in data:
            pitcher_count = len(data['pitchers']) if isinstance(data['pitchers'], list) else 0
            print(f"  ⚾ Pitchers data: {pitcher_count} players")
        
        if 'contextual_data' in data:
            print(f"  🌤️  Contextual data: Available")
            
        if 'team_level_stats' in data:
            team_count = len(data['team_level_stats']) if isinstance(data['team_level_stats'], list) else 0
            print(f"  🏆 Team stats: {team_count} teams")
        
    except Exception as e:
        print(f"❌ Error with direct extraction: {e}")
        print("   This is expected if FIRECRAWL_API_KEY is not set or invalid.")

def show_cache_status():
    """Show the status of cached Firecrawl data."""
    
    print("\n📦 Cache Status")
    print("=" * 15)
    
    try:
        from pipeline.data_sources.firecrawl_client import get_firecrawl_client
        
        client = get_firecrawl_client()
        cache_status = client.get_cache_status()
        
        if not cache_status:
            print("  No cached data found.")
        else:
            for status in cache_status:
                validity = "✅ Valid" if status['is_valid'] else "❌ Expired"
                print(f"  • {status['cache_key']}: {validity}")
                print(f"    Cached: {status['cached_at']}")
                print(f"    Expires: {status['expires_at']}")
                print(f"    Size: {status['file_size_kb']:.1f} KB")
                print()
    
    except Exception as e:
        print(f"❌ Error checking cache: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced MLB DFS Tools with Firecrawl Integration")
    print("=" * 55)
    print()
    
    # Run examples
    example_enhanced_projections()
    example_direct_firecrawl_usage() 
    show_cache_status()
    
    print("\n💡 Tips:")
    print("  • Set FIRECRAWL_API_KEY environment variable for live data")
    print("  • Provide actual DraftKings slate CSV for full projection pipeline")
    print("  • Data is cached intelligently to optimize API usage")
    print("  • Contextual adjustments improve projection accuracy")
    print("  • Ownership factors help with tournament strategy")