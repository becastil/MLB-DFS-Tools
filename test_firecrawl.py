#!/usr/bin/env python3
"""Simple test script to verify Firecrawl API key works."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_firecrawl_api():
    """Test if Firecrawl API key is working."""
    
    # Check if API key is set
    api_key = os.getenv('FIRECRAWL_API_KEY')
    if not api_key:
        print("❌ FIRECRAWL_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        # Try to import and initialize Firecrawl
        from firecrawl import FirecrawlApp
        
        app = FirecrawlApp(api_key=api_key)
        print("✅ Firecrawl client initialized successfully")
        
        # Test with a simple scrape
        print("🔄 Testing scrape functionality...")
        result = app.scrape("https://httpbin.org/json")
        
        print(f"🔍 Response type: {type(result)}")
        print(f"🔍 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if result:
            # Check different possible response structures
            content = None
            if hasattr(result, 'markdown') and result.markdown:
                content = result.markdown
            elif hasattr(result, 'content') and result.content:
                content = result.content
            elif isinstance(result, dict):
                content = result.get('content') or result.get('markdown') or result.get('text') or result.get('data')
            
            if content:
                print("✅ Scrape test successful!")
                print(f"📄 Content length: {len(str(content))} characters")
                
                # Check credits used
                if hasattr(result, 'metadata') and hasattr(result.metadata, 'credits_used'):
                    print(f"💳 Credits used: {result.metadata.credits_used}")
                
                return True
            else:
                print("❌ Scrape test failed - no content found")
                return False
        else:
            print("❌ Scrape test failed - no response")
            return False
            
    except ImportError:
        print("❌ firecrawl-py package not installed")
        print("💡 Run: pip install firecrawl-py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

def test_credits_info():
    """Check remaining credits if possible."""
    try:
        from firecrawl import FirecrawlApp
        
        api_key = os.getenv('FIRECRAWL_API_KEY')
        app = FirecrawlApp(api_key=api_key)
        
        # Note: This may not work depending on the API version
        print("🔄 Attempting to check credits...")
        
        # Try a small operation to see response
        result = app.scrape("https://httpbin.org/status/200")
        print("✅ Small operation successful - API is working")
        
    except Exception as e:
        print(f"ℹ️ Credits check not available: {str(e)}")

if __name__ == "__main__":
    print("🔥 Firecrawl API Key Test")
    print("=" * 30)
    
    if test_firecrawl_api():
        print("\n🎉 Your Firecrawl API key is working correctly!")
        test_credits_info()
    else:
        print("\n💸 Your API key may not be working - check your subscription status")
        
    print("\n💡 Tips:")
    print("  • Check your Firecrawl dashboard for usage limits")
    print("  • Free tier has limited requests per month")
    print("  • Monitor your usage to avoid unexpected charges")