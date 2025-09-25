"""
MLB DFS PyTorch Pipeline

Extends existing pipeline with PyTorch plate appearance models,
Monte Carlo simulation with team correlation, and advanced blending.

Builds on:
- pipeline.scoring (DK/FD scoring)
- pipeline.features (feature engineering) 
- pipeline.data_sources (MLB Stats API, Firecrawl)
- pipeline.projection_pipeline (base pipeline)
"""

__version__ = "1.0.0"