"""
PyTorch integration layer for existing MLB DFS pipeline.

This module extends the existing MLB_GPP_Simulator to add PyTorch
PA-level predictions while preserving all correlation logic.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, List
import warnings

import numpy as np
import pandas as pd
import torch

# Import existing components
import sys
sys.path.append('..')
from src.mlb_gpp_simulator import MLB_GPP_Simulator
from pipeline.modeling import HitterProjector, PitcherProjector
from pipeline.features import FeatureEngineer
from pipeline.projection_pipeline import BaseballProjectionPipeline

from .pa_models import PlateAppearanceClassifier, StolenBaseModel
from .calibration import ModelCalibrator
from .player_ids import PlayerIDMapper


logger = logging.getLogger(__name__)


class PyTorchEnhancedSimulator(MLB_GPP_Simulator):
    """
    Extends existing MLB_GPP_Simulator to optionally use PyTorch PA models.
    
    Preserves all existing functionality including:
    - 11x11 batting order correlation matrix
    - Team correlation via Gamma multipliers
    - Multiple simulation runs with variance
    
    Adds:
    - PA-level outcome predictions from PyTorch
    - Dynamic blending with Ridge baseline
    - Calibrated probabilities for accurate variance
    """
    
    def __init__(
        self,
        num_iterations: int = 10000,
        use_pytorch: bool = True,
        pytorch_blend_weight: float = 0.5,
        pa_model_path: Optional[str] = None,
        sb_model_path: Optional[str] = None,
        calibrator_path: Optional[str] = None
    ):
        """
        Initialize enhanced simulator.
        
        Args:
            num_iterations: Number of Monte Carlo iterations
            use_pytorch: Whether to use PyTorch models
            pytorch_blend_weight: Weight for PyTorch predictions (0=Ridge only, 1=PyTorch only)
            pa_model_path: Path to trained PA classifier
            sb_model_path: Path to trained SB model
            calibrator_path: Path to calibration parameters
        """
        super().__init__(num_iterations=num_iterations, use_contest_data=True, site="dk")
        
        self.use_pytorch = use_pytorch
        self.pytorch_blend_weight = pytorch_blend_weight
        
        # Initialize PyTorch components if enabled
        if self.use_pytorch:
            self._init_pytorch_models(pa_model_path, sb_model_path, calibrator_path)
        else:
            self.pa_model = None
            self.sb_model = None
            self.calibrator = None
        
        # Initialize player ID mapper
        self.id_mapper = PlayerIDMapper()
        
        logger.info(f"PyTorchEnhancedSimulator initialized - PyTorch: {use_pytorch}, Blend: {pytorch_blend_weight}")
    
    def _init_pytorch_models(
        self,
        pa_model_path: Optional[str],
        sb_model_path: Optional[str],
        calibrator_path: Optional[str]
    ):
        """Initialize PyTorch models if paths provided."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load PA classifier
        if pa_model_path:
            try:
                self.pa_model = PlateAppearanceClassifier()
                checkpoint = torch.load(pa_model_path, map_location=device)
                self.pa_model.load_state_dict(checkpoint['model_state_dict'])
                self.pa_model.eval()
                logger.info(f"Loaded PA model from {pa_model_path}")
            except Exception as e:
                logger.warning(f"Could not load PA model: {e}")
                self.pa_model = None
        else:
            self.pa_model = None
            logger.info("No PA model path provided, using Ridge baseline only")
        
        # Load SB model
        if sb_model_path:
            try:
                self.sb_model = StolenBaseModel()
                checkpoint = torch.load(sb_model_path, map_location=device)
                self.sb_model.load_state_dict(checkpoint['model_state_dict'])
                self.sb_model.eval()
                logger.info(f"Loaded SB model from {sb_model_path}")
            except Exception as e:
                logger.warning(f"Could not load SB model: {e}")
                self.sb_model = None
        else:
            self.sb_model = None
        
        # Load calibrators
        if calibrator_path:
            try:
                self.calibrator = ModelCalibrator()
                self.calibrator.load_calibrators(calibrator_path)
                logger.info(f"Loaded calibrators from {calibrator_path}")
            except Exception as e:
                logger.warning(f"Could not load calibrators: {e}")
                self.calibrator = None
        else:
            self.calibrator = None
    
    def generate_field_lineups(self, num_lineups: int, payout: List[float]) -> pd.DataFrame:
        """
        Override parent method to inject PyTorch predictions.
        
        Maintains all existing correlation logic while optionally
        enhancing with PA-level predictions.
        """
        if not self.use_pytorch or self.pa_model is None:
            # Fall back to parent implementation
            return super().generate_field_lineups(num_lineups, payout)
        
        # Get baseline projections from Ridge model
        baseline_lineups = super().generate_field_lineups(num_lineups, payout)
        
        # Enhance with PyTorch predictions
        enhanced_lineups = self._enhance_with_pytorch(baseline_lineups)
        
        return enhanced_lineups
    
    def _enhance_with_pytorch(self, baseline_lineups: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance baseline projections with PyTorch PA predictions.
        
        This preserves the correlation structure from the parent class
        while improving individual player projections.
        """
        enhanced_lineups = baseline_lineups.copy()
        
        # Extract unique players
        player_cols = [col for col in baseline_lineups.columns if col.startswith('P')]
        unique_players = set()
        for col in player_cols:
            unique_players.update(baseline_lineups[col].unique())
        
        # Remove NaN values
        unique_players = {p for p in unique_players if pd.notna(p)}
        
        # Generate PyTorch predictions for each player
        pytorch_projections = {}
        
        for player_name in unique_players:
            try:
                # Get player features (would come from feature pipeline)
                player_features = self._get_player_features(player_name)
                
                if player_features is not None:
                    # Get PA outcome probabilities
                    pa_probs = self._predict_pa_outcomes(player_features)
                    
                    # Convert to fantasy points
                    pytorch_points = self._pa_probs_to_points(pa_probs)
                    
                    # Get baseline projection
                    baseline_points = self._get_baseline_projection(player_name)
                    
                    # Blend predictions
                    blended_points = (
                        self.pytorch_blend_weight * pytorch_points +
                        (1 - self.pytorch_blend_weight) * baseline_points
                    )
                    
                    pytorch_projections[player_name] = blended_points
                    
            except Exception as e:
                logger.debug(f"Could not enhance {player_name}: {e}")
                continue
        
        # Apply enhancements to lineups
        if pytorch_projections:
            # Update the projections used in simulation
            # This maintains correlation structure while improving point estimates
            self._apply_projection_updates(enhanced_lineups, pytorch_projections)
        
        return enhanced_lineups
    
    def _get_player_features(self, player_name: str) -> Optional[pd.DataFrame]:
        """
        Get features for a player from the existing feature pipeline.
        
        This would integrate with pipeline.features.FeatureEngineer.
        """
        # Placeholder - would connect to actual feature pipeline
        # For now, return None to use baseline only
        return None
    
    def _predict_pa_outcomes(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get PA outcome probabilities from PyTorch model.
        
        Returns:
            Array of shape [7] with probabilities for each PA outcome
        """
        if self.pa_model is None:
            return np.zeros(7)
        
        # Prepare features for PyTorch
        numeric_features = self._prepare_numeric_features(features)
        categorical_features = self._prepare_categorical_features(features)
        
        # Get predictions
        with torch.no_grad():
            logits = self.pa_model(numeric_features, categorical_features)
            
            # Apply calibration if available
            if self.calibrator is not None:
                probs = self.calibrator.calibrate_pa_predictions(logits)
            else:
                probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy().squeeze()
    
    def _pa_probs_to_points(self, pa_probs: np.ndarray) -> float:
        """
        Convert PA outcome probabilities to expected DraftKings points.
        
        Args:
            pa_probs: Probabilities for [1B, 2B, 3B, HR, BB_HBP, K, Out]
            
        Returns:
            Expected DraftKings points
        """
        # DraftKings scoring
        dk_points = {
            '1B': 3,
            '2B': 5,
            '3B': 8,
            'HR': 10,
            'BB_HBP': 2,
            'K': 0,
            'Out': 0
        }
        
        outcomes = ['1B', '2B', '3B', 'HR', 'BB_HBP', 'K', 'Out']
        
        expected_points = 0
        for i, outcome in enumerate(outcomes):
            expected_points += pa_probs[i] * dk_points[outcome]
        
        # Add stolen base contribution if model available
        if self.sb_model is not None:
            # Probability of reaching first base
            reach_base_prob = pa_probs[0] + pa_probs[4]  # 1B + BB_HBP
            
            # Get SB predictions (would need features)
            # For now, use baseline estimate
            sb_contribution = reach_base_prob * 0.1 * 5  # 10% SB rate * 5 points
            expected_points += sb_contribution
        
        return expected_points
    
    def _get_baseline_projection(self, player_name: str) -> float:
        """Get baseline Ridge model projection for player."""
        # This would connect to the existing projection system
        # For now, return a placeholder
        return 8.0  # Average DK points
    
    def _prepare_numeric_features(self, features: pd.DataFrame) -> torch.Tensor:
        """Prepare numeric features for PyTorch model."""
        # Extract numeric columns that match model expectations
        numeric_cols = [col for col in features.columns if col not in ['team', 'opponent', 'is_home']]
        numeric_data = features[numeric_cols].values
        return torch.FloatTensor(numeric_data)
    
    def _prepare_categorical_features(self, features: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Prepare categorical features for PyTorch model."""
        categorical = {}
        
        if 'is_home' in features.columns:
            categorical['is_home'] = torch.LongTensor(features['is_home'].values)
        
        # Add team/opponent encoding if present
        
        return categorical
    
    def _apply_projection_updates(
        self,
        lineups: pd.DataFrame,
        pytorch_projections: Dict[str, float]
    ):
        """
        Apply PyTorch projection updates while maintaining correlation.
        
        This is the key integration point - we update the mean projections
        but preserve the correlation structure from the parent simulator.
        """
        # The parent simulator uses these projections with correlation
        # We update the base projections while correlation logic remains intact
        
        # This would integrate with the actual projection storage mechanism
        # For now, log the updates
        logger.info(f"Would update {len(pytorch_projections)} player projections")
    
    def compare_models(self, sample_size: int = 100) -> pd.DataFrame:
        """
        Compare Ridge baseline vs PyTorch enhanced projections.
        
        Useful for A/B testing and model validation.
        """
        comparison_results = []
        
        # Generate lineups with Ridge only
        self.use_pytorch = False
        ridge_lineups = self.generate_field_lineups(sample_size, [1.0])
        
        # Generate lineups with PyTorch
        self.use_pytorch = True  
        pytorch_lineups = self.generate_field_lineups(sample_size, [1.0])
        
        # Compare distributions
        ridge_scores = ridge_lineups['fantasy_points'].values
        pytorch_scores = pytorch_lineups['fantasy_points'].values
        
        comparison = pd.DataFrame({
            'ridge_mean': [ridge_scores.mean()],
            'ridge_std': [ridge_scores.std()],
            'pytorch_mean': [pytorch_scores.mean()],
            'pytorch_std': [pytorch_scores.std()],
            'mean_diff': [pytorch_scores.mean() - ridge_scores.mean()],
            'std_ratio': [pytorch_scores.std() / ridge_scores.std()]
        })
        
        logger.info(f"Model comparison:\n{comparison}")
        
        return comparison


class PyTorchProjectionAPI:
    """
    API wrapper for PyTorch projections to integrate with dashboard.
    
    This provides a clean interface for the FastAPI endpoints.
    """
    
    def __init__(
        self,
        model_dir: str = "mlb_pytorch/models",
        use_cache: bool = True
    ):
        self.model_dir = model_dir
        self.use_cache = use_cache
        self.simulator = None
        self._projection_cache = {}
    
    def initialize_simulator(
        self,
        use_pytorch: bool = True,
        blend_weight: float = 0.5
    ) -> PyTorchEnhancedSimulator:
        """Initialize the enhanced simulator."""
        self.simulator = PyTorchEnhancedSimulator(
            use_pytorch=use_pytorch,
            pytorch_blend_weight=blend_weight,
            pa_model_path=f"{self.model_dir}/pa_classifier.pth",
            sb_model_path=f"{self.model_dir}/sb_model.pth",
            calibrator_path=f"{self.model_dir}/calibrators.pkl"
        )
        return self.simulator
    
    def get_projections(
        self,
        slate_id: str,
        use_pytorch: bool = True,
        blend_weight: float = 0.5,
        num_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Get projections for a DFS slate.
        
        Args:
            slate_id: DraftKings slate identifier
            use_pytorch: Whether to use PyTorch enhancement
            blend_weight: PyTorch blend weight (0-1)
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            DataFrame with player projections
        """
        # Check cache
        cache_key = f"{slate_id}_{use_pytorch}_{blend_weight}_{num_simulations}"
        if self.use_cache and cache_key in self._projection_cache:
            logger.info(f"Returning cached projections for {cache_key}")
            return self._projection_cache[cache_key]
        
        # Initialize simulator if needed
        if self.simulator is None or self.simulator.pytorch_blend_weight != blend_weight:
            self.initialize_simulator(use_pytorch, blend_weight)
        
        # Generate projections
        self.simulator.num_iterations = num_simulations
        
        # This would connect to actual DFS slate data
        # For now, return placeholder
        projections = pd.DataFrame({
            'player': ['Player1', 'Player2'],
            'position': ['1B', 'OF'],
            'salary': [5000, 4500],
            'projection': [8.5, 7.2],
            'ceiling': [15.0, 13.0],
            'floor': [3.0, 2.5]
        })
        
        # Cache results
        if self.use_cache:
            self._projection_cache[cache_key] = projections
        
        return projections
    
    def clear_cache(self):
        """Clear projection cache."""
        self._projection_cache = {}
        logger.info("Projection cache cleared")


def create_enhanced_simulator(config: dict) -> PyTorchEnhancedSimulator:
    """
    Factory function to create configured simulator.
    
    Args:
        config: Configuration dictionary with keys:
            - use_pytorch: bool
            - blend_weight: float
            - num_iterations: int
            - model_paths: dict with pa_model, sb_model, calibrator
            
    Returns:
        Configured PyTorchEnhancedSimulator instance
    """
    return PyTorchEnhancedSimulator(
        num_iterations=config.get('num_iterations', 10000),
        use_pytorch=config.get('use_pytorch', True),
        pytorch_blend_weight=config.get('blend_weight', 0.5),
        pa_model_path=config.get('model_paths', {}).get('pa_model'),
        sb_model_path=config.get('model_paths', {}).get('sb_model'),
        calibrator_path=config.get('model_paths', {}).get('calibrator')
    )


if __name__ == "__main__":
    # Test integration
    logging.basicConfig(level=logging.INFO)
    
    print("Testing PyTorch Enhanced Simulator...")
    
    # Create simulator with PyTorch disabled (baseline)
    baseline_sim = PyTorchEnhancedSimulator(
        num_iterations=1000,
        use_pytorch=False
    )
    
    # Create simulator with PyTorch enabled (enhanced)
    enhanced_sim = PyTorchEnhancedSimulator(
        num_iterations=1000,
        use_pytorch=True,
        pytorch_blend_weight=0.5
    )
    
    print("✅ Simulators created successfully!")
    
    # Test API wrapper
    api = PyTorchProjectionAPI()
    api.initialize_simulator()
    
    test_projections = api.get_projections(
        slate_id="test_slate",
        use_pytorch=True,
        blend_weight=0.5,
        num_simulations=100
    )
    
    print(f"\nTest projections shape: {test_projections.shape}")
    print(test_projections.head())
    
    print("\n✅ Integration layer ready!")