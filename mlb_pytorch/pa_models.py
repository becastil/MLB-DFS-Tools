"""
PyTorch plate appearance models.

Builds on existing feature engineering from pipeline.features and 
pipeline.modeling to create PA-level classification models.

Two model heads:
1. Classification: Predicts PA outcome probabilities 
2. Regression: Direct fantasy points prediction (baseline)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Import existing components to reuse
import sys
sys.path.append('..')
from pipeline.modeling import HITTER_FEATURES, PITCHER_FEATURES
from pipeline.features import FeatureEngineer
from pipeline.scoring import calculate_hitter_points, calculate_pitcher_points


logger = logging.getLogger(__name__)


class PlateAppearanceClassifier(nn.Module):
    """
    7-class plate appearance outcome classifier.
    
    Uses existing hitter features from pipeline.modeling.HITTER_FEATURES
    and extends with PyTorch-specific enhancements.
    """
    
    # DFS-aligned PA outcomes (excluding stolen bases - handled separately)
    PA_OUTCOMES = ['1B', '2B', '3B', 'HR', 'BB_HBP', 'K', 'Out']
    NUM_OUTCOMES = len(PA_OUTCOMES)
    
    def __init__(
        self,
        numeric_features: int = 25,
        embedding_dims: Dict[str, Tuple[int, int]] = None,
        hidden_sizes: List[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.numeric_features = numeric_features
        self.embedding_dims = embedding_dims or {
            'team': (30, 8),      # 30 teams, 8-dim embedding
            'opponent': (30, 8),
            'is_home': (2, 1),    # Home/away
        }
        self.hidden_sizes = hidden_sizes or [128, 64, 32]
        self.dropout = dropout
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        embedding_total_dim = 0
        
        for feature, (vocab_size, embed_dim) in self.embedding_dims.items():
            self.embeddings[feature] = nn.Embedding(vocab_size, embed_dim)
            embedding_total_dim += embed_dim
        
        # Input dimension = numeric + embeddings
        input_dim = self.numeric_features + embedding_total_dim
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.NUM_OUTCOMES))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, numeric_inputs: torch.Tensor, categorical_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            numeric_inputs: Shape [batch_size, numeric_features]
            categorical_inputs: Dict of category tensors
            
        Returns:
            Logits for PA outcomes: Shape [batch_size, NUM_OUTCOMES]
        """
        # Process embeddings
        embedded_features = []
        
        for feature_name, embedding_layer in self.embeddings.items():
            if feature_name in categorical_inputs:
                embedded = embedding_layer(categorical_inputs[feature_name])
                embedded_features.append(embedded)
        
        # Concatenate numeric and embedded features
        if embedded_features:
            embedded_concat = torch.cat(embedded_features, dim=1)
            combined = torch.cat([numeric_inputs, embedded_concat], dim=1)
        else:
            combined = numeric_inputs
        
        # Forward through MLP
        logits = self.mlp(combined)
        
        return logits
    
    def predict_proba(self, numeric_inputs: torch.Tensor, categorical_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(numeric_inputs, categorical_inputs)
            probs = F.softmax(logits, dim=1)
            return probs
    
    def get_feature_names(self) -> List[str]:
        """Get list of expected numeric feature names (from existing pipeline)."""
        # Use existing hitter features as base
        return [feat for feat in HITTER_FEATURES if feat not in ['is_home']]  # is_home is categorical


class StolenBaseModel(nn.Module):
    """
    Conditional stolen base model.
    
    Only predicts SB for players who reach first base (1B or BB/HBP).
    Two-stage: attempt probability, then success probability.
    """
    
    def __init__(
        self,
        numeric_features: int = 15,
        hidden_sizes: List[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.numeric_features = numeric_features
        self.hidden_sizes = hidden_sizes or [32, 16]
        
        # Attempt model (will player attempt SB?)
        attempt_layers = []
        prev_dim = numeric_features
        
        for hidden_dim in self.hidden_sizes:
            attempt_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        attempt_layers.append(nn.Linear(prev_dim, 1))  # Binary classification
        self.attempt_head = nn.Sequential(*attempt_layers)
        
        # Success model (given attempt, will it succeed?)
        success_layers = []
        prev_dim = numeric_features
        
        for hidden_dim in self.hidden_sizes:
            success_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        success_layers.append(nn.Linear(prev_dim, 1))  # Binary classification
        self.success_head = nn.Sequential(*success_layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Features [batch_size, numeric_features]
            
        Returns:
            (attempt_logits, success_logits): Both [batch_size, 1]
        """
        attempt_logits = self.attempt_head(x)
        success_logits = self.success_head(x)
        
        return attempt_logits, success_logits
    
    def predict_sb_proba(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get SB attempt and success probabilities."""
        with torch.no_grad():
            attempt_logits, success_logits = self.forward(x)
            attempt_prob = torch.sigmoid(attempt_logits)
            success_prob = torch.sigmoid(success_logits)
            return attempt_prob, success_prob


class DirectRegressionModel(nn.Module):
    """
    Direct fantasy points regression model (baseline).
    
    Uses same features as classification model but predicts DK points directly.
    """
    
    def __init__(
        self,
        numeric_features: int = 25,
        embedding_dims: Dict[str, Tuple[int, int]] = None,
        hidden_sizes: List[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.numeric_features = numeric_features
        self.embedding_dims = embedding_dims or {
            'team': (30, 8),
            'opponent': (30, 8),
            'is_home': (2, 1),
        }
        self.hidden_sizes = hidden_sizes or [128, 64, 32]
        
        # Same architecture as classifier but single output
        self.embeddings = nn.ModuleDict()
        embedding_total_dim = 0
        
        for feature, (vocab_size, embed_dim) in self.embedding_dims.items():
            self.embeddings[feature] = nn.Embedding(vocab_size, embed_dim)
            embedding_total_dim += embed_dim
        
        input_dim = self.numeric_features + embedding_total_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Single output for DK points
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, numeric_inputs: torch.Tensor, categorical_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning DK points prediction."""
        # Same embedding logic as classifier
        embedded_features = []
        
        for feature_name, embedding_layer in self.embeddings.items():
            if feature_name in categorical_inputs:
                embedded = embedding_layer(categorical_inputs[feature_name])
                embedded_features.append(embedded)
        
        if embedded_features:
            embedded_concat = torch.cat(embedded_features, dim=1)
            combined = torch.cat([numeric_inputs, embedded_concat], dim=1)
        else:
            combined = numeric_inputs
        
        points = self.mlp(combined)
        return points.squeeze(-1)  # [batch_size]


class PAModelTrainer:
    """Trainer class for PyTorch PA models using existing data pipeline."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scaler = StandardScaler()
        
        # Track training history
        self.train_losses = []
        self.val_losses = []
    
    def prepare_features(self, feature_df: pd.DataFrame) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert feature DataFrame to PyTorch tensors.
        
        Uses existing HITTER_FEATURES from pipeline.modeling.
        """
        # Get numeric features (exclude categorical ones)
        numeric_cols = [col for col in HITTER_FEATURES 
                       if col in feature_df.columns and col != 'is_home']
        
        numeric_data = feature_df[numeric_cols].values
        numeric_tensor = torch.FloatTensor(numeric_data).to(self.device)
        
        # Categorical features
        categorical_tensors = {}
        
        if 'is_home' in feature_df.columns:
            categorical_tensors['is_home'] = torch.LongTensor(
                feature_df['is_home'].values
            ).to(self.device)
        
        if 'team' in feature_df.columns:
            # Convert team names to indices
            unique_teams = feature_df['team'].unique()
            team_to_idx = {team: idx for idx, team in enumerate(unique_teams)}
            team_indices = feature_df['team'].map(team_to_idx).fillna(0)
            categorical_tensors['team'] = torch.LongTensor(team_indices.values).to(self.device)
        
        if 'opponent' in feature_df.columns:
            # Convert opponent names to indices
            unique_opponents = feature_df['opponent'].unique()
            opp_to_idx = {opp: idx for idx, opp in enumerate(unique_opponents)}
            opp_indices = feature_df['opponent'].map(opp_to_idx).fillna(0)
            categorical_tensors['opponent'] = torch.LongTensor(opp_indices.values).to(self.device)
        
        return numeric_tensor, categorical_tensors
    
    def train_classification(
        self,
        train_features: pd.DataFrame,
        train_outcomes: pd.DataFrame,
        val_features: pd.DataFrame,
        val_outcomes: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 512
    ) -> Dict[str, List[float]]:
        """Train PA classification model."""
        
        # Prepare training data
        X_train_num, X_train_cat = self.prepare_features(train_features)
        y_train = torch.LongTensor(train_outcomes.values).to(self.device)
        
        X_val_num, X_val_cat = self.prepare_features(val_features)
        y_val = torch.LongTensor(val_outcomes.values).to(self.device)
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            # Training
            total_train_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train_num), batch_size):
                batch_end = min(i + batch_size, len(X_train_num))
                
                batch_num = X_train_num[i:batch_end]
                batch_cat = {k: v[i:batch_end] for k, v in X_train_cat.items()}
                batch_y = y_train[i:batch_end]
                
                self.optimizer.zero_grad()
                
                logits = self.model(batch_num, batch_cat)
                loss = F.cross_entropy(logits, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_num, X_val_cat)
                val_loss = F.cross_entropy(val_logits, y_val).item()
            
            self.model.train()
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, filepath: Path | str) -> None:
        """Save model state dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path | str) -> None:
        """Load model state dict."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Model loaded from {filepath}")


def create_pa_outcomes_from_stats(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert box score stats to PA outcome labels.
    
    Uses existing scoring logic to create training targets.
    """
    outcomes = []
    
    for _, row in game_logs.iterrows():
        # Extract basic stats
        singles = max(row.get('bat_hits', 0) - row.get('bat_doubles', 0) - 
                     row.get('bat_triples', 0) - row.get('bat_homeRuns', 0), 0)
        
        doubles = row.get('bat_doubles', 0)
        triples = row.get('bat_triples', 0) 
        home_runs = row.get('bat_homeRuns', 0)
        bb = row.get('bat_baseOnBalls', 0)
        hbp = row.get('bat_hitByPitch', 0)
        so = row.get('bat_strikeOuts', 0)
        pas = row.get('bat_plateAppearances', 0)
        
        # Simple approximation of PA outcomes
        total_events = singles + doubles + triples + home_runs + bb + hbp + so
        outs = max(pas - total_events, 0) if pas > 0 else 0
        
        # Create outcome distribution for this game
        player_outcomes = {
            'player_name': row.get('player_name', ''),
            'game_datetime': row.get('game_datetime'),
            '1B': singles,
            '2B': doubles, 
            '3B': triples,
            'HR': home_runs,
            'BB_HBP': bb + hbp,
            'K': so,
            'Out': outs,
            'total_pas': pas
        }
        
        outcomes.append(player_outcomes)
    
    return pd.DataFrame(outcomes)


if __name__ == "__main__":
    # Test model creation
    logging.basicConfig(level=logging.INFO)
    
    # Create models
    pa_classifier = PlateAppearanceClassifier()
    sb_model = StolenBaseModel()
    regression_model = DirectRegressionModel()
    
    print(f"PA Classifier parameters: {sum(p.numel() for p in pa_classifier.parameters())}")
    print(f"SB Model parameters: {sum(p.numel() for p in sb_model.parameters())}")
    print(f"Regression Model parameters: {sum(p.numel() for p in regression_model.parameters())}")
    
    print("\nModel architectures created successfully!")
    print(f"Expected PA outcomes: {pa_classifier.PA_OUTCOMES}")
    print(f"Expected numeric features: {pa_classifier.numeric_features}")