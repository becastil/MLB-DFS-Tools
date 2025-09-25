"""
Probability calibration for PyTorch models.

Implements temperature scaling and isotonic regression for post-hoc calibration
of PA outcome probabilities. Essential for Monte Carlo simulation accuracy.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.
    
    Learns a single temperature parameter T to scale logits before softmax:
    calibrated_probs = softmax(logits / T)
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
        """
        Learn temperature parameter on validation set.
        
        Args:
            logits: Model outputs before softmax [batch_size, num_classes]
            labels: True class labels [batch_size]
            max_iter: Maximum optimization iterations
            
        Returns:
            Final validation NLL loss
        """
        # Use LBFGS optimizer for temperature parameter
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        # Return final loss
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            final_loss = F.cross_entropy(scaled_logits, labels).item()
        
        logger.info(f"Temperature scaling fitted: T={self.temperature.item():.4f}, NLL={final_loss:.4f}")
        return final_loss
    
    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Get calibrated probabilities."""
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            return F.softmax(scaled_logits, dim=1)


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for binary probabilities.
    
    Useful for stolen base model calibration.
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False
    
    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        """
        Fit isotonic regression to probabilities.
        
        Args:
            probs: Predicted probabilities [N]
            y_true: True binary labels [N]
        """
        self.calibrator.fit(probs, y_true)
        self.fitted = True
        
        logger.info("Isotonic calibrator fitted")
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to probabilities."""
        if not self.fitted:
            logger.warning("Calibrator not fitted, returning original probabilities")
            return probs
        
        return self.calibrator.predict(probs)


def compute_reliability_diagram(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram for calibration assessment.
    
    Args:
        y_true: True binary labels [N]
        y_prob: Predicted probabilities [N]  
        n_bins: Number of bins for reliability curve
        
    Returns:
        (bin_boundaries, bin_lowers, bin_uppers): 
        - bin_boundaries: Bin boundary points
        - bin_lowers: Lower confidence bound per bin
        - bin_uppers: Upper confidence bound per bin
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = (bin_lowers + bin_uppers) / 2
    bin_true_probs = []
    bin_pred_probs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_true_prob = y_true[in_bin].mean()
            bin_pred_prob = y_prob[in_bin].mean()
            bin_count = in_bin.sum()
        else:
            bin_true_prob = 0.0
            bin_pred_prob = bin_lower  # Assign bin center
            bin_count = 0
        
        bin_true_probs.append(bin_true_prob)
        bin_pred_probs.append(bin_pred_prob)
        bin_counts.append(bin_count)
    
    return np.array(bin_pred_probs), np.array(bin_true_probs), np.array(bin_counts)


def compute_expected_calibration_error(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels [N]
        y_prob: Predicted probabilities [N]
        n_bins: Number of bins
        
    Returns:
        ECE score (lower is better)
    """
    bin_pred_probs, bin_true_probs, bin_counts = compute_reliability_diagram(
        y_true, y_prob, n_bins
    )
    
    total_count = len(y_true)
    ece = 0.0
    
    for pred_prob, true_prob, count in zip(bin_pred_probs, bin_true_probs, bin_counts):
        if count > 0:
            ece += (count / total_count) * abs(pred_prob - true_prob)
    
    return ece


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute multi-class Brier score.
    
    Args:
        y_true: True class labels [N] (0 to num_classes-1)
        y_prob: Predicted probabilities [N, num_classes]
        
    Returns:
        Brier score (lower is better)
    """
    n_classes = y_prob.shape[1]
    
    # Convert labels to one-hot
    y_true_onehot = np.eye(n_classes)[y_true]
    
    # Compute Brier score
    brier = np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
    
    return brier


def evaluate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    task: str = 'multiclass'
) -> dict:
    """
    Comprehensive calibration evaluation.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities  
        task: 'binary' or 'multiclass'
        
    Returns:
        Dict of calibration metrics
    """
    metrics = {}
    
    if task == 'binary':
        # Binary classification metrics
        metrics['ece'] = compute_expected_calibration_error(y_true, y_prob)
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        
        # Reliability diagram data
        bin_pred, bin_true, bin_counts = compute_reliability_diagram(y_true, y_prob)
        metrics['reliability_diagram'] = {
            'bin_pred_probs': bin_pred.tolist(),
            'bin_true_probs': bin_true.tolist(),
            'bin_counts': bin_counts.tolist()
        }
        
    elif task == 'multiclass':
        # Multi-class metrics
        if y_prob.ndim == 2:
            metrics['brier_score'] = multiclass_brier_score(y_true, y_prob)
            
            # ECE for predicted class probabilities
            pred_class_probs = np.max(y_prob, axis=1)
            pred_classes = np.argmax(y_prob, axis=1)
            correct = (pred_classes == y_true).astype(int)
            
            metrics['ece'] = compute_expected_calibration_error(correct, pred_class_probs)
        else:
            logger.warning("Invalid probability array shape for multiclass evaluation")
    
    return metrics


class ModelCalibrator:
    """
    Combined calibrator for PyTorch PA models.
    
    Handles both classification and regression model calibration.
    """
    
    def __init__(self):
        self.temperature_scaler: Optional[TemperatureScaling] = None
        self.sb_calibrators: dict = {}  # For SB attempt/success calibration
        self.fitted = False
    
    def fit_pa_classifier(
        self,
        model: nn.Module,
        val_features_num: torch.Tensor,
        val_features_cat: dict,
        val_labels: torch.Tensor
    ) -> dict:
        """
        Fit temperature scaling for PA classifier.
        
        Args:
            model: Trained PA classifier
            val_features_num: Validation numeric features
            val_features_cat: Validation categorical features
            val_labels: Validation PA outcome labels
            
        Returns:
            Calibration metrics
        """
        model.eval()
        
        with torch.no_grad():
            logits = model(val_features_num, val_features_cat)
        
        # Fit temperature scaling
        self.temperature_scaler = TemperatureScaling()
        final_loss = self.temperature_scaler.fit(logits, val_labels)
        
        # Evaluate calibration
        calibrated_logits = self.temperature_scaler(logits)
        calibrated_probs = F.softmax(calibrated_logits, dim=1)
        
        metrics = evaluate_calibration_metrics(
            val_labels.cpu().numpy(),
            calibrated_probs.cpu().numpy(),
            task='multiclass'
        )
        
        self.fitted = True
        
        logger.info(f"PA classifier calibrated: Brier={metrics['brier_score']:.4f}, ECE={metrics['ece']:.4f}")
        
        return metrics
    
    def fit_sb_model(
        self,
        attempt_probs: np.ndarray,
        attempt_labels: np.ndarray,
        success_probs: np.ndarray,
        success_labels: np.ndarray
    ) -> dict:
        """
        Fit isotonic calibration for SB model.
        
        Args:
            attempt_probs: SB attempt probabilities
            attempt_labels: True SB attempt labels
            success_probs: SB success probabilities  
            success_labels: True SB success labels
            
        Returns:
            Calibration metrics for both heads
        """
        # Fit calibrators
        attempt_calibrator = IsotonicCalibrator()
        attempt_calibrator.fit(attempt_probs, attempt_labels)
        
        success_calibrator = IsotonicCalibrator()
        success_calibrator.fit(success_probs, success_labels)
        
        self.sb_calibrators = {
            'attempt': attempt_calibrator,
            'success': success_calibrator
        }
        
        # Evaluate calibration
        attempt_cal_probs = attempt_calibrator.calibrate(attempt_probs)
        success_cal_probs = success_calibrator.calibrate(success_probs)
        
        attempt_metrics = evaluate_calibration_metrics(
            attempt_labels, attempt_cal_probs, task='binary'
        )
        success_metrics = evaluate_calibration_metrics(
            success_labels, success_cal_probs, task='binary'
        )
        
        logger.info(f"SB model calibrated - Attempt ECE: {attempt_metrics['ece']:.4f}, "
                   f"Success ECE: {success_metrics['ece']:.4f}")
        
        return {
            'attempt': attempt_metrics,
            'success': success_metrics
        }
    
    def calibrate_pa_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get calibrated PA outcome probabilities."""
        if self.temperature_scaler is None:
            logger.warning("Temperature scaler not fitted, returning uncalibrated probabilities")
            return F.softmax(logits, dim=1)
        
        return self.temperature_scaler.calibrate_probs(logits)
    
    def calibrate_sb_predictions(
        self, 
        attempt_probs: np.ndarray, 
        success_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibrated SB probabilities."""
        if 'attempt' not in self.sb_calibrators or 'success' not in self.sb_calibrators:
            logger.warning("SB calibrators not fitted, returning uncalibrated probabilities")
            return attempt_probs, success_probs
        
        cal_attempt = self.sb_calibrators['attempt'].calibrate(attempt_probs)
        cal_success = self.sb_calibrators['success'].calibrate(success_probs)
        
        return cal_attempt, cal_success
    
    def save_calibrators(self, filepath: str) -> None:
        """Save calibration parameters."""
        import joblib
        
        calibration_data = {
            'temperature_scaler': self.temperature_scaler.state_dict() if self.temperature_scaler else None,
            'sb_calibrators': self.sb_calibrators,
            'fitted': self.fitted
        }
        
        joblib.dump(calibration_data, filepath)
        logger.info(f"Calibrators saved to {filepath}")
    
    def load_calibrators(self, filepath: str) -> None:
        """Load calibration parameters."""
        import joblib
        
        calibration_data = joblib.load(filepath)
        
        if calibration_data['temperature_scaler']:
            self.temperature_scaler = TemperatureScaling()
            self.temperature_scaler.load_state_dict(calibration_data['temperature_scaler'])
        
        self.sb_calibrators = calibration_data['sb_calibrators']
        self.fitted = calibration_data['fitted']
        
        logger.info(f"Calibrators loaded from {filepath}")


if __name__ == "__main__":
    # Test calibration functionality
    logging.basicConfig(level=logging.INFO)
    
    # Test temperature scaling
    print("Testing Temperature Scaling...")
    
    # Generate sample data
    torch.manual_seed(42)
    logits = torch.randn(1000, 7)  # 7 PA outcomes
    labels = torch.randint(0, 7, (1000,))
    
    temp_scaler = TemperatureScaling()
    loss = temp_scaler.fit(logits, labels)
    
    calibrated_probs = temp_scaler.calibrate_probs(logits)
    
    print(f"Temperature: {temp_scaler.temperature.item():.4f}")
    print(f"Final loss: {loss:.4f}")
    print(f"Calibrated probs shape: {calibrated_probs.shape}")
    
    # Test multi-class Brier score
    y_true_np = labels.numpy()
    y_prob_np = calibrated_probs.numpy()
    
    brier = multiclass_brier_score(y_true_np, y_prob_np)
    print(f"Multi-class Brier score: {brier:.4f}")
    
    print("✅ Calibration module test completed!")