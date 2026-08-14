import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import joblib

from src.config import ARTIFACTS_DIR


class ProbabilityCalibrator:
    """Handles Isotonic Regression and Platt (Sigmoid) Scaling for raw classifier outputs."""

    def __init__(self, method: str = "isotonic"):
        if method not in ["isotonic", "sigmoid", "platt"]:
            raise ValueError(f"Invalid calibration method: {method}. Must be 'isotonic' or 'sigmoid' / 'platt'.")
        self.method = "sigmoid" if method == "platt" else method
        self.calibrator: Any = None
        self.base_model: Any = None

    def fit(self, base_estimator: Any, X_val: np.ndarray, y_val: np.ndarray):
        """Fit the calibrator on validation predictions."""
        self.base_model = base_estimator
        raw_val_probs = self.base_model.predict_proba(X_val)[:, 1]

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_val_probs, y_val)
        else:  # sigmoid / platt
            self.calibrator = LogisticRegression()
            self.calibrator.fit(raw_val_probs.reshape(-1, 1), y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities array of shape (N, 2)."""
        if self.calibrator is None or self.base_model is None:
            raise RuntimeError("Calibrator not fitted.")

        raw_probs = self.base_model.predict_proba(X)[:, 1]

        if self.method == "isotonic":
            calibrated_p1 = self.calibrator.predict(raw_probs)
        else:  # sigmoid / platt
            calibrated_p1 = self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        calibrated_p1 = np.clip(calibrated_p1, 1e-6, 1 - 1e-6)
        calibrated_p0 = 1.0 - calibrated_p1
        return np.column_stack([calibrated_p0, calibrated_p1])

    def save(self, dataset_key: str, model_type: str):
        """Save calibrator model to artifacts directory."""
        file_path = ARTIFACTS_DIR / f"{dataset_key}_{model_type}_{self.method}_calibrator.joblib"
        joblib.dump(self.calibrator, file_path)

    @classmethod
    def load(cls, dataset_key: str, model_type: str, method: str, base_estimator: Any):
        """Load calibrator from artifacts directory."""
        actual_method = "sigmoid" if method == "platt" else method
        file_path = ARTIFACTS_DIR / f"{dataset_key}_{model_type}_{actual_method}_calibrator.joblib"
        inst = cls(method=actual_method)
        inst.calibrator = joblib.load(file_path)
        inst.base_model = base_estimator
        return inst


def evaluate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Calculate Brier score, Log Loss, ECE, and reliability curve coordinates."""
    y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    brier = brier_score_loss(y_true, y_prob_clipped)
    loss = log_loss(y_true, y_prob_clipped)

    prob_true, prob_pred = calibration_curve(y_true, y_prob_clipped, n_bins=n_bins, strategy="uniform")

    # Expected Calibration Error (ECE) calculation
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob_clipped >= bin_boundaries[i]) & (y_prob_clipped < bin_boundaries[i+1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob_clipped[bin_mask])
            ece += (bin_size / len(y_prob_clipped)) * np.abs(bin_acc - bin_conf)

    return {
        "brier_score": float(brier),
        "log_loss": float(loss),
        "ece": float(ece),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


if __name__ == "__main__":
    y_true_dummy = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    y_prob_dummy = np.array([0.1, 0.8, 0.65, 0.2, 0.9, 0.15, 0.3, 0.7, 0.85, 0.05])
    metrics = evaluate_calibration_metrics(y_true_dummy, y_prob_dummy)
    print("Calibration Metrics Test:")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  ECE:         {metrics['ece']:.4f}")
