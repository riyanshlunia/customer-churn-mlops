import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats
import joblib

from src.config import ARTIFACTS_DIR, DATASET_CONFIGS, PSI_THRESHOLDS
from src.data_pipeline import DataPipeline


def calculate_psi(
    expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, is_categorical: bool = False
) -> float:
    """Calculate Population Stability Index (PSI) between baseline expected and production actual arrays."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    if is_categorical:
        categories = np.unique(np.concatenate([expected, actual]))
        exp_counts = pd.Series(expected).value_counts(normalize=True).reindex(categories, fill_value=0.0).values
        act_counts = pd.Series(actual).value_counts(normalize=True).reindex(categories, fill_value=0.0).values
    else:
        # Quantile bucketing based on expected distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(expected, percentiles)
        buckets[0] -= 1e-5
        buckets[-1] += 1e-5
        buckets = np.unique(buckets)

        if len(buckets) < 2:
            return 0.0

        exp_counts, _ = np.histogram(expected, bins=buckets)
        act_counts, _ = np.histogram(actual, bins=buckets)

        exp_counts = exp_counts / len(expected)
        act_counts = act_counts / len(actual)

    # Avoid zero division and log(0) using smooth epsilon
    eps = 1e-4
    exp_counts = np.where(exp_counts == 0, eps, exp_counts)
    act_counts = np.where(act_counts == 0, eps, act_counts)

    psi_val = np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts))
    return float(psi_val)


class DriftDetector:
    """Statistical Drift Monitoring Engine using PSI and Kolmogorov-Smirnov (KS) tests."""

    def __init__(self, dataset_key: str = "telco"):
        if dataset_key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset_key: {dataset_key}")
        self.dataset_key = dataset_key
        self.config = DATASET_CONFIGS[dataset_key]

        # Load reference baseline dataframe
        ref_path = ARTIFACTS_DIR / f"{self.dataset_key}_reference_baseline.csv"
        if not ref_path.exists():
            # Generate reference split if missing
            pipe = DataPipeline(self.dataset_key)
            splits = pipe.get_train_val_test_splits()
            self.ref_df = splits["X_train_raw"]
        else:
            self.ref_df = pd.read_csv(ref_path)

    def detect_drift(self, prod_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare production payload distribution against training baseline distribution."""
        feature_results = []
        high_drift_count = 0
        moderate_drift_count = 0

        num_cols = self.config["numerical_features"]
        cat_cols = self.config["categorical_features"]

        for col in num_cols:
            if col in prod_df.columns and col in self.ref_df.columns:
                ref_vals = self.ref_df[col].dropna().values
                prod_vals = prod_df[col].dropna().values

                psi = calculate_psi(ref_vals, prod_vals, is_categorical=False)
                ks_stat, p_val = stats.ks_2samp(ref_vals, prod_vals)

                status = "stable"
                if psi >= PSI_THRESHOLDS["moderate"]:
                    status = "high_drift"
                    high_drift_count += 1
                elif psi >= PSI_THRESHOLDS["stable"]:
                    status = "moderate_drift"
                    moderate_drift_count += 1

                feature_results.append({
                    "feature": col,
                    "type": "numerical",
                    "psi": round(psi, 4),
                    "ks_p_value": round(float(p_val), 4),
                    "status": status,
                    "ref_mean": round(float(np.mean(ref_vals)), 2),
                    "prod_mean": round(float(np.mean(prod_vals)), 2),
                })

        for col in cat_cols:
            if col in prod_df.columns and col in self.ref_df.columns:
                ref_vals = self.ref_df[col].astype(str).values
                prod_vals = prod_df[col].astype(str).values

                psi = calculate_psi(ref_vals, prod_vals, is_categorical=True)

                status = "stable"
                if psi >= PSI_THRESHOLDS["moderate"]:
                    status = "high_drift"
                    high_drift_count += 1
                elif psi >= PSI_THRESHOLDS["stable"]:
                    status = "moderate_drift"
                    moderate_drift_count += 1

                feature_results.append({
                    "feature": col,
                    "type": "categorical",
                    "psi": round(psi, 4),
                    "ks_p_value": None,
                    "status": status,
                    "ref_mean": None,
                    "prod_mean": None,
                })

        overall_status = "STABLE"
        if high_drift_count > 0:
            overall_status = "CRITICAL_DRIFT_ALERT"
        elif moderate_drift_count > 0:
            overall_status = "MODERATE_DRIFT_WARNING"

        return {
            "dataset_key": self.dataset_key,
            "overall_status": overall_status,
            "total_features": len(feature_results),
            "high_drift_features": high_drift_count,
            "moderate_drift_features": moderate_drift_count,
            "feature_drift_details": feature_results,
        }


if __name__ == "__main__":
    detector = DriftDetector("telco")

    # Simulate production data with drift (e.g. higher MonthlyCharges & more Month-to-month contracts)
    prod_sim = detector.ref_df.copy()
    prod_sim["MonthlyCharges"] = prod_sim["MonthlyCharges"] * 1.40  # 40% cost increase shift
    prod_sim["Contract"] = "Month-to-month"

    report = detector.detect_drift(prod_sim)
    print(f"Drift Analysis for TELCO:")
    print(f"  Overall Status: {report['overall_status']}")
    print(f"  High Drift Count: {report['high_drift_features']}")
    print(f"  Moderate Drift Count: {report['moderate_drift_features']}\n")

    print("Top Feature Drift Details:")
    for f in sorted(report["feature_drift_details"], key=lambda x: x["psi"], reverse=True)[:5]:
        print(f"  {f['feature']} ({f['type']}): PSI={f['psi']}, Status={f['status']}")
