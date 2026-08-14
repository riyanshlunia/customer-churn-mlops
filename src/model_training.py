import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import mlflow
import joblib

from src.config import ARTIFACTS_DIR, MLRUNS_DIR, MLFLOW_DB_PATH, DATASET_CONFIGS
from src.data_pipeline import DataPipeline
from src.calibration import ProbabilityCalibrator, evaluate_calibration_metrics


class ModelTrainer:
    """Trains Gradient Boosting classifiers, applies probability calibration, and logs to MLflow."""

    def __init__(self, dataset_key: str = "telco", model_type: str = "lightgbm"):
        if dataset_key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset_key: {dataset_key}")
        if model_type not in ["lightgbm", "xgboost"]:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.dataset_key = dataset_key
        self.model_type = model_type
        self.model = None
        self.isotonic_calibrator = None
        self.platt_calibrator = None

    def _instantiate_model(self, random_state: int = 42):
        if self.model_type == "lightgbm":
            self.model = LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=random_state,
                verbose=-1
            )
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                random_state=random_state,
                eval_metric="logloss"
            )

    def train_and_evaluate(self, random_state: int = 42) -> Dict[str, Any]:
        """Run complete model training, calibration, MLflow logging, and evaluation."""
        pipeline = DataPipeline(self.dataset_key)
        splits = pipeline.get_train_val_test_splits(random_state=random_state)

        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val, y_val = splits["X_val"], splits["y_val"]
        X_test, y_test = splits["X_test"], splits["y_test"]

        self._instantiate_model(random_state=random_state)
        self.model.fit(X_train, y_train)

        # Fit Calibrators on Validation Set
        self.isotonic_calibrator = ProbabilityCalibrator(method="isotonic")
        self.isotonic_calibrator.fit(self.model, X_val, y_val)

        self.platt_calibrator = ProbabilityCalibrator(method="platt")
        self.platt_calibrator.fit(self.model, X_val, y_val)

        # Raw Uncalibrated Predictions
        raw_prob_test = self.model.predict_proba(X_test)[:, 1]
        iso_prob_test = self.isotonic_calibrator.predict_proba(X_test)[:, 1]
        platt_prob_test = self.platt_calibrator.predict_proba(X_test)[:, 1]

        # Calculate Classification Metrics
        roc_auc = roc_auc_score(y_test, raw_prob_test)
        pr_auc = average_precision_score(y_test, raw_prob_test)
        raw_preds = (raw_prob_test >= 0.5).astype(int)
        acc = accuracy_score(y_test, raw_preds)
        f1 = f1_score(y_test, raw_preds)

        # Evaluate Calibration Metrics
        raw_cal_metrics = evaluate_calibration_metrics(y_test, raw_prob_test)
        iso_cal_metrics = evaluate_calibration_metrics(y_test, iso_prob_test)
        platt_cal_metrics = evaluate_calibration_metrics(y_test, platt_prob_test)

        # Save artifacts
        model_path = ARTIFACTS_DIR / f"{self.dataset_key}_{self.model_type}_model.joblib"
        joblib.dump(self.model, model_path)
        self.isotonic_calibrator.save(self.dataset_key, self.model_type)
        self.platt_calibrator.save(self.dataset_key, self.model_type)

        # Save Baseline Reference Data for Drift Detection
        ref_df = splits["X_train_raw"]
        ref_df.to_csv(ARTIFACTS_DIR / f"{self.dataset_key}_reference_baseline.csv", index=False)

        # Log to MLflow
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
        mlflow.set_experiment(f"Churn_Prediction_{self.dataset_key.upper()}")

        with mlflow.start_run(run_name=f"{self.model_type}_calibrated"):
            mlflow.log_param("dataset", self.dataset_key)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            mlflow.log_param("num_features", X_train.shape[1])

            # Metrics
            mlflow.log_metric("roc_auc", float(roc_auc))
            mlflow.log_metric("pr_auc", float(pr_auc))
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("f1_score", float(f1))

            mlflow.log_metric("raw_brier_score", raw_cal_metrics["brier_score"])
            mlflow.log_metric("raw_log_loss", raw_cal_metrics["log_loss"])
            mlflow.log_metric("isotonic_brier_score", iso_cal_metrics["brier_score"])
            mlflow.log_metric("isotonic_log_loss", iso_cal_metrics["log_loss"])
            mlflow.log_metric("platt_brier_score", platt_cal_metrics["brier_score"])
            mlflow.log_metric("platt_log_loss", platt_cal_metrics["log_loss"])

        return {
            "dataset_key": self.dataset_key,
            "model_type": self.model_type,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": acc,
            "f1": f1,
            "raw_cal_metrics": raw_cal_metrics,
            "iso_cal_metrics": iso_cal_metrics,
            "platt_cal_metrics": platt_cal_metrics,
        }


def train_all_models():
    """Train LightGBM and XGBoost for both Telco and Bank churn datasets."""
    results = {}
    for dataset_key in ["telco", "bank"]:
        for model_type in ["lightgbm", "xgboost"]:
            print(f"\n================ Training {model_type.upper()} on {dataset_key.upper()} ================")
            trainer = ModelTrainer(dataset_key=dataset_key, model_type=model_type)
            res = trainer.train_and_evaluate()
            results[f"{dataset_key}_{model_type}"] = res

            print(f"  ROC-AUC: {res['roc_auc']:.4f} | PR-AUC: {res['pr_auc']:.4f} | F1: {res['f1']:.4f}")
            print(f"  Uncalibrated Brier Score: {res['raw_cal_metrics']['brier_score']:.4f} | LogLoss: {res['raw_cal_metrics']['log_loss']:.4f}")
            print(f"  Isotonic     Brier Score: {res['iso_cal_metrics']['brier_score']:.4f} | LogLoss: {res['iso_cal_metrics']['log_loss']:.4f}")
            print(f"  Platt        Brier Score: {res['platt_cal_metrics']['brier_score']:.4f} | LogLoss: {res['platt_cal_metrics']['log_loss']:.4f}")

    return results


if __name__ == "__main__":
    train_all_models()
