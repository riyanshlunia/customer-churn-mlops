import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from src.config import DATASET_CONFIGS
from src.data_pipeline import DataPipeline
from src.calibration import ProbabilityCalibrator
from src.explainability import ChurnExplainer
from src.drift_detector import DriftDetector
from src.model_training import ModelTrainer, train_all_models

app = FastAPI(
    title="Enterprise Churn Prediction & Drift API",
    description="Low-latency REST microservice for calibrated churn prediction, SHAP feature attributions, and population stability drift detection.",
    version="1.0.0"
)

# Global Cached Pipeliners & Explainers
pipelines: Dict[str, DataPipeline] = {}
explainers: Dict[str, ChurnExplainer] = {}
drift_detectors: Dict[str, DriftDetector] = {}


def get_pipeline(dataset_key: str) -> DataPipeline:
    if dataset_key not in pipelines:
        pipelines[dataset_key] = DataPipeline(dataset_key)
    return pipelines[dataset_key]


def get_explainer(dataset_key: str, model_type: str) -> ChurnExplainer:
    key = f"{dataset_key}_{model_type}"
    if key not in explainers:
        explainers[key] = ChurnExplainer(dataset_key, model_type)
    return explainers[key]


def get_drift_detector(dataset_key: str) -> DriftDetector:
    if dataset_key not in drift_detectors:
        drift_detectors[dataset_key] = DriftDetector(dataset_key)
    return drift_detectors[dataset_key]


class PredictRequest(BaseModel):
    dataset: str = Field(default="telco", description="Dataset key: 'telco' or 'bank'")
    model_type: str = Field(default="lightgbm", description="Model engine: 'lightgbm' or 'xgboost'")
    calibration_method: str = Field(default="isotonic", description="Calibration method: 'uncalibrated', 'isotonic', or 'platt'")
    customer_data: Dict[str, Any] = Field(..., description="Customer profile feature dictionary")


class ExplainRequest(BaseModel):
    dataset: str = Field(default="telco", description="Dataset key: 'telco' or 'bank'")
    model_type: str = Field(default="lightgbm", description="Model engine: 'lightgbm' or 'xgboost'")
    customer_data: Dict[str, Any] = Field(..., description="Customer profile feature dictionary")


class DriftCheckRequest(BaseModel):
    dataset: str = Field(default="telco", description="Dataset key: 'telco' or 'bank'")
    batch_data: List[Dict[str, Any]] = Field(..., description="List of production customer profiles")


@app.get("/")
def root_index():
    """API Index Endpoint."""
    return {
        "message": "⚡ Enterprise Churn Prediction & Monitoring API",
        "interactive_docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "datasets": "GET /datasets",
            "predict": "POST /predict",
            "explain": "POST /explain",
            "drift_check": "POST /drift/check"
        }
    }


@app.get("/health")
def health_check():
    """Health check status endpoint."""
    return {
        "status": "healthy",
        "service": "Churn Prediction & Monitoring Microservice",
        "supported_datasets": list(DATASET_CONFIGS.keys()),
    }


@app.get("/datasets")
def list_datasets():
    """Get metadata and schemas for supported domain datasets."""
    return DATASET_CONFIGS


@app.post("/predict")
def predict_churn(req: PredictRequest):
    """Predict calibrated customer churn risk score and confidence level."""
    if req.dataset not in DATASET_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {req.dataset}")

    try:
        pipeline = get_pipeline(req.dataset)
        explainer = get_explainer(req.dataset, req.model_type)

        df_input = pd.DataFrame([req.customer_data])
        X_proc = pipeline.transform(df_input)

        # Raw base model prediction
        raw_prob = float(explainer.model.predict_proba(X_proc)[0, 1])

        # Calibration
        if req.calibration_method == "uncalibrated":
            calibrated_prob = raw_prob
        else:
            calibrator = ProbabilityCalibrator.load(
                dataset_key=req.dataset,
                model_type=req.model_type,
                method=req.calibration_method,
                base_estimator=explainer.model,
            )
            calibrated_prob = float(calibrator.predict_proba(X_proc)[0, 1])

        # Risk Classification
        thresholds = DATASET_CONFIGS[req.dataset]["default_risk_thresholds"]
        if calibrated_prob < thresholds["low"]:
            risk_level = "Low Risk"
        elif calibrated_prob < thresholds["high"]:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        churn_pred = int(calibrated_prob >= 0.5)
        confidence_score = round(abs(calibrated_prob - 0.5) * 2, 4)  # 0 to 1 confidence scale

        return {
            "dataset": req.dataset,
            "model_type": req.model_type,
            "calibration_method": req.calibration_method,
            "raw_probability": round(raw_prob, 4),
            "calibrated_probability": round(calibrated_prob, 4),
            "churn_prediction": churn_pred,
            "risk_level": risk_level,
            "confidence_score": confidence_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain_churn(req: ExplainRequest):
    """Compute per-customer SHAP feature attribution and retention recommendations."""
    if req.dataset not in DATASET_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {req.dataset}")

    try:
        pipeline = get_pipeline(req.dataset)
        explainer = get_explainer(req.dataset, req.model_type)

        df_input = pd.DataFrame([req.customer_data])
        X_proc = pipeline.transform(df_input)

        explanation = explainer.explain_sample(X_proc[0], req.customer_data)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift/check")
def check_drift(req: DriftCheckRequest):
    """Perform PSI and KS statistical distribution shift detection on incoming batch payload."""
    if req.dataset not in DATASET_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {req.dataset}")

    try:
        detector = get_drift_detector(req.dataset)
        df_batch = pd.DataFrame(req.batch_data)
        report = detector.detect_drift(df_batch)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
