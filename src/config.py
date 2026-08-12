import os
from pathlib import Path

# Base Paths
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

# Datasets Config
DATASET_CONFIGS = {
    "telco": {
        "name": "Telco Subscription Churn",
        "file_name": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "target_col": "Churn",
        "positive_val": 1,
        "id_col": "customerID",
        "numerical_features": ["tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_features": [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
        "default_risk_thresholds": {"low": 0.30, "high": 0.65},
    },
    "bank": {
        "name": "Bank Customer Churn",
        "file_name": "Churn_Modelling.csv",
        "target_col": "Exited",
        "positive_val": 1,
        "id_col": "CustomerId",
        "numerical_features": [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "EstimatedSalary",
        ],
        "categorical_features": [
            "Geography",
            "Gender",
            "HasCrCard",
            "IsActiveMember",
        ],
        "default_risk_thresholds": {"low": 0.25, "high": 0.60},
    },
}

# Calibration Config
CALIBRATION_METHODS = ["uncalibrated", "isotonic", "platt"]

# PSI Bucket Thresholds
PSI_THRESHOLDS = {
    "stable": 0.10,      # PSI < 0.10: No significant shift
    "moderate": 0.25,    # 0.10 <= PSI < 0.25: Moderate shift
    # PSI >= 0.25: High shift / alert
}

# Actionable Retention Playbooks per Feature (for consulting recommendations)
RETENTION_PLAYBOOKS = {
    "Contract": {
        "Month-to-month": "Offer an upgrade to a 1-year or 2-year contract with a 15% promotional loyalty discount.",
    },
    "TechSupport": {
        "No": "Provide a 3-month complimentary Tech Support & Security package add-on.",
    },
    "InternetService": {
        "Fiber optic": "Conduct line quality diagnostic & offer high-speed reliability guarantee.",
    },
    "OnlineSecurity": {
        "No": "Send proactive privacy & digital security setup guide with bundled anti-virus protection.",
    },
    "PaymentMethod": {
        "Electronic check": "Encourage automated recurring auto-pay billing with a $5/month bill credit.",
    },
    "IsActiveMember": {
        "0": "Deploy re-engagement campaign offering personalized wealth advisory or higher savings yields.",
    },
    "NumOfProducts": {
        "1": "Offer a multi-product bundle discount (e.g., credit card cashback + high-yield savings account).",
    },
    "Age": {
        "high": "Provide dedicated senior customer service hotline & prioritized support routing.",
    },
    "MonthlyCharges": {
        "high": "Perform a tailored bill audit to optimize feature add-ons and lower monthly costs.",
    },
    "Balance": {
        "low": "Introduce low-threshold investment options or fee-waived account status.",
    },
}
