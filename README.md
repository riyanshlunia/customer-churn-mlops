# Enterprise Churn Prediction Platform

Welcome to the **Enterprise Churn Prediction Platform**! This project provides a complete, end-to-end MLOps pipeline for predicting customer churn. It encompasses model training, experiment tracking, explainability, data drift detection, a REST API for real-time predictions, and an interactive dashboard for insights.

## Features

- **Model Training**: Robust training pipeline utilizing LightGBM and XGBoost, tested on Telco and Bank datasets.
- **MLOps & Tracking**: Seamless integration with MLflow for tracking experiments, metrics, and managing the model lifecycle.
- **Explainable AI (XAI)**: Integrated SHAP (SHapley Additive exPlanations) values to interpret model predictions and understand feature importance.
- **Data Drift Detection**: Built-in mechanisms to detect shifts in data distribution over time, ensuring model reliability.
- **Model Calibration**: Probability calibration to provide reliable confidence scores for churn predictions.
- **REST API**: A blazing-fast FastAPI microservice for serving real-time predictions.
- **Interactive Dashboard**: A user-friendly Streamlit dashboard for business users to interact with the models and visualize insights.

## Technology Stack

- **Machine Learning**: `scikit-learn`, `LightGBM`, `XGBoost`
- **MLOps**: `MLflow`
- **Explainability**: `SHAP`
- **Backend Service**: `FastAPI`, `Uvicorn`
- **Frontend Dashboard**: `Streamlit`, `Plotly`
- **Data Manipulation**: `pandas`, `numpy`, `scipy`

## Project Structure

```text
customer-churn-mlops/
├── api/                  # FastAPI service for model serving
├── dashboard/            # Streamlit interactive web dashboard
├── data/                 # Datasets used for training and testing
├── src/                  # Core ML pipeline source code
│   ├── data_pipeline.py  # Data preprocessing and feature engineering
│   ├── model_training.py # Model training and MLflow tracking
│   ├── explainability.py # SHAP value computations
│   ├── drift_detector.py # Data drift detection logic
│   └── calibration.py    # Model output calibration
├── mlruns/               # MLflow tracking directory
├── mlflow.db             # MLflow SQLite database
├── requirements.txt      # Python dependencies
└── run_app.py            # Main entry point for running the platform
```

## Setup and Installation

1. **Clone the repository**:
   Navigate to your workspace and clone the project (if not already done).

2. **Install Dependencies**:
   Ensure you have Python installed. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

We provide a convenient centralized script (`run_app.py`) to manage the platform's components.

### 1. Train the Models
To run the data pipeline, train the models, and log experiments to MLflow:
```bash
python run_app.py train
```

### 2. Launch the API Service
To start the FastAPI microservice for real-time inference (runs on `http://127.0.0.1:8000`):
```bash
python run_app.py api
```
*Note: You can access the interactive Swagger API documentation at `http://127.0.0.1:8000/docs`.*

### 3. Launch the Dashboard
To start the interactive Streamlit dashboard for visual insights (runs on `http://localhost:8501`):
```bash
python run_app.py dashboard
```

### 4. Train & Show Instructions
To train models and print out instructions for launching the services:
```bash
python run_app.py all
```

## Contributing

Contributions are welcome! If you'd like to improve the pipeline, add new models, or enhance the dashboard, please feel free to fork the repository and submit a pull request.
