import streamlit as st
import pandas as pd
import mlflow

from src.config import MLFLOW_DB_PATH, DATASET_CONFIGS


# Static fallback results - updated after each training run
_STATIC_RESULTS = {
    "telco": [
        {"Run Name": "lightgbm_calibrated", "Model": "lightgbm", "ROC AUC": 0.8424, "PR AUC": 0.6464, "Accuracy": 0.8057, "F1 Score": 0.5749, "Brier Score": 0.1372, "Status": "FINISHED"},
        {"Run Name": "xgboost_calibrated",  "Model": "xgboost",  "ROC AUC": 0.8460, "PR AUC": 0.6602, "Accuracy": 0.8013, "F1 Score": 0.5719, "Brier Score": 0.1356, "Status": "FINISHED"},
    ],
    "bank": [
        {"Run Name": "lightgbm_calibrated", "Model": "lightgbm", "ROC AUC": 0.8657, "PR AUC": 0.7083, "Accuracy": 0.8648, "F1 Score": 0.5855, "Brier Score": 0.1011, "Status": "FINISHED"},
        {"Run Name": "xgboost_calibrated",  "Model": "xgboost",  "ROC AUC": 0.8684, "PR AUC": 0.7108, "Accuracy": 0.8641, "F1 Score": 0.5696, "Brier Score": 0.1008, "Status": "FINISHED"},
    ],
}


def _render_static_results(dataset_key: str):
    """Render a static experiment results table when MLflow DB is unavailable."""
    rows = _STATIC_RESULTS.get(dataset_key, [])
    df_exp = pd.DataFrame(rows)

    best_idx = df_exp["ROC AUC"].idxmax()
    best_run = df_exp.loc[best_idx, "Run Name"]
    best_auc = df_exp.loc[best_idx, "ROC AUC"]

    st.markdown(f"""
    <div class="info-panel">
        <span class="badge-normal">CHAMPION MODEL</span> &nbsp;
        <strong>{best_run}</strong> achieved highest score (ROC-AUC = <strong>{best_auc:.4f}</strong>).
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Logged Training Runs")
    st.dataframe(df_exp, use_container_width=True)

    st.info(
        "MLflow tracking database is not available in this deployment environment. "
        "The table above reflects results from the last local training run. "
        "Run `python run_app.py train` locally to regenerate."
    )


def render_experiments(dataset_key: str):
    """Render MLflow experiment tracking registry and champion model highlights."""
    st.subheader("Experiment Registry")
    st.caption("MLflow experiment tracking history, logged hyperparameters, evaluation metrics, and champion model selection.")

    exp_name = f"Churn_Prediction_{dataset_key.upper()}"

    # On Streamlit Cloud there is no SQLite DB - fall back to static results
    if not MLFLOW_DB_PATH.exists():
        _render_static_results(dataset_key)
        return

    try:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
        exp = mlflow.get_experiment_by_name(exp_name)

        if exp is None:
            _render_static_results(dataset_key)
            return

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

        if runs.empty:
            _render_static_results(dataset_key)
            return

        # Prepare clean display table
        display_cols = []
        rename_map = {}

        if "tags.mlflow.runName" in runs.columns:
            display_cols.append("tags.mlflow.runName")
            rename_map["tags.mlflow.runName"] = "Run Name"
        if "params.model_type" in runs.columns:
            display_cols.append("params.model_type")
            rename_map["params.model_type"] = "Model"
        elif "params.n_estimators" in runs.columns:
            display_cols.append("params.n_estimators")
            rename_map["params.n_estimators"] = "n_estimators"

        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        for mc in metric_cols:
            display_cols.append(mc)
            clean_name = mc.replace("metrics.", "").replace("_", " ").title()
            rename_map[mc] = clean_name

        if "status" in runs.columns:
            display_cols.append("status")
            rename_map["status"] = "Status"

        df_exp = runs[display_cols].rename(columns=rename_map)

        # Highlight Champion Model
        st.markdown("#### Logged Training Runs")
        auc_col = [c for c in df_exp.columns if "Roc" in c or "Auc" in c]
        if auc_col:
            best_idx = df_exp[auc_col[0]].idxmax()
            best_run_name = df_exp.loc[best_idx, "Run Name"] if "Run Name" in df_exp.columns else "Active Run"
            best_auc = df_exp.loc[best_idx, auc_col[0]]

            st.markdown(f"""
            <div class="info-panel">
                <span class="badge-normal">CHAMPION MODEL</span> &nbsp;
                <strong>{best_run_name}</strong> achieved highest score (ROC-AUC = <strong>{best_auc:.4f}</strong>).
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(df_exp, use_container_width=True)

    except Exception:
        _render_static_results(dataset_key)
