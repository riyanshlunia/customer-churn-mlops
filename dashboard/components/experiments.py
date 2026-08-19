import streamlit as st
import pandas as pd
import mlflow

from src.config import MLFLOW_DB_PATH, DATASET_CONFIGS


def render_experiments(dataset_key: str):
    """Render MLflow experiment tracking registry and champion model highlights."""
    st.subheader("Experiment Registry")
    st.caption("MLflow experiment tracking history, logged hyperparameters, evaluation metrics, and champion model selection.")

    exp_name = f"Churn_Prediction_{dataset_key.upper()}"

    try:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
        exp = mlflow.get_experiment_by_name(exp_name)

        if exp is None:
            st.info(f"No logged MLflow experiments found for dataset '{dataset_key.upper()}'. Run training script first.")
            return

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

        if runs.empty:
            st.info("No runs found in experiment registry.")
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

        # Highlight Champion Model (highest ROC-AUC or first row)
        st.markdown("#### Logged Training Runs")

        # Check if ROC-AUC column exists
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

    except Exception as e:
        st.warning(f"Could not connect to MLflow database at '{MLFLOW_DB_PATH}': {e}")
