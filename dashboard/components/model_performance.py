import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from src.config import DATASET_CONFIGS
from src.data_pipeline import DataPipeline
from src.calibration import ProbabilityCalibrator, evaluate_calibration_metrics
from src.explainability import ChurnExplainer


def render_model_performance(pipeline: DataPipeline, explainer: ChurnExplainer, dataset_key: str, model_type: str, calibration_method: str):
    """Render dedicated Model Performance page with real ROC/PR curves, calibration diagrams, confusion matrix, and model comparison table."""
    st.subheader("Model Performance")
    st.caption("Comprehensive evaluation metrics, ROC/PR curves, reliability diagrams, and decision thresholds for the active model.")

    splits = pipeline.get_train_val_test_splits()
    X_test, y_test = splits["X_test"], splits["y_test"]

    # Active Model Evaluation
    raw_probs = explainer.model.predict_proba(X_test)[:, 1]

    if calibration_method == "uncalibrated":
        calibrated_probs = raw_probs
    else:
        calibrator = ProbabilityCalibrator.load(dataset_key, model_type, calibration_method, explainer.model)
        calibrated_probs = calibrator.predict_proba(X_test)[:, 1]

    preds = (calibrated_probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, calibrated_probs)
    pr_auc = average_precision_score(y_test, calibrated_probs)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    cal_metrics = evaluate_calibration_metrics(y_test, calibrated_probs)

    # Top Metrics Bar
    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
    with mcol1:
        st.metric("ROC-AUC", f"{roc_auc:.4f}")
    with mcol2:
        st.metric("PR-AUC", f"{pr_auc:.4f}")
    with mcol3:
        st.metric("F1 Score", f"{f1:.4f}")
    with mcol4:
        st.metric("Precision / Recall", f"{prec:.2f} / {rec:.2f}")
    with mcol5:
        st.metric("Brier Score", f"{cal_metrics['brier_score']:.4f}")
    with mcol6:
        st.metric("ECE (Calibration)", f"{cal_metrics['ece']:.4f}")

    st.markdown("---")

    # CHARTS ROW 1: ROC & PR CURVES
    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
        fpr, tpr, _ = roc_curve(y_test, calibrated_probs)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Active Model (AUC = {roc_auc:.4f})", line=dict(color="#6366F1", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline", line=dict(color="#64748B", dash="dash")))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption("The ROC curve measures the model's ability to discriminate between churning and non-churning customers across all classification thresholds.")

    with col_pr:
        st.markdown("#### Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_test, calibrated_probs)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"Active Model (PR-AUC = {pr_auc:.4f})", line=dict(color="#38BDF8", width=2)))
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.caption("The Precision-Recall curve evaluates model accuracy on the positive churn class under class imbalance.")

    st.markdown("---")

    # CHARTS ROW 2: CALIBRATION CURVE & CONFUSION MATRIX
    col_cal, col_cm = st.columns(2)

    with col_cal:
        st.markdown("#### Calibration / Reliability Curve")
        raw_metrics = evaluate_calibration_metrics(y_test, raw_probs)

        iso_cal = ProbabilityCalibrator.load(dataset_key, model_type, "isotonic", explainer.model)
        iso_probs = iso_cal.predict_proba(X_test)[:, 1]
        iso_metrics = evaluate_calibration_metrics(y_test, iso_probs)

        platt_cal = ProbabilityCalibrator.load(dataset_key, model_type, "platt", explainer.model)
        platt_probs = platt_cal.predict_proba(X_test)[:, 1]
        platt_metrics = evaluate_calibration_metrics(y_test, platt_probs)

        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly Calibrated", line=dict(dash="dash", color="#64748B")))
        fig_rel.add_trace(go.Scatter(x=raw_metrics["prob_pred"], y=raw_metrics["prob_true"], mode="lines+markers", name="Uncalibrated Raw", line=dict(color="#EF4444")))
        fig_rel.add_trace(go.Scatter(x=iso_metrics["prob_pred"], y=iso_metrics["prob_true"], mode="lines+markers", name="Isotonic Regression", line=dict(color="#10B981")))
        fig_rel.add_trace(go.Scatter(x=platt_metrics["prob_pred"], y=platt_metrics["prob_true"], mode="lines+markers", name="Platt Scaling", line=dict(color="#6366F1")))

        fig_rel.update_layout(
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Observed Churn Fraction",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_rel, use_container_width=True)
        st.caption("Reliability curves check whether predicted probability matches empirical frequency (e.g. 70% predicted churn = 70% actual churn).")

    with col_cm:
        st.markdown("#### Confusion Matrix (Threshold = 0.50)")
        cm = confusion_matrix(y_test, preds)
        cm_df = pd.DataFrame(cm, index=["Actual Non-Churn", "Actual Churn"], columns=["Pred Non-Churn", "Pred Churn"])

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Non-Churn", "Predicted Churn"],
            y=["Actual Non-Churn", "Actual Churn"],
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("The confusion matrix details True Positive, False Positive, True Negative, and False Negative prediction counts.")

    st.markdown("---")

    # THRESHOLD PERFORMANCE ANALYSIS
    st.markdown("#### Threshold-Performance Analysis")

    threshold_grid = np.linspace(0.1, 0.9, 17)
    thresh_precisions = []
    thresh_recalls = []
    thresh_f1s = []

    for th in threshold_grid:
        t_preds = (calibrated_probs >= th).astype(int)
        thresh_precisions.append(precision_score(y_test, t_preds, zero_division=0))
        thresh_recalls.append(recall_score(y_test, t_preds, zero_division=0))
        thresh_f1s.append(f1_score(y_test, t_preds, zero_division=0))

    fig_th = go.Figure()
    fig_th.add_trace(go.Scatter(x=threshold_grid, y=thresh_precisions, mode="lines+markers", name="Precision", line=dict(color="#38BDF8")))
    fig_th.add_trace(go.Scatter(x=threshold_grid, y=thresh_recalls, mode="lines+markers", name="Recall", line=dict(color="#F59E0B")))
    fig_th.add_trace(go.Scatter(x=threshold_grid, y=thresh_f1s, mode="lines+markers", name="F1 Score", line=dict(color="#10B981")))
    fig_th.update_layout(
        xaxis_title="Decision Threshold",
        yaxis_title="Metric Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_th, use_container_width=True)
    st.caption("Threshold tuning allows business risk managers to trade off Precision (avoiding false alarms) against Recall (capturing more churners).")

    st.markdown("---")

    # MODEL COMPARISON TABLE
    st.markdown("#### Model Comparison Table (Active Dataset)")

    # Load and compute real metrics for both LightGBM and XGBoost
    lgb_explainer = ChurnExplainer(dataset_key, "lightgbm")
    lgb_probs = lgb_explainer.model.predict_proba(X_test)[:, 1]
    lgb_preds = (lgb_probs >= 0.5).astype(int)
    lgb_cal = evaluate_calibration_metrics(y_test, lgb_probs)

    xgb_explainer = ChurnExplainer(dataset_key, "xgboost")
    xgb_probs = xgb_explainer.model.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    xgb_cal = evaluate_calibration_metrics(y_test, xgb_probs)

    comparison_df = pd.DataFrame([
        {
            "Model": "LightGBM",
            "ROC-AUC": round(roc_auc_score(y_test, lgb_probs), 4),
            "PR-AUC": round(average_precision_score(y_test, lgb_probs), 4),
            "F1 Score": round(f1_score(y_test, lgb_preds, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_test, lgb_preds), 4),
            "Raw Brier Score": round(lgb_cal["brier_score"], 4),
            "Raw Log Loss": round(lgb_cal["log_loss"], 4),
        },
        {
            "Model": "XGBoost",
            "ROC-AUC": round(roc_auc_score(y_test, xgb_probs), 4),
            "PR-AUC": round(average_precision_score(y_test, xgb_probs), 4),
            "F1 Score": round(f1_score(y_test, xgb_preds, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_test, xgb_preds), 4),
            "Raw Brier Score": round(xgb_cal["brier_score"], 4),
            "Raw Log Loss": round(xgb_cal["log_loss"], 4),
        },
    ])

    st.dataframe(comparison_df, use_container_width=True)
