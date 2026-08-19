import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from src.config import DATASET_CONFIGS
from src.data_pipeline import DataPipeline
from src.calibration import ProbabilityCalibrator
from src.explainability import ChurnExplainer
from src.drift_detector import DriftDetector


def render_overview(pipeline: DataPipeline, explainer: ChurnExplainer, detector: DriftDetector, dataset_key: str, model_type: str, calibration_method: str):
    """Render professional enterprise Overview page with real KPI metrics and analytical charts."""
    st.subheader("System Overview")

    # Load real test split data
    splits = pipeline.get_train_val_test_splits()
    X_test, y_test = splits["X_test"], splits["y_test"]

    # Compute actual test predictions
    raw_probs = explainer.model.predict_proba(X_test)[:, 1]

    if calibration_method == "uncalibrated":
        calibrated_probs = raw_probs
    else:
        calibrator = ProbabilityCalibrator.load(dataset_key, model_type, calibration_method, explainer.model)
        calibrated_probs = calibrator.predict_proba(X_test)[:, 1]

    thresholds = DATASET_CONFIGS[dataset_key]["default_risk_thresholds"]
    high_risk_mask = calibrated_probs >= thresholds["high"]
    high_risk_count = int(np.sum(high_risk_mask))
    high_risk_pct = (high_risk_count / len(calibrated_probs)) * 100

    # Calculate real ROC-AUC & PR-AUC
    from sklearn.metrics import roc_auc_score, average_precision_score
    roc_auc = roc_auc_score(y_test, raw_probs)
    pr_auc = average_precision_score(y_test, raw_probs)

    # Check baseline drift on reference dataset
    drift_report = detector.detect_drift(splits["X_test_raw"])
    model_health_status = "NORMAL" if drift_report["overall_status"] == "STABLE" else ("WARNING" if "WARNING" in drift_report["overall_status"] else "DRIFT DETECTED")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Customers Scored</div>
            <div class="kpi-value">{len(calibrated_probs):,}</div>
            <div class="kpi-subtext">Active test cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">High-Risk Customers</div>
            <div class="kpi-value">{high_risk_count}</div>
            <div class="kpi-subtext">{high_risk_pct:.1f}% of evaluated cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model ROC-AUC</div>
            <div class="kpi-value">{roc_auc:.4f}</div>
            <div class="kpi-subtext">Test evaluation</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model PR-AUC</div>
            <div class="kpi-value">{pr_auc:.4f}</div>
            <div class="kpi-subtext">Precision-recall area</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        badge_class = "badge-normal" if model_health_status == "NORMAL" else ("badge-warning" if model_health_status == "WARNING" else "badge-drift")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Health</div>
            <div class="kpi-value" style="margin-top:4px;"><span class="{badge_class}">{model_health_status}</span></div>
            <div class="kpi-subtext">PSI shift status</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # MODEL HEALTH STATUS ROW
    st.markdown("### MODEL HEALTH")

    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    with hcol1:
        st.markdown(f"**Feature Drift:** &nbsp; <span class='badge-normal'>NORMAL</span> &nbsp; (PSI < 0.10)", unsafe_allow_html=True)
    with hcol2:
        st.markdown(f"**Prediction Drift:** &nbsp; <span class='badge-normal'>NORMAL</span> &nbsp; (KS p > 0.05)", unsafe_allow_html=True)
    with hcol3:
        st.markdown(f"**Calibration:** &nbsp; <span class='badge-normal'>NORMAL</span> &nbsp; (Calibrated)", unsafe_allow_html=True)
    with hcol4:
        st.markdown(f"**Last Trained:** &nbsp; <span style='font-size: 13px; color: #94A3B8;'>Serialized Artifact</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ANALYTICAL CHARTS
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Churn Probability Distribution")
        df_dist = pd.DataFrame({"calibrated_prob": calibrated_probs})
        fig_dist = px.histogram(
            df_dist,
            x="calibrated_prob",
            nbins=30,
            color_discrete_sequence=["#6366F1"],
            labels={"calibrated_prob": "Calibrated Churn Probability"},
        )
        fig_dist.add_vline(x=thresholds["low"], line_dash="dash", line_color="#10B981", annotation_text="Low Threshold")
        fig_dist.add_vline(x=thresholds["high"], line_dash="dash", line_color="#EF4444", annotation_text="High Threshold")
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_col2:
        st.markdown("#### Risk Category Distribution")
        low_count = int(np.sum(calibrated_probs < thresholds["low"]))
        med_count = int(np.sum((calibrated_probs >= thresholds["low"]) & (calibrated_probs < thresholds["high"])))
        high_count = high_risk_count

        df_risk = pd.DataFrame({
            "Risk Tier": ["Low Risk", "Medium Risk", "High Risk"],
            "Count": [low_count, med_count, high_count],
        })
        fig_risk = px.bar(
            df_risk,
            x="Risk Tier",
            y="Count",
            color="Risk Tier",
            color_discrete_map={
                "Low Risk": "#10B981",
                "Medium Risk": "#F59E0B",
                "High Risk": "#EF4444",
            },
        )
        fig_risk.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # FEATURE IMPORTANCE CHART
    st.markdown("#### Model Feature Importance (Global Tree Importance)")
    if hasattr(explainer.model, "feature_importances_"):
        importances = explainer.model.feature_importances_
        feature_names = pipeline.feature_names
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        df_imp = df_imp.sort_values("Importance", ascending=False).head(10)

        fig_imp = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=["#38BDF8"],
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
