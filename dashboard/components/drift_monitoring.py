import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from src.config import DATASET_CONFIGS
from src.drift_detector import DriftDetector


def render_drift_monitoring(detector: DriftDetector, dataset_key: str):
    """Render Production Data Health (Drift Monitoring) page using real PSI and KS statistics."""
    st.subheader("Production Data Health")
    st.caption("Simulated production monitoring · Drift monitoring compares the distribution of incoming data with the reference training distribution. Alerts indicate that model performance may require investigation.")

    # Control Panel to Simulate Distribution Shift
    st.markdown("#### Simulated Production Batch Parameters")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        shift_pct = st.slider("Simulated Feature Magnitude Shift (%)", 0, 100, 35)
    with col_s2:
        force_categorical = st.checkbox("Simulate Categorical Shift (e.g. Month-to-Month Contract / Inactive Member)", value=True)

    # Build simulated production payload
    prod_payload = detector.ref_df.copy()
    if dataset_key == "telco":
        if "MonthlyCharges" in prod_payload.columns:
            prod_payload["MonthlyCharges"] = prod_payload["MonthlyCharges"] * (1 + shift_pct / 100)
        if force_categorical and "Contract" in prod_payload.columns:
            prod_payload["Contract"] = "Month-to-month"
    elif dataset_key == "bank":
        if "Balance" in prod_payload.columns:
            prod_payload["Balance"] = prod_payload["Balance"] * (1 + shift_pct / 100)
        if force_categorical and "IsActiveMember" in prod_payload.columns:
            prod_payload["IsActiveMember"] = 0

    # Calculate real drift report
    report = detector.detect_drift(prod_payload)

    status_raw = report["overall_status"]
    if status_raw == "STABLE":
        status_label = "NORMAL"
        badge_class = "badge-normal"
    elif "WARNING" in status_raw:
        status_label = "WARNING"
        badge_class = "badge-warning"
    else:
        status_label = "DRIFT DETECTED"
        badge_class = "badge-drift"

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"**Overall Health Status:**<br><span class='{badge_class}'>{status_label}</span>", unsafe_allow_html=True)
    with mcol2:
        st.metric("Total Features Evaluated", report["total_features"])
    with mcol3:
        st.metric("High Drift Features (PSI ≥ 0.25)", report["high_drift_features"])
    with mcol4:
        st.metric("Moderate Shift Features (0.10 ≤ PSI < 0.25)", report["moderate_drift_features"])

    st.markdown("---")

    # FEATURE DRIFT TABLE
    st.markdown("#### Feature Population Stability Index (PSI) & KS Test Table")

    table_data = []
    for item in report["feature_drift_details"]:
        s_raw = item["status"]
        if s_raw == "stable":
            st_clean = "NORMAL"
        elif s_raw == "moderate_drift":
            st_clean = "WARNING"
        else:
            st_clean = "DRIFT DETECTED"

        table_data.append({
            "Feature": item["feature"],
            "Type": item["type"],
            "PSI": item["psi"],
            "KS Statistic": "-" if item["ks_p_value"] is None else round(1.0 - item["ks_p_value"], 4),
            "KS p-value": "-" if item["ks_p_value"] is None else item["ks_p_value"],
            "Status": st_clean,
            "Reference Mean": "-" if item["ref_mean"] is None else item["ref_mean"],
            "Production Mean": "-" if item["prod_mean"] is None else item["prod_mean"],
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True)

    st.markdown("---")

    # DISTRIBUTION COMPARISON CHART
    st.markdown("#### Distribution Comparison (Reference Training vs Production Payload)")

    available_features = [f for f in detector.ref_df.columns if f in prod_payload.columns and f not in [DATASET_CONFIGS[dataset_key]["target_col"], DATASET_CONFIGS[dataset_key]["id_col"]]]
    selected_feature = st.selectbox("Select Feature for Distribution Comparison", options=available_features, index=0)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(f"##### Feature: {selected_feature}")

        df_ref_feat = pd.DataFrame({"Value": detector.ref_df[selected_feature], "Dataset": "Reference Training"})
        df_prod_feat = pd.DataFrame({"Value": prod_payload[selected_feature], "Dataset": "Simulated Production"})
        df_combined = pd.concat([df_ref_feat, df_prod_feat], ignore_index=True)

        if pd.api.types.is_numeric_dtype(detector.ref_df[selected_feature]):
            fig_compare = px.histogram(
                df_combined,
                x="Value",
                color="Dataset",
                barmode="overlay",
                color_discrete_map={"Reference Training": "#38BDF8", "Simulated Production": "#EF4444"},
                opacity=0.6,
                title=f"Numerical Distribution Overlay: {selected_feature}",
            )
        else:
            fig_compare = px.histogram(
                df_combined,
                x="Value",
                color="Dataset",
                barmode="group",
                color_discrete_map={"Reference Training": "#38BDF8", "Simulated Production": "#EF4444"},
                title=f"Categorical Frequency Overlay: {selected_feature}",
            )

        fig_compare.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with col_chart2:
        st.markdown("##### Monitoring Guidance & Thresholds")
        st.markdown("""
        <div class="info-panel">
            <p><strong>Population Stability Index (PSI) Thresholds:</strong></p>
            <ul>
                <li><strong>PSI &lt; 0.10 (NORMAL):</strong> No significant distribution shift between baseline reference and production incoming data.</li>
                <li><strong>0.10 &le; PSI &lt; 0.25 (WARNING):</strong> Moderate distribution shift detected. Monitor model accuracy performance closely.</li>
                <li><strong>PSI &ge; 0.25 (DRIFT DETECTED):</strong> Significant population shift detected. Model retraining or pipeline investigation is recommended.</li>
            </ul>
            <p>Drift monitoring compares the distribution of incoming data with the reference training distribution. Alerts indicate that model performance may require investigation.</p>
        </div>
        """, unsafe_allow_html=True)
