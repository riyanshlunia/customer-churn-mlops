import streamlit as st
from src.config import DATASET_CONFIGS


def render_header(dataset_key: str = "telco", model_type: str = "lightgbm", calibration_method: str = "isotonic", current_page: str = "Overview", **kwargs):
    """Render enterprise top header, navigation breadcrumbs, and compact metadata bar."""
    if current_page != "Overview":
        col_back, _ = st.columns([1, 4])
        with col_back:
            if st.button("← Back to Overview", key="btn_back_overview"):
                st.session_state["nav_page"] = "Overview"
                st.query_params["page"] = "Overview"
                st.session_state["_last_query_page"] = "Overview"
                st.rerun()

    st.title("Customer Churn Risk Analytics")
    st.markdown("##### Telco subscription retention · ML prediction, explainability & model monitoring")
    st.caption("Predict churn risk, understand model drivers, and monitor production data health.")

    dataset_name = DATASET_CONFIGS.get(dataset_key, {}).get("name", dataset_key)
    model_name = "LightGBM" if model_type == "lightgbm" else "XGBoost"
    cal_name = "Isotonic Regression" if calibration_method == "isotonic" else ("Platt Scaling" if calibration_method == "platt" else "Uncalibrated Raw")

    st.markdown(f"""
    <div class="meta-bar">
        <span class="meta-item">Model: <strong>{model_name}</strong></span>
        <span class="meta-item">Calibration: <strong>{cal_name}</strong></span>
        <span class="meta-item">Dataset: <strong>{dataset_name}</strong></span>
        <span class="meta-item">View: <strong>{current_page}</strong></span>
    </div>
    """, unsafe_allow_html=True)
