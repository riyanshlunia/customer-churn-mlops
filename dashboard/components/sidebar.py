import streamlit as st
from src.config import DATASET_CONFIGS


def render_sidebar():
    """Render enterprise-style sidebar navigation with browser URL query parameter history synchronization."""
    st.sidebar.markdown("### CUSTOMER CHURN RISK")
    st.sidebar.markdown("---")

    # Dataset Selector
    dataset_key = st.sidebar.selectbox(
        "Dataset",
        options=["telco", "bank"],
        format_func=lambda x: DATASET_CONFIGS[x]["name"],
        index=0,
    )

    # Model Selector
    model_type = st.sidebar.selectbox(
        "Model",
        options=["lightgbm", "xgboost"],
        format_func=lambda x: "LightGBM" if x == "lightgbm" else "XGBoost",
        index=0,
    )

    # Calibration Selector
    calibration_method = st.sidebar.selectbox(
        "Calibration",
        options=["isotonic", "platt", "uncalibrated"],
        format_func=lambda x: "Isotonic Regression" if x == "isotonic" else ("Platt Scaling" if x == "platt" else "Uncalibrated"),
        index=0,
    )

    # Static Enterprise Environment Metadata
    st.sidebar.markdown("""
    <div style='font-size: 12px; color: #94A3B8; margin-top: 8px;'>
        <p style='margin: 4px 0;'><strong>Version:</strong> v1.0</p>
        <p style='margin: 4px 0;'><strong>Environment:</strong> Local / Development</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    page_options = ["Overview", "Customer Risk", "Model Performance", "Drift Monitoring", "Experiments"]

    # Read current page from browser URL query parameter (enables browser Back / Forward buttons)
    query_page = st.query_params.get("page", "Overview")
    if query_page not in page_options:
        query_page = "Overview"

    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = query_page
    else:
        # If user pressed browser Back/Forward button, query parameter changes
        if st.session_state.get("_last_query_page") != query_page:
            st.session_state["nav_page"] = query_page

    st.session_state["_last_query_page"] = query_page

    # Callback when user clicks sidebar radio
    def on_page_change():
        selected = st.session_state["nav_radio_key"]
        st.session_state["nav_page"] = selected
        st.query_params["page"] = selected
        st.session_state["_last_query_page"] = selected

    default_index = page_options.index(st.session_state["nav_page"])

    page = st.sidebar.radio(
        "Select Page",
        options=page_options,
        index=default_index,
        key="nav_radio_key",
        on_change=on_page_change,
        label_visibility="collapsed",
    )

    return {
        "dataset_key": dataset_key,
        "model_type": model_type,
        "calibration_method": calibration_method,
        "page": page,
    }
