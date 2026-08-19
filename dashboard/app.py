import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.components.styles import apply_enterprise_styles
from dashboard.components.sidebar import render_sidebar
from dashboard.components.header import render_header
from dashboard.components.overview import render_overview
from dashboard.components.customer_risk import render_customer_risk
from dashboard.components.model_performance import render_model_performance
from dashboard.components.drift_monitoring import render_drift_monitoring
from dashboard.components.experiments import render_experiments

from src.data_pipeline import DataPipeline
from src.explainability import ChurnExplainer
from src.drift_detector import DriftDetector


# Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Churn Risk Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Clean Enterprise Styling System
apply_enterprise_styles()


@st.cache_resource
def get_cached_components(dataset_key: str, model_type: str):
    pipeline = DataPipeline(dataset_key)
    explainer = ChurnExplainer(dataset_key, model_type)
    detector = DriftDetector(dataset_key)
    return pipeline, explainer, detector


def main():
    # Render Enterprise Sidebar & Navigation Controls
    controls = render_sidebar()
    dataset_key = controls["dataset_key"]
    model_type = controls["model_type"]
    calibration_method = controls["calibration_method"]
    selected_page = controls["page"]

    # Render Top Header & Compact Metadata Bar
    render_header(dataset_key, model_type, calibration_method, current_page=selected_page)

    # Load ML Pipeline Components
    pipeline, explainer, detector = get_cached_components(dataset_key, model_type)

    # Router to Selected Page
    if selected_page == "Overview":
        render_overview(pipeline, explainer, detector, dataset_key, model_type, calibration_method)
    elif selected_page == "Customer Risk":
        render_customer_risk(pipeline, explainer, dataset_key, model_type, calibration_method)
    elif selected_page == "Model Performance":
        render_model_performance(pipeline, explainer, dataset_key, model_type, calibration_method)
    elif selected_page == "Drift Monitoring":
        render_drift_monitoring(detector, dataset_key)
    elif selected_page == "Experiments":
        render_experiments(dataset_key)


if __name__ == "__main__":
    main()
