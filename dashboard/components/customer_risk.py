import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Any

from src.config import DATASET_CONFIGS
from src.data_pipeline import DataPipeline
from src.calibration import ProbabilityCalibrator
from src.explainability import ChurnExplainer


def render_customer_risk(pipeline: DataPipeline, explainer: ChurnExplainer, dataset_key: str, model_type: str, calibration_method: str):
    """Render main interactive Customer Risk Assessment and SHAP explanation page."""
    st.subheader("Customer Risk Assessment")
    st.caption("Evaluate an individual customer's calibrated churn probability and the factors influencing the prediction.")

    # SECTION A: CUSTOMER PROFILE
    st.markdown("### SECTION A: Customer Profile")

    input_dict = {}

    if dataset_key == "telco":
        col1, col2, col3 = st.columns(3)
        with col1:
            input_dict["Contract"] = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
            input_dict["tenure"] = st.slider("Tenure (Months)", 1, 72, 12)
            input_dict["MonthlyCharges"] = st.slider("Monthly Charges ($)", 18.0, 120.0, 85.0)
            input_dict["TotalCharges"] = round(input_dict["tenure"] * input_dict["MonthlyCharges"], 2)
        with col2:
            input_dict["InternetService"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
            input_dict["PaymentMethod"] = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
            input_dict["TechSupport"] = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
            input_dict["OnlineSecurity"] = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
        with col3:
            input_dict["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            input_dict["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1], index=0)
            input_dict["Partner"] = st.selectbox("Partner", ["Yes", "No"], index=1)
            input_dict["Dependents"] = st.selectbox("Dependents", ["Yes", "No"], index=1)
            input_dict["gender"] = st.selectbox("Gender", ["Female", "Male"], index=0)

    elif dataset_key == "bank":
        col1, col2, col3 = st.columns(3)
        with col1:
            input_dict["Age"] = st.slider("Customer Age", 18, 90, 48)
            input_dict["Geography"] = st.selectbox("Geography", ["France", "Germany", "Spain"], index=1)
            input_dict["IsActiveMember"] = st.selectbox("Is Active Member", [1, 0], index=1)
        with col2:
            input_dict["NumOfProducts"] = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
            input_dict["Balance"] = st.number_input("Account Balance ($)", 0.0, 250000.0, 95000.0)
            input_dict["CreditScore"] = st.slider("Credit Score", 350, 850, 610)
        with col3:
            input_dict["Gender"] = st.selectbox("Gender", ["Female", "Male"], index=0)
            input_dict["Tenure"] = st.slider("Bank Tenure (Years)", 0, 10, 3)
            input_dict["EstimatedSalary"] = st.number_input("Estimated Salary ($)", 10000.0, 200000.0, 110000.0)
            input_dict["HasCrCard"] = st.selectbox("Has Credit Card", [1, 0], index=0)

    st.markdown("---")

    # SECTION B: PREDICTION
    st.markdown("### SECTION B: Prediction")

    # Process input sample
    df_single = pd.DataFrame([input_dict])
    X_single_proc = pipeline.transform(df_single)

    raw_prob = float(explainer.model.predict_proba(X_single_proc)[0, 1])

    if calibration_method == "uncalibrated":
        calibrated_prob = raw_prob
    else:
        calibrator = ProbabilityCalibrator.load(dataset_key, model_type, calibration_method, explainer.model)
        calibrated_prob = float(calibrator.predict_proba(X_single_proc)[0, 1])

    thresholds = DATASET_CONFIGS[dataset_key]["default_risk_thresholds"]
    if calibrated_prob < thresholds["low"]:
        risk_label = "LOW RISK"
        badge_class = "badge-normal"
    elif calibrated_prob < thresholds["high"]:
        risk_label = "MEDIUM RISK"
        badge_class = "badge-warning"
    else:
        risk_label = "HIGH RISK"
        badge_class = "badge-drift"

    cal_adjustment = (calibrated_prob - raw_prob) * 100

    col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
    with col_pred1:
        st.markdown(f"**Risk Classification:**<br><span class='{badge_class}'>{risk_label}</span>", unsafe_allow_html=True)
    with col_pred2:
        st.markdown(f"**Calibrated Churn Probability:**<br><span style='font-size:22px; font-weight:700;'>{calibrated_prob * 100:.1f}%</span>", unsafe_allow_html=True)
    with col_pred3:
        st.markdown(f"**Raw Model Score:**<br><span style='font-size:22px; font-weight:700; color:#94A3B8;'>{raw_prob * 100:.1f}%</span>", unsafe_allow_html=True)
    with col_pred4:
        adj_color = "#10B981" if cal_adjustment <= 0 else "#EF4444"
        st.markdown(f"**Calibration Adjustment:**<br><span style='font-size:22px; font-weight:700; color:{adj_color};'>{cal_adjustment:+.1f} pts</span>", unsafe_allow_html=True)

    st.markdown("---")

    # SECTION C: SHAP EXPLANATION ("Why this prediction?")
    st.markdown("### Why this prediction?")
    st.caption("SHAP feature contributions influencing the predicted churn probability.")

    explanation = explainer.explain_sample(X_single_proc[0], input_dict)
    attributions = explanation["attributions"][:8]

    df_shap = pd.DataFrame(attributions)
    df_shap["Impact"] = np.where(df_shap["shap_value"] > 0, "Increases Churn Risk (+)", "Reduces Churn Risk (-)")

    # Dynamic Natural-Language Summary generated directly from actual SHAP outputs
    pos_drivers = [item["feature"] for item in attributions if item["shap_value"] > 0][:3]
    neg_drivers = [item["feature"] for item in attributions if item["shap_value"] < 0][:2]

    if pos_drivers:
        summary_text = f"Primary churn drivers increasing risk are **{', '.join(pos_drivers)}**."
        if neg_drivers:
            summary_text += f" Mitigating factors reducing risk include **{', '.join(neg_drivers)}**."
    else:
        summary_text = "This customer profile shows a low risk distribution across evaluated features."

    st.markdown(f"<div class='info-panel'>{summary_text}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_recs = st.columns([1.4, 1])

    with col_chart:
        fig_shap = px.bar(
            df_shap,
            x="shap_value",
            y="feature",
            orientation="h",
            color="Impact",
            color_discrete_map={
                "Increases Churn Risk (+)": "#EF4444",
                "Reduces Churn Risk (-)": "#10B981",
            },
            title="Feature SHAP Contribution Value",
            labels={"shap_value": "SHAP Impact Value", "feature": "Feature"},
        )
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_recs:
        st.markdown("#### Actionable Retention Interventions")
        for rec in explanation["recommendations"]:
            st.markdown(f"<div class='info-panel'>{rec}</div>", unsafe_allow_html=True)
