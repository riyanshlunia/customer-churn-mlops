import streamlit as st

def render_introduction():
    """Render a recruiter-focused landing page for the project."""
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Enterprise Customer Churn MLOps Platform</h1>
        <p class="hero-subtitle">An End-to-End Machine Learning Pipeline Built for Scale and Reliability</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # The Problem & Solution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 class="card-title">The Business Problem</h3>
            <p class="card-text">
            Customer churn is a silent revenue killer. Identifying which customers are at risk of leaving <i>before</i> they actually churn is critical for proactive retention strategies. However, standard models often lack the reliability, explainability, and continuous monitoring required for enterprise deployment.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3 class="card-title">The Engineering Solution</h3>
            <p class="card-text">
            This project implements a robust <b>MLOps pipeline</b>. It doesn't just train a model; it provides a production-ready ecosystem featuring automated data drift detection, strict probability calibration, interactive SHAP explainability, and comprehensive experiment tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Architectural Highlights
    st.markdown("### Architecture & Engineering Highlights")
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <h4>Robust Model Training</h4>
            <p>Utilizes state-of-the-art gradient boosting algorithms (LightGBM & XGBoost). Features probability calibration (Isotonic/Platt) ensuring that raw model scores reflect true real-world risk probabilities.</p>
        </div>
        <div class="feature-card">
            <h4>Explainable AI (XAI)</h4>
            <p>Integrates SHAP (SHapley Additive exPlanations) to provide global and local interpretability. Stakeholders can understand exactly <i>why</i> a specific customer was flagged as high-risk.</p>
        </div>
        <div class="feature-card">
            <h4>Data Drift Monitoring</h4>
            <p>Built-in statistical tests (Kolmogorov-Smirnov & Population Stability Index) automatically monitor incoming data distributions against the training baseline to detect model degradation.</p>
        </div>
        <div class="feature-card">
            <h4>MLOps & Tracking</h4>
            <p>Seamlessly integrated with MLflow for tracking parameters, metrics, and serialized model artifacts. Ensures complete reproducibility and streamlined model lifecycle management.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech Stack
    st.markdown("### Technology Stack")
    st.markdown("""
    <div class="tech-stack-container">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">LightGBM</span>
        <span class="tech-badge">XGBoost</span>
        <span class="tech-badge">Scikit-Learn</span>
        <span class="tech-badge">MLflow</span>
        <span class="tech-badge">FastAPI</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">SHAP</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">Pandas</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-hint">
        <b>Explore the platform:</b> Use the sidebar on the left to navigate to the <i>Overview</i>, inspect <i>Customer Risk</i> profiles, or view <i>Drift Monitoring</i> metrics.
    </div>
    """, unsafe_allow_html=True)
