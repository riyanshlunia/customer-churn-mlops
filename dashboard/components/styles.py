import streamlit as st

def apply_enterprise_styles():
    """Apply clean, restrained enterprise analytics visual styles to Streamlit."""
    st.markdown("""
    <style>
        /* Global Container & Typography */
        .main {
            background-color: #0B0F19;
            color: #F8FAFC;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }

        /* Metric Cards */
        .stMetric {
            background: #1E293B;
            padding: 16px 20px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: none;
        }

        /* KPI Card Container */
        .kpi-card {
            background-color: #1E293B;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }
        .kpi-label {
            font-size: 12px;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 24px;
            font-weight: 700;
            color: #F8FAFC;
        }
        .kpi-subtext {
            font-size: 12px;
            color: #64748B;
            margin-top: 4px;
        }

        /* Restrained Status Badges */
        .badge-normal {
            display: inline-block;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 700;
            color: #10B981;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid #10B981;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-warning {
            display: inline-block;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 700;
            color: #F59E0B;
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid #F59E0B;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-drift {
            display: inline-block;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 700;
            color: #EF4444;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid #EF4444;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Recommendation & Explanation Boxes */
        .info-panel {
            background-color: #1E293B;
            border-left: 3px solid #6366F1;
            padding: 12px 16px;
            margin-top: 8px;
            border-radius: 0 4px 4px 0;
            font-size: 13px;
            color: #E2E8F0;
        }

        /* Section Headings */
        .section-header {
            font-size: 16px;
            font-weight: 700;
            color: #F8FAFC;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
            margin-top: 16px;
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Compact Metadata Bar */
        .meta-bar {
            background-color: #1E293B;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 12px;
            color: #94A3B8;
            margin-bottom: 24px;
        }
        .meta-item {
            display: inline-block;
            margin-right: 24px;
        }
        .meta-item strong {
            color: #F8FAFC;
        }

        /* Clean Table Headers */
        .stDataFrame {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 4px;
        /* Introduction Page Styles */
        .hero-section {
            text-align: center;
            padding: 40px 20px 30px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(30, 41, 59, 0) 100%);
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        .hero-title {
            font-size: 36px;
            font-weight: 800;
            color: #F8FAFC;
            margin-bottom: 12px;
            background: -webkit-linear-gradient(45deg, #818CF8, #38BDF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-subtitle {
            font-size: 18px;
            color: #94A3B8;
            font-weight: 500;
        }
        .info-card {
            background-color: #1E293B;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 24px;
            height: 100%;
        }
        .card-title {
            font-size: 18px;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 8px;
        }
        .card-text {
            font-size: 14px;
            color: #CBD5E1;
            line-height: 1.6;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .feature-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.2s, background 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-2px);
            background: rgba(30, 41, 59, 0.8);
            border-color: rgba(99, 102, 241, 0.4);
        }
        .feature-card h4 {
            font-size: 16px;
            color: #E2E8F0;
            margin-bottom: 10px;
        }
        .feature-card p {
            font-size: 13px;
            color: #94A3B8;
            line-height: 1.5;
        }
        .tech-stack-container {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 16px;
        }
        .tech-badge {
            background: rgba(56, 189, 248, 0.1);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.2);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .nav-hint {
            background-color: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10B981;
            padding: 16px;
            border-radius: 4px;
            color: #E2E8F0;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)
