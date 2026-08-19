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
        }
    </style>
    """, unsafe_allow_html=True)
