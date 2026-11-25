"""Customer Churn Intelligence — Streamlit Application Entrypoint."""

import streamlit as st

from src.ui.data_loader import get_cached_model_artifacts
from src.ui.pages import (
    render_about_model_page,
    render_analytics_page,
    render_dashboard_page,
    render_model_insights_page,
    render_prediction_page,
)
from src.ui.styles import inject_custom_css

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

try:
    pipeline, metadata = get_cached_model_artifacts()
except Exception as exc:
    st.error(
        f"Unable to load trained model artifacts: {exc}\n\n"
        "Run `python run_pipeline.py` from the project root to generate the pipeline."
    )
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-title">Customer Churn Intelligence</div>
            <div class="brand-subtitle">Predictive Retention Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=[
            "Dashboard",
            "Customer Prediction",
            "Analytics",
            "Model Insights",
            "About Model",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(
        f"**Active Model:** {metadata.get('best_model_name', 'Gradient Boosting')}\n\n"
        f"**Decision Threshold:** {metadata.get('optimal_threshold', 0.35):.2f} (CV-Tuned)\n\n"
        f"**Test ROC-AUC:** {metadata.get('test_metrics_tuned_threshold', {}).get('roc_auc', 0.8554):.4f}"
    )

if page == "Dashboard":
    render_dashboard_page(pipeline, metadata)
elif page == "Customer Prediction":
    render_prediction_page(pipeline, metadata)
elif page == "Analytics":
    render_analytics_page()
elif page == "Model Insights":
    render_model_insights_page(metadata)
elif page == "About Model":
    render_about_model_page(metadata)
