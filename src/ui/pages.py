"""Page renderers for the Customer Churn Intelligence Streamlit dashboard."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src.config import FEATURE_COLUMNS, REPORTS_DIR
from src.predict import explain_prediction, predict_churn
from src.ui.charts import (
    make_churn_by_category,
    make_churn_donut,
    make_churn_rate_by_group,
    make_confusion_matrix_chart,
    make_model_comparison_chart,
    make_monthly_charges_box,
    make_probability_gauge,
    make_risk_distribution_chart,
    make_shap_bar,
    make_tenure_histogram,
    make_threshold_tradeoff_chart,
)
from src.ui.data_loader import (
    get_cleaned_dataset,
    get_model_comparison,
    get_test_risk_distribution,
    get_threshold_analysis,
)
from src.ui.styles import (
    render_kpi_card,
    render_page_header,
    render_section_header,
)

DEFAULT_FORM_VALUES: Dict[str, Any] = {
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "DSL",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 50.0,
    "Total Charges": 600.0,
    "CLTV": 4000,
}

HIGH_RISK_EXAMPLE: Dict[str, Any] = {
    "Gender": "Female",
    "Senior Citizen": "Yes",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 3,
    "Phone Service": "Yes",
    "Multiple Lines": "Yes",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 98.5,
    "Total Charges": 295.0,
    "CLTV": 2400,
}

LOW_RISK_EXAMPLE: Dict[str, Any] = {
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "Yes",
    "Tenure Months": 64,
    "Phone Service": "Yes",
    "Multiple Lines": "Yes",
    "Internet Service": "DSL",
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "Yes",
    "Tech Support": "Yes",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Two year",
    "Paperless Billing": "No",
    "Payment Method": "Credit card (automatic)",
    "Monthly Charges": 62.0,
    "Total Charges": 3968.0,
    "CLTV": 5600,
}


def _init_form_state() -> None:
    """Initialize session state keys for the Customer Prediction form."""
    for key, val in DEFAULT_FORM_VALUES.items():
        state_key = f"form_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = val


def _apply_form_preset(preset: Dict[str, Any]) -> None:
    """Populate form session state with a preset dictionary."""
    for key, val in preset.items():
        st.session_state[f"form_{key}"] = val


def build_retention_recommendations(profile: Dict[str, Any], risk_tier: str) -> List[str]:
    """Generate transparent, rule-based retention suggestions from customer attributes."""
    if risk_tier == "Low Risk":
        return [
            "Maintain standard engagement; no immediate retention discount required.",
            "Consider periodic loyalty check-ins or cross-sell offers based on usage.",
        ]

    recs: List[str] = []
    if profile.get("Contract") == "Month-to-month":
        recs.append("Offer an annual or two-year contract migration incentive to reduce month-to-month flexibility risk.")
    if profile.get("Payment Method") == "Electronic check":
        recs.append("Encourage switching to automatic bank transfer or credit card billing with a small auto-pay credit.")
    if profile.get("Internet Service") == "Fiber optic" and profile.get("Tech Support") == "No":
        recs.append("Bundle complimentary or discounted Tech Support / Online Security to improve perceived service value.")
    if int(profile.get("Tenure Months", 0)) <= 12:
        recs.append("Prioritize early-tenure onboarding check-in during the first 12 months.")
    if not recs:
        recs.append("Schedule proactive customer success outreach to review plan fit and billing satisfaction.")
    return recs


# ==============================================================================
# 1. DASHBOARD HOME
# ==============================================================================
def render_dashboard_page(pipeline: Any, metadata: Dict[str, Any]) -> None:
    render_page_header(
        "Executive Churn Dashboard",
        "Verified overview of the IBM Telco Customer Churn dataset and deployed ML model performance.",
    )

    df = get_cleaned_dataset()
    test_tuned = metadata.get("test_metrics_tuned_threshold", {})
    test_default = metadata.get("test_metrics_default_0_50", {})
    thr = float(metadata.get("optimal_threshold", 0.35))
    low_max = float(metadata.get("risk_tiers", {}).get("low_max", 0.25))
    risk_df = get_test_risk_distribution(pipeline, thr, low_max)

    if df is None:
        st.warning("Cleaned dataset (`data/processed/cleaned_customer_churn.csv`) not found. Run `python run_pipeline.py` to generate it.")
        return

    total_customers = len(df)
    churned = int((df["Churn Label"] == 1).sum())
    stayed = int((df["Churn Label"] == 0).sum())
    churn_rate = churned / total_customers
    avg_tenure = float(df["Tenure Months"].mean())
    hist_mrr_lost = float(df.loc[df["Churn Label"] == 1, "Monthly Charges"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card("Total Customers", f"{total_customers:,}", "IBM Telco dataset")
    with c2:
        render_kpi_card("Churn Rate", f"{churn_rate:.2%}", "Historical baseline")
    with c3:
        render_kpi_card("Customers Stayed", f"{stayed:,}", "Retained subscribers")
    with c4:
        render_kpi_card("Customers Churned", f"{churned:,}", "Lost subscribers")
    with c5:
        acc = test_tuned.get("accuracy", test_default.get("accuracy", 0.0))
        render_kpi_card("Model Accuracy", f"{acc:.2%}", f"Test set (thr = {thr:.2f})")
    with c6:
        auc = test_tuned.get("roc_auc", test_default.get("roc_auc", 0.0))
        render_kpi_card("Model ROC-AUC", f"{auc:.4f}", metadata.get("best_model_name", "Gradient Boosting"))

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        render_kpi_card("Average Tenure", f"{avg_tenure:.1f} mo", "Across all 7,043 customers")
    with b2:
        render_kpi_card("Historical Churned MRR", f"${hist_mrr_lost:,.0f}/mo", "Sum of churned Monthly Charges")
    with b3:
        if risk_df is not None:
            high_risk_df = risk_df[risk_df["Risk Tier"] == "High Risk"]
            hr_mrr = float(high_risk_df["Monthly Charges"].sum())
            render_kpi_card(
                "High-Risk Test Cohort",
                f"{len(high_risk_df):,} / {len(risk_df):,}",
                f"${hr_mrr:,.0f}/mo at risk (thr ≥ {thr:.2f})",
            )
        else:
            render_kpi_card("High-Risk Test Cohort", "N/A", "Test split not found")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    r1_col1, r1_col2 = st.columns([1, 1.35])
    with r1_col1:
        st.plotly_chart(make_churn_donut(df), use_container_width=True)
    with r1_col2:
        st.plotly_chart(
            make_churn_by_category(df, "Contract", "Churn Volume by Contract Type"),
            use_container_width=True,
        )

    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.plotly_chart(make_tenure_histogram(df), use_container_width=True)
    with r2_col2:
        st.plotly_chart(make_monthly_charges_box(df), use_container_width=True)

    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        st.plotly_chart(
            make_churn_rate_by_group(df, "Tenure Group", "Churn Rate (%) Across Customer Tenure Cohorts"),
            use_container_width=True,
        )
    with r3_col2:
        if risk_df is not None:
            st.plotly_chart(make_risk_distribution_chart(risk_df), use_container_width=True)


# ==============================================================================
# 2. CUSTOMER PREDICTION PAGE (MOST IMPORTANT)
# ==============================================================================
def render_prediction_page(pipeline: Any, metadata: Dict[str, Any]) -> None:
    render_page_header(
        "Customer Churn Prediction",
        "Enter customer attributes or upload a batch CSV to estimate churn probability and inspect model drivers.",
    )

    mode_single, mode_batch = st.tabs(["Single Customer Assessment", "Batch CSV Scoring"])

    with mode_single:
        _init_form_state()

        # Sample / Reset Toolbar
        st.caption("Quick-fill with an example profile (synthetic demo profiles, not real customers) or reset all fields:")
        btn_c1, btn_c2, btn_c3, _ = st.columns([1.4, 1.4, 1.0, 1.2])
        with btn_c1:
            if st.button("Load High-Risk Example", use_container_width=True):
                _apply_form_preset(HIGH_RISK_EXAMPLE)
                st.session_state["last_prediction"] = None
                st.rerun()
        with btn_c2:
            if st.button("Load Low-Risk Example", use_container_width=True):
                _apply_form_preset(LOW_RISK_EXAMPLE)
                st.session_state["last_prediction"] = None
                st.rerun()
        with btn_c3:
            if st.button("Reset Form", use_container_width=True):
                _apply_form_preset(DEFAULT_FORM_VALUES)
                st.session_state["last_prediction"] = None
                st.rerun()

        # Section A: Customer Profile
        render_section_header(
            "Section A — Customer Profile",
            "Demographic attributes and relationship length with the telecom provider.",
        )
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.radio("Gender", ["Male", "Female"], horizontal=True, key="form_Gender")
        with p2:
            st.radio("Senior Citizen", ["No", "Yes"], horizontal=True, key="form_Senior Citizen", help="Whether the customer is aged 65 or older.")
        with p3:
            st.radio("Partner", ["No", "Yes"], horizontal=True, key="form_Partner", help="Whether the customer has a partner/spouse.")
        with p4:
            st.radio("Dependents", ["No", "Yes"], horizontal=True, key="form_Dependents", help="Whether the customer lives with dependents.")

        st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            key="form_Tenure Months",
            help="Number of months the customer has stayed with the company (0–72).",
        )

        # Section B: Services & Subscriptions
        render_section_header(
            "Section B — Services & Subscriptions",
            "Active phone, internet, security, and streaming add-on services.",
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            st.selectbox("Phone Service", ["Yes", "No"], key="form_Phone Service")
            st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], key="form_Multiple Lines")
            st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="form_Internet Service")
        with s2:
            st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="form_Online Security")
            st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="form_Online Backup")
            st.selectbox("Device Protection", ["No", "Yes", "No internet service"], key="form_Device Protection")
        with s3:
            st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="form_Tech Support")
            st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], key="form_Streaming TV")
            st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], key="form_Streaming Movies")

        # Section C: Contract & Billing
        render_section_header(
            "Section C — Contract & Billing",
            "Agreement terms, payment preferences, charges, and Customer Lifetime Value.",
        )
        b1, b2, b3 = st.columns(3)
        with b1:
            st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                key="form_Contract",
                help="Longer contracts typically correlate with lower churn.",
            )
            st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=500.0,
                step=1.0,
                format="%.2f",
                key="form_Monthly Charges",
                help="Current recurring monthly bill in USD.",
            )
        with b2:
            st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                key="form_Paperless Billing",
            )
            st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=50000.0,
                step=25.0,
                format="%.2f",
                key="form_Total Charges",
                help="Cumulative amount billed over the customer's tenure.",
            )
        with b3:
            st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                key="form_Payment Method",
            )
            st.number_input(
                "CLTV Score",
                min_value=0,
                max_value=15000,
                step=100,
                key="form_CLTV",
                help="Customer Lifetime Value score from the IBM Telco schema.",
            )

        customer_payload = {col: st.session_state[f"form_{col}"] for col in FEATURE_COLUMNS}
        customer_df = pd.DataFrame([customer_payload])

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        predict_clicked = st.button(
            "Predict Customer Churn",
            type="primary",
            use_container_width=True,
        )

        if predict_clicked:
            with st.spinner("Running preprocessing pipeline and model inference..."):
                try:
                    result_df = predict_churn(customer_df, pipeline=pipeline, metadata=metadata)
                    drivers = explain_prediction(customer_df, pipeline=pipeline, top_n=6)
                    res_payload = {
                        "probability": float(result_df["Churn Probability"].iloc[0]),
                        "prediction": int(result_df["Churn Prediction"].iloc[0]),
                        "risk_tier": str(result_df["Risk Tier"].iloc[0]),
                        "drivers": drivers,
                        "profile": customer_payload,
                    }
                    st.session_state["last_prediction"] = res_payload
                    history = st.session_state.setdefault("prediction_history", [])
                    history.insert(
                        0,
                        {
                            "Contract": customer_payload["Contract"],
                            "Tenure (mo)": customer_payload["Tenure Months"],
                            "Monthly ($)": customer_payload["Monthly Charges"],
                            "Internet": customer_payload["Internet Service"],
                            "Churn Prob": f"{res_payload['probability']:.1%}",
                            "Risk Tier": res_payload["risk_tier"],
                            "Outcome": "Churn" if res_payload["prediction"] == 1 else "Stay",
                        },
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        last_res = st.session_state.get("last_prediction")
        if last_res is not None:
            _render_prediction_result_card(last_res, metadata)

        history = st.session_state.get("prediction_history", [])
        if history:
            with st.expander(f"Session Prediction History ({len(history)} predictions this session)", expanded=False):
                st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
                if st.button("Clear Session History"):
                    st.session_state["prediction_history"] = []
                    st.rerun()

    with mode_batch:
        st.subheader("Batch Customer Scoring")
        st.write(
            "Upload a CSV file containing the 20 required customer feature columns to score multiple customers at once."
        )
        with st.expander("View Required CSV Column Names"):
            st.code(", ".join(FEATURE_COLUMNS))

        uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                scored = predict_churn(batch_df, pipeline=pipeline, metadata=metadata)
                m1, m2, m3 = st.columns(3)
                with m1:
                    render_kpi_card("Customers Scored", f"{len(scored):,}", "Uploaded rows")
                with m2:
                    churn_cnt = int((scored["Churn Prediction"] == 1).sum())
                    render_kpi_card("Predicted Churners", f"{churn_cnt:,}", f"At {metadata.get('optimal_threshold', 0.35):.2f} threshold")
                with m3:
                    high_share = (scored["Risk Tier"] == "High Risk").mean()
                    render_kpi_card("High Risk Share", f"{high_share:.1%}", "Immediate retention focus")

                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                st.dataframe(scored.head(50), use_container_width=True)
                st.download_button(
                    "Download Scored Predictions CSV",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Could not process uploaded CSV: {exc}")


def _render_prediction_result_card(res: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Render the styled prediction result banner, gauge, summary, and SHAP drivers."""
    prob = res["probability"]
    retain_prob = 1.0 - prob
    pred = res["prediction"]
    tier = res["risk_tier"]
    threshold = float(metadata.get("optimal_threshold", 0.35))
    low_max = float(metadata.get("risk_tiers", {}).get("low_max", 0.25))
    profile = res["profile"]

    st.divider()
    if pred == 1:
        st.markdown(
            f"""
            <div class="result-card-churn">
                <div class="result-badge-churn">HIGH CHURN RISK</div>
                <div class="result-heading">Customer Predicted to Churn ({prob:.1%} Probability)</div>
                <p class="result-copy">
                    Based on the customer's profile, the model estimates an elevated likelihood of churn
                    (above the CV-validated {threshold:.0%} decision threshold). Proactive retention outreach is recommended.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-stay">
                <div class="result-badge-stay">CUSTOMER LIKELY TO STAY</div>
                <div class="result-heading">Customer Predicted to Stay ({retain_prob:.1%} Retention Probability)</div>
                <p class="result-copy">
                    Based on the information provided, the model estimates a {prob:.1%} churn probability
                    (below the CV-validated {threshold:.0%} decision threshold).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1, 1.25])
    with left:
        st.plotly_chart(make_probability_gauge(prob, threshold), use_container_width=True)
        pill_cls = (
            "risk-high" if tier == "High Risk" else ("risk-medium" if tier == "Medium Risk" else "risk-low")
        )
        st.markdown(
            f"**Risk Classification:** <span class='risk-pill {pill_cls}'>{tier}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Thresholds are anchored to the 5-fold CV-validated decision threshold ({threshold:.2f}): "
            f"Low (< {low_max:.2f}), Medium ({low_max:.2f}–{threshold:.2f}), High (≥ {threshold:.2f})."
        )

        # Concise Customer Profile Summary
        active_addons = sum(
            1
            for k in [
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies",
            ]
            if profile.get(k) == "Yes"
        )
        st.markdown("**Customer Summary**")
        st.write(
            f"- **Tenure:** {profile['Tenure Months']} months\n"
            f"- **Contract:** {profile['Contract']}\n"
            f"- **Monthly Billing:** ${profile['Monthly Charges']:.2f} (Total: ${profile['Total Charges']:.2f})\n"
            f"- **Internet & Add-ons:** {profile['Internet Service']} ({active_addons} add-ons active)\n"
            f"- **Payment Method:** {profile['Payment Method']}"
        )

    with right:
        drivers = res.get("drivers", [])
        if drivers:
            st.plotly_chart(make_shap_bar(drivers), use_container_width=True)
            st.caption(
                "Note: Local SHAP values measure how each feature shifted this model's log-odds prediction "
                "relative to the baseline. They indicate model influence and statistical association, not causation."
            )
        else:
            st.info("Local feature explanation is unavailable for this model type.")

        st.markdown("**Suggested Retention Actions (Rule-Based Business Guidance)**")
        for rec in build_retention_recommendations(profile, tier):
            st.write(f"- {rec}")


# ==============================================================================
# 3. ANALYTICS PAGE
# ==============================================================================
def render_analytics_page() -> None:
    render_page_header(
        "Customer Analytics & Segmentation",
        "Interactive exploration of historical churn patterns across contracts, tenure, services, and billing.",
    )

    df = get_cleaned_dataset()
    if df is None:
        st.warning("Cleaned dataset not found. Run `python run_pipeline.py` first.")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        contracts = st.multiselect(
            "Contract Type",
            options=sorted(df["Contract"].unique()),
            default=sorted(df["Contract"].unique()),
        )
    with f2:
        internet_opts = st.multiselect(
            "Internet Service",
            options=sorted(df["Internet Service"].unique()),
            default=sorted(df["Internet Service"].unique()),
        )
    with f3:
        tenure_opts = st.multiselect(
            "Tenure Cohort",
            options=list(df["Tenure Group"].cat.categories),
            default=list(df["Tenure Group"].cat.categories),
        )

    filtered = df[
        df["Contract"].isin(contracts)
        & df["Internet Service"].isin(internet_opts)
        & df["Tenure Group"].isin(tenure_opts)
    ]

    if filtered.empty:
        st.info("No customers match the selected filters. Adjust filter selections above.")
        return

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Filtered Customers", f"{len(filtered):,}", f"of {len(df):,} total")
    with k2:
        render_kpi_card("Segment Churn Rate", f"{filtered['Churn Label'].mean():.2%}", "Within selected filters")
    with k3:
        render_kpi_card("Avg Monthly Bill", f"${filtered['Monthly Charges'].mean():.2f}", "Mean recurring charge")
    with k4:
        render_kpi_card("Avg Tenure", f"{filtered['Tenure Months'].mean():.1f} mo", "Mean customer lifetime")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_churn_by_category(filtered, "Contract", "Churn by Contract Type"), use_container_width=True)
    with c2:
        st.plotly_chart(make_churn_by_category(filtered, "Internet Service", "Churn by Internet Service"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(make_churn_by_category(filtered, "Payment Method", "Churn by Payment Method"), use_container_width=True)
    with c4:
        st.plotly_chart(make_churn_rate_by_group(filtered, "Tenure Group", "Churn Rate by Tenure Group"), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(make_tenure_histogram(filtered), use_container_width=True)
    with c6:
        st.plotly_chart(make_monthly_charges_box(filtered), use_container_width=True)


# ==============================================================================
# 4. MODEL INSIGHTS PAGE
# ==============================================================================
def render_model_insights_page(metadata: Dict[str, Any]) -> None:
    render_page_header(
        "Model Performance & Insights",
        "Verified evaluation metrics from 5-fold Stratified Cross-Validation and the untouched test set (n = 1,409).",
    )

    thr_choice = st.radio(
        "Inspect Test Metrics At Decision Threshold:",
        [
            f"CV-Tuned Threshold ({metadata.get('optimal_threshold', 0.35):.2f}) — Active in App",
            "Default Threshold (0.50)",
        ],
        horizontal=True,
    )
    use_tuned = thr_choice.startswith("CV-Tuned")
    metrics = (
        metadata.get("test_metrics_tuned_threshold", {})
        if use_tuned
        else metadata.get("test_metrics_default_0_50", {})
    )

    if metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            render_kpi_card("Accuracy", f"{metrics.get('accuracy', 0):.2%}", "Overall correct predictions")
        with m2:
            render_kpi_card("Precision", f"{metrics.get('precision', 0):.2%}", "Reliability of churn alerts")
        with m3:
            render_kpi_card("Recall", f"{metrics.get('recall', 0):.2%}", "Actual churners caught")
        with m4:
            render_kpi_card("F1 Score", f"{metrics.get('f1', 0):.4f}", "Precision–Recall balance")
        with m5:
            render_kpi_card("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}", "Class separation quality")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    comp_df = get_model_comparison()
    thr_df = get_threshold_analysis()

    col_left, col_right = st.columns(2)
    with col_left:
        if comp_df is not None:
            st.plotly_chart(make_model_comparison_chart(comp_df), use_container_width=True)
    with col_right:
        if thr_df is not None:
            st.plotly_chart(
                make_threshold_tradeoff_chart(thr_df, float(metadata.get("optimal_threshold", 0.35))),
                use_container_width=True,
            )

    cm_col, roc_col = st.columns(2)
    with cm_col:
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            st.plotly_chart(
                make_confusion_matrix_chart(
                    np.array(cm),
                    f"Test Confusion Matrix (thr = {metrics.get('threshold', 0.35):.2f})",
                ),
                use_container_width=True,
            )
    with roc_col:
        roc_path = REPORTS_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Verified Test ROC Curve", use_container_width=True)

    if comp_df is not None:
        st.subheader("5-Fold Stratified Cross-Validation Results")
        display_df = comp_df.drop(columns=["Best Params"]) if "Best Params" in comp_df.columns else comp_df
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    feat_path = REPORTS_DIR / "feature_importance.png"
    if feat_path.exists():
        st.subheader("Global Feature Importance (Permutation Importance on Test Set)")
        st.image(str(feat_path), use_container_width=True)


# ==============================================================================
# 5. ABOUT MODEL PAGE
# ==============================================================================
def render_about_model_page(metadata: Dict[str, Any]) -> None:
    render_page_header(
        "About This Model & Methodology",
        "Architecture, leakage prevention, threshold rationale, and known limitations.",
    )

    st.markdown(
        f"""
        ### 1. Selected Pipeline
        - **Estimator:** `{metadata.get('best_model_name', 'Gradient Boosting')}` (`GradientBoostingClassifier`)
        - **Best Hyperparameters (5-Fold CV):** `{metadata.get('best_params', {})}`
        - **Preprocessing (`ColumnTransformer`):**
          - Numerical features (`Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`): Median Imputation + `StandardScaler`
          - Categorical features (16 demographic, service, and billing attributes): Mode Imputation + `OneHotEncoder(handle_unknown="ignore")`
        - **Leakage Prevention:** All imputation, scaling, encoding, model selection, and threshold optimization are fit **strictly on the 80% training split** (`n = 5,634`). The 20% test split (`n = 1,409`) is evaluated once at the end.

        ### 2. Why Decision Threshold `{metadata.get('optimal_threshold', 0.35):.2f}` Was Selected
        In telecom churn, **False Negatives** (missing a customer who leaves) cost recurring revenue and Customer Lifetime Value, while **False Positives** only trigger a low-cost retention outreach. Using out-of-fold CV probabilities on the training set, lowering the threshold from `0.50` to **`{metadata.get('optimal_threshold', 0.35):.2f}`** raised test-set Recall from **52.41% to 71.93%** (reducing missed churners from 178 to 105 out of 374).

        ### 3. Explainability & Limitations
        - **Global Importance:** Permutation importance on the untouched test set measures each feature's contribution to ROC-AUC.
        - **Local Explanations:** SHAP values (`TreeExplainer`) explain individual predictions relative to the model baseline. These reflect **statistical model influence, not causal proof**.
        - **Dataset Scope:** Trained on the public IBM Telco Customer Churn dataset (7,043 rows). Real-world deployment on new telecom cohorts requires monitoring for data drift and periodic recalibration.
        """
    )
