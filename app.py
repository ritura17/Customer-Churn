import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# CUSTOMER CHURN PREDICTION - STREAMLIT APP
# ==========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ----------------------------------------------------------
# Load Model and Preprocessor
# ----------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/best_churn_model.pkl"
    )

    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    return model, preprocessor


model, preprocessor = load_model()


# ----------------------------------------------------------
# Header
# ----------------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn "
    "using a machine learning model."
)

st.divider()


# ----------------------------------------------------------
# Customer Information
# ----------------------------------------------------------

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )


with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# ----------------------------------------------------------
# Financial Information
# ----------------------------------------------------------

st.divider()

st.header("💰 Financial Information")

col4, col5, col6 = st.columns(3)


with col4:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=1.0
    )


with col5:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0,
        step=10.0
    )


with col6:

    cltv = st.number_input(
        "CLTV",
        min_value=0,
        value=4000,
        step=100
    )


# ----------------------------------------------------------
# Create Customer DataFrame
# ----------------------------------------------------------

customer = pd.DataFrame([{

    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges,
    "CLTV": cltv

}])


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
):

    # Preprocess customer data
    customer_processed = preprocessor.transform(
        customer
    )

    # Get feature names
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # Convert to DataFrame
    customer_processed = pd.DataFrame(
        customer_processed,
        columns=feature_names
    )

    # Prediction
    prediction = model.predict(
        customer_processed
    )[0]

    # Churn probability
    probability = model.predict_proba(
        customer_processed
    )[0][1]


    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    st.divider()

    st.header("🎯 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn"
        )

    else:

        st.success(
            "✅ Customer is likely to stay"
        )


    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )


    st.progress(
        float(probability)
    )


    if probability >= 0.5:

        st.warning(
            "This customer has a relatively high "
            "probability of churn."
        )

    else:

        st.info(
            "This customer has a relatively low "
            "probability of churn."
        )