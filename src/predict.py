import pandas as pd
import joblib


# ==========================================================
# CUSTOMER CHURN - PREDICTION
# ==========================================================

print("=" * 60)
print("CUSTOMER CHURN - PREDICTION")
print("=" * 60)


# ----------------------------------------------------------
# 1. Load trained model
# ----------------------------------------------------------

model = joblib.load(
    "models/best_churn_model.pkl"
)


# ----------------------------------------------------------
# 2. Load preprocessor
# ----------------------------------------------------------

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

print("\nModel and preprocessor loaded successfully.")


# ----------------------------------------------------------
# 3. Create sample customer
# ----------------------------------------------------------

customer = pd.DataFrame([{

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

    "CLTV": 4000

}])


print("\nCustomer Information:")
print(customer.to_string(index=False))


# ----------------------------------------------------------
# 4. Preprocess customer data
# ----------------------------------------------------------

customer_processed = preprocessor.transform(
    customer
)


# ----------------------------------------------------------
# 5. Convert processed data to DataFrame
# ----------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

customer_processed = pd.DataFrame(
    customer_processed,
    columns=feature_names
)


# ----------------------------------------------------------
# 6. Make prediction
# ----------------------------------------------------------

prediction = model.predict(
    customer_processed
)[0]


# ----------------------------------------------------------
# 7. Calculate churn probability
# ----------------------------------------------------------

churn_probability = model.predict_proba(
    customer_processed
)[0][1]


# ----------------------------------------------------------
# 8. Display prediction
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)


if prediction == 1:

    print("\nPrediction: CUSTOMER LIKELY TO CHURN")

else:

    print("\nPrediction: CUSTOMER LIKELY TO STAY")


print(
    f"Churn Probability: {churn_probability:.2%}"
)


# ----------------------------------------------------------
# 9. Final message
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 60)