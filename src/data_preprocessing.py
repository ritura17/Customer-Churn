import pandas as pd

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/raw/Customer_churn.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("CUSTOMER CHURN - DATA PREPROCESSING")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(df.shape)


# --------------------------------------------------
# 2. Remove unnecessary and leakage columns
# --------------------------------------------------

columns_to_drop = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "Lat Long",
    "Churn Value",
    "Churn Score",
    "Churn Reason"
]

df = df.drop(columns=columns_to_drop)

print("\nShape after removing unnecessary/leakage columns:")
print(df.shape)


# --------------------------------------------------
# 3. Convert Total Charges to numeric
# --------------------------------------------------

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

print("\nMissing values after converting Total Charges:")
print(df["Total Charges"].isnull().sum())


# --------------------------------------------------
# 4. Handle missing values
# --------------------------------------------------

df["Total Charges"] = df["Total Charges"].fillna(
    df["Total Charges"].median()
)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())


# --------------------------------------------------
# 5. Convert target variable
# --------------------------------------------------

df["Churn Label"] = df["Churn Label"].map({
    "Yes": 1,
    "No": 0
})

print("\nTarget variable:")
print(df["Churn Label"].value_counts())


# --------------------------------------------------
# 6. Save cleaned dataset
# --------------------------------------------------

output_path = "data/processed/cleaned_customer_churn.csv"

df.to_csv(output_path, index=False)

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nCleaned dataset saved successfully!")
print(output_path)