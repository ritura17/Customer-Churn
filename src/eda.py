import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

file_path = "data/processed/cleaned_customer_churn.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# 2. Churn Distribution
# --------------------------------------------------

print("\nChurn Distribution:")
print(df["Churn Label"].value_counts())

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Churn Label"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("reports/churn_distribution.png")

plt.show()


# --------------------------------------------------
# 3. Churn by Contract Type
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn Label"
)

plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("reports/churn_by_contract.png")

plt.show()


# --------------------------------------------------
# 4. Churn by Internet Service
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Internet Service",
    hue="Churn Label"
)

plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig("reports/churn_by_internet_service.png")

plt.show()


# --------------------------------------------------
# 5. Churn by Tenure
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn Label",
    y="Tenure Months"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig("reports/churn_by_tenure.png")

plt.show()


# --------------------------------------------------
# 6. Churn by Monthly Charges
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn Label",
    y="Monthly Charges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig("reports/churn_by_monthly_charges.png")

plt.show()


# --------------------------------------------------
# 7. Churn by Payment Method
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="Payment Method",
    hue="Churn Label"
)

plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig("reports/churn_by_payment_method.png")

plt.show()


print("\nEDA completed successfully!")
print("Charts saved in reports/ folder.")