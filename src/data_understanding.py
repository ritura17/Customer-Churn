import pandas as pd

# Load dataset
file_path = "data/raw/Customer_churn.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("CUSTOMER CHURN DATASET - DATA UNDERSTANDING")
print("=" * 60)

# 1. Dataset shape
print("\n1. Dataset Shape:")
print(df.shape)

# 2. Column names
print("\n2. Column Names:")
print(df.columns.tolist())

# 3. Dataset information
print("\n3. Dataset Information:")
print(df.info())

# 4. Missing values
print("\n4. Missing Values:")
print(df.isnull().sum())

# 5. Duplicate rows
print("\n5. Duplicate Rows:")
print(df.duplicated().sum())

# 6. Statistical summary
print("\n6. Statistical Summary:")
print(df.describe())

# 7. Churn distribution
print("\n7. Churn Distribution:")
print(df["Churn Label"].value_counts())

# 8. Churn percentage
print("\n8. Churn Percentage:")
print(df["Churn Label"].value_counts(normalize=True) * 100)