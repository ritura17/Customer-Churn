import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================================
# CUSTOMER CHURN - FEATURE ENGINEERING
# ==========================================================

print("=" * 60)
print("CUSTOMER CHURN - FEATURE ENGINEERING")
print("=" * 60)


# ----------------------------------------------------------
# 1. Load cleaned dataset
# ----------------------------------------------------------

file_path = "data/processed/cleaned_customer_churn.csv"

df = pd.read_csv(file_path)

print("\nDataset Shape:")
print(df.shape)


# ----------------------------------------------------------
# 2. Separate features and target
# ----------------------------------------------------------

# Remove target and unnecessary geographic features
columns_to_drop = [
    "Churn Label",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude"
]

X = df.drop(columns=columns_to_drop)

y = df["Churn Label"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ----------------------------------------------------------
# 3. Identify numerical and categorical features
# ----------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ----------------------------------------------------------
# 4. Train/Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ----------------------------------------------------------
# 5. Create preprocessing transformer
# ----------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ----------------------------------------------------------
# 6. Fit preprocessing ONLY on training data
# ----------------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ----------------------------------------------------------
# 7. Save preprocessing pipeline
# ----------------------------------------------------------

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\nPreprocessor saved:")
print("models/preprocessor.pkl")


# ----------------------------------------------------------
# 8. Get processed feature names
# ----------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()


# ----------------------------------------------------------
# 9. Convert processed data to DataFrames
# ----------------------------------------------------------

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# ----------------------------------------------------------
# 10. Save processed datasets
# ----------------------------------------------------------

X_train_processed.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test_processed.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)


# ----------------------------------------------------------
# 11. Display final information
# ----------------------------------------------------------

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)

print("\nTarget Distribution - Training:")
print(y_train.value_counts())

print("\nTarget Distribution - Testing:")
print(y_test.value_counts())


# ----------------------------------------------------------
# 12. Final message
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles created:")

print("data/processed/X_train.csv")
print("data/processed/X_test.csv")
print("data/processed/y_train.csv")
print("data/processed/y_test.csv")
print("models/preprocessor.pkl")