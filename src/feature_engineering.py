"""Feature validation, stratified splitting, and sklearn preprocessing pipeline."""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CLEANED_DATA_PATH,
    DROP_GEO_COLUMNS,
    FEATURE_COLUMNS,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)


def validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that input DataFrame contains all required features and valid numeric values."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    validated = df[FEATURE_COLUMNS].copy()
    for col in NUMERICAL_FEATURES:
        validated[col] = pd.to_numeric(validated[col], errors="coerce")
        if (validated[col].dropna() < 0).any():
            raise ValueError(f"Numerical feature '{col}' cannot contain negative values.")

    for col in CATEGORICAL_FEATURES:
        validated[col] = validated[col].astype("object")

    return validated


def build_preprocessor() -> ColumnTransformer:
    """Build leakage-free ColumnTransformer with imputation, scaling, and one-hot encoding."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=True,
    )


def prepare_train_test_splits(
    cleaned_path: Path = CLEANED_DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load cleaned dataset, perform stratified train/test split, and save raw split CSVs."""
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_path}. Run data_preprocessing.py first."
        )

    df = pd.read_csv(cleaned_path)
    drop_cols = [c for c in [TARGET_COLUMN, *DROP_GEO_COLUMNS] if c in df.columns]
    X = validate_input_dataframe(df.drop(columns=drop_cols))
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)

    # Fit and save standalone preprocessor solely on X_train for inspection/compatibility
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("=" * 60)
    print("CUSTOMER CHURN - FEATURE ENGINEERING & SPLIT")
    print("=" * 60)
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.4f} | Test churn rate: {y_test.mean():.4f}")
    print(f"Saved split datasets to: {PROCESSED_DATA_DIR}")
    print(f"Saved preprocessor to: {PREPROCESSOR_PATH}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_train_test_splits()
