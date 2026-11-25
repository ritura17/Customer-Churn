"""Data cleaning and schema preparation for the IBM Telco Customer Churn dataset.

Note: Missing-value imputation is intentionally NOT performed here on the full
dataset to prevent train/test data leakage. Missing values in numeric columns
are coerced to NaN and imputed inside the scikit-learn Pipeline fit on X_train.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_DATA_PATH,
    DROP_RAW_COLUMNS,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco churn DataFrame without leaking test-set statistics."""
    cleaned = df.copy()

    cols_to_drop = [c for c in DROP_RAW_COLUMNS if c in cleaned.columns]
    cleaned = cleaned.drop(columns=cols_to_drop)

    if "Total Charges" in cleaned.columns:
        cleaned["Total Charges"] = pd.to_numeric(
            cleaned["Total Charges"], errors="coerce"
        )

    if TARGET_COLUMN in cleaned.columns:
        if cleaned[TARGET_COLUMN].dtype == object or str(cleaned[TARGET_COLUMN].dtype) == "str":
            cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"Yes": 1, "No": 0})
        if cleaned[TARGET_COLUMN].isnull().any():
            raise ValueError(
                f"Unexpected values in target column '{TARGET_COLUMN}'; "
                "expected 'Yes'/'No' or 1/0."
            )
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    return cleaned


def load_and_clean_data(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """Load raw CSV, clean it, and save to processed directory."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    df = pd.read_csv(raw_path)
    cleaned = clean_raw_data(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print("=" * 60)
    print("CUSTOMER CHURN - DATA PREPROCESSING")
    print("=" * 60)
    print(f"Original Shape: {df.shape}")
    print(f"Cleaned Shape : {cleaned.shape}")
    print(f"Coerced NaNs in Total Charges (deferred to Pipeline imputer): {cleaned['Total Charges'].isnull().sum()}")
    print(f"Saved cleaned dataset to: {output_path}")

    return cleaned


if __name__ == "__main__":
    load_and_clean_data()
