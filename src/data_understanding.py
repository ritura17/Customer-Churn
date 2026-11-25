"""Exploratory summary of the raw IBM Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN


def run_data_understanding(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset and print shape, types, missing values, and class balance."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    print("=" * 60)
    print("CUSTOMER CHURN DATASET - DATA UNDERSTANDING")
    print("=" * 60)
    print(f"\n1. Dataset Shape: {df.shape}")
    print(f"\n2. Column Names ({len(df.columns)}): {df.columns.tolist()}")
    print("\n3. Missing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\n4. Duplicate Rows: {df.duplicated().sum()}")
    print("\n5. Statistical Summary:")
    print(df.describe())

    if TARGET_COLUMN in df.columns:
        print("\n6. Churn Distribution:")
        print(df[TARGET_COLUMN].value_counts())
        print("\n7. Churn Percentage:")
        print((df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2))

    return df


if __name__ == "__main__":
    run_data_understanding()
