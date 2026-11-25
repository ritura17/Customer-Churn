"""Exploratory Data Analysis chart generation for Customer Churn."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CLEANED_DATA_PATH, REPORTS_DIR, TARGET_COLUMN


def run_eda(
    data_path: Path = CLEANED_DATA_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> None:
    """Generate and save EDA plots without blocking execution."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {data_path}. Run data_preprocessing.py first."
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    charts = [
        ("churn_distribution.png", lambda: sns.countplot(data=df, x=TARGET_COLUMN),
         "Customer Churn Distribution", "Churn (0 = No, 1 = Yes)", "Number of Customers", 0),
        ("churn_by_contract.png", lambda: sns.countplot(data=df, x="Contract", hue=TARGET_COLUMN),
         "Churn by Contract Type", "Contract Type", "Number of Customers", 15),
        ("churn_by_internet_service.png", lambda: sns.countplot(data=df, x="Internet Service", hue=TARGET_COLUMN),
         "Churn by Internet Service", "Internet Service", "Number of Customers", 0),
        ("churn_by_tenure.png", lambda: sns.boxplot(data=df, x=TARGET_COLUMN, y="Tenure Months"),
         "Tenure Distribution by Churn", "Churn (0 = No, 1 = Yes)", "Tenure (Months)", 0),
        ("churn_by_monthly_charges.png", lambda: sns.boxplot(data=df, x=TARGET_COLUMN, y="Monthly Charges"),
         "Monthly Charges by Churn", "Churn (0 = No, 1 = Yes)", "Monthly Charges", 0),
        ("churn_by_payment_method.png", lambda: sns.countplot(data=df, x="Payment Method", hue=TARGET_COLUMN),
         "Churn by Payment Method", "Payment Method", "Number of Customers", 25),
    ]

    for filename, plot_fn, title, xlabel, ylabel, rotation in charts:
        plt.figure(figsize=(8, 5))
        plot_fn()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if rotation:
            plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.savefig(reports_dir / filename, dpi=200)
        plt.close()

    print(f"EDA completed successfully. {len(charts)} charts saved in {reports_dir}")


if __name__ == "__main__":
    run_eda()
