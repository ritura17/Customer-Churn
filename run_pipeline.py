"""End-to-end reproducible runner for the Customer Churn ML pipeline."""

from src.data_preprocessing import load_and_clean_data
from src.eda import run_eda
from src.evaluate_model import evaluate_final_model
from src.feature_engineering import prepare_train_test_splits
from src.train_model import train_and_select_model


def main() -> None:
    """Execute preprocessing, splitting, CV training/tuning, evaluation, and EDA."""
    load_and_clean_data()
    prepare_train_test_splits()
    train_and_select_model()
    evaluate_final_model()
    run_eda()
    print("\nFull Customer Churn pipeline completed successfully.")


if __name__ == "__main__":
    main()
