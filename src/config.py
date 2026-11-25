"""Central configuration, paths, and feature schema for Customer Churn Prediction."""

from pathlib import Path

# Project root directory (independent of current working directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Customer_churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_customer_churn.csv"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_PIPELINE_PATH = MODELS_DIR / "best_churn_pipeline.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_churn_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COLUMN = "Churn Label"

DROP_RAW_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "Lat Long",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
]

DROP_GEO_COLUMNS = [
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
]

NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
