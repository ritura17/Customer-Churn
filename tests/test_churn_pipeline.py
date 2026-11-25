"""Automated pytest suite for Customer Churn Prediction."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    BEST_PIPELINE_PATH,
    FEATURE_COLUMNS,
    MODEL_METADATA_PATH,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.data_preprocessing import clean_raw_data
from src.feature_engineering import build_preprocessor, validate_input_dataframe
from src.predict import explain_prediction, load_artifacts, predict_churn


@pytest.fixture
def sample_customer_df() -> pd.DataFrame:
    """Return a valid single-row customer DataFrame."""
    return pd.DataFrame(
        [
            {
                "Gender": "Male",
                "Senior Citizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure Months": 12,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "DSL",
                "Online Security": "No",
                "Online Backup": "Yes",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Monthly Charges": 50.0,
                "Total Charges": 600.0,
                "CLTV": 4000,
            }
        ]
    )


def test_raw_dataset_exists_and_valid():
    assert RAW_DATA_PATH.exists(), "Raw dataset is missing."
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.shape == (7043, 33)
    assert TARGET_COLUMN in df.columns


def test_clean_raw_data_defers_imputation():
    df = pd.read_csv(RAW_DATA_PATH).head(20).copy()
    df.loc[0, "Total Charges"] = " "
    cleaned = clean_raw_data(df)

    assert "Churn Score" not in cleaned.columns
    assert "Churn Reason" not in cleaned.columns
    assert np.isnan(cleaned.loc[0, "Total Charges"])
    assert set(cleaned[TARGET_COLUMN].unique()).issubset({0, 1})


def test_validate_input_dataframe_rejects_invalid(sample_customer_df):
    with pytest.raises(ValueError, match="empty"):
        validate_input_dataframe(sample_customer_df.iloc[0:0])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_input_dataframe(sample_customer_df.drop(columns=["Contract"]))

    bad_num = sample_customer_df.copy()
    bad_num["Monthly Charges"] = -10.0
    with pytest.raises(ValueError, match="negative"):
        validate_input_dataframe(bad_num)


def test_preprocessor_handles_missing_and_unknown_categories(sample_customer_df):
    train_df = pd.concat([sample_customer_df] * 5, ignore_index=True)
    preprocessor = build_preprocessor()
    preprocessor.fit(train_df)

    test_row = sample_customer_df.copy()
    test_row["Total Charges"] = np.nan
    test_row["Payment Method"] = "Unknown Crypto Wallet"

    transformed = preprocessor.transform(test_row)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_saved_pipeline_prediction_and_probabilities(sample_customer_df):
    assert BEST_PIPELINE_PATH.exists()
    assert MODEL_METADATA_PATH.exists()

    pipeline, metadata = load_artifacts()
    result = predict_churn(sample_customer_df, pipeline=pipeline, metadata=metadata)

    prob = float(result["Churn Probability"].iloc[0])
    pred = int(result["Churn Prediction"].iloc[0])
    tier = str(result["Risk Tier"].iloc[0])

    assert 0.0 <= prob <= 1.0
    assert pred in {0, 1}
    assert tier in {"Low Risk", "Medium Risk", "High Risk"}
    assert set(sample_customer_df.columns) == set(FEATURE_COLUMNS)


def test_explain_prediction_returns_valid_drivers(sample_customer_df):
    pipeline, _ = load_artifacts()
    drivers = explain_prediction(sample_customer_df, pipeline=pipeline, top_n=5)
    assert len(drivers) == 5
    for item in drivers:
        assert {"feature", "value", "shap_impact", "direction"}.issubset(item.keys())


def test_missing_artifacts_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Trained pipeline not found"):
        load_artifacts(pipeline_path=tmp_path / "nonexistent.pkl")


def test_streamlit_app_navigation_and_prediction_flows():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=20)
    at.run()
    assert not at.exception

    # Navigate through all 5 sidebar pages
    for page_name in ["Dashboard", "Analytics", "Model Insights", "About Model", "Customer Prediction"]:
        at.sidebar.radio[0].set_value(page_name).run()
        assert not at.exception, f"Exception on page {page_name}: {at.exception}"

    # 1. Load High-Risk Example and Predict
    at.button[0].click().run()  # "Load High-Risk Example"
    assert not at.exception
    at.button[3].click().run()  # "Predict Customer Churn"
    assert not at.exception
    assert at.session_state["last_prediction"]["prediction"] == 1
    assert at.session_state["last_prediction"]["risk_tier"] == "High Risk"

    # 2. Load Low-Risk Example and Predict
    at.button[1].click().run()  # "Load Low-Risk Example"
    assert not at.exception
    at.button[3].click().run()  # "Predict Customer Churn"
    assert not at.exception
    assert at.session_state["last_prediction"]["prediction"] == 0
    assert at.session_state["last_prediction"]["risk_tier"] == "Low Risk"

    # 3. Reset Form
    at.button[2].click().run()  # "Reset Form"
    assert not at.exception
    assert at.session_state["last_prediction"] is None
    assert at.session_state["form_Contract"] == "Month-to-month"

    # 4. Manual input change + Predict
    at.slider[0].set_value(48).run()
    at.button[3].click().run()
    assert not at.exception
    assert at.session_state["last_prediction"] is not None
    assert len(at.session_state["prediction_history"]) >= 3


def test_batch_csv_scoring_and_export(tmp_path):
    sample_csv = PROJECT_ROOT / "data" / "test" / "sample_batch_customers.csv"
    assert sample_csv.exists()
    batch_df = pd.read_csv(sample_csv)
    pipeline, metadata = load_artifacts()

    scored = predict_churn(batch_df, pipeline=pipeline, metadata=metadata)
    assert len(scored) == len(batch_df)
    assert {"Churn Probability", "Churn Prediction", "Risk Tier"}.issubset(scored.columns)

    export_path = tmp_path / "scored_export.csv"
    scored.to_csv(export_path, index=False)
    reloaded = pd.read_csv(export_path)
    assert len(reloaded) == len(batch_df)


def test_core_analytics_and_retention_recommendations():
    from src.ui.data_loader import get_cleaned_dataset, get_test_risk_distribution
    from src.ui.pages import build_retention_recommendations

    df = get_cleaned_dataset()
    assert df is not None
    assert len(df) == 7043
    assert abs(df["Churn Label"].mean() - 0.26537) < 1e-4

    pipeline, metadata = load_artifacts()
    thr = float(metadata["optimal_threshold"])
    low_max = float(metadata["risk_tiers"]["low_max"])
    risk_df = get_test_risk_distribution(pipeline, thr, low_max)
    assert risk_df is not None
    assert len(risk_df) == 1409

    high_recs = build_retention_recommendations(
        {"Contract": "Month-to-month", "Payment Method": "Electronic check", "Internet Service": "Fiber optic", "Tech Support": "No", "Tenure Months": 3},
        "High Risk",
    )
    assert len(high_recs) >= 3
    low_recs = build_retention_recommendations({"Contract": "Two year"}, "Low Risk")
    assert len(low_recs) >= 1
