"""Prediction and local SHAP explainability utilities for Customer Churn."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import (
    BEST_PIPELINE_PATH,
    CATEGORICAL_FEATURES,
    MODEL_METADATA_PATH,
)
from src.feature_engineering import validate_input_dataframe


def load_artifacts(
    pipeline_path: Path = BEST_PIPELINE_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the trained sklearn Pipeline and model metadata JSON."""
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Trained pipeline not found at {pipeline_path}. Run python run_pipeline.py first."
        )
    pipeline = joblib.load(pipeline_path)

    metadata: Dict[str, Any] = {"optimal_threshold": 0.35, "risk_tiers": {"low_max": 0.25, "medium_max": 0.35}}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return pipeline, metadata


def classify_risk_tier(probability: float, metadata: Dict[str, Any]) -> str:
    """Map churn probability to a business risk tier using validated thresholds."""
    tiers = metadata.get("risk_tiers", {})
    low_max = float(tiers.get("low_max", 0.25))
    medium_max = float(metadata.get("optimal_threshold", tiers.get("medium_max", 0.35)))

    if probability >= medium_max:
        return "High Risk"
    if probability >= low_max:
        return "Medium Risk"
    return "Low Risk"


def predict_churn(
    input_df: pd.DataFrame,
    pipeline: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Validate input DataFrame and return churn predictions, probabilities, and risk tiers."""
    if pipeline is None or metadata is None:
        pipeline, metadata = load_artifacts()

    validated = validate_input_dataframe(input_df)
    threshold = float(metadata.get("optimal_threshold", 0.35))

    probabilities = pipeline.predict_proba(validated)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    risk_tiers = [classify_risk_tier(float(p), metadata) for p in probabilities]

    result = validated.copy()
    result["Churn Probability"] = np.round(probabilities, 4)
    result["Churn Prediction"] = predictions
    result["Risk Tier"] = risk_tiers
    return result


def _map_transformed_to_original(transformed_name: str) -> str:
    """Map ColumnTransformer output feature name back to its original column name."""
    if transformed_name.startswith("numerical__"):
        return transformed_name.replace("numerical__", "", 1)
    if transformed_name.startswith("categorical__"):
        rest = transformed_name.replace("categorical__", "", 1)
        for cat in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            if rest.startswith(f"{cat}_"):
                return cat
    return transformed_name


def explain_prediction(
    input_df: pd.DataFrame,
    pipeline: Optional[Any] = None,
    top_n: int = 6,
) -> List[Dict[str, Any]]:
    """Compute local SHAP feature contributions aggregated to original input features.

    Returns model influence values (not causal effects).
    """
    if pipeline is None:
        pipeline, _ = load_artifacts()

    validated = validate_input_dataframe(input_df).iloc[[0]]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_proc = preprocessor.transform(validated)
    proc_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "estimators_") or hasattr(classifier, "tree_"):
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_proc)
        if isinstance(shap_vals, list):
            # Binary classifier returning [class0, class1]
            vals = np.asarray(shap_vals[1])[0]
        elif np.asarray(shap_vals).ndim == 3:
            vals = np.asarray(shap_vals)[0, :, 1]
        else:
            vals = np.asarray(shap_vals)[0]
    elif hasattr(classifier, "coef_"):
        vals = (classifier.coef_[0] * X_proc[0])
    else:
        return []

    grouped: Dict[str, float] = {}
    for name, val in zip(proc_names, vals):
        orig = _map_transformed_to_original(str(name))
        grouped[orig] = grouped.get(orig, 0.0) + float(val)

    contributions = []
    for feat, impact in grouped.items():
        raw_val = validated.iloc[0][feat] if feat in validated.columns else ""
        contributions.append(
            {
                "feature": feat,
                "value": raw_val,
                "shap_impact": round(impact, 4),
                "direction": "Increases churn risk" if impact > 0 else "Decreases churn risk",
            }
        )

    contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
    return contributions[:top_n]


if __name__ == "__main__":
    sample_customer = pd.DataFrame(
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

    preds = predict_churn(sample_customer)
    prob = float(preds["Churn Probability"].iloc[0])
    label = int(preds["Churn Prediction"].iloc[0])
    tier = str(preds["Risk Tier"].iloc[0])

    print("=" * 60)
    print("CUSTOMER CHURN - SAMPLE PREDICTION")
    print("=" * 60)
    print(f"Prediction       : {'LIKELY TO CHURN' if label == 1 else 'LIKELY TO STAY'}")
    print(f"Churn Probability: {prob:.2%}")
    print(f"Risk Tier        : {tier}")
    print("\nTop Local Model Drivers (SHAP influence, not causation):")
    for item in explain_prediction(sample_customer):
        print(f"  - {item['feature']} = {item['value']}: {item['shap_impact']:+.4f} ({item['direction']})")
