"""Cached data and artifact loaders for the Streamlit dashboard."""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from src.config import (
    CLEANED_DATA_PATH,
    MODEL_COMPARISON_PATH,
    REPORTS_DIR,
    X_TEST_PATH,
    Y_TEST_PATH,
)
from src.predict import load_artifacts


@st.cache_resource
def get_cached_model_artifacts() -> Tuple[Any, Dict[str, Any]]:
    """Load and cache the trained sklearn Pipeline and metadata."""
    return load_artifacts()


@st.cache_data
def get_cleaned_dataset() -> Optional[pd.DataFrame]:
    """Load the cleaned IBM Telco Customer Churn dataset if available."""
    if not CLEANED_DATA_PATH.exists():
        return None
    df = pd.read_csv(CLEANED_DATA_PATH)
    df["Churn Status"] = df["Churn Label"].map({0: "Stayed", 1: "Churned"})
    df["Tenure Group"] = pd.cut(
        df["Tenure Months"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"],
    )
    return df


@st.cache_data
def get_model_comparison() -> Optional[pd.DataFrame]:
    """Load 5-fold CV model comparison table."""
    if not MODEL_COMPARISON_PATH.exists():
        return None
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data
def get_threshold_analysis() -> Optional[pd.DataFrame]:
    """Load CV threshold sweep results."""
    path = REPORTS_DIR / "threshold_analysis.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def get_test_split() -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Load untouched test split for verified ROC/CM rendering."""
    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        return None, None
    return pd.read_csv(X_TEST_PATH), pd.read_csv(Y_TEST_PATH).squeeze().astype(int)


@st.cache_data
def get_test_risk_distribution(_pipeline: Any, threshold: float, low_max: float) -> Optional[pd.DataFrame]:
    """Score the untouched test split to compute verified risk-tier distribution and MRR at risk."""
    X_test, _ = get_test_split()
    if X_test is None or _pipeline is None:
        return None
    probs = _pipeline.predict_proba(X_test)[:, 1]
    tiers = [
        "High Risk" if p >= threshold else ("Medium Risk" if p >= low_max else "Low Risk")
        for p in probs
    ]
    return pd.DataFrame(
        {
            "Probability": probs,
            "Risk Tier": tiers,
            "Monthly Charges": X_test["Monthly Charges"].values,
        }
    )
