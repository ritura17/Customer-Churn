"""Model training, Stratified K-Fold CV, hyperparameter tuning, and threshold selection."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    BEST_MODEL_PATH,
    BEST_PIPELINE_PATH,
    CV_FOLDS,
    FEATURE_COLUMNS,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
)
from src.feature_engineering import build_preprocessor


def _build_search_spaces(cv: StratifiedKFold) -> Dict[str, Any]:
    """Define candidate model pipelines and hyperparameter search spaces."""
    scoring = "roc_auc"

    lr_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    dt_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )
    rf_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    gb_pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )

    return {
        "Logistic Regression": GridSearchCV(
            lr_pipe,
            param_grid={
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        ),
        "Decision Tree": GridSearchCV(
            dt_pipe,
            param_grid={
                "classifier__max_depth": [3, 5, 7, 10],
                "classifier__min_samples_leaf": [5, 10, 20, 40],
                "classifier__class_weight": [None, "balanced"],
            },
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        ),
        "Random Forest": RandomizedSearchCV(
            rf_pipe,
            param_distributions={
                "classifier__n_estimators": [150, 250, 350],
                "classifier__max_depth": [6, 8, 10, 14, None],
                "classifier__min_samples_leaf": [2, 5, 10],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
            n_iter=12,
            cv=cv,
            scoring=scoring,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        ),
        "Gradient Boosting": RandomizedSearchCV(
            gb_pipe,
            param_distributions={
                "classifier__n_estimators": [100, 150, 200],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [2, 3, 4],
                "classifier__subsample": [0.8, 1.0],
            },
            n_iter=12,
            cv=cv,
            scoring=scoring,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        ),
    }


def optimize_threshold_from_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[float, pd.DataFrame]:
    """Select probability threshold using out-of-fold CV predictions on training data only."""
    oof_probs = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    rows = []
    for thr in np.round(np.arange(0.15, 0.76, 0.05), 2):
        preds = (oof_probs >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": float(accuracy_score(y_train, preds)),
                "precision": float(precision_score(y_train, preds, zero_division=0)),
                "recall": float(recall_score(y_train, preds, zero_division=0)),
                "f1": float(f1_score(y_train, preds, zero_division=0)),
                "f2": float(fbeta_score(y_train, preds, beta=2, zero_division=0)),
                "predicted_churn_rate": float(preds.mean()),
            }
        )

    sweep_df = pd.DataFrame(rows)
    # Business objective: prioritize catching churners (F1-optimal among thresholds with recall >= 0.65,
    # falling back to overall F1-optimal if none meet 0.65).
    candidates = sweep_df[sweep_df["recall"] >= 0.65]
    if not candidates.empty:
        best_row = candidates.loc[candidates["f1"].idxmax()]
    else:
        best_row = sweep_df.loc[sweep_df["f1"].idxmax()]

    return float(best_row["threshold"]), sweep_df


def train_and_select_model(
    x_train_path: Path = X_TRAIN_PATH,
    y_train_path: Path = Y_TRAIN_PATH,
) -> Dict[str, Any]:
    """Run CV model comparison, hyperparameter tuning, and threshold optimization."""
    if not x_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(
            "Training splits not found. Run data_preprocessing.py and feature_engineering.py first."
        )

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).squeeze().astype(int)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    searches = _build_search_spaces(cv)

    cv_metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    comparison_rows = []
    best_estimators: Dict[str, Pipeline] = {}
    best_params_map: Dict[str, Dict[str, Any]] = {}

    print("=" * 60)
    print("CUSTOMER CHURN - CV MODEL TRAINING & TUNING")
    print("=" * 60)

    for name, search in searches.items():
        print(f"\nTuning {name} with {CV_FOLDS}-fold Stratified CV...")
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        best_estimators[name] = best_est
        best_params_map[name] = search.best_params_

        scores = cross_validate(
            best_est, X_train, y_train, cv=cv, scoring=cv_metrics, n_jobs=-1
        )
        row = {
            "Model": name,
            "CV Accuracy": float(scores["test_accuracy"].mean()),
            "CV Precision": float(scores["test_precision"].mean()),
            "CV Recall": float(scores["test_recall"].mean()),
            "CV F1 Score": float(scores["test_f1"].mean()),
            "CV ROC-AUC": float(scores["test_roc_auc"].mean()),
            "CV ROC-AUC Std": float(scores["test_roc_auc"].std()),
            "Best Params": json.dumps(search.best_params_),
        }
        comparison_rows.append(row)
        print(
            f"  Best Params: {search.best_params_}\n"
            f"  CV ROC-AUC: {row['CV ROC-AUC']:.4f} (+/- {row['CV ROC-AUC Std']:.4f}) | "
            f"CV F1: {row['CV F1 Score']:.4f} | CV Recall: {row['CV Recall']:.4f}"
        )

    results_df = pd.DataFrame(comparison_rows).sort_values(
        by="CV ROC-AUC", ascending=False
    )

    best_model_name = str(results_df.iloc[0]["Model"])
    best_pipeline = best_estimators[best_model_name]

    # Optimize classification threshold using CV out-of-fold probabilities on X_train
    optimal_threshold, threshold_df = optimize_threshold_from_cv(
        best_pipeline, X_train, y_train, cv
    )

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, BEST_PIPELINE_PATH)
    joblib.dump(best_pipeline.named_steps["classifier"], BEST_MODEL_PATH)
    joblib.dump(best_pipeline.named_steps["preprocessor"], PREPROCESSOR_PATH)

    results_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    threshold_df.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    metadata: Dict[str, Any] = {
        "best_model_name": best_model_name,
        "best_params": best_params_map[best_model_name],
        "selection_metric": "CV ROC-AUC (5-fold Stratified)",
        "cv_results": results_df.drop(columns=["Best Params"]).to_dict(orient="records"),
        "optimal_threshold": optimal_threshold,
        "risk_tiers": {
            "low_max": round(max(0.25, optimal_threshold - 0.10), 2),
            "medium_max": optimal_threshold,
        },
        "feature_columns": FEATURE_COLUMNS,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"SELECTED BEST MODEL (by CV ROC-AUC): {best_model_name}")
    print(f"CV-OPTIMIZED THRESHOLD: {optimal_threshold:.2f}")
    print("=" * 60)
    print(results_df.drop(columns=["Best Params"]).to_string(index=False))

    return metadata


if __name__ == "__main__":
    train_and_select_model()
