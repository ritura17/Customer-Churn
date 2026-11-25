"""Model-agnostic final evaluation on the untouched test set."""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    BEST_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    X_TEST_PATH,
    Y_TEST_PATH,
)


def _compute_metrics(
    y_true: pd.Series, y_prob: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Compute binary classification metrics at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_final_model(
    pipeline_path: Path = BEST_PIPELINE_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    x_test_path: Path = X_TEST_PATH,
    y_test_path: Path = Y_TEST_PATH,
    reports_dir: Path = REPORTS_DIR,
) -> Dict[str, Any]:
    """Evaluate the selected pipeline once on the untouched test set and save reports."""
    for p in (pipeline_path, metadata_path, x_test_path, y_test_path):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    pipeline = joblib.load(pipeline_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze().astype(int)

    model_name = metadata.get("best_model_name", "Selected Model")
    optimal_threshold = float(metadata.get("optimal_threshold", 0.50))

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    default_metrics = _compute_metrics(y_test, y_prob, 0.50)
    tuned_metrics = _compute_metrics(y_test, y_prob, optimal_threshold)

    print("=" * 60)
    print(f"FINAL TEST EVALUATION - {model_name}")
    print("=" * 60)
    print(f"\nDefault Threshold (0.50): {default_metrics}")
    print(f"CV-Tuned Threshold ({optimal_threshold:.2f}): {tuned_metrics}")
    print("\nClassification Report (at CV-tuned threshold):")
    print(
        classification_report(
            y_test,
            (y_prob >= optimal_threshold).astype(int),
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix (at CV-tuned threshold)
    cm = np.array(tuned_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Churn", "Churn"]
    )
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({model_name}, thr={optimal_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (ROC-AUC = {tuned_metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Customer Churn (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Precision-Recall Curve
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(
        rec_curve,
        prec_curve,
        label=f"{model_name} (PR-AUC = {tuned_metrics['pr_auc']:.4f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Customer Churn (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Threshold Analysis Plot (from CV sweep)
    thr_csv = reports_dir / "threshold_analysis.csv"
    if thr_csv.exists():
        thr_df = pd.read_csv(thr_csv)
        plt.figure(figsize=(8, 5))
        plt.plot(thr_df["threshold"], thr_df["precision"], marker="o", label="Precision")
        plt.plot(thr_df["threshold"], thr_df["recall"], marker="s", label="Recall")
        plt.plot(thr_df["threshold"], thr_df["f1"], marker="^", label="F1 Score")
        plt.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            label=f"Selected Threshold ({optimal_threshold:.2f})",
        )
        plt.xlabel("Probability Threshold")
        plt.ylabel("Score (5-Fold CV OOF)")
        plt.title("Threshold Trade-off Analysis (Training CV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(reports_dir / "threshold_analysis.png", dpi=200, bbox_inches="tight")
        plt.close()

    # 5. Model-agnostic Permutation Importance on X_test
    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    importance_df = (
        pd.DataFrame(
            {
                "Feature": X_test.columns,
                "Importance": perm.importances_mean,
                "Std": perm.importances_std,
            }
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    top_features = importance_df.head(15)
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
    plt.xlabel("Mean ROC-AUC Decrease (Permutation Importance)")
    plt.ylabel("Feature")
    plt.title(f"Top 15 Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Persist verified test metrics into metadata
    metadata["test_metrics_default_0_50"] = default_metrics
    metadata["test_metrics_tuned_threshold"] = tuned_metrics
    metadata["top_permutation_features"] = importance_df.head(10).to_dict(orient="records")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved evaluation plots and updated {metadata_path}")
    return metadata


if __name__ == "__main__":
    evaluate_final_model()
