import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


# ==========================================================
# CUSTOMER CHURN - MODEL EVALUATION
# ==========================================================

print("=" * 60)
print("CUSTOMER CHURN - MODEL EVALUATION")
print("=" * 60)


# ----------------------------------------------------------
# 1. Load test data
# ----------------------------------------------------------

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()


print("\nTest Data Shape:")
print(X_test.shape)

print("\nActual Target Distribution:")
print(y_test.value_counts())


# ----------------------------------------------------------
# 2. Load trained model
# ----------------------------------------------------------

model = joblib.load(
    "models/best_churn_model.pkl"
)

print("\nBest Model Loaded:")
print("Random Forest")


# ----------------------------------------------------------
# 3. Make predictions
# ----------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ----------------------------------------------------------
# 4. Classification Report
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
)

print(report)


# ----------------------------------------------------------
# 5. Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


fig, ax = plt.subplots(figsize=(6, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(
    ax=ax,
    cmap="Blues"
)

ax.set_title("Customer Churn - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------------------
# 6. ROC Curve
# ----------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC Score:")
print(f"{roc_auc:.4f}")


plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Customer Churn")

plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------------------
# 7. Feature Importance
# ----------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# ----------------------------------------------------------
# 8. Feature Importance Plot
# ----------------------------------------------------------

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 15 Feature Importance - Random Forest"
)

plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------------------
# 9. Final message
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles saved:")

print("reports/confusion_matrix.png")
print("reports/roc_curve.png")
print("reports/feature_importance.png")