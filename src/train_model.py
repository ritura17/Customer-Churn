import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================================
# CUSTOMER CHURN - MODEL TRAINING
# ==========================================================

print("=" * 60)
print("CUSTOMER CHURN - MODEL TRAINING")
print("=" * 60)


# ----------------------------------------------------------
# 1. Load processed data
# ----------------------------------------------------------

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ----------------------------------------------------------
# 2. Define models
# ----------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


# ----------------------------------------------------------
# 3. Train and evaluate models
# ----------------------------------------------------------

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {name}")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = model

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ----------------------------------------------------------
# 4. Model comparison
# ----------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ----------------------------------------------------------
# 5. Select best model using ROC-AUC
# ----------------------------------------------------------

best_model_name = results_df.loc[
    results_df["ROC-AUC"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nBest Model:")
print(best_model_name)


# ----------------------------------------------------------
# 6. Save best model
# ----------------------------------------------------------

joblib.dump(
    best_model,
    "models/best_churn_model.pkl"
)

print("\nBest model saved:")
print("models/best_churn_model.pkl")


# ----------------------------------------------------------
# 7. Save model comparison
# ----------------------------------------------------------

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nModel comparison saved:")
print("reports/model_comparison.csv")


print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)