"""
MLOps Pulse - Training Script
Trains a credit risk model on the UCI German Credit dataset.
Logs all experiments, metrics, and artifacts to MLflow.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ── MLflow setup ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
EXPERIMENT_NAME = "credit-risk-model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Generates a synthetic credit risk dataset.
    In a real MLOps pipeline this would pull from a feature store or data lake.
    Features mirror the UCI German Credit dataset structure.
    """
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame(
        {
            "age": np.random.randint(18, 75, n),
            "credit_amount": np.random.randint(250, 18000, n),
            "duration_months": np.random.randint(4, 72, n),
            "employment_years": np.random.randint(0, 10, n),
            "installment_rate": np.random.randint(1, 5, n),
            "residence_years": np.random.randint(1, 5, n),
            "existing_credits": np.random.randint(1, 4, n),
            "num_dependents": np.random.randint(1, 3, n),
            "has_telephone": np.random.randint(0, 2, n),
            "is_foreign_worker": np.random.randint(0, 2, n),
        }
    )

    # Target: 1 = good credit, 0 = bad credit
    # Higher credit amount + shorter duration = riskier
    risk_score = (
        -0.3 * (data["credit_amount"] / data["credit_amount"].max())
        + 0.2 * (data["duration_months"] / data["duration_months"].max())
        + 0.2 * (data["employment_years"] / data["employment_years"].max())
        + 0.1 * (data["age"] / data["age"].max())
        + np.random.normal(0, 0.1, n)
    )
    data["target"] = (risk_score > risk_score.median()).astype(int)

    return data


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    df = load_data()

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameters — in a real setup these come from a config or sweep
    params = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
    }

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(**params)),
        ]
    )

    with mlflow.start_run(run_name="gbt-credit-risk-v1"):
        # ── Log params
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", feature_cols)

        # ── Train
        pipeline.fit(X_train, y_train)

        # ── Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_metrics(metrics)

        print("\n── Model Metrics ─────────────────────────────")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")

        # ── Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="credit-risk-classifier",
            input_example=X_test.iloc[:3],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Run ID: {run_id}")
        print(f"✅ Model registered as: credit-risk-classifier")
        print(f"✅ Tracking UI: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    train()
