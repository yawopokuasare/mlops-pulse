"""
MLOps Pulse - Training Script
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
EXPERIMENT_NAME = "credit-risk-model"
MODEL_SAVE_PATH = "/tmp/model.pkl"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data():
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
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
    })
    risk_score = (
        -0.3 * (data["credit_amount"] / data["credit_amount"].max())
        + 0.2 * (data["duration_months"] / data["duration_months"].max())
        + 0.2 * (data["employment_years"] / data["employment_years"].max())
        + 0.1 * (data["age"] / data["age"].max())
        + np.random.normal(0, 0.1, n)
    )
    data["target"] = (risk_score > risk_score.median()).astype(int)
    return data

def train():
    df = load_data()
    feature_cols = [c for c in df.columns if c != "target"]
    X, y = df[feature_cols], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params = {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "random_state": 42}
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", GradientBoostingClassifier(**params))])

    with mlflow.start_run(run_name="gbt-credit-risk-v1"):
        mlflow.log_params(params)
        pipeline.fit(X_train, y_train)
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

        # Save model file and log as artifact
        joblib.dump(pipeline, MODEL_SAVE_PATH)
        mlflow.log_artifact(MODEL_SAVE_PATH, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Run ID: {run_id}")
        print(f"✅ Model saved and logged as artifact")
        print(f"✅ Tracking UI: {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    train()
