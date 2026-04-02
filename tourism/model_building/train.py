"""train.py
Trains multiple classifiers with MLflow tracking, selects the best by F1-score,
and uploads the winning model to a Hugging Face model repository.
Triggered as the third job (train-model) in the GitHub Actions pipeline.
"""
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings("ignore")  # Suppress warning messages for cleaner output

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Set up environment variables and configuration parameters
HF_USERNAME   = os.getenv("HF_USERNAME", "vivekkumar-hf")  # Default username if env var not set
DATASET_REPO  = f"{HF_USERNAME}/tourism-data"  # Repository for the dataset
MODEL_REPO    = f"{HF_USERNAME}/tourism-model"  # Repository for the model
HF_TOKEN      = os.getenv("HF_TOKEN")  # Authentication token for Hugging Face
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")  # MLflow tracking server URI
MODEL_FNAME   = "best_tourism_model_v1.joblib"  # Filename for the saved model

# ── MLFLOW SETUP ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)  # Configure MLflow to use the specified tracking server
mlflow.set_experiment("Tourism_Experiment")  # Set the experiment name for tracking

# ── LOAD DATA FROM HUGGING FACE ───────────────────────────────────────────────
print("Loading processed data from Hugging Face...")
os.makedirs("tourism/data", exist_ok=True)  # Create directory for data if it doesn't exist

for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    local_p = hf_hub_download(
        repo_id=DATASET_REPO, filename=fname,
        repo_type="dataset", token=HF_TOKEN,
    )
    # Copy to working dir
    import shutil
    shutil.copy(local_p, f"tourism_project/data/{fname}")

X_train = pd.read_csv("tourism/data/X_train.csv")
X_test  = pd.read_csv("tourism/data/X_test.csv")
y_train = pd.read_csv("tourism/data/y_train.csv").squeeze()
y_test  = pd.read_csv("tourism/data/y_test.csv").squeeze()
print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples.")


# Define a list of models to experiment with, each with specific hyperparameters to tune
experiments = [
    {
        "name": "Decision_Tree",
        "model": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "params": {"max_depth": [5, 7, None], "min_samples_split": [2, 10]},
    },
    {
        "name": "Random_Forest",
        "model": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        "params": {"n_estimators": [100], "max_depth": [5, 10]},
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(
            scale_pos_weight=4, eval_metric="logloss",  # scale_pos_weight helps with class imbalance
            random_state=42, verbosity=0,  # Set verbosity to 0 to suppress XGBoost outputs
        ),
        "params": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    },
]

# ── TRAIN & TRACK ─────────────────────────────────────────────────────────────
# Initialize variables to keep track of the best performing model
best_f1    = 0
best_model = None
best_name  = ""

# Iterate through each model configuration
for exp in experiments:
    # Use MLflow to track each experiment run
    with mlflow.start_run(run_name=exp["name"]):
        # Set up GridSearchCV to find optimal hyperparameters
        gs = GridSearchCV(
            exp["model"], exp["params"],
            scoring="f1", cv=5, n_jobs=-1, refit=True,  # 5-fold cross-validation, use all CPU cores
        )
        # Train the model with grid search
        gs.fit(X_train, y_train)
        # Get the best model from grid search
        fitted = gs.best_estimator_

        # Make predictions on test data
        y_pred = fitted.predict(X_test)
        y_prob = fitted.predict_proba(X_test)[:, 1]  # Get probability of positive class

        # Calculate various performance metrics
        metrics = {
            "accuracy"  : accuracy_score(y_test, y_pred),
            "precision" : precision_score(y_test, y_pred, zero_division=0),  # Handle division by zero
            "recall"    : recall_score(y_test, y_pred),
            "f1"        : f1_score(y_test, y_pred),
            "roc_auc"   : roc_auc_score(y_test, y_prob),
        }

        # Log the best parameters and metrics to MLflow
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(metrics)
        # Save the model in MLflow
        mlflow.sklearn.log_model(fitted, artifact_path="model")

        # Print the performance of the current model
        print(f"{exp['name']}: F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}")

        # Update the best model if current one has better F1 score
        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_model = fitted
            best_name  = exp["name"]

# Print information about the best performing model
print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
os.makedirs("tourism/model_building", exist_ok=True)
local_model_path = f"tourism/model_building/{MODEL_FNAME}"
joblib.dump(best_model, local_model_path)
print(f"Model saved locally: {local_model_path}")

# ── UPLOAD MODEL TO HUGGING FACE ──────────────────────────────────────────────
api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model",
                private=False, token=HF_TOKEN)
    print(f"Created model repo: {MODEL_REPO}")

api.upload_file(
    path_or_fileobj=local_model_path,
    path_in_repo=MODEL_FNAME,
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Uploaded model '{MODEL_FNAME}' → '{MODEL_REPO}'.")
print("Training complete.")
