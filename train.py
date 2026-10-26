from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
TARGET_COLUMN = "target"
RANDOM_STATE = 2026


def build_models() -> dict[str, Pipeline]:
    scaled_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "Logistic Regression": Pipeline(
            scaled_steps
            + [("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            scaled_steps + [("model", KNeighborsClassifier(n_neighbors=11))]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GaussianNB()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            scaled_steps
            + [("model", SVC(C=1.5, probability=True, random_state=RANDOM_STATE))]
        ),
    }


def probability_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    probabilities = model.predict_proba(features)
    positive_index = list(model.classes_).index(1)
    return probabilities[:, positive_index]


def main() -> None:
    for directory in (DATA_DIR, MODEL_DIR, EXPERIMENT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.rename(TARGET_COLUMN)
    full_dataset = pd.concat([features, target], axis=1)
    full_dataset.to_csv(DATA_DIR / "breast_cancer_wdbc.csv", index=False)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        PROJECT_ROOT / "test_data.csv", index=False
    )

    results: list[dict[str, float | str]] = []
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores = probability_scores(model, x_test)
        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "auc": roc_auc_score(y_test, scores),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "mcc": matthews_corrcoef(y_test, predictions),
            }
        )
        joblib.dump(model, MODEL_DIR / f"{model_name.lower().replace(' ', '_').replace('-', '_')}.joblib")

    results_frame = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_frame.to_csv(EXPERIMENT_DIR / "results.csv", index=False)
    metadata = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (UCI)",
        "dataset_source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "instances": len(full_dataset),
        "features": list(features.columns),
        "target_column": TARGET_COLUMN,
        "target_labels": {"0": "malignant", "1": "benign"},
        "test_size": len(x_test),
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved dataset, test split, five trained models, metadata, and experiments/results.csv.")
    print(results_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()