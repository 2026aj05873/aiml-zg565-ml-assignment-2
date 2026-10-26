from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
TARGET_COLUMN = "target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(model_file: str):
    return joblib.load(MODEL_DIR / model_file)


@st.cache_data
def load_metadata() -> dict:
    return json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="ML Assignment 2 Evaluator", page_icon="ML", layout="wide")
    st.title("Machine Learning Assignment 2: Classifier Evaluator")
    st.caption("A trained-model evaluator for the Wisconsin Diagnostic Breast Cancer dataset.")

    if not (MODEL_DIR / "model_metadata.json").exists():
        st.error("Trained artifacts are missing. Run `python train.py` before starting the app.")
        st.stop()

    metadata = load_metadata()
    with st.sidebar:
        st.header("Evaluation settings")
        selected_model = st.selectbox("Choose a trained model", list(MODEL_FILES))
        st.write(f"Dataset: {metadata['dataset']}")
        st.write(f"Expected features: {len(metadata['features'])}")

    uploaded_file = st.file_uploader(
        "Upload test data as CSV",
        type="csv",
        help="The CSV must include all 30 feature columns and the target column named 'target'.",
    )
    if uploaded_file is None:
        st.info("Upload a labelled test CSV to evaluate a saved model.")
        return

    uploaded_data = pd.read_csv(uploaded_file)
    missing_columns = set(metadata["features"] + [TARGET_COLUMN]) - set(uploaded_data.columns)
    if missing_columns:
        st.error("The uploaded CSV is missing: " + ", ".join(sorted(missing_columns)))
        return

    features = uploaded_data[metadata["features"]]
    target = uploaded_data[TARGET_COLUMN]
    if not set(target.dropna().unique()).issubset({0, 1}):
        st.error("The target column must contain binary labels 0 (malignant) or 1 (benign).")
        return

    model = load_model(MODEL_FILES[selected_model])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, probabilities),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }

    st.subheader(f"{selected_model} evaluation")
    columns = st.columns(3)
    for column, (metric_name, metric_value) in zip(columns * 2, metrics.items()):
        column.metric(metric_name, f"{metric_value:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        figure, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            target,
            predictions,
            display_labels=["Malignant (0)", "Benign (1)"],
            cmap="Blues",
            ax=axis,
        )
        st.pyplot(figure, clear_figure=True)
    with right:
        st.subheader("Classification report")
        report = classification_report(
            target,
            predictions,
            target_names=["Malignant (0)", "Benign (1)"],
            zero_division=0,
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.subheader("Uploaded data preview")
    st.dataframe(uploaded_data.head(), use_container_width=True)


if __name__ == "__main__":
    main()