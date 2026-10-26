# AIML ZG565 Machine Learning: Assignment 2

## Problem statement

This project compares five classifiers on the Wisconsin Diagnostic Breast Cancer dataset. The goal is to predict whether a tumour is malignant (`0`) or benign (`1`) from 30 numerical features.

## Dataset description

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Instances:** 569
- **Features:** 30 numerical predictors, satisfying the assignment minimum of 12
- **Task:** Binary classification
- **Split:** Stratified 80:20 train/test split, `random_state=2026`

`train.py` creates a local dataset copy at `data/breast_cancer_wdbc.csv` and saves the evaluation split as `test_data.csv` in the project root.

## Repository structure

```text
.
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── breast_cancer_wdbc.csv
├── experiments/
│   └── results.csv
└── models/
    ├── *.joblib
    └── model_metadata.json
```

## Run locally

```powershell
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

The Streamlit app uses the saved models. Upload a labelled CSV with the 30 feature columns and the `target` column. The app evaluates the uploaded file and does not retrain a model.

## Models and metrics

The metrics below were calculated on the saved stratified test split. The full values are also in `experiments/results.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9737 | 0.9927 | 0.9726 | 0.9861 | 0.9793 | 0.9433 |
| Decision Tree | 0.9298 | 0.9395 | 0.9848 | 0.9028 | 0.9420 | 0.8588 |
| K-Nearest Neighbors | 0.9737 | 0.9924 | 0.9726 | 0.9861 | 0.9793 | 0.9433 |
| Gaussian Naive Bayes | 0.9386 | 0.9838 | 0.9577 | 0.9444 | 0.9510 | 0.8689 |
| Support Vector Machine | 0.9649 | 0.9960 | 0.9857 | 0.9583 | 0.9718 | 0.9260 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It gave the joint-best accuracy and F1 score. It also had the highest recall. |
| Decision Tree | It had the highest precision, but its recall was lower than the other models. |
| K-Nearest Neighbors | It matched Logistic Regression on accuracy, recall, F1, and MCC for this test split. |
| Gaussian Naive Bayes | Its AUC was good, but its accuracy and F1 score were lower than the top models. |
| Support Vector Machine | It had the highest AUC. Its precision was high, but recall was lower than Logistic Regression and KNN. |
| Overall winner for Wisconsin Diagnostic Breast Cancer dataset dataset | **Logistic Regression** was selected because it had high accuracy, high recall, and the joint-best F1 score. |

## Deployment

**GitHub repository:** https://github.com/2026aj05873/aiml-zg565-ml-assignment-2

**Live Streamlit app:** https://aiml-zg565-ml-assignment-2-jaqn7tc4jg9fa3wubxdvql.streamlit.app/

| Deployment item | Current status |
|---|---|
| GitHub repository | The first commit has been pushed. The repository includes the code, trained models, `test_data.csv`, and experiment results. |
| Streamlit Community Cloud | The app is deployed from `main` with `app.py` as the entry point. The live app was checked after deployment. |
| Final submission PDF | Pending. It needs the GitHub URL, Streamlit URL, README content, and the BITS Virtual Lab execution screenshot. |