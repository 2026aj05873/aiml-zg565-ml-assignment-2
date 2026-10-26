# AIML ZG565 Machine Learning: Assignment 2

## Problem statement

This project compares five classification models on the Wisconsin Diagnostic Breast Cancer dataset: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. The objective is to predict whether a tumour is malignant (`0`) or benign (`1`) using 30 numerical measurements.

## Dataset description

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Instances:** 569
- **Features:** 30 numerical predictors, satisfying the assignment minimum of 12
- **Task:** Binary classification
- **Split:** Stratified 80:20 train/test split, `random_state=2026`

The reproducible local copy is created at `data/breast_cancer_wdbc.csv` by `train.py`; the persisted evaluation split is `test_data.csv` in the project root.

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

The Streamlit app loads saved models only. Upload a labelled CSV that includes the 30 expected features and the `target` column; it does not retrain any model.

## Models and metrics

The six required metrics are Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC). The values below come from the persisted stratified test split; the source data is available in `experiments/results.csv`.

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
| Logistic Regression | Achieved the joint-best accuracy and F1 score, with the highest recall. It is a strong linear baseline for this standardized feature set. |
| Decision Tree | Delivered the highest precision, but its lower recall and MCC indicate that its constrained tree structure missed more benign cases than the other models. |
| K-Nearest Neighbors | Matched Logistic Regression on accuracy, recall, F1, and MCC. It performs well after feature scaling but is more sensitive to the neighbourhood-size choice. |
| Gaussian Naive Bayes | Produced a competitive AUC and reasonable F1 score, but its conditional-independence assumption limits its accuracy on correlated medical measurements. |
| Support Vector Machine | Achieved the best AUC and the second-best F1/MCC. Its high precision makes it a strong alternative where false-positive predictions are costly. |
| Overall winner for your dataset | **Logistic Regression** is selected because it ties for the best accuracy and F1 score while achieving the highest recall and offering a transparent, interpretable decision boundary. |

## Deployment

1. Create a GitHub repository and push this project, including the generated `models/`, `data/test_data.csv`, and `experiments/results.csv` files.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Create an app from the repository, select the main branch, and set `app.py` as the entry point.
4. Add the deployed app URL and GitHub URL to the final submission PDF.