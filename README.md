# Healthcare-Predictive-Analytics-Disease-Detection-
ML pipeline to predict diabetes &amp; heart disease risk using UCI/Kaggle data, with feature importance analysis, age-group fairness checks, and privacy-first data handling. Educational project, not a diagnostic tool.
# Healthcare Predictive Analytics: Disease Risk Detection

An end-to-end machine learning pipeline that predicts the risk of **diabetes** and **heart disease** from clinical measurements, built with an emphasis on **explainability**, **fairness**, and **ethical data handling**.

> **Disclaimer:** This project is for educational and research purposes only. It is **not** a medical device and must not be used for diagnosis or clinical decisions.

---

## Table of Contents
- [Overview](#overview)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Results](#sample-results)
- [Ethics and Privacy](#ethics-and-privacy)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Overview

Early risk detection can help clinicians prioritise screening. This project builds and compares several classification models on public medical datasets, then goes beyond accuracy by asking two more questions:

1. **Which features drive the prediction?** (feature importance analysis)
2. **Does the model work equally well for everyone?** (age-group fairness check)

## Datasets

| Dataset | Source | Target |
|---|---|---|
| Pima Indians Diabetes | UCI / Kaggle | Diabetes (0 = no, 1 = yes) |
| Heart Disease (Cleveland) | UCI (via `ucimlrepo`) | Heart disease (0 = no, 1 = yes) |

Both datasets are public and de-identified. No real patient health information (PHI) is used.

## Pipeline

```
Load data -> Privacy controls -> Clean + normalise -> Train models
          -> Evaluate -> Feature importance -> Fairness check
```

1. **Privacy controls:** drops direct identifiers and pseudonymises patient IDs with a salted hash.
2. **Cleaning:** medically impossible zeros (for example Glucose, BMI, Blood Pressure) are treated as missing values; duplicates are removed.
3. **Normalisation:** median imputation followed by z-score standardisation, wrapped inside a scikit-learn `Pipeline` so no information leaks from the test set into training.
4. **Models:** Logistic Regression, Random Forest, Gradient Boosting, with class weighting to handle imbalance.
5. **Evaluation:** 5-fold stratified cross-validation, then a held-out test set. Metrics: ROC-AUC and recall (sensitivity), since missing a sick patient is the costliest error.
6. **Feature importance:** random forest impurity importance, permutation importance on test data, and logistic regression coefficients (direction of effect).
7. **Fairness check:** recall and ROC-AUC compared across age groups.

## Installation

```bash
git clone https://github.com/<your-username>/healthcare-risk-prediction.git
cd healthcare-risk-prediction
pip install -r requirements.txt
```

## Usage

```bash
# Diabetes (Pima Indians dataset)
python healthcare_risk_prediction.py --dataset diabetes

# Heart disease (UCI Cleveland dataset)
python healthcare_risk_prediction.py --dataset heart
```

All plots and tables are saved to the `outputs/` folder:

| File | Description |
|---|---|
| `model_comparison.csv` | CV and test metrics for every model |
| `best_model_eval.png` | Confusion matrix and ROC curve of the best model |
| `feature_importance.png` | Three views of feature importance |
| `correlation.png` | Feature correlation heatmap |

**Optional:** set a secret salt for ID pseudonymisation instead of using the default.

```bash
export PATIENT_ID_SALT="your-secret-salt"
```

## Sample Results

Results on the diabetes dataset (80/20 stratified split):

| Model | CV ROC-AUC | Test ROC-AUC | Test Recall |
|---|---|---|---|
| Gradient Boosting | 0.821 +/- 0.015 | 0.830 | 0.556 |
| Random Forest | 0.832 +/- 0.024 | 0.824 | 0.704 |
| Logistic Regression | 0.844 +/- 0.016 | 0.813 | 0.704 |

**Top risk drivers (permutation importance):** Glucose, BMI, Age, Diabetes Pedigree Function, Insulin. This ordering matches established clinical understanding of diabetes risk.

**Fairness check by age group (Gradient Boosting):**

| Age group | n | Recall | ROC-AUC |
|---|---|---|---|
| 30 and under | 89 | 0.278 | 0.761 |
| 31-45 | 40 | 0.654 | 0.758 |
| 46-60 | 20 | 0.778 | 0.859 |

### Visual outputs

![Model evaluation](outputs/best_model_eval.png)
![Feature importance](outputs/feature_importance.png)

> Exact numbers may vary slightly with library versions.

## Ethics and Privacy

- **Data minimisation:** direct identifiers (name, phone, email, address, national IDs) are dropped before any modelling.
- **Pseudonymisation:** patient IDs are replaced with a salted SHA-256 hash; the salt is read from an environment variable and never hard-coded in a real deployment.
- **Public data only:** no real PHI is processed.
- **Fairness auditing:** model performance is broken down by age group to surface hidden bias.
- **Transparency:** multiple feature importance methods make predictions explainable.
- **Regulatory awareness:** any real-world use would require compliance with laws such as HIPAA, GDPR, and India's DPDP Act, along with informed consent, encryption, access control, and audit logging.

## Limitations

- The models are noticeably **less sensitive for patients aged 30 and under** (recall about 0.28 vs 0.78 for ages 46-60). Part of this reflects how few young patients with the disease appear in the data.
- The Pima dataset is small (768 rows) and covers a specific population (women of Pima Indian heritage, aged 21 and above), so results may not generalise to other groups.
- Default 0.5 decision threshold is used; a screening tool would tune the threshold toward higher recall.
- No external validation on a separate hospital or population.

## Future Work

- Hyperparameter tuning with `GridSearchCV` or `Optuna`
- SHAP values for per-patient explanations
- Threshold tuning to prioritise recall
- Handling imbalance with SMOTE or resampling
- Probability calibration plots
- A Streamlit app for an interactive risk calculator
- Fairness analysis across additional attributes where the data allows

## Project Structure

```
healthcare-risk-prediction/
├── healthcare_risk_prediction.py   # main script
├── requirements.txt
├── README.md
└── outputs/                        # generated plots and tables
    ├── model_comparison.csv
    ├── best_model_eval.png
    ├── feature_importance.png
    └── correlation.png
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, ucimlrepo

## License

Released under the MIT License. See `LICENSE` for details.

## Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for the datasets
- Original Pima Indians Diabetes data from the National Institute of Diabetes and Digestive and Kidney Diseases
