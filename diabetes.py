"""
Healthcare Predictive Analytics - Disease Risk Detection
=========================================================
Datasets (choose with --dataset):
  diabetes : Pima Indians Diabetes (UCI / Kaggle)
  heart    : Heart Disease, Cleveland (UCI, via `ucimlrepo`)

Pipeline:
  1. Load data  ->  2. Privacy step (drop / pseudonymise identifiers)
  3. Clean + normalise  ->  4. Train several classifiers
  5. Evaluate (recall-focused)  ->  6. Feature importance
  7. Simple fairness check across age groups

Usage:
  pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo
  python healthcare_risk_prediction.py --dataset diabetes
  python healthcare_risk_prediction.py --dataset heart

NOTE: Educational project only. NOT a medical device / diagnostic tool.
"""

import argparse
import hashlib
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # works on servers / without display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
def load_diabetes() -> pd.DataFrame:
    """Pima Indians Diabetes dataset (UCI). Also on Kaggle as diabetes.csv."""
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
           "master/pima-indians-diabetes.data.csv")
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "target"]
    # If you downloaded Kaggle's diabetes.csv, use:
    # df = pd.read_csv("diabetes.csv").rename(columns={"Outcome": "target"})
    df = pd.read_csv(url, header=None, names=cols)
    return df


def load_heart() -> pd.DataFrame:
    """UCI Heart Disease (Cleveland) via ucimlrepo."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    X, y = ds.data.features.copy(), ds.data.targets.copy()
    df = X.copy()
    df["target"] = (y.iloc[:, 0] > 0).astype(int)  # 0 = no disease, 1 = disease
    return df


# ----------------------------------------------------------------------
# 2. PRIVACY / ETHICAL DATA HANDLING
# ----------------------------------------------------------------------
DIRECT_IDENTIFIERS = ["name", "patient_name", "phone", "email", "address",
                      "ssn", "aadhaar", "mrn"]


def pseudonymise_id(value: str, salt: str) -> str:
    """One-way salted hash so records can be linked without exposing identity."""
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()[:16]


def apply_privacy_controls(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop direct identifiers if present (data minimisation).
    - Replace patient_id with a salted hash (pseudonymisation).
    - Generalise exact age into bands for reporting (reduces re-identification).
    Real-world: also follow HIPAA / GDPR / India's DPDP Act, get consent,
    use access control + encryption, and keep an audit trail.
    """
    df = df.copy()
    dropped = [c for c in df.columns if c.lower() in DIRECT_IDENTIFIERS]
    df.drop(columns=dropped, inplace=True)

    if "patient_id" in df.columns:
        salt = os.environ.get("PATIENT_ID_SALT", "change-me-and-keep-secret")
        df["patient_id"] = df["patient_id"].map(lambda v: pseudonymise_id(v, salt))

    print(f"[Privacy] Dropped identifier columns: {dropped or 'none found'}")
    print("[Privacy] Dataset used is public & de-identified; no real PHI handled.")
    return df


# ----------------------------------------------------------------------
# 3. CLEANING + NORMALISATION
# ----------------------------------------------------------------------
def clean_medical_values(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Medically impossible zeros are really missing values -> NaN."""
    df = df.copy()
    if dataset == "diabetes":
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness",
                           "Insulin", "BMI"]
        df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    df = df.drop_duplicates()
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median imputation (robust to outliers) + standardisation (z-score)."""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("num", numeric_pipe, num_cols)])


# ----------------------------------------------------------------------
# 4. MODELS
# ----------------------------------------------------------------------
def get_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


# ----------------------------------------------------------------------
# 5. EVALUATION
# ----------------------------------------------------------------------
def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, rows = {}, []

    for name, model in get_models().items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "CV ROC-AUC": f"{cv_auc.mean():.3f} +/- {cv_auc.std():.3f}",
            "Test ROC-AUC": round(roc_auc_score(y_test, proba), 3),
            "Test Recall (sensitivity)": round(recall_score(y_test, pred), 3),
        })
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False)
    print("\n=== Model comparison ===")
    print(results.to_string(index=False))
    results.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    return fitted, results


def plot_best_model(pipe, name, X_test, y_test):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax[0],
                                          cmap="Blues", colorbar=False)
    ax[0].set_title(f"{name} - Confusion Matrix")
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax[1])
    ax[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax[1].set_title(f"{name} - ROC Curve")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/best_model_eval.png", dpi=150)
    plt.close()

    print(f"\n=== Classification report: {name} ===")
    print(classification_report(y_test, pipe.predict(X_test),
                                target_names=["No disease", "Disease"]))


# ----------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ----------------------------------------------------------------------
def feature_importance_analysis(fitted, X_test, y_test, feature_names):
    # (a) Model-native importance for tree models
    rf = fitted["RandomForest"].named_steps["model"]
    native = pd.Series(rf.feature_importances_, index=feature_names)

    # (b) Permutation importance on held-out data (model-agnostic, more reliable)
    best = fitted["RandomForest"]
    perm = permutation_importance(best, X_test, y_test, n_repeats=20,
                                  scoring="roc_auc", random_state=RANDOM_STATE)
    perm_s = pd.Series(perm.importances_mean, index=X_test.columns)

    # (c) Logistic regression coefficients (direction of effect, on scaled data)
    lr = fitted["LogisticRegression"].named_steps["model"]
    coefs = pd.Series(lr.coef_[0], index=feature_names)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    native.sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title("Random Forest (impurity) importance")
    perm_s.sort_values().plot.barh(ax=axes[1], color="darkorange")
    axes[1].set_title("Permutation importance (test ROC-AUC drop)")
    coefs.sort_values().plot.barh(
        ax=axes[2], color=["tomato" if v < 0 else "seagreen" for v in coefs.sort_values()])
    axes[2].set_title("Logistic Regression coefficients\n(+ raises risk, - lowers risk)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

    print("\n=== Top risk drivers (permutation importance) ===")
    print(perm_s.sort_values(ascending=False).head(5).round(4).to_string())


# ----------------------------------------------------------------------
# 7. FAIRNESS CHECK (ethics)
# ----------------------------------------------------------------------
def fairness_check(pipe, X_test, y_test, age_col="Age"):
    """Compare recall / AUC across age groups to spot performance gaps."""
    if age_col not in X_test.columns:
        return
    groups = pd.cut(X_test[age_col], bins=[0, 30, 45, 60, 120],
                    labels=["<=30", "31-45", "46-60", "60+"])
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    rows = []
    for g in groups.cat.categories:
        m = (groups == g).values
        if m.sum() < 10 or y_test[m].nunique() < 2:
            continue
        rows.append({"Age group": g, "n": int(m.sum()),
                     "Recall": round(recall_score(y_test[m], pred[m]), 3),
                     "ROC-AUC": round(roc_auc_score(y_test[m], proba[m]), 3)})
    print("\n=== Fairness check by age group ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Large gaps between groups = possible bias; investigate before any deployment.")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["diabetes", "heart"], default="diabetes")
    args = parser.parse_args()

    df = load_diabetes() if args.dataset == "diabetes" else load_heart()
    print(f"Loaded '{args.dataset}' dataset: {df.shape[0]} rows x {df.shape[1]} cols")

    df = apply_privacy_controls(df)
    df = clean_medical_values(df, args.dataset)

    print("\nClass balance:\n", df["target"].value_counts(normalize=True).round(3).to_string())
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0].to_string() or "none")

    # Quick EDA heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature correlation")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation.png", dpi=150)
    plt.close()

    X, y = df.drop(columns="target"), df["target"]
    # Stratified split + preprocessing inside the pipeline => no data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    preprocessor = build_preprocessor(X)
    fitted, results = evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    best_name = results.iloc[0]["Model"]
    plot_best_model(fitted[best_name], best_name, X_test, y_test)
    feature_importance_analysis(fitted, X_test, y_test, X.columns)
    fairness_check(fitted[best_name], X_test, y_test)

    print(f"\nPlots + results saved in ./{OUT_DIR}/")
    print("Disclaimer: research/educational use only; clinical use needs "
          "validation, regulatory approval and clinician oversight.")


if __name__ == "__main__":
    main()