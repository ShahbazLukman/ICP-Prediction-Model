
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "synthetic_course_leads.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

TARGET = "purchased_within_30_days"

# IMPORTANT: these are available before purchase in our synthetic scenario.
# Do not include purchased_course or any post-purchase information.
FEATURES = [
    "age", "education", "occupation", "city", "course_enquiry", "lead_source",
    "website_visits", "course_page_views", "pricing_page_views",
    "brochure_downloaded", "email_opens", "email_clicks",
    "whatsapp_responses", "calls_answered", "followup_count",
    "webinar_attended", "demo_attended", "video_watches",
    "days_since_last_activity"
]

df = pd.read_csv(DATA)

X = df[FEATURES]
y = df[TARGET]

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

models = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", random_state=42, n_jobs=4
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall"
        },
        n_jobs=-1
    )
    results.append({
        "model": name,
        "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
        "cv_pr_auc_mean": scores["test_pr_auc"].mean(),
        "cv_f1_mean": scores["test_f1"].mean(),
        "cv_precision_mean": scores["test_precision"].mean(),
        "cv_recall_mean": scores["test_recall"].mean(),
    })

results_df = pd.DataFrame(results).sort_values("cv_pr_auc_mean", ascending=False)
results_df.to_csv(REPORTS / "model_comparison.csv", index=False)
print("\nMODEL COMPARISON\n", results_df)

# Choose XGBoost for the tuning stage as the main candidate.
base_xgb = XGBClassifier(
    eval_metric="logloss", random_state=42, n_jobs=4
)

xgb_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", base_xgb)
])

param_dist = {
    "model__n_estimators": [200, 300, 500, 700],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
    "model__min_child_weight": [1, 3, 5, 8],
}

search = RandomizedSearchCV(
    xgb_pipe,
    param_distributions=param_dist,
    n_iter=25,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
joblib.dump(best_model, MODELS / "purchase_model.joblib")

proba = best_model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "f1": f1_score(y_test, pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, proba),
    "pr_auc": average_precision_score(y_test, proba),
    "brier_score": brier_score_loss(y_test, proba)
}
with open(REPORTS / "final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nBEST PARAMETERS:\n", search.best_params_)
print("\nFINAL TEST METRICS:\n", json.dumps(metrics, indent=2))
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, pred, zero_division=0))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred))
disp.plot()
plt.title("Purchase Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORTS / "confusion_matrix.png", dpi=160)
plt.close()

# Top-K lift / business metric.
eval_df = pd.DataFrame({"y": y_test.values, "p": proba})
eval_df = eval_df.sort_values("p", ascending=False).reset_index(drop=True)
baseline_rate = eval_df["y"].mean()
for pct in [0.05, 0.10, 0.20]:
    k = max(1, int(len(eval_df) * pct))
    top_rate = eval_df.iloc[:k]["y"].mean()
    lift = top_rate / baseline_rate if baseline_rate > 0 else np.nan
    print(f"Top {int(pct*100)}%: conversion rate={top_rate:.4f}, lift={lift:.2f}x")

# A simple threshold table for business use.
threshold_rows = []
for threshold in np.arange(0.20, 0.81, 0.05):
    p = (proba >= threshold).astype(int)
    threshold_rows.append({
        "threshold": round(float(threshold), 2),
        "precision": precision_score(y_test, p, zero_division=0),
        "recall": recall_score(y_test, p, zero_division=0),
        "f1": f1_score(y_test, p, zero_division=0),
        "leads_selected_pct": p.mean()
    })
pd.DataFrame(threshold_rows).to_csv(REPORTS / "threshold_analysis.csv", index=False)

print("\nSaved model to:", MODELS / "purchase_model.joblib")
