# src/train_evaluate.py
# ------------------------------------------------------------
# Train LogReg, GBM, MLP on Pima; save hold-out metrics (ROC,
# CM) and 5-fold cross-validation metrics + charts.
# ------------------------------------------------------------
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score, RocCurveDisplay,
    precision_recall_fscore_support, accuracy_score,
    ConfusionMatrixDisplay, make_scorer,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils import load_pima, split, RANDOM_STATE

# ---------- Paths
OUT_DIR = "outputs"
FIG_DIR = f"{OUT_DIR}/figures"
MET_DIR = f"{OUT_DIR}/metrics"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)

# ---------- Data
X, y = load_pima()
X_train, X_test, y_train, y_test = split(X, y)

# ---------- Models (pipelines)
logreg = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

gbm = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    ))
])

mlp = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                          activation="relu",
                          solver="adam",
                          alpha=1e-4,
                          learning_rate_init=1e-3,
                          max_iter=1000,
                          random_state=RANDOM_STATE))
])

models = {"logreg": logreg, "gbm": gbm, "mlp": mlp}

# ============================================================
# Hold-out Test Evaluation (80/20 split)
# ============================================================
metrics = {}
fig, ax = plt.subplots()

for name, model in models.items():
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    metrics[name] = {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f)
    }

    RocCurveDisplay.from_predictions(y_test, proba,
                                     name=f"{name.upper()} (AUC={auc:.2f})",
                                     ax=ax)

plt.title("ROC Curves — LogReg vs GBM vs MLP")
plt.tight_layout()
roc_path = f"{FIG_DIR}/roc_all.png"
plt.savefig(roc_path, dpi=200, bbox_inches="tight")
plt.close()
with open(f"{MET_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved:", roc_path, f"{MET_DIR}/metrics.json")

# Confusion matrices
for name, model in models.items():
    y_pred = model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix — {name.upper()}")
    plt.tight_layout()
    cm_path = f"{FIG_DIR}/cm_{name}.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", cm_path)

# ============================================================
# 5-Fold Cross-Validation (performance metrics ONLY)
# ============================================================
print("\nRunning 5-fold cross-validation for performance metrics...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# NOTE: use 'roc_auc' string scorer (auto uses predict_proba)
scorers = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": "roc_auc",
}

cv_results = {}
model_order = list(models.keys())  # ["logreg", "gbm", "mlp"]

for name in model_order:
    model = models[name]
    cv_results[name] = {}
    for metric_name, scorer in scorers.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        cv_results[name][metric_name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std())
        }
        print(f"{name.upper()} - {metric_name}: {scores.mean():.3f} ± {scores.std():.3f}")

# Save CV table
cv_df = pd.DataFrame({
    m: {k: f"{v['mean']:.3f} ± {v['std']:.3f}" for k, v in d.items()}
    for m, d in cv_results.items()
}).T
cv_csv = f"{MET_DIR}/crossval_metrics.csv"
cv_df.to_csv(cv_csv)
print("Saved:", cv_csv)

# ============================================================
# CV Visualization: bar charts with error bars (Acc, F1, AUC)
# ============================================================
metrics_to_plot = ["accuracy", "f1", "roc_auc"]
colors = ["#4c72b0", "#55a868", "#c44e52"]  # readable palette

for metric_name in metrics_to_plot:
    means = [cv_results[m][metric_name]["mean"] for m in model_order]
    stds = [cv_results[m][metric_name]["std"] for m in model_order]

    plt.figure(figsize=(6.2, 4.2))
    plt.bar([m.upper() for m in model_order], means, yerr=stds,
            capsize=6, color=colors)
    plt.ylim(0, 1)
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} — 5-Fold Cross-Validation")
    plt.tight_layout()

    out_path = f"{FIG_DIR}/cv_{metric_name}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

print("\nAll figures & metrics saved under:", OUT_DIR)
