# src/interpret.py
# ------------------------------------------------------------
# Interpretable ML: LIME + SHAP (local/global) + Permutation Importance
# Models: Logistic Regression (interpretable), GBM, MLP (black-box)
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular as lt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

from src.utils import load_pima, split, RANDOM_STATE

# ---------- Paths
OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Data
X, y = load_pima()
X_train, X_test, y_train, y_test = split(X, y)

# ---------- Models (pipelines)
logreg = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
]).fit(X_train, y_train)

gbm = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    ))
]).fit(X_train, y_train)

mlp = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
]).fit(X_train, y_train)

# ============================================================
# LIME — LOCAL (GBM & MLP)
# ============================================================
patient_idx = 16  # fixed for comparability

# ---- GBM LIME (feed imputed data)
def gbm_pred(Xn):
    return gbm.predict_proba(pd.DataFrame(Xn, columns=X_train.columns))

X_train_imp_gbm = pd.DataFrame(
    gbm.named_steps["imp"].transform(X_train),
    columns=X_train.columns, index=X_train.index
)
X_test_imp_gbm = pd.DataFrame(
    gbm.named_steps["imp"].transform(X_test),
    columns=X_test.columns, index=X_test.index
)

expl_gbm = lt.LimeTabularExplainer(
    X_train_imp_gbm.values,
    feature_names=X_train_imp_gbm.columns.tolist(),
    class_names=["neg", "pos"],
    mode="classification",
    discretize_continuous=False,
    random_state=RANDOM_STATE,
)
exp_gbm = expl_gbm.explain_instance(
    X_test_imp_gbm.iloc[patient_idx].values, gbm_pred, num_features=10
)
fig = exp_gbm.as_pyplot_figure()
plt.title("LIME — GBM"); plt.tight_layout()
fig.savefig(f"{OUT_DIR}/lime_gbm.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- MLP LIME (feed imputed data; scaler is inside pipeline for model)
def mlp_pred(Xn):
    return mlp.predict_proba(pd.DataFrame(Xn, columns=X_train.columns))

X_train_imp_mlp = pd.DataFrame(
    mlp.named_steps["imp"].transform(X_train),
    columns=X_train.columns, index=X_train.index
)
X_test_imp_mlp = pd.DataFrame(
    mlp.named_steps["imp"].transform(X_test),
    columns=X_test.columns, index=X_test.index
)

expl_mlp = lt.LimeTabularExplainer(
    X_train_imp_mlp.values,
    feature_names=X_train_imp_mlp.columns.tolist(),
    class_names=["neg", "pos"],
    mode="classification",
    discretize_continuous=False,
    random_state=RANDOM_STATE,
)
exp_mlp = expl_mlp.explain_instance(
    X_test_imp_mlp.iloc[patient_idx].values, mlp_pred, num_features=10
)
fig = exp_mlp.as_pyplot_figure()
plt.title("LIME — MLP"); plt.tight_layout()
fig.savefig(f"{OUT_DIR}/lime_mlp.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============================================================
# SHAP — GBM (TreeExplainer): LOCAL + GLOBAL
# ============================================================
explainer_tree = shap.TreeExplainer(gbm.named_steps["clf"])
sv_gbm = explainer_tree(X_test_imp_gbm)  # Explanation object (uses imputed data)

# Local waterfall
shap.plots.waterfall(sv_gbm[patient_idx], max_display=12, show=False)
plt.gcf().savefig(f"{OUT_DIR}/shap_waterfall_gbm.png", dpi=200, bbox_inches="tight")
plt.close()

# Global bar
shap.plots.bar(sv_gbm, max_display=15, show=False)
plt.gcf().savefig(f"{OUT_DIR}/shap_bar_gbm.png", dpi=200, bbox_inches="tight")
plt.close()

# Global beeswarm
shap.plots.beeswarm(sv_gbm, max_display=15, show=False)
plt.gcf().savefig(f"{OUT_DIR}/shap_beeswarm_gbm.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# SHAP — MLP (KernelExplainer): LOCAL + GLOBAL (bar + beeswarm)
# ============================================================
# Work with the exact inputs the MLP saw: imputed + scaled
X_train_imp = mlp.named_steps["imp"].transform(X_train)
X_test_imp = mlp.named_steps["imp"].transform(X_test)
X_train_std = mlp.named_steps["sc"].transform(X_train_imp)
X_test_std = mlp.named_steps["sc"].transform(X_test_imp)

# Prediction function on STANDARDIZED inputs for class 1
def _mlp_proba_std(x_std: np.ndarray) -> np.ndarray:
    return mlp.named_steps["clf"].predict_proba(x_std)[:, 1]

# Background for KernelExplainer (speed/stability)
rng = np.random.RandomState(RANDOM_STATE)
bg_idx = rng.choice(X_train_std.shape[0], size=min(100, X_train_std.shape[0]), replace=False)
background = X_train_std[bg_idx]

# Subsample test points if needed (KernelExplainer can be slow)
test_subset = X_test_std[:200]

explainer_mlp = shap.KernelExplainer(_mlp_proba_std, background, link="logit")
sv_mlp = explainer_mlp.shap_values(test_subset, nsamples=200)  # array: (n, n_features)

# Local waterfall (first instance)
from shap import Explanation
shap.plots.waterfall(
    Explanation(
        values=sv_mlp[0],
        base_values=explainer_mlp.expected_value,
        data=test_subset[0],
        feature_names=X.columns,
    ),
    max_display=12,
    show=False
)
plt.gcf().savefig(f"{OUT_DIR}/shap_waterfall_mlp.png", dpi=200, bbox_inches="tight")
plt.close()

# Global bar: mean |SHAP|
mean_abs = np.abs(sv_mlp).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
plt.figure(figsize=(9, 6))
plt.bar(range(len(order)), mean_abs[order])
plt.xticks(range(len(order)), X.columns[order], rotation=45, ha="right")
plt.title("MLP — mean |SHAP| (KernelExplainer)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_bar_mlp.png", dpi=200, bbox_inches="tight")
plt.close()

# Global beeswarm for MLP
plt.figure()
shap.summary_plot(sv_mlp, test_subset, feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_beeswarm_mlp.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# PERMUTATION IMPORTANCE — GLOBAL for ALL THREE
# ============================================================
models = {
    "logreg": logreg,
    "gbm": gbm,
    "mlp": mlp,
}

for name, model in models.items():
    print(f"Running permutation importance for {name.upper()}...")
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE
    )
    pi = pd.Series(perm.importances_mean, index=X_test.columns).sort_values()
    ax = pi.plot(kind="barh", figsize=(10, 7),
                 title=f"Permutation Importance — {name.upper()}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/perm_importance_{name}.png", dpi=200, bbox_inches="tight")
    plt.close()

print("Saved all interpretability figures to:", OUT_DIR)
