# Common utility functions (e.g., load data, preprocess)
# src/utils.py
import os
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Map short → long, for report-friendly CSVs
COLUMN_MAP = {
    "preg": "Pregnancies",
    "plas": "PlasmaGlucose",
    "pres": "BloodPressure",
    "skin": "SkinThickness",
    "insu": "Insulin",
    "mass": "BMI",
    "pedi": "DiabetesPedigree",
    "age":  "Age",
    "class":"Diabetes"
}

DATA_DIR = "data"
SHORT_CSV = os.path.join(DATA_DIR, "pima_diabetes_clean.csv")
LONG_CSV  = os.path.join(DATA_DIR, "pima_diabetes_clean_long.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train_long.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_long.csv")

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _encode_target_to_numeric(y, name="class"):
    y = pd.Series(y, name=name)
    if y.dtype == object or str(y.dtype) == "category":
        y = y.astype("category").cat.codes
    return pd.Series(y.values, index=y.index, name=name)

def load_pima(save_clean=True, save_splits=False, test_size=0.2):
    """
    Load the Pima Indians Diabetes dataset (OpenML #37), clean zeros-as-missing,
    save cleaned CSVs (short and long names), and optionally save train/test splits (long names).
    Returns: X (DataFrame, short names), y (Series numeric)
    """
    _ensure_data_dir()

    # --- Download from OpenML
    ds = openml.datasets.get_dataset(37)  # Pima Indians Diabetes
    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)

    # --- Numeric target (0/1)
    y = _encode_target_to_numeric(y, name="class")

    # --- Treat zeros as missing for medical columns
    zero_as_missing_cols = [c for c in ['plas','pres','skin','insu','mass'] if c in X.columns]
    for col in zero_as_missing_cols:
        X.loc[X[col] == 0, col] = np.nan

    # --- Print totals ---
    counts = y.value_counts()
    print("\n=== Dataset totals ===")
    print(f"Total patients: {len(y)}")
    print(f"Affected (diabetes=1): {counts.get(1, 0)}")
    print(f"Not affected (diabetes=0): {counts.get(0, 0)}")

    # --- Save class balance plot ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    counts.rename({0: "Not Affected", 1: "Affected"}).plot(
        kind="bar", color=["skyblue", "salmon"], ax=ax
    )
    plt.title("Diabetes Class Distribution")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig_path = os.path.join("outputs", "figures", "class_balance.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Class balance figure saved: {fig_path}")

    # --- Save cleaned datasets ---
    if save_clean:
        df_short = X.copy(); df_short["class"] = y
        df_short.to_csv(SHORT_CSV, index=False)

        df_long = df_short.rename(columns=COLUMN_MAP)
        df_long.to_csv(LONG_CSV, index=False)

        print(f"✅ Cleaned (short) saved: {SHORT_CSV}")
        print(f"✅ Cleaned (long)  saved: {LONG_CSV}")

    if save_splits:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        train_long = X_train.copy(); train_long["class"] = y_train
        test_long  = X_test.copy();  test_long["class"]  = y_test
        train_long = train_long.rename(columns=COLUMN_MAP)
        test_long  = test_long.rename(columns=COLUMN_MAP)
        train_long.to_csv(TRAIN_CSV, index=False)
        test_long.to_csv(TEST_CSV, index=False)
        print(f"✅ Train split (long) saved: {TRAIN_CSV}")
        print(f"✅ Test split  (long) saved: {TEST_CSV}")

    return X, y

def split(X, y, test_size=0.2):
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
