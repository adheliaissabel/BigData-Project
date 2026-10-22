"""Compare baseline and engineered-feature models using the same stratified validation split."""
from pathlib import Path
import joblib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
TARGET = "churn"
DROP = {"customer_id", "dataset_split", TARGET}
ENGINEERED = {"high_support_calls_6plus", "high_payment_delay_21plus", "support_delay_combined_risk"}

def build_preprocessor(X):
    cat = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, num), ("cat", categorical_pipe, cat)])

def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42
        ),
    }

def metrics_for(y, pred, prob):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }

def run_experiment(df, feature_mode, split_seed=42):
    X = df.drop(columns=[c for c in DROP if c in df.columns]).copy()
    if feature_mode == "baseline":
        X = X.drop(columns=[c for c in ENGINEERED if c in X.columns], errors="ignore")
    y = df[TARGET].astype(int)
    # Remove rows with missing target only; preprocessing handles feature missing values.
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=split_seed
    )
    results = []
    for model_name, model in get_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", model)])
        print(f"Training {model_name} ({feature_mode}) ...")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        prob = pipe.predict_proba(X_val)[:, 1]
        row = {"feature_mode": feature_mode, "model": model_name, **metrics_for(y_val, pred, prob)}
        results.append(row)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    path = DATA / "training_engineered.parquet"
    if not path.exists():
        raise SystemExit(
            "Jalankan 03_feature_engineering.py terlebih dahulu."
        )

    df = pd.read_parquet(path)

    # Evaluasi validation untuk semua kombinasi model dan fitur.
    all_results = (
        run_experiment(df, "baseline")
        + run_experiment(df, "engineered")
    )

    clean_results = [
        {k: v for k, v in row.items() if k != "confusion_matrix"}
        for row in all_results
    ]

    results_df = pd.DataFrame(clean_results)
    results_df = results_df.sort_values("f1", ascending=False)

    results_df.to_csv(
        OUT / "model_comparison_validation.csv",
        index=False
    )

    with open(
        OUT / "model_confusion_matrices.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(all_results, f, indent=2)

    print("\nPerbandingan model (validation training):")
    print(results_df.round(4).to_string(index=False))

    # Pilih kombinasi model dan fitur dengan F1 validation tertinggi.
    best_row = results_df.iloc[0]
    best_model_name = best_row["model"]
    best_feature_mode = best_row["feature_mode"]

    print("\nModel terpilih:", best_model_name)
    print("Mode fitur:", best_feature_mode)

    # Siapkan seluruh data training untuk melatih model final.
    X = df.drop(
        columns=[c for c in DROP if c in df.columns],
        errors="ignore"
    ).copy()

    # Pastikan target dan ID pelanggan tidak menjadi fitur.
    X = X.drop(
        columns=[
            c for c in X.columns
            if c.lower() in {"churn", "customer_id", "customerid"}
        ],
        errors="ignore"
    )

    if best_feature_mode == "baseline":
        X = X.drop(
            columns=[
                c for c in ENGINEERED if c in X.columns
            ],
            errors="ignore"
        )

    y = df[TARGET].astype(int)

    # Fit ulang pipeline terpilih menggunakan seluruh data training.
    final_model = get_models()[best_model_name]
    final_pipeline = Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("model", final_model)
    ])

    print("\nMelatih ulang model dengan seluruh data training...")
    final_pipeline.fit(X, y)

    model_path = OUT / "best_churn_model.joblib"
    joblib.dump(final_pipeline, model_path)

    metadata = {
        "model": best_model_name,
        "feature_mode": best_feature_mode,
        "target": TARGET,
        "features": list(X.columns),
        "validation_f1": float(best_row["f1"]),
        "validation_accuracy": float(best_row["accuracy"])
    }

    with open(
        OUT / "best_model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nModel tersimpan:", model_path)
    print("Metadata tersimpan:", OUT / "best_model_metadata.json")
    print(
        "\nCatatan: skor validation bukan skor testing final."
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
