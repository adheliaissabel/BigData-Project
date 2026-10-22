
from pathlib import Path
import importlib.util
import json

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

# Import fungsi preprocessing dan konfigurasi dari script training.
spec = importlib.util.spec_from_file_location(
    "train_compare", ROOT / "src" / "04_train_compare.py"
)
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def evaluate(name, feature_mode, model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    return {
        "model": name,
        "feature_mode": feature_mode,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main():
    train_path = DATA / "training_engineered.parquet"
    test_path = DATA / "testing_engineered.parquet"
    saved_model_path = OUT / "best_churn_model.joblib"
    metadata_path = OUT / "best_model_metadata.json"

    for path in [
        train_path, test_path, saved_model_path, metadata_path
    ]:
        if not path.exists():
            raise SystemExit(f"File tidak ditemukan: {path}")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    y_train = train[tc.TARGET].astype(int)
    y_test = test[tc.TARGET].astype(int)

    # Model engineered yang sudah dilatih dan disimpan sebelumnya.
    engineered_model = joblib.load(saved_model_path)
    engineered_features = metadata["features"]

    X_test_engineered = test[engineered_features].copy()

    # Baseline: buang fitur rekayasa dan ID/kolom administratif.
    excluded = {
        tc.TARGET, "customer_id", "customerid", "dataset_split"
    }
    excluded.update(tc.ENGINEERED)

    baseline_features = [
        c for c in train.columns if c.lower() not in excluded
    ]

    X_train_baseline = train[baseline_features].copy()
    X_test_baseline = test[baseline_features].copy()

    # Latih satu model HistGradientBoosting baseline pada seluruh training.
    baseline_model = Pipeline([
        ("preprocess", tc.build_preprocessor(X_train_baseline)),
        ("model", tc.get_models()["HistGradientBoosting"]),
    ])

    print("Melatih HistGradientBoosting baseline...")
    baseline_model.fit(X_train_baseline, y_train)

    print("Mengevaluasi baseline dan engineered pada testing...")
    baseline_result = evaluate(
        "HistGradientBoosting", "baseline",
        baseline_model, X_test_baseline, y_test
    )

    engineered_result = evaluate(
        metadata["model"], "engineered",
        engineered_model, X_test_engineered, y_test
    )

    results = [baseline_result, engineered_result]
    summary = pd.DataFrame([
        {k: v for k, v in row.items() if k != "confusion_matrix"}
        for row in results
    ])

    summary.to_csv(OUT / "baseline_vs_engineered_testing.csv", index=False)

    with open(
        OUT / "baseline_vs_engineered_confusion.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nPERBANDINGAN MODEL PADA DATA TESTING")
    print(summary.round(4).to_string(index=False))

    print("\nConfusion matrix:")
    for row in results:
        print(f"\n{row['feature_mode']}:")
        print(row["confusion_matrix"])

    print("\nHasil tersimpan di outputs/")


if __name__ == "__main__":
    main()
