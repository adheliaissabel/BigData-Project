
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

TARGET = "churn"


def main():
    model_path = OUT / "best_churn_model.joblib"
    metadata_path = OUT / "best_model_metadata.json"
    test_path = DATA / "testing_engineered.parquet"

    for path in [model_path, metadata_path, test_path]:
        if not path.exists():
            raise SystemExit(
                f"File tidak ditemukan: {path}\n"
                "Pastikan training dan feature engineering "
                "sudah berhasil dijalankan."
            )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = joblib.load(model_path)
    df = pd.read_parquet(test_path)

    if TARGET not in df.columns:
        raise SystemExit(f"Kolom target '{TARGET}' tidak ditemukan.")

    # Gunakan hanya fitur yang dipakai ketika model dilatih.
    features = metadata["features"]
    missing = [col for col in features if col not in df.columns]

    if missing:
        raise SystemExit(
            f"Kolom fitur testing tidak lengkap: {missing}"
        )

    X_test = df[features].copy()
    y_test = df[TARGET].astype(int)

    print("Model:", metadata["model"])
    print("Feature mode:", metadata["feature_mode"])
    print("Jumlah data testing:", len(df))
    print("Jumlah fitur:", len(features))
    print("Customer ID digunakan sebagai fitur:",
          any(c.lower() in {"customer_id", "customerid"}
              for c in features))

    # Tidak ada proses fit pada data testing.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "model": metadata["model"],
        "feature_mode": metadata["feature_mode"],
        "rows_test": len(df),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    print("\nHASIL EVALUASI TESTING")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    pd.DataFrame([results]).to_csv(
        OUT / "model_evaluation_testing.csv",
        index=False
    )

    pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"]
    ).to_csv(OUT / "testing_confusion_matrix.csv")

    print("\nHasil tersimpan:")
    print(OUT / "model_evaluation_testing.csv")
    print(OUT / "testing_confusion_matrix.csv")


if __name__ == "__main__":
    main()
