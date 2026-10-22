
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "contract_length",
    "support_calls",
    "payment_delay",
    "subscription_type",
]

for dataset_name in ["training", "testing"]:
    path = DATA_DIR / f"{dataset_name}.parquet"
    df = pd.read_parquet(path)

    print(f"\n{'=' * 60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # Periksa jumlah dan proporsi masing-masing label
    print("\nDistribusi churn:")
    print(
        df["churn"]
        .value_counts(dropna=False)
        .rename_axis("churn")
        .to_frame("count")
        .assign(
            proportion=lambda x: x["count"] / len(df)
        )
        .to_string()
    )

    # Periksa jumlah data di balik setiap tingkat churn
    for feature in FEATURES:
        if feature not in df.columns:
            print(f"\nKolom {feature} tidak ditemukan, dilewati.")
            continue

        audit = (
            df.groupby([feature, "churn"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )

        for label in [0, 1]:
            if label not in audit.columns:
                audit[label] = 0

        audit = audit.rename(
            columns={0: "not_churn_count", 1: "churn_count"}
        )

        audit["total"] = (
            audit["not_churn_count"] + audit["churn_count"]
        )

        audit["churn_rate"] = (
            audit["churn_count"] / audit["total"]
        )

        audit.to_csv(
            OUTPUT_DIR / f"{dataset_name}_{feature}_audit.csv"
        )

        print(f"\n--- {feature} ---")
        print(audit.to_string())

    # Periksa apakah CustomerID berkorelasi dengan churn
    id_candidates = [
        col for col in df.columns
        if col.replace("_", "").lower() == "customerid"
    ]

    if id_candidates:
        id_col = id_candidates[0]
        numeric_id = pd.to_numeric(df[id_col], errors="coerce")

        print(f"\nPemeriksaan {id_col}:")
        print("Jumlah ID unik:", df[id_col].nunique())
        print("Jumlah baris:", len(df))
        print(
            "Korelasi ID dengan churn:",
            round(numeric_id.corr(df["churn"]), 4)
        )
        print(
            "Korelasi ID dengan churn hanya indikasi, "
            "bukan bukti data leakage."
        )
    else:
        print("\nKolom CustomerID tidak ditemukan.")

print("\nAudit selesai. Hasil detail tersimpan di outputs/.")
