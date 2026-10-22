
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

for name in ["training", "testing"]:
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet")

    print(f"\n{'=' * 50}")
    print(f"{name.upper()}: URUTAN CUSTOMER ID")
    print("=" * 50)

    df["customer_id"] = pd.to_numeric(
        df["customer_id"], errors="coerce"
    )

    # Periksa churn berdasarkan kelompok ID
    df["id_group"] = pd.qcut(
        df["customer_id"].rank(method="first"),
        q=10,
        labels=False
    )

    summary = (
        df.groupby("id_group", observed=True)
        .agg(
            rows=("churn", "size"),
            id_min=("customer_id", "min"),
            id_max=("customer_id", "max"),
            churn_rate=("churn", "mean"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / f"{name}_id_order_audit.csv",
        index=False
    )

    # Periksa perubahan label berurutan setelah ID diurutkan
    ordered = df.sort_values("customer_id")
    labels = ordered["churn"]

    changes = labels.ne(labels.shift()).sum() - 1

    print("\nPerubahan label sepanjang urutan ID:", int(changes))
    print("\nContoh 10 baris pertama setelah ID diurutkan:")

    print(
        ordered[["customer_id", "churn"]]
        .head(10)
        .to_string(index=False)
    )

print("\nAudit urutan selesai.")
