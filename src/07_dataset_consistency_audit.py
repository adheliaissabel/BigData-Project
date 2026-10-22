
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

TARGET = "churn"

NUMERIC = [
    "age", "tenure", "usage_frequency", "support_calls",
    "payment_delay", "total_spend", "last_interaction"
]

CATEGORICAL = [
    "gender", "subscription_type", "contract_length"
]


def audit_split(name, df):
    rows = []

    for col in NUMERIC:
        if col not in df.columns:
            continue

        summary = df.groupby(TARGET)[col].agg(
            ["count", "mean", "median", "std"]
        )

        for label, values in summary.iterrows():
            rows.append({
                "dataset": name,
                "feature": col,
                "group": f"churn={label}",
                "count": int(values["count"]),
                "mean": values["mean"],
                "median": values["median"],
                "std": values["std"],
            })

    for col in CATEGORICAL:
        if col not in df.columns:
            continue

        summary = df.groupby(col, dropna=False)[TARGET].agg(
            ["count", "mean"]
        )

        for value, stats in summary.iterrows():
            rows.append({
                "dataset": name,
                "feature": col,
                "group": str(value),
                "count": int(stats["count"]),
                "mean": stats["mean"],
                "median": np.nan,
                "std": np.nan,
            })

    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA / "training.parquet")
    test = pd.read_parquet(DATA / "testing.parquet")

    all_rows = []
    summary_rows = []

    for name, df in [("training", train), ("testing", test)]:
        df = df.copy()
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET])

        summary_rows.append({
            "dataset": name,
            "rows": len(df),
            "churn_rate": df[TARGET].mean(),
            "unique_customer_id": (
                df["customer_id"].nunique()
                if "customer_id" in df.columns else np.nan
            ),
            "duplicate_rows": int(df.duplicated().sum()),
        })

        all_rows.extend(audit_split(name, df))

        if "customer_id" in df.columns:
            id_num = pd.to_numeric(
                df["customer_id"], errors="coerce"
            )
            valid = id_num.notna() & df[TARGET].notna()

            corr = id_num[valid].corr(df.loc[valid, TARGET])

            print(f"\n{name.upper()}")
            print("Rows:", len(df))
            print("Churn rate:", round(df[TARGET].mean(), 4))
            print("Unique customer_id:", df["customer_id"].nunique())
            print("Correlation ID vs churn:", round(corr, 4))

            # Pola target berdasarkan rentang ID, bukan penyebab kausal.
            temp = pd.DataFrame({
                "customer_id": id_num[valid],
                "churn": df.loc[valid, TARGET]
            }).sort_values("customer_id")

            temp["id_group"] = pd.qcut(
                temp["customer_id"].rank(method="first"),
                q=10,
                labels=False
            )

            grouped = temp.groupby("id_group").agg(
                rows=("churn", "size"),
                id_min=("customer_id", "min"),
                id_max=("customer_id", "max"),
                churn_rate=("churn", "mean")
            ).reset_index()

            grouped.to_csv(
                OUT / f"{name}_id_target_consistency.csv",
                index=False
            )
            print("\nChurn rate menurut kelompok ID:")
            print(grouped.round(4).to_string(index=False))

    pd.DataFrame(summary_rows).to_csv(
        OUT / "dataset_consistency_summary.csv",
        index=False
    )

    pd.DataFrame(all_rows).to_csv(
        OUT / "feature_target_consistency.csv",
        index=False
    )

    print("\nRingkasan dataset:")
    print(pd.DataFrame(summary_rows).round(4).to_string(index=False))

    print("\nFile audit tersimpan di outputs/:")
    print("- dataset_consistency_summary.csv")
    print("- feature_target_consistency.csv")
    print("- training_id_target_consistency.csv")
    print("- testing_id_target_consistency.csv")


if __name__ == "__main__":
    main()
