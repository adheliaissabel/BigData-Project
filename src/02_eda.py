"""EDA outputs for training and testing; EDA is diagnostic, not proof of causation."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
NUMERIC = ["age", "tenure", "usage_frequency", "support_calls",
           "payment_delay", "total_spend", "last_interaction"]

def main():
    train_path, test_path = DATA / "training.parquet", DATA / "testing.parquet"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("Jalankan dulu: python src/01_ingest_clean_store.py")
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)

    # Basic quality summary
    summaries = []
    for split, df in [("training", train), ("testing", test)]:
        summaries.append({
            "dataset": split, "rows": len(df), "columns": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "churn_rate": float(df["churn"].mean()),
            "duplicate_rows": int(df.duplicated().sum()),
        })
    pd.DataFrame(summaries).to_csv(OUT / "data_quality_summary.csv", index=False)

    # Label distribution
    label_rows = []
    for split, df in [("training", train), ("testing", test)]:
        counts = df["churn"].value_counts(dropna=False)
        for label, count in counts.items():
            label_rows.append({"dataset": split, "churn": label, "count": int(count),
                               "proportion": float(count / len(df))})
    pd.DataFrame(label_rows).to_csv(OUT / "churn_distribution.csv", index=False)

    # Feature distribution by target and split
    for split, df in [("training", train), ("testing", test)]:
        numeric = [c for c in NUMERIC if c in df.columns]
        df[numeric + ["churn"]].groupby("churn").agg(["mean", "median", "std"]).to_csv(
            OUT / f"{split}_numeric_by_churn.csv"
        )
        for col in ["support_calls", "payment_delay", "contract_length", "subscription_type"]:
            if col in df.columns:
                pd.crosstab(df[col], df["churn"], normalize="index").to_csv(
                    OUT / f"{split}_{col}_churn_rate.csv"
                )

    # Compare numerical distributions
    comparison = []
    for col in NUMERIC:
        if col in train.columns and col in test.columns:
            comparison.append({
                "feature": col,
                "training_mean": train[col].mean(),
                "testing_mean": test[col].mean(),
                "training_median": train[col].median(),
                "testing_median": test[col].median(),
                "training_missing": int(train[col].isna().sum()),
                "testing_missing": int(test[col].isna().sum()),
            })
    pd.DataFrame(comparison).to_csv(OUT / "train_test_numeric_comparison.csv", index=False)

    sns.set_theme()
    plt.figure(figsize=(7, 4))
    sns.countplot(data=train, x="churn")
    plt.title("Training: Distribusi Label Churn")
    plt.tight_layout()
    plt.savefig(OUT / "training_churn_distribution.png", dpi=150)
    plt.close()

    for col in ["support_calls", "payment_delay"]:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=train, x=col, hue="churn", bins=30, stat="density",
                     common_norm=False, element="step")
        plt.title(f"Training: Distribusi {col} berdasarkan Churn")
        plt.tight_layout()
        plt.savefig(OUT / f"training_{col}_by_churn.png", dpi=150)
        plt.close()

    print("EDA selesai. Periksa file CSV/PNG di outputs/.")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nCatatan: perbedaan train-test dan korelasi tidak otomatis membuktikan penyebab atau data leakage.")

if __name__ == "__main__":
    main()
