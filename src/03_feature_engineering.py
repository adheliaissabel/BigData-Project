"""Create testable derived features; do not change the original target labels."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

def add_features(df):
    df = df.copy()
    # Candidate thresholds based on the team's EDA discussion.
    # Treat these as hypotheses and compare against a baseline; do not use them to relabel churn.
    df["high_support_calls_6plus"] = (df["support_calls"] >= 6).astype("int8")
    df["high_payment_delay_21plus"] = (df["payment_delay"] >= 21).astype("int8")
    df["support_delay_combined_risk"] = (
        (df["support_calls"] >= 6) & (df["payment_delay"] >= 21)
    ).astype("int8")
    return df

def main():
    for split in ["training", "testing"]:
        path = DATA / f"{split}.parquet"
        if not path.exists():
            raise SystemExit("Jalankan dulu: python src/01_ingest_clean_store.py")
        df = pd.read_parquet(path)
        engineered = add_features(df)
        out = DATA / f"{split}_engineered.parquet"
        engineered.to_parquet(out, index=False)
        print(f"{split}: {len(engineered):,} baris -> {out}")
    print("Fitur tambahan dibuat. Label churn asli tidak diubah.")

if __name__ == "__main__":
    main()
