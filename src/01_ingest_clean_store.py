"""Read raw churn CSVs, standardize columns, clean empty rows, save Parquet."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "training": "customer_churn_dataset-training-master.csv",
    "testing": "customer_churn_dataset-testing-master.csv",
}
RENAME = {
    "customerid": "customer_id",
    "age": "age",
    "gender": "gender",
    "tenure": "tenure",
    "usage frequency": "usage_frequency",
    "support calls": "support_calls",
    "payment delay": "payment_delay",
    "subscription type": "subscription_type",
    "contract length": "contract_length",
    "total spend": "total_spend",
    "last interaction": "last_interaction",
    "churn": "churn",
}
NUMERIC = ["age", "tenure", "usage_frequency", "support_calls", "payment_delay",
           "total_spend", "last_interaction", "churn"]

def normalize_columns(df):
    df.columns = [str(c).strip().lower().replace("_", " ") for c in df.columns]
    return df.rename(columns=RENAME)

def load_one(name, filename):
    path = RAW / filename
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\nLetakkan CSV asli di folder data/raw/."
        )
    df = normalize_columns(pd.read_csv(path))
    before = len(df)
    df = df.dropna(how="all").copy()
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = {"churn", "age", "tenure", "usage_frequency", "support_calls",
                "payment_delay", "total_spend", "last_interaction",
                "gender", "subscription_type", "contract_length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: kolom tidak ditemukan setelah normalisasi: {sorted(missing)}")
    df["churn"] = df["churn"].astype("Int64")
    df["dataset_split"] = name
    print(f"{name}: {before:,} baris awal -> {len(df):,} setelah menghapus baris kosong")
    print("  missing target:", int(df["churn"].isna().sum()))
    print("  churn counts:", df["churn"].value_counts(dropna=False).to_dict())
    return df

def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, filename in FILES.items():
        df = load_one(name, filename)
        out = PROCESSED / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"Disimpan: {out}")
    print("\nSelesai. Data mentah tetap tersimpan di data/raw; hasil olahan di data/processed.")

if __name__ == "__main__":
    main()
