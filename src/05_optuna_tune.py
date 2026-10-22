"""Optional Optuna tuning for Random Forest on a stratified training validation split."""
from pathlib import Path
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

def main():
    path = DATA / "training_engineered.parquet"
    if not path.exists():
        raise SystemExit("Jalankan 03_feature_engineering.py terlebih dahulu.")
    df = pd.read_parquet(path)
    drop = [c for c in ["customer_id", "dataset_split", "churn"] if c in df.columns]
    X, y = df.drop(columns=drop), df["churn"].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cat = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    def objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 350, step=50),
            max_depth=trial.suggest_int("max_depth", 5, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        pipe = Pipeline([("preprocess", clone(prep)), ("model", model)])
        pipe.fit(X_train, y_train)
        return average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(study.trials_dataframe()).to_csv(OUT / "optuna_trials.csv", index=False)
    result = {"best_pr_auc": study.best_value, "best_params": study.best_params, "n_trials": len(study.trials)}
    pd.Series(result, dtype="object").to_json(OUT / "optuna_best_result.json", indent=2)
    print("Hasil terbaik:", result)

if __name__ == "__main__":
    main()
