# Big Data Analytics — Customer Churn

Repository coding untuk pipeline analisis churn pelanggan. Fokus proyek: **data ingestion → storage → cleaning → EDA → feature engineering → model comparison → optional Optuna tuning**. Machine learning adalah salah satu tahap dari pipeline, bukan keseluruhan proyek.

## Dataset input

Letakkan dua file CSV berikut di `data/raw/`:
- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

Data mentah tidak disertakan dalam repository. `customer_id` dipertahankan untuk audit/EDA, tetapi dikeluarkan dari fitur model karena merupakan identifier, bukan fitur perilaku pelanggan.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline — jalankan sesuai urutan

```powershell
python src/01_ingest_clean_store.py
python src/02_eda.py
python src/03_feature_engineering.py
python src/04_train_compare.py
```

Optuna bersifat opsional dan dijalankan setelah baseline/model comparison:

```powershell
python src/05_optuna_tune.py
```

## Output

Skrip menghasilkan:
- `data/processed/training.parquet`, `testing.parquet`
- `data/processed/training_engineered.parquet`, `testing_engineered.parquet`
- `outputs/data_quality_summary.csv`
- `outputs/churn_distribution.csv`
- `outputs/*_by_churn.csv` dan grafik EDA PNG
- `outputs/model_comparison_validation.csv`
- `outputs/model_confusion_matrices.json`
- Jika Optuna dijalankan: `outputs/optuna_trials.csv`, `outputs/optuna_best_result.json`

## Catatan analisis penting

- Periksa EDA sebelum memutuskan fitur yang dipakai. Fitur ambang pada skrip (`support_calls >= 6`, `payment_delay >= 21`) adalah kandidat untuk diuji, bukan aturan kebenaran universal.
- Jangan mengubah label `churn` berdasarkan ambang fitur karena itu akan membuat model meniru aturan yang dipakai membuat label.
- Bandingkan baseline dan engineered features pada split yang sama. Gunakan PR-AUC, F1, precision, recall, ROC-AUC, serta confusion matrix.
- Model comparison menggunakan stratified random split dari training untuk validasi. Testing asli belum otomatis digunakan untuk memilih model; gunakan testing hanya setelah pendekatan dipilih agar evaluasi final tidak bias.
- Hasil validasi tinggi tidak menjamin generalisasi. Dataset ini sebelumnya menunjukkan perbedaan pola antara training dan testing; interpretasikan hasil testing dengan hati-hati.
- Jangan memakai `CustomerID` sebagai fitur prediksi.
"# BigData-Project" 
