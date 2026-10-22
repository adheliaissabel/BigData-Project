# Customer Churn Analysis — Big Data Project

## 📌 Deskripsi Proyek

Proyek ini berfokus pada pengolahan dan analisis dataset *customer churn* untuk mengidentifikasi pola yang berkaitan dengan pelanggan yang berhenti menggunakan layanan. Proses dilakukan melalui tahapan data ingestion, preprocessing, penyimpanan data, exploratory data analysis (EDA), audit konsistensi dataset, feature engineering, serta pembangunan dan evaluasi model machine learning.

Proyek ini juga mengevaluasi konsistensi antara dataset training dan testing untuk memahami pengaruh kualitas serta distribusi data terhadap performa model prediksi churn.

## 🎯 Tujuan Proyek

1. Mengolah dan membersihkan data pelanggan agar siap dianalisis.
2. Menyimpan data hasil preprocessing dalam format Parquet.
3. Mengeksplorasi karakteristik pelanggan dan pola yang berkaitan dengan churn.
4. Memeriksa kualitas, distribusi, dan konsistensi data training dan testing.
5. Membuat fitur tambahan berdasarkan hasil eksplorasi data.
6. Membandingkan beberapa algoritma machine learning dan mengevaluasi performanya pada data testing.

## 📂 Sumber Dataset

Dataset yang digunakan adalah **Customer Churn Dataset** dari Kaggle.

- Sumber: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
- Data training: 440.832 baris setelah pembersihan baris kosong.
- Data testing: 64.374 baris.
- Target: `Churn`, yang menunjukkan status churn pelanggan.

Atribut yang digunakan mencakup `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, `Last Interaction`, dan `Churn`.

## 🛠️ Teknologi yang Digunakan

- **Python** — pemrosesan dan analisis data.
- **Pandas** — manipulasi, pembersihan, dan eksplorasi data.
- **NumPy** — operasi numerik.
- **Parquet** — penyimpanan data terstruktur yang telah diproses.
- **Scikit-learn** — feature engineering, pelatihan, dan evaluasi model machine learning.
- **Joblib** — penyimpanan model terlatih.
- **Matplotlib/Seaborn** — visualisasi data, apabila digunakan dalam skrip EDA.

## ⚙️ Alur Pengolahan Data

1. **Data Ingestion dan Cleaning**  
   Membaca dataset mentah, memeriksa target, menangani baris kosong, dan menyimpan hasil preprocessing ke format Parquet.

2. **Exploratory Data Analysis (EDA)**  
   Menganalisis distribusi churn, statistik numerik, hubungan atribut dengan churn, serta perbedaan karakteristik training dan testing.

3. **Data Consistency Audit**  
   Memeriksa jumlah baris, nilai duplikat, distribusi target, hubungan fitur dengan churn, dan pola churn berdasarkan rentang `CustomerID`.

4. **Feature Engineering**  
   Membuat fitur tambahan berdasarkan jumlah panggilan dukungan pelanggan dan keterlambatan pembayaran.

5. **Model Training dan Comparison**  
   Membandingkan Random Forest, Gradient Boosting, dan HistGradientBoosting menggunakan data training dan validation.

6. **Testing Evaluation**  
   Mengevaluasi model terpilih pada dataset testing yang terpisah dan membandingkan hasil model baseline dengan model yang menggunakan fitur tambahan.

## 🧪 Feature Engineering

Tiga fitur tambahan dibuat berdasarkan hasil eksplorasi data:

- `high_support_calls_6plus`: menandai pelanggan dengan jumlah panggilan dukungan minimal enam kali.
- `high_payment_delay_21plus`: menandai pelanggan dengan keterlambatan pembayaran minimal 21 hari.
- `support_delay_combined_risk`: menandai pelanggan yang memenuhi kedua kondisi tersebut.

Fitur-fitur ini merupakan kandidat fitur berbasis ambang batas yang ditemukan pada EDA. Karena hubungan atribut dengan churn berbeda antara training dan testing, fitur tersebut belum dapat dianggap sebagai indikator risiko yang konsisten pada semua data.

## 🤖 Pemodelan

Tiga algoritma dibandingkan:

- Random Forest
- Gradient Boosting
- HistGradientBoosting

Evaluasi validation menggunakan pembagian data training dengan rasio 80:20 dan stratifikasi target. Pemilihan model didasarkan pada F1-score validation.

Model dengan hasil validation terbaik adalah **HistGradientBoosting** dengan fitur hasil feature engineering. Model terpilih disimpan menggunakan Joblib untuk digunakan pada tahap evaluasi testing.

Atribut `CustomerID` tidak digunakan sebagai fitur input model.

## 📊 Hasil Evaluasi

### Validation

Model HistGradientBoosting menghasilkan metrik berikut pada data validation:

| Metrik | Hasil |
|---|---:|
| Accuracy | 99,97% |
| Precision | 99,99% |
| Recall | 99,96% |
| F1-score | 99,97% |
| ROC-AUC | 1,0000 |
| PR-AUC | 1,0000 |

### Testing

Ketika dievaluasi pada dataset testing, hasilnya menurun secara signifikan:

| Metrik | Hasil |
|---|---:|
| Accuracy | 50,33% |
| Precision | 48,81% |
| Recall | 99,88% |
| F1-score | 65,58% |
| ROC-AUC | 0,6989 |
| PR-AUC | 0,5897 |

Model memiliki recall yang tinggi untuk kelas churn, tetapi juga menghasilkan banyak false positive. Dari 33.881 pelanggan non-churn, sebanyak 31.938 diprediksi sebagai churn.

Perbedaan hasil validation dan testing menunjukkan bahwa performa validation yang tinggi belum mencerminkan kemampuan generalisasi model pada dataset testing.

### Perbandingan Baseline dan Feature Engineering

Pengujian HistGradientBoosting baseline dan versi dengan fitur tambahan menghasilkan metrik yang sama pada dataset testing. Dengan demikian, feature engineering yang diterapkan belum menunjukkan peningkatan performa terukur pada pengujian ini.

## 🔍 Temuan Audit Data

Audit terhadap file CSV mentah dan data hasil preprocessing menemukan pola hubungan `CustomerID` dan `Churn` yang berbeda antara training dan testing.

| Pemeriksaan | Training | Testing |
|---|---:|---:|
| Jumlah baris | 440.832 | 64.374 |
| Tingkat churn | 56,71% | 47,37% |
| ID pelanggan unik | 440.832 | 64.374 |
| Korelasi ID dengan churn | -0,8394 | +0,5298 |
| Baris duplikat | 0 | 0 |

Pada training, beberapa kelompok ID dengan rentang lebih rendah memiliki tingkat churn mendekati 100%, sedangkan kelompok ID dengan rentang lebih tinggi memiliki tingkat churn 0%. Pada testing, pola yang ditemukan berbeda: kelompok ID awal memiliki tingkat churn sekitar 25–26%, sementara kelompok ID akhir mencapai sekitar 93–100%.

Pola tersebut sudah ditemukan pada CSV mentah dan tetap terlihat setelah data disimpan dalam format Parquet. Oleh karena itu, anomali ini tidak disebabkan oleh proses konversi ke Parquet.

Temuan tersebut menunjukkan adanya ketidakkonsistenan pola label berdasarkan rentang ID dan menjadi indikasi bahwa konstruksi dataset atau distribusi data training dan testing perlu diperiksa lebih lanjut. Temuan ini tidak dengan sendirinya membuktikan bahwa `CustomerID` menyebabkan churn atau bahwa terjadi data leakage secara langsung.

## 📁 Struktur Proyek

```text
BigData-Project/
├── data/
│   ├── raw/
│   │   ├── customer_churn_dataset-training-master.csv
│   │   └── customer_churn_dataset-testing-master.csv
│   └── processed/
│       ├── training.parquet
│       ├── testing.parquet
│       ├── training_engineered.parquet
│       └── testing_engineered.parquet
├── outputs/
│   ├── best_churn_model.joblib
│   ├── best_model_metadata.json
│   ├── dataset_consistency_summary.csv
│   ├── feature_target_consistency.csv
│   ├── training_id_target_consistency.csv
│   └── testing_id_target_consistency.csv
├── src/
│   ├── 01_ingest_clean_store.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 03_data_audit.py
│   ├── 04_order_audit.py
│   ├── 04_train_compare.py
│   ├── 05_evaluate_testing.py
│   ├── 06_compare_testing.py
│   └── 07_dataset_consistency_audit.py
└── README.md
```

*Catatan: struktur di atas menggambarkan file utama yang digunakan dalam pipeline. Sesuaikan jika terdapat file atau output tambahan pada repository.*

## ▶️ Cara Menjalankan Proyek

### 1. Clone repository

```bash
git clone https://github.com/adheliaissabel/BigData-Project.git
cd BigData-Project
```

### 2. Instal dependensi

Pastikan Python sudah terpasang, kemudian instal pustaka yang dibutuhkan:

```bash
pip install pandas numpy pyarrow scikit-learn joblib matplotlib seaborn
```

### 3. Jalankan pipeline

Jalankan skrip dari direktori utama proyek secara berurutan:

```bash
python src/01_ingest_clean_store.py
python src/02_eda.py
python src/03_feature_engineering.py
python src/03_data_audit.py
python src/04_order_audit.py
python src/04_train_compare.py
python src/05_evaluate_testing.py
python src/06_compare_testing.py
python src/07_dataset_consistency_audit.py
```

Pastikan dataset mentah tersedia di folder `data/raw/` sebelum menjalankan pipeline.

**Catatan:** `04_train_compare.py` dapat membutuhkan waktu cukup lama karena melatih dan membandingkan beberapa model. Jika menjalankan ulang skrip, model dan hasil evaluasi akan dihitung kembali.

## 📝 Kesimpulan

Proyek ini berhasil membangun alur pengolahan data customer churn mulai dari ingestion dan preprocessing hingga EDA, audit konsistensi data, feature engineering, serta evaluasi machine learning.

Temuan utama proyek adalah perbedaan pola hubungan `CustomerID` dan `Churn` antara training dan testing, disertai penurunan performa yang signifikan pada data testing. Hasil ini menunjukkan bahwa evaluasi model tidak cukup hanya mengandalkan validation, tetapi perlu mempertimbangkan konsistensi dan karakteristik dataset yang digunakan.

Pemeriksaan lebih lanjut terhadap sumber dan proses pembentukan dataset diperlukan sebelum model dapat direkomendasikan untuk penggunaan prediksi churn pada kondisi nyata.

## 👥 Kontributor

Proyek ini dikerjakan sebagai bagian dari tugas mata kuliah Big Data.

Repository: https://github.com/adheliaissabel/BigData-Project
