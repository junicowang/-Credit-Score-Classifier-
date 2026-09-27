# Credit Score Classification — Final Project Model Deployment

Klasifikasi `Credit_Score` (Good / Standard / Poor) untuk nasabah institusi keuangan menggunakan dataset A (nasabah laki-laki, NIM ganjil).

---

## 📁 Struktur File

```
.
├── README.md
├── requirements.txt
│
├── exploration.ipynb       # EDA + perbandingan model (tidak berbasis OOP)
│
├── training.py             # Re-training pipeline + MLflow (entry point)
├── app.py                  # Streamlit deployment (entry point)
│
├── data_cleaner.py         # OOP: custom transformer untuk cleaning
├── preprocessor.py         # OOP: PREPROCESSING (ColumnTransformer wrapper)
├── model_trainer.py        # OOP: TRAINING (CV, GridSearch) + MLflow logging
├── evaluator.py            # OOP: EVALUATION (metrik, report, confusion matrix)
├── predictor.py            # OOP: inferencing wrapper untuk Streamlit
│
├── model.pkl               # Bundle: model + preprocessor + cleaner + label_encoder
├── mlflow.db               # Database MLflow (riwayat eksperimen)
│
└── screenshots/            # Test case screenshots (untuk PDF)
```

---

## 🚀 Cara Menjalankan

### 1. Instal dependensi
```bash
pip install -r requirements.txt
```

### 2. (Opsional) Re-training dari awal
```bash
# Ubah DATA_PATH di training.py ke lokasi CSV Anda terlebih dahulu
python training.py
```
Ini menghasilkan `model.pkl` baru dan mencatat eksperimen ke MLflow (`mlflow.db`).

### 3. Lihat hasil eksperimen MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Buka `http://localhost:5000` untuk membandingkan semua run (4 model + final) dalam tabel.

### 4. Jalankan Streamlit
```bash
streamlit run app.py
```
Buka `http://localhost:8501` di browser.

---

## 🧱 Arsitektur OOP

Pipeline dirancang sebagai sklearn custom transformers yang reusable:

```
Data Mentah → DataCleaner.transform()      [class custom transformer]
            → Preprocessor.transform()     [wrap ColumnTransformer]
            → Model.predict()              [RandomForestClassifier]
```

Semua komponen di-pickle dalam **satu file bundle** (`model.pkl`), sehingga
`CreditScorePredictor` cukup `joblib.load()` saja untuk inferencing.

| Class | File | Tanggung jawab |
|---|---|---|
| `DataCleaner` | `data_cleaner.py` | Strip karakter `_`, parse `Credit_History_Age`, multi-label encode `Type_of_Loan`, ganti placeholder & outlier mustahil jadi NaN |
| `Preprocessor` | `preprocessor.py` | **(Preprocessing)** Imputasi median/modus, scaling numerik, OneHot/Ordinal encoding kategorik |
| `ModelTrainer` | `model_trainer.py` | **(Training)** 5-fold CV bandingkan 4 model, GridSearchCV tune yang terbaik, save artifacts, logging MLflow |
| `Evaluator` | `evaluator.py` | **(Evaluation)** Hitung macro-F1, accuracy, precision/recall, F1 per kelas, classification report, confusion matrix |
| `CreditScorePredictor` | `predictor.py` | Bangun row lengkap dari input parsial Streamlit, jalankan full pipeline, return prediksi + probabilitas |

> Memenuhi requirement soal: minimal class mencakup **preprocessing** (`Preprocessor`),
> **training** (`ModelTrainer`), dan **evaluation** (`Evaluator`).

---

## 📊 Performa Model

- **Algoritma terpilih:** RandomForest (mengungguli LogisticRegression, XGBoost, LightGBM dalam 5-fold CV)
- **Best hyperparameters:** `n_estimators=200`, `max_depth=10`, `min_samples_split=2`
- **Test set macro-F1:** ~0.69 (Good 0.61 | Poor 0.71 | Standard 0.76)
- **Ukuran model:** 15 MB (dikompresi dari ~140 MB dengan membatasi `max_depth=10` — trade-off akurasi vs deployability)

---

## 📈 Experiment Tracking (MLflow)

Pipeline training (`training.py`) mencatat setiap eksperimen ke MLflow (backend SQLite `mlflow.db`):

- **4 run perbandingan** (`compare_LogisticRegression`, `compare_RandomForest`, `compare_XGBoost`, `compare_LightGBM`) — masing-masing mencatat parameter model + CV macro-F1.
- **1 run final** (`final_RandomForest`) — mencatat best hyperparameters, metrik test set (macro-F1, accuracy, F1 per kelas), dan **model artifact** (model terbaik tersimpan di MLflow).

Lihat dengan: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

> Catatan: MLflow 3.x men-deprecate file store (`./mlruns`), sehingga pipeline ini
> memakai backend SQLite (`sqlite:///mlflow.db`) yang merupakan pendekatan modern.

---

## 🔑 Keputusan Desain Penting

1. **`max_depth=10` (bukan unconstrained):** Mengurangi ukuran model 90% (140 MB → 15 MB) dengan penurunan akurasi minimal (~1%). Lebih praktis untuk deployment di Streamlit Cloud / AWS.
2. **Macro-F1 sebagai metrik utama:** Data imbalanced (Standard 53%, Good 17%) — accuracy bisa menyesatkan. Macro-F1 memperlakukan semua kelas setara.
3. **OrdinalEncoder untuk `Credit_Mix`:** Ada urutan natural (Bad < Standard < Good) — OneHot akan menghilangkan informasi ini.
4. **Imputasi di dalam pipeline:** Mencegah data leakage — median dihitung hanya dari training set tiap fold.
5. **`handle_unknown='ignore'` di OneHot:** Streamlit user bisa input kategori yang belum ada di training tanpa membuat app crash.

---

## 🧪 Test Cases (Lihat `screenshots/`)

| Profil | Hasil Prediksi |
|---|---|
| Profil Bagus (income tinggi, debt rendah, delay minim) | ✅ Good |
| Profil Sedang (semua nilai medium) | ✅ Standard |
| Profil Buruk (income rendah, debt tinggi, banyak delay) | ✅ Poor |

Klik tombol 🟢 / 🟡 / 🔴 di app untuk auto-fill profil contoh.
