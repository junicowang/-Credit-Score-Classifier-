"""
training.py — Script utama untuk training pipeline Credit Score Classification.

Workflow:
1. Load data mentah (.csv)
2. Cleaning via DataCleaner
3. Split train/test (80/20, stratified)
4. Fit Preprocessor di training, transform train+test
5. Bandingkan 4 model via 5-fold CV
6. Tune model terbaik via GridSearchCV
7. Evaluasi di test set
8. Save artifacts ke model.pkl

Jalankan: python training.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_cleaner import DataCleaner
from preprocessor import Preprocessor
from model_trainer import ModelTrainer
import os

# ===== Konfigurasi =====
DATA_PATH = os.environ.get('DATA_PATH', 'data/data_A.csv')
MODEL_PATH = 'model.pkl'
TARGET_COL = 'Credit_Score'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    print("="*60)
    print("CREDIT SCORE CLASSIFICATION — TRAINING PIPELINE")
    print("="*60)

    # ----- 1. Load data -----
    print("\n[1] Loading data...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"    Shape: {df_raw.shape}")

    # ----- 2. Cleaning -----
    print("\n[2] Cleaning data...")
    raw_input_columns = [c for c in df_raw.columns if c != TARGET_COL]
    cleaner = DataCleaner()
    df_clean = cleaner.fit_transform(df_raw)
    print(f"    Shape setelah cleaning: {df_clean.shape}")

    # ----- 3. Pisahkan fitur dan target, encode target -----
    print("\n[3] Memisahkan fitur dan target...")
    X = df_clean.drop(columns=[TARGET_COL])
    y_raw = df_clean[TARGET_COL]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"    Target mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # ----- 4. Train/test split (stratified untuk pertahankan proporsi kelas) -----
    print("\n[4] Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

    # ----- 5. Fit Preprocessor di train, transform keduanya -----
    print("\n[5] Preprocessing fitur...")
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"    Train processed: {X_train_processed.shape}")
    print(f"    Test  processed: {X_test_processed.shape}")

    # ----- 6. Bandingkan 4 model (dengan MLflow logging) -----
    print("\n[6] Membandingkan model...")
    trainer = ModelTrainer(random_state=RANDOM_STATE, use_mlflow=True,
                           experiment_name='credit_score_classification')
    cv_results = trainer.compare_models(X_train_processed, y_train)
    print(f"\nHasil CV (sorted by score):")
    print(cv_results[['model', 'cv_mean', 'cv_std']].to_string(index=False))

    # ----- 7. Tune model terbaik -----
    print("\n[7] Tuning hyperparameter...")
    trainer.tune_best_model(X_train_processed, y_train)

    # ----- 8. Evaluasi di test set -----
    print("\n[8] Evaluasi di test set...")
    trainer.evaluate_on_test(
        X_test_processed, y_test,
        label_names=label_encoder.classes_
    )

    # ----- 9. Save artifacts -----
    print("\n[9] Menyimpan artifacts...")
    trainer.save_artifacts(
        path=MODEL_PATH,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        cleaner=cleaner,
        raw_columns=raw_input_columns
    )

    print(f"\n{'='*60}")
    print("TRAINING SELESAI ✓")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
