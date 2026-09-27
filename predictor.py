"""
predictor.py — Class OOP untuk inferencing (prediksi data baru).

CreditScorePredictor memuat bundle model.pkl dan menyediakan API sederhana:
    predictor = CreditScorePredictor('model.pkl')
    result = predictor.predict({'Age': 35, 'Annual_Income': 50000, ...})
    # result = {'prediction': 'Good', 'probabilities': {'Good': 0.7, ...}}

Predictor menerima input PARSIAL (user cukup isi sebagian fitur).
Fitur yang tidak diisi otomatis di-NaN dan diimputasi median/modus
oleh Preprocessor (yang sudah belajar dari data training).
"""

import joblib
import numpy as np
import pandas as pd
import os

class CreditScorePredictor:
    """
    Wrapper inferencing yang meng-encapsulate seluruh pipeline:
    raw input -> DataCleaner -> Preprocessor -> Model -> label.

    Semua komponen dimuat dari 1 file bundle (.pkl) hasil training.py.
    """

    def __init__(self, model_path=None):
        """Load bundle dan ekstrak semua komponen."""
        if model_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base, 'model.pkl')
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.model_name = bundle['model_name']
        self.preprocessor = bundle['preprocessor']
        self.label_encoder = bundle['label_encoder']
        self.cleaner = bundle['cleaner']
        self.raw_columns = bundle['raw_columns']
        self.class_names = list(self.label_encoder.classes_)

    def _build_full_row(self, user_input: dict) -> pd.DataFrame:
        """
        Bangun DataFrame 1 baris dengan SEMUA kolom asli.
        Kolom yang tidak diisi user -> NaN (akan diimputasi nanti).

        Penanganan khusus Credit_History_Age: user input angka tahun ->
        format ke string "X Years and 0 Months" supaya bisa di-parse DataCleaner.
        """
        row = {col: np.nan for col in self.raw_columns}

        for key, value in user_input.items():
            if key in row:
                row[key] = value

        if 'Credit_History_Age' in user_input:
            years = user_input['Credit_History_Age']
            if years is not None and not pd.isna(years):
                row['Credit_History_Age'] = f"{int(years)} Years and 0 Months"

        return pd.DataFrame([row])

    def predict(self, user_input: dict) -> dict:
        """
        Prediksi credit score dari input user (boleh parsial).

        Returns:
            {
                'prediction': 'Good',
                'probabilities': {'Good': 0.72, 'Poor': 0.10, 'Standard': 0.18},
                'model_name': 'RandomForest'
            }
        """
        df = self._build_full_row(user_input)              # 1. DataFrame lengkap
        df_clean = self.cleaner.transform(df)              # 2. Cleaning
        X = self.preprocessor.transform(df_clean)          # 3. Preprocessing

        pred_encoded = self.model.predict(X)[0]            # 4. Prediksi
        pred_proba = self.model.predict_proba(X)[0]

        prediction = self.label_encoder.inverse_transform([pred_encoded])[0]  # 5. Decode
        probabilities = {
            self.class_names[i]: float(pred_proba[i])
            for i in range(len(self.class_names))
        }

        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'model_name': self.model_name
        }


if __name__ == '__main__':
    predictor = CreditScorePredictor('model.pkl')

    sample = {
        'Age': 35, 'Annual_Income': 80000, 'Outstanding_Debt': 200,
        'Interest_Rate': 5, 'Delay_from_due_date': 2, 'Num_of_Delayed_Payment': 1,
        'Num_Credit_Card': 3, 'Num_Credit_Inquiries': 1, 'Changed_Credit_Limit': 5,
        'Monthly_Balance': 500, 'Credit_Mix': 'Good', 'Credit_History_Age': 20,
    }

    result = predictor.predict(sample)
    print(f"Model: {result['model_name']}")
    print(f"Prediksi: {result['prediction']}")
    print("Probabilitas:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.2%}")
