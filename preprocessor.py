"""
Preprocessor: Custom sklearn transformer untuk encoding & scaling fitur.

Class ini menangani semua transformasi setelah DataCleaner:
- Imputasi missing values (median untuk numerik, modus untuk kategorik)
- Scaling numerik (StandardScaler)
- Encoding kategorik:
    * Nominal → OneHotEncoder
    * Ordinal (Credit_Mix) → OrdinalEncoder
- Kolom binary (has_*) langsung passthrough (sudah di-encode di DataCleaner)

Output: numpy array siap dipakai oleh model ML.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Bungkus ColumnTransformer dalam class kustom untuk modularitas.

    Kenapa pakai ColumnTransformer di dalam?
    - Bisa apply transformasi berbeda untuk kolom berbeda (numerik vs kategorik)
    - Imputer + scaler/encoder jadi 1 mini-pipeline → tidak ada data leakage
    - Lebih clean dibanding loop manual
    """

    # Kolom numerik (akan di-impute median + di-scale)
    NUMERIC_COLS = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
        'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio',
        'Credit_History_Age', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance', 'Num_Loan_Types'
    ]

    # Kategorik nominal (akan di-OneHot encode)
    NOMINAL_COLS = ['Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour']

    # Kategorik ordinal (ada urutan: Bad < Standard < Good)
    ORDINAL_COLS = ['Credit_Mix']
    ORDINAL_ORDER = [['Bad', 'Standard', 'Good']]  # urutan kategori per kolom

    # Kolom binary yang sudah di-encode di DataCleaner (passthrough saja)
    BINARY_COLS = [
        'has_Auto_Loan', 'has_Credit_Builder_Loan', 'has_Debt_Consolidation_Loan',
        'has_Home_Equity_Loan', 'has_Mortgage_Loan', 'has_Not_Specified',
        'has_Payday_Loan', 'has_Personal_Loan', 'has_Student_Loan'
    ]

    def __init__(self):
        # ColumnTransformer dibuat di __init__ supaya bisa diakses dari luar
        # (misal untuk debugging / inspeksi feature names)
        self.column_transformer = self._build_column_transformer()
        self.feature_names_ = None

    def _build_column_transformer(self):
        """Bangun ColumnTransformer dengan 4 pipeline berbeda."""

        # Pipeline 1: numerik → impute median + standardize
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline 2: nominal kategorik → impute modus + one-hot
        nominal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Pipeline 3: ordinal kategorik → impute modus + ordinal encode
        # handle_unknown='use_encoded_value' supaya tidak error kalau ada kategori baru
        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                categories=self.ORDINAL_ORDER,
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])

        # Gabungkan semua via ColumnTransformer
        return ColumnTransformer(
            transformers=[
                ('numeric', numeric_pipeline, self.NUMERIC_COLS),
                ('nominal', nominal_pipeline, self.NOMINAL_COLS),
                ('ordinal', ordinal_pipeline, self.ORDINAL_COLS),
                ('binary', 'passthrough', self.BINARY_COLS),
            ],
            remainder='drop',  # kolom yang tidak disebut dibuang
            verbose_feature_names_out=False
        )

    def fit(self, X, y=None):
        """Fit semua imputer, scaler, dan encoder pada data training."""
        self.column_transformer.fit(X)
        self.feature_names_ = self.column_transformer.get_feature_names_out().tolist()
        return self

    def transform(self, X):
        """Transform X menggunakan transformer yang sudah di-fit."""
        return self.column_transformer.transform(X)

    def get_feature_names_out(self):
        """Return nama-nama fitur setelah transformasi (berguna untuk inspeksi)."""
        if self.feature_names_ is None:
            raise ValueError("Preprocessor belum di-fit. Panggil .fit(X) dulu.")
        return self.feature_names_
