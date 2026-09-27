"""
DataCleaner: Custom sklearn transformer untuk membersihkan data Credit Score.

Class ini menangani semua masalah data mentah:
- Strip karakter '_' dari kolom numerik string
- Ganti placeholder ('_______', 'NM', '!@9#%8') jadi NaN
- Cap outlier mustahil ke NaN (Age > 100, Bank > 50, dst)
- Parse 'Credit_History_Age' dari "X Years and Y Months" ke total bulan
- Multi-label encode 'Type_of_Loan'
- Drop kolom non-prediktif (ID, Name, SSN, dll)

Dipakai sebagai langkah pertama di sklearn Pipeline:
    pipe = Pipeline([('cleaner', DataCleaner()), ('preprocessor', ...), ('model', ...)])
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer kustom untuk membersihkan dataset Credit Score yang kotor.

    Mewarisi BaseEstimator (untuk get_params/set_params) dan TransformerMixin
    (untuk fit_transform otomatis). Ini bikin classnya kompatibel dengan
    sklearn Pipeline, GridSearchCV, dll.
    """

    # Kolom yang akan dibuang (tidak punya nilai prediktif / PII)
    COLS_TO_DROP = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN', 'Month']

    # Kolom numerik yang disimpan sebagai string dengan karakter '_'
    NUMERIC_STR_COLS = [
        'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Outstanding_Debt',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]

    # Batas wajar untuk outlier mustahil (di atas batas → NaN → imputasi nanti)
    OUTLIER_BOUNDS = {
        'Age': (0, 100),
        'Num_Bank_Accounts': (0, 20),
        'Num_Credit_Card': (0, 20),
        'Interest_Rate': (0, 50),
        'Num_of_Loan': (0, 20),
        'Num_of_Delayed_Payment': (0, 50),
        'Num_Credit_Inquiries': (0, 50),
        'Annual_Income': (0, 1_000_000),
        'Monthly_Balance': (-10_000, 100_000),
    }

    # Placeholder yang sebenarnya artinya NaN
    PLACEHOLDERS = {
        'Occupation': ['_______'],
        'Credit_Mix': ['_'],
        'Payment_of_Min_Amount': ['NM'],
        'Payment_Behaviour': ['!@9#%8'],
    }

    # Jenis pinjaman untuk multi-label encoding Type_of_Loan
    LOAN_TYPES = [
        'Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan',
        'Home Equity Loan', 'Mortgage Loan', 'Not Specified',
        'Payday Loan', 'Personal Loan', 'Student Loan'
    ]

    def fit(self, X, y=None):
        """
        DataCleaner tidak belajar parameter apapun dari data training
        (cleaning rules-nya hardcoded). Tapi tetap implement fit() supaya
        kompatibel dengan sklearn Pipeline.
        """
        return self

    def transform(self, X):
        """Aplikasikan semua langkah cleaning ke DataFrame X."""
        df = X.copy()

        df = self._drop_useless_columns(df)
        df = self._clean_numeric_strings(df)
        df = self._replace_placeholders(df)
        df = self._cap_outliers(df)
        df = self._parse_credit_history_age(df)
        df = self._encode_type_of_loan(df)

        return df

    # ===== Method-method privat (helper internal) =====

    def _drop_useless_columns(self, df):
        """Buang kolom non-prediktif: ID, Name, SSN, dll."""
        cols = [c for c in self.COLS_TO_DROP if c in df.columns]
        return df.drop(columns=cols)

    def _clean_numeric_strings(self, df):
        """
        Strip karakter '_' dari kolom numerik string, lalu convert ke float.
        Contoh: '39_' → 39.0, '_' → NaN, '__-333...__' → -3.3e26 (kena cap nanti)
        """
        for col in self.NUMERIC_STR_COLS:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace('_', '', regex=False)
                    .replace('', np.nan)
                )
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _replace_placeholders(self, df):
        """Ganti placeholder ('_______', 'NM', dll) jadi NaN."""
        for col, bad_values in self.PLACEHOLDERS.items():
            if col in df.columns:
                df[col] = df[col].replace(bad_values, np.nan)
        return df

    def _cap_outliers(self, df):
        """
        Untuk nilai di luar batas wajar, ganti jadi NaN
        (akan diimputasi median di step selanjutnya).
        """
        for col, (lo, hi) in self.OUTLIER_BOUNDS.items():
            if col in df.columns:
                df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
        return df

    def _parse_credit_history_age(self, df):
        """
        Konversi 'Credit_History_Age' dari "20 Years and 5 Months" → 245 (bulan).
        Kalau NaN/format aneh → tetap NaN.
        """
        if 'Credit_History_Age' not in df.columns:
            return df

        def to_months(val):
            if pd.isna(val):
                return np.nan
            try:
                parts = str(val).split()
                years = int(parts[0])
                months = int(parts[3])
                return years * 12 + months
            except (ValueError, IndexError):
                return np.nan

        df['Credit_History_Age'] = df['Credit_History_Age'].apply(to_months)
        return df

    def _encode_type_of_loan(self, df):
        """
        Type_of_Loan berisi multi-loan dipisah koma:
            "Auto Loan, Personal Loan, Student Loan"
        Kita encode jadi 9 kolom binary: has_Auto_Loan, has_Personal_Loan, dll.
        Plus 1 kolom Num_Loan_Types (total jenis pinjaman).
        """
        if 'Type_of_Loan' not in df.columns:
            return df

        loans_str = df['Type_of_Loan'].fillna('').astype(str)

        for loan_type in self.LOAN_TYPES:
            col_name = 'has_' + loan_type.replace(' ', '_').replace('-', '_')
            df[col_name] = loans_str.str.contains(loan_type, regex=False).astype(int)

        df['Num_Loan_Types'] = (
            loans_str
            .str.split(',')
            .apply(lambda x: len([t for t in x if t.strip()]))
        )

        df = df.drop(columns=['Type_of_Loan'])
        return df
