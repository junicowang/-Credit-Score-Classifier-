"""
evaluator.py — Class OOP untuk evaluasi model klasifikasi.

Cara pakai:
    evaluator = Evaluator(label_names=['Good', 'Poor', 'Standard'])
    metrics = evaluator.evaluate(model, X_test, y_test)
    print(evaluator.text_report(y_test, y_pred))
"""

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)


class Evaluator:
    """
    Menghitung metrik evaluasi untuk model klasifikasi multi-kelas.

    Dipisah dari ModelTrainer agar tanggung jawab jelas:
    - ModelTrainer  -> melatih model
    - Evaluator     -> menilai model
    """

    def __init__(self, label_names=None):
        """
        Args:
            label_names: list nama kelas (mis. ['Good', 'Poor', 'Standard'])
                         untuk laporan yang mudah dibaca.
        """
        self.label_names = label_names
        self.last_metrics_ = None

    def evaluate(self, model, X_test, y_test):
        """
        Hitung metrik utama untuk model pada test set.

        Returns dict berisi:
            macro_f1, weighted_f1, accuracy, macro_precision, macro_recall,
            dan f1 per kelas (mis. 'f1_Good', 'f1_Poor', 'f1_Standard').
        """
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
            'macro_precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        }

        # F1 per kelas
        per_class_f1 = f1_score(y_test, y_pred, average=None)
        for i, score in enumerate(per_class_f1):
            cls_name = self.label_names[i] if self.label_names is not None else str(i)
            metrics[f'f1_{cls_name}'] = float(score)

        self.last_metrics_ = metrics
        return metrics

    def text_report(self, y_test, y_pred):
        """Classification report dalam bentuk teks (precision/recall/f1 per kelas)."""
        return classification_report(
            y_test, y_pred,
            target_names=self.label_names if self.label_names is not None else None
        )

    def confusion_matrix(self, y_test, y_pred):
        """Return confusion matrix sebagai numpy array."""
        return confusion_matrix(y_test, y_pred)
