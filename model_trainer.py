"""
ModelTrainer: Class OOP untuk melatih, membandingkan, dan tuning model ML.

Workflow class ini:
1. compare_models()    → 5-fold CV semua model dengan default params
2. tune_best_model()   → GridSearchCV pada model terbaik dari step 1
3. evaluate_on_test()  → classification report di held-out test set
4. save_artifacts()    → simpan model + label encoder ke .pkl

Cara pakai:
    trainer = ModelTrainer(random_state=42)
    trainer.compare_models(X_train, y_train)
    trainer.tune_best_model(X_train, y_train)
    trainer.evaluate_on_test(X_test, y_test)
    trainer.save_artifacts('model.pkl', preprocessor, label_encoder)
"""

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from evaluator import Evaluator


class ModelTrainer:
    """
    Orchestrator untuk training pipeline.

    Encapsulates:
    - Daftar kandidat model
    - Hyperparameter grid untuk tuning
    - State (hasil CV, model terbaik, model ter-tuning)
    - Methods untuk evaluasi & penyimpanan
    """

    def __init__(self, random_state=42, cv_folds=5, scoring='f1_macro',
                 use_mlflow=False, experiment_name='credit_score_classification',
                 tracking_uri='sqlite:///mlflow.db'):
        """
        Args:
            random_state: untuk reproducibility
            cv_folds: jumlah fold cross-validation
            scoring: metric utama untuk seleksi (macro-F1 karena imbalanced)
            use_mlflow: jika True, catat semua eksperimen ke MLflow
            experiment_name: nama experiment di MLflow
            tracking_uri: backend MLflow. Pakai SQLite (bukan file store)
                          karena MLflow 3.x men-deprecate file store './mlruns'.
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if self.use_mlflow:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

        # Daftar model kandidat dengan default params (tapi dengan random_state)
        self.models = {
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=random_state
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, random_state=random_state,
                eval_metric='mlogloss', n_jobs=-1, verbosity=0
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100, random_state=random_state,
                n_jobs=-1, verbosity=-1
            )
        }

        # Grid hyperparameter untuk tuning (per model)
        # Sengaja dibatasi 2-3 parameter saja supaya tidak terlalu lama
        self.param_grids = {
            'LogisticRegression': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            },
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10],           # dibatasi 10 untuk model ringan (~15MB, deployable)
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'num_leaves': [31, 63],
                'learning_rate': [0.05, 0.1]
            }
        }

        # State (diisi setelah training)
        self.cv_results_ = None
        self.best_model_name_ = None
        self.tuned_model_ = None
        self.best_params_ = None
        self.test_report_ = None

    def compare_models(self, X, y):
        """
        Step 1: 5-fold CV semua model dengan default params.
        Return DataFrame perbandingan, simpan nama model terbaik.
        """
        print(f"Membandingkan {len(self.models)} model via {self.cv_folds}-fold CV...")
        print(f"Scoring: {self.scoring}\n")

        results = []
        for name, model in self.models.items():
            print(f"  Training {name}...", end=' ', flush=True)
            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds, scoring=self.scoring, n_jobs=-1
            )
            results.append({
                'model': name,
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'scores': scores.tolist()
            })
            print(f"mean={scores.mean():.4f} ± {scores.std():.4f}")

            # Catat tiap model sebagai 1 run MLflow (eksperimen perbandingan)
            if self.use_mlflow:
                with mlflow.start_run(run_name=f"compare_{name}"):
                    mlflow.set_tag("stage", "model_comparison")
                    mlflow.log_param("model_type", name)
                    mlflow.log_param("cv_folds", self.cv_folds)
                    mlflow.log_param("scoring", self.scoring)
                    # Log parameter default model
                    for pname, pval in model.get_params().items():
                        mlflow.log_param(f"param_{pname}", pval)
                    mlflow.log_metric("cv_macro_f1_mean", scores.mean())
                    mlflow.log_metric("cv_macro_f1_std", scores.std())

        self.cv_results_ = (
            pd.DataFrame(results)
            .sort_values('cv_mean', ascending=False)
            .reset_index(drop=True)
        )
        self.best_model_name_ = self.cv_results_.iloc[0]['model']

        print(f"\n>>> Model terbaik (default params): {self.best_model_name_}")
        print(f"    CV {self.scoring}: {self.cv_results_.iloc[0]['cv_mean']:.4f}")

        return self.cv_results_

    def tune_best_model(self, X, y):
        """
        Step 2: GridSearchCV pada model terbaik dari step 1.
        Tuning hanya 1 model supaya efisien.
        """
        if self.best_model_name_ is None:
            raise ValueError("Panggil compare_models() dulu sebelum tuning.")

        print(f"\nTuning hyperparameter untuk {self.best_model_name_}...")
        base_model = self.models[self.best_model_name_]
        param_grid = self.param_grids[self.best_model_name_]

        n_combinations = 1
        for v in param_grid.values():
            n_combinations *= len(v)
        print(f"Total kombinasi: {n_combinations} × {self.cv_folds} folds "
              f"= {n_combinations * self.cv_folds} fits")

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X, y)

        self.tuned_model_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        print(f"\n>>> Best params: {self.best_params_}")
        print(f"    Best CV score: {grid_search.best_score_:.4f}")

        # Bandingkan dengan default. Catatan: kalau param grid sengaja membatasi
        # kompleksitas model (misal max_depth) demi ukuran/deployment, skor bisa
        # sedikit TURUN. Ini trade-off yang disengaja, bukan kegagalan tuning.
        default_score = self.cv_results_.iloc[0]['cv_mean']
        diff = grid_search.best_score_ - default_score
        print(f"    CV score default (unconstrained): {default_score:.4f}")
        print(f"    Selisih: {diff:+.4f}")
        if diff < 0:
            print(f"    (Negatif karena model dibatasi untuk deployment ringan "
                  f"— trade-off akurasi vs ukuran)")

        return self.tuned_model_

    def evaluate_on_test(self, X_test, y_test, label_names=None):
        """
        Step 3: Evaluasi final di test set yang belum pernah dilihat.
        Memakai class Evaluator (terpisah) untuk menghitung metrik.
        Jika MLflow aktif: catat best params + test metrics + model artifact
        sebagai 1 run final.
        """
        if self.tuned_model_ is None:
            raise ValueError("Panggil tune_best_model() dulu sebelum evaluasi.")

        # Gunakan class Evaluator untuk menghitung metrik
        evaluator = Evaluator(label_names=label_names)
        metrics = evaluator.evaluate(self.tuned_model_, X_test, y_test)
        y_pred = self.tuned_model_.predict(X_test)
        report = evaluator.text_report(y_test, y_pred)

        print(f"\n{'='*50}")
        print(f"EVALUASI FINAL DI TEST SET")
        print(f"{'='*50}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}  |  Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nClassification Report:")
        print(report)

        self.test_report_ = report
        self.test_metrics_ = metrics

        # Catat run final ke MLflow: best params + test metrics + model artifact
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"final_{self.best_model_name_}"):
                mlflow.set_tag("stage", "final_model")
                mlflow.log_param("model_type", self.best_model_name_)
                if self.best_params_:
                    for k, v in self.best_params_.items():
                        mlflow.log_param(f"best_{k}", v)
                for mname, mval in metrics.items():
                    mlflow.log_metric(f"test_{mname}", mval)
                # Simpan model terbaik sebagai artifact MLflow
                mlflow.sklearn.log_model(self.tuned_model_, name="best_model")
                print(">>> Model + metrics dicatat ke MLflow (run: final)")

        return report

    def save_artifacts(self, path, preprocessor, label_encoder, cleaner=None,
                       raw_columns=None):
        """
        Step 4: Simpan semua artifak yang dibutuhkan predictor sebagai 1 file .pkl.

        Bundle berisi:
        - tuned_model: model ML terbaik yang sudah ter-tuning
        - preprocessor: Preprocessor yang sudah di-fit
        - label_encoder: untuk decode prediksi balik ke string (Good/Standard/Poor)
        - cleaner: DataCleaner (stateless, tapi disimpan untuk konsistensi)
        - raw_columns: daftar kolom input mentah (supaya predictor bisa bangun
                       DataFrame lengkap meski user hanya isi sebagian fitur)
        """
        if self.tuned_model_ is None:
            raise ValueError("Tidak ada model yang ter-train. Jalankan training dulu.")

        bundle = {
            'model': self.tuned_model_,
            'model_name': self.best_model_name_,
            'best_params': self.best_params_,
            'preprocessor': preprocessor,
            'label_encoder': label_encoder,
            'cleaner': cleaner,
            'raw_columns': raw_columns,
            'cv_results': self.cv_results_,
        }
        joblib.dump(bundle, path)
        print(f"\n>>> Bundle disimpan ke: {path}")
        print(f"    Isi bundle: {list(bundle.keys())}")
