from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ModelTrainer:
    """CMCS 회귀 및 위험 등급 분류 모델 비교 파이프라인."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_model = None
        self.best_model_name: str | None = None
        self.results: dict[str, dict[str, dict[str, Any]]] = {}

    @staticmethod
    def _imports():
        try:
            import joblib
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
            )
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                mean_absolute_error,
                mean_squared_error,
                precision_score,
                r2_score,
                recall_score,
                roc_auc_score,
            )
            from sklearn.model_selection import (
                KFold,
                StratifiedKFold,
                cross_validate,
                train_test_split,
            )
        except ImportError as exc:
            raise RuntimeError(
                "모델 학습에는 scikit-learn과 joblib이 필요합니다."
            ) from exc
        return locals()

    def _regression_models(self) -> dict[str, object]:
        modules = self._imports()
        models: dict[str, object] = {
            "RandomForest": modules["RandomForestRegressor"](
                n_estimators=160,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "GradientBoosting": modules["GradientBoostingRegressor"](
                n_estimators=160,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state,
            ),
        }
        try:
            from xgboost import XGBRegressor
        except ImportError:
            return models
        models["XGBoost"] = XGBRegressor(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.05,
            random_state=self.random_state,
            verbosity=0,
        )
        return models

    def _classification_models(self) -> dict[str, object]:
        modules = self._imports()
        models: dict[str, object] = {
            "RandomForest": modules["RandomForestClassifier"](
                n_estimators=160,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "GradientBoosting": modules["GradientBoostingClassifier"](
                n_estimators=160,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state,
            ),
        }
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return models
        models["XGBoost"] = XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.05,
            random_state=self.random_state,
            verbosity=0,
            eval_metric="mlogloss",
        )
        return models

    def train_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        save_path: str | Path = "models/best_model.pkl",
    ) -> dict[str, dict[str, Any]]:
        modules = self._imports()
        X_train, X_test, y_train, y_test = modules["train_test_split"](
            X, y, test_size=test_size, random_state=self.random_state
        )
        results: dict[str, dict[str, Any]] = {}
        for name, model in self._regression_models().items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            results[name] = {
                "MAE": float(modules["mean_absolute_error"](y_test, prediction)),
                "RMSE": float(
                    np.sqrt(modules["mean_squared_error"](y_test, prediction))
                ),
                "R2": float(modules["r2_score"](y_test, prediction)),
                "model": model,
            }

        best_name = max(results, key=lambda name: results[name]["R2"])
        self.best_model_name = best_name
        self.best_model = results[best_name]["model"]
        self.results["regression"] = results
        self._save_model(save_path)
        return results

    def train_classification(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        save_path: str | Path = "models/best_classifier.pkl",
    ) -> dict[str, dict[str, Any]]:
        modules = self._imports()
        y = pd.Series(y).astype(int)
        class_counts = y.value_counts()
        stratify = y if len(class_counts) > 1 and class_counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = modules["train_test_split"](
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        results: dict[str, dict[str, Any]] = {}
        for name, model in self._classification_models().items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            metrics = {
                "Accuracy": float(modules["accuracy_score"](y_test, prediction)),
                "F1_macro": float(
                    modules["f1_score"](
                        y_test, prediction, average="macro", zero_division=0
                    )
                ),
                "Precision_macro": float(
                    modules["precision_score"](
                        y_test, prediction, average="macro", zero_division=0
                    )
                ),
                "Recall_macro": float(
                    modules["recall_score"](
                        y_test, prediction, average="macro", zero_division=0
                    )
                ),
                "model": model,
            }
            if hasattr(model, "predict_proba") and y_test.nunique() > 1:
                try:
                    probability = model.predict_proba(X_test)
                    metrics["AUC_ROC_OvR"] = float(
                        modules["roc_auc_score"](
                            y_test, probability, multi_class="ovr"
                        )
                    )
                except ValueError:
                    metrics["AUC_ROC_OvR"] = None
            results[name] = metrics

        best_name = max(results, key=lambda name: results[name]["F1_macro"])
        self.best_model_name = best_name
        self.best_model = results[best_name]["model"]
        self.results["classification"] = results
        self._save_model(save_path)
        return results

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = "classification",
        folds: int = 5,
    ) -> dict[str, dict[str, Any]]:
        modules = self._imports()
        if task == "classification":
            model = self._classification_models()["RandomForest"]
            class_counts = pd.Series(y).value_counts()
            folds = min(folds, int(class_counts.min()))
            if folds < 2:
                raise ValueError("교차 검증을 위한 클래스별 표본이 부족합니다.")
            cv = modules["StratifiedKFold"](
                n_splits=folds, shuffle=True, random_state=self.random_state
            )
            scoring = ["accuracy", "f1_macro"]
        elif task == "regression":
            model = self._regression_models()["RandomForest"]
            folds = min(folds, len(X))
            if folds < 2:
                raise ValueError("교차 검증 표본이 부족합니다.")
            cv = modules["KFold"](
                n_splits=folds, shuffle=True, random_state=self.random_state
            )
            scoring = ["neg_mean_absolute_error", "r2"]
        else:
            raise ValueError("task는 classification 또는 regression이어야 합니다.")

        raw = modules["cross_validate"](model, X, y, cv=cv, scoring=scoring)
        return {
            metric: {
                "mean": float(np.mean(raw[f"test_{metric}"])),
                "std": float(np.std(raw[f"test_{metric}"])),
                "folds": raw[f"test_{metric}"].astype(float).tolist(),
            }
            for metric in scoring
        }

    def _save_model(self, save_path: str | Path) -> None:
        modules = self._imports()
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        modules["joblib"].dump(self.best_model, path)

    def explain_with_shap(
        self,
        X: pd.DataFrame,
        save_path: str | Path = "outputs/charts/shap_importance.png",
    ) -> Path:
        if self.best_model is None:
            raise ValueError("모델을 먼저 학습해야 합니다.")
        try:
            import matplotlib.pyplot as plt
            import shap
        except ImportError as exc:
            raise RuntimeError("SHAP 분석에는 shap과 matplotlib이 필요합니다.") from exc

        sample = X.sample(min(len(X), 1000), random_state=self.random_state)
        explainer = shap.TreeExplainer(self.best_model)
        values = explainer.shap_values(sample)
        shap.summary_plot(values, sample, plot_type="bar", show=False)
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def generate_report(
        self, save_path: str | Path = "outputs/reports/model_report.json"
    ) -> dict[str, Any]:
        report = {
            "best_model": self.best_model_name,
            "results": {
                task: {
                    name: {
                        key: round(value, 6) if isinstance(value, float) else value
                        for key, value in metrics.items()
                        if key != "model"
                    }
                    for name, metrics in task_results.items()
                }
                for task, task_results in self.results.items()
            },
        }
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

