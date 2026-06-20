"""
Module 3 — AI Detection Engine
================================
Trains, evaluates, and persists multiple ML classifiers on the
feature matrix produced by Module 2.

Models:
    - Logistic Regression  (fast baseline, great with SHAP)
    - Random Forest        (robust, handles non-linearity)
    - XGBoost              (best raw performance)

Usage:
    engine = DetectionEngine()
    engine.train(X_train, y_train)
    results = engine.evaluate(X_test, y_test)
    proba = engine.predict_proba(X_new)
    engine.save("models/")
    engine.load("models/")
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

import xgboost as xgb


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list
    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0

    def summary(self) -> str:
        tn, fp = self.confusion[0][0], self.confusion[0][1]
        fn, tp = self.confusion[1][0], self.confusion[1][1]
        return (
            f"\n{'─'*52}\n"
            f"  Model     : {self.model_name}\n"
            f"  Accuracy  : {self.accuracy:.4f}\n"
            f"  Precision : {self.precision:.4f}\n"
            f"  Recall    : {self.recall:.4f}\n"
            f"  F1        : {self.f1:.4f}\n"
            f"  ROC-AUC   : {self.roc_auc:.4f}\n"
            f"  CV F1     : {self.cv_f1_mean:.4f} ± {self.cv_f1_std:.4f}\n"
            f"  Confusion :\n"
            f"              Predicted\n"
            f"              Legit  Phish\n"
            f"  Actual Legit  {tn:>5}  {fp:>5}\n"
            f"  Actual Phish  {fn:>5}  {tp:>5}\n"
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "cv_f1_mean": round(self.cv_f1_mean, 4),
            "cv_f1_std": round(self.cv_f1_std, 4),
            "confusion": self.confusion,
        }


class DetectionEngine:
    """Multi-model phishing detection engine."""

    MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]

    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models: dict = {}
        self.feature_names: list = []
        self.eval_results: dict = {}
        self._best_model: str = ""
        self._cv_cache: dict = {}

    def _build_pipelines(self, neg_pos_ratio: float) -> dict:
        scale_pos = max(neg_pos_ratio, 1.0)
        return {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0, max_iter=1000, class_weight="balanced",
                    solver="lbfgs", random_state=self.random_state, n_jobs=-1,
                )),
            ]),
            "random_forest": Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=self.random_state, n_jobs=-1,
                )),
            ]),
            "xgboost": Pipeline([
                ("clf", xgb.XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos,
                    eval_metric="logloss", random_state=self.random_state,
                    n_jobs=-1, verbosity=0,
                )),
            ]),
        }

    def train(self, X_train, y_train, models=None, verbose=True):
        if models is None:
            models = self.MODEL_NAMES

        X = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
        y = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
        self.feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else []

        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        ratio = n_neg / max(n_pos, 1)

        if verbose:
            logger.info(f"Training on {len(y)} samples | {n_pos} phishing | {n_neg} legit")

        pipelines = self._build_pipelines(ratio)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for name in models:
            if name not in pipelines:
                continue
            pipe = pipelines[name]
            if verbose:
                logger.info(f"Training {name}...")

            cv_scores = cross_validate(
                pipe, X, y, cv=cv,
                scoring={"f1": "f1", "roc_auc": "roc_auc"},
                n_jobs=-1,
            )
            pipe.fit(X, y)
            self.models[name] = pipe
            self._cv_cache[name] = {
                "f1_mean": float(cv_scores["test_f1"].mean()),
                "f1_std":  float(cv_scores["test_f1"].std()),
            }

            if verbose:
                logger.info(
                    f"  {name} | CV F1: "
                    f"{cv_scores['test_f1'].mean():.4f} +/- {cv_scores['test_f1'].std():.4f}"
                )

    def evaluate(self, X_test, y_test, verbose=True):
        if not self.models:
            raise RuntimeError("No models trained. Call .train() first.")

        X = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
        y = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)

        best_f1 = -1.0
        results = {}

        for name, pipe in self.models.items():
            y_pred  = pipe.predict(X)
            y_proba = pipe.predict_proba(X)[:, 1]
            cm = confusion_matrix(y, y_pred).tolist()
            cv = self._cv_cache.get(name, {})

            result = EvalResult(
                model_name=name,
                accuracy=float(accuracy_score(y, y_pred)),
                precision=float(precision_score(y, y_pred, zero_division=0)),
                recall=float(recall_score(y, y_pred, zero_division=0)),
                f1=float(f1_score(y, y_pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, y_proba)),
                confusion=cm,
                cv_f1_mean=cv.get("f1_mean", 0.0),
                cv_f1_std=cv.get("f1_std", 0.0),
            )
            results[name] = result
            self.eval_results[name] = result

            if result.f1 > best_f1:
                best_f1 = result.f1
                self._best_model = name

            if verbose:
                print(result.summary())

        if verbose:
            logger.info(f"Best model: {self._best_model} (F1={best_f1:.4f})")

        return results

    def predict_proba(self, X, model=None):
        if not self.models:
            raise RuntimeError("No models trained.")
        if isinstance(X, pd.DataFrame):
            arr = X.values
        elif isinstance(X, dict):
            arr = np.array([list(X.values())])
        else:
            arr = np.atleast_2d(X)

        names = [model] if model else list(self.models.keys())
        results = {}
        for name in names:
            proba = self.models[name].predict_proba(arr)[:, 1]
            results[name] = float(proba[0]) if len(proba) == 1 else proba.tolist()
        return results

    def predict(self, X, model=None):
        return {name: int(p >= 0.5) for name, p in self.predict_proba(X, model).items()}

    @property
    def best_model(self):
        return self._best_model or (list(self.models.keys())[-1] if self.models else "")

    def best_predict_proba(self, X):
        return self.predict_proba(X, model=self.best_model)[self.best_model]

    def save(self, directory="models"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, pipe in self.models.items():
            joblib.dump(pipe, directory / f"{name}.joblib")
            logger.info(f"Saved {name}")
        meta = {
            "feature_names": self.feature_names,
            "best_model": self._best_model,
            "eval_results": {k: v.to_dict() for k, v in self.eval_results.items()},
        }
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved metadata to {directory}/metadata.json")

    def load(self, directory="models"):
        directory = Path(directory)
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_names = meta.get("feature_names", [])
            self._best_model = meta.get("best_model", "")
            for name, d in meta.get("eval_results", {}).items():
                self.eval_results[name] = EvalResult(
                    model_name=d["model"], accuracy=d["accuracy"],
                    precision=d["precision"], recall=d["recall"],
                    f1=d["f1"], roc_auc=d["roc_auc"], confusion=d["confusion"],
                    cv_f1_mean=d["cv_f1_mean"], cv_f1_std=d["cv_f1_std"],
                )
        for path in directory.glob("*.joblib"):
            self.models[path.stem] = joblib.load(path)
            logger.info(f"Loaded {path.stem}")

    def feature_importance(self, model=None):
        name = model or self.best_model
        if name not in self.models:
            return None
        clf = self.models[name].named_steps["clf"]
        names = self.feature_names or [f"f{i}" for i in range(200)]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            return None
        return pd.Series(imp, index=names[:len(imp)]).sort_values(ascending=False)
