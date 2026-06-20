"""
Tests for Module 3 (DetectionEngine), Module 4 (ThreatScorer),
Module 5 (Explainability), and full pipeline integration.

Run with: pytest tests/test_ml_pipeline.py -v
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from modules.detection_engine import DetectionEngine, PredictionResult
from modules.threat_scorer import ThreatScorer, ThreatReport
from pipeline import PhishingDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """Generate a synthetic binary dataset that mimics phishing feature space."""
    X_arr, y_arr = make_classification(
        n_samples=600,
        n_features=43,
        n_informative=20,
        n_redundant=10,
        weights=[0.6, 0.4],    # 60% legit, 40% phishing
        random_state=42
    )
    feature_names = [f"feat_{i:02d}" for i in range(43)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    # Clip to non-negative (features are counts/ratios)
    X = X.clip(lower=0)
    y = pd.Series(y_arr)
    return X, y


@pytest.fixture(scope="module")
def trained_engine(synthetic_data):
    X, y = synthetic_data
    engine = DetectionEngine(model_dir="/tmp/test_models")
    engine.train(X, y, eval_split=0.2, cv_folds=3)
    return engine


@pytest.fixture(scope="module")
def sample_features(synthetic_data):
    X, _ = synthetic_data
    return dict(X.iloc[0])


# ── Module 3: DetectionEngine ─────────────────────────────────────────────────

class TestDetectionEngine:
    def test_trains_without_error(self, trained_engine):
        assert trained_engine.is_trained

    def test_metrics_returned(self, synthetic_data):
        X, y = synthetic_data
        engine = DetectionEngine(model_dir="/tmp/test_models2")
        metrics = engine.train(X, y, eval_split=0.2, cv_folds=3)
        assert "ensemble" in metrics
        assert "logistic_regression" in metrics
        assert "xgboost" in metrics
        assert "random_forest" in metrics

    def test_auc_above_random(self, trained_engine, synthetic_data):
        X, y = synthetic_data
        # Use held-out 20%
        from sklearn.model_selection import train_test_split
        _, X_eval, _, y_eval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        probs = trained_engine._ensemble.predict_proba(X_eval)[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_eval, probs)
        assert auc > 0.6, f"AUC {auc:.3f} below 0.6 on synthetic data"

    def test_predict_single_returns_result(self, trained_engine, sample_features):
        result = trained_engine.predict_single(sample_features)
        assert isinstance(result, PredictionResult)

    def test_predict_single_score_in_range(self, trained_engine, sample_features):
        result = trained_engine.predict_single(sample_features)
        assert 0 <= result.risk_score <= 100

    def test_predict_single_confidence_in_range(self, trained_engine, sample_features):
        result = trained_engine.predict_single(sample_features)
        assert 0.0 <= result.confidence <= 1.0

    def test_model_votes_all_present(self, trained_engine, sample_features):
        result = trained_engine.predict_single(sample_features)
        for key in ("logistic_regression", "random_forest", "xgboost", "ensemble"):
            assert key in result.model_votes

    def test_predict_batch_shape(self, trained_engine, synthetic_data):
        X, _ = synthetic_data
        batch = trained_engine.predict_batch(X.head(50))
        assert len(batch) == 50
        assert "risk_score" in batch.columns
        assert "threat_level" in batch.columns

    def test_threat_level_categories(self, trained_engine, synthetic_data):
        X, _ = synthetic_data
        batch = trained_engine.predict_batch(X.head(100))
        valid = {"Safe", "Suspicious", "High Risk"}
        assert set(batch["threat_level"].unique()).issubset(valid)

    def test_save_and_load(self, trained_engine, sample_features, tmp_path):
        trained_engine.save(tmp_path)
        loaded = DetectionEngine(model_dir=tmp_path)
        loaded.load(tmp_path)
        r1 = trained_engine.predict_single(sample_features)
        r2 = loaded.predict_single(sample_features)
        assert abs(r1.confidence - r2.confidence) < 1e-5

    def test_untrained_raises(self, sample_features):
        engine = DetectionEngine(model_dir="/tmp/unused")
        with pytest.raises(RuntimeError):
            engine.predict_single(sample_features)

    def test_feature_importances(self, trained_engine):
        fi = trained_engine.feature_importances(top_n=10)
        assert len(fi) == 10
        assert "feature" in fi.columns
        assert "importance" in fi.columns


# ── Module 4: ThreatScorer ────────────────────────────────────────────────────

class TestThreatScorer:
    def _make_result(self, score: int, features: dict = None) -> PredictionResult:
        if features is None:
            features = {}
        return PredictionResult(
            is_phishing=score > 30,
            confidence=score / 100,
            risk_score=score,
            threat_level=DetectionEngine._risk_level(score),
            model_votes={"ensemble": score / 100},
            feature_vector=features
        )

    def test_safe_email(self):
        result = self._make_result(15)
        report = ThreatScorer().score(result)
        assert report.badge_colour == "green"
        assert report.threat_level == "Safe"

    def test_suspicious_email(self):
        result = self._make_result(50)
        report = ThreatScorer().score(result)
        assert report.badge_colour == "amber"

    def test_high_risk_email(self):
        features = {
            "sender_brand_impersonation": 1,
            "reply_to_domain_mismatch": 1,
            "url_suspicious_tld": 1,
            "urgency_score": 0.8,
            "credential_request_score": 0.9,
        }
        result = self._make_result(92, features)
        report = ThreatScorer().score(result)
        assert report.badge_colour == "red"
        assert len(report.triggered_factors) >= 4

    def test_triggered_factors_sorted(self):
        features = {
            "urgency_score": 0.5,
            "fear_score": 0.9,
            "sender_brand_impersonation": 1,
        }
        result = self._make_result(75, features)
        report = ThreatScorer().score(result)
        contribs = [f.contribution for f in report.triggered_factors]
        assert contribs == sorted(contribs, reverse=True)

    def test_report_has_summary(self):
        result = self._make_result(85, {"urgency_score": 0.7})
        report = ThreatScorer().score(result)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 10

    def test_all_factors_present(self):
        result = self._make_result(50, {})
        report = ThreatScorer().score(result)
        from modules.threat_scorer import SCORING_FACTORS
        assert len(report.factors) == len(SCORING_FACTORS)


# ── Full pipeline integration ─────────────────────────────────────────────────

class TestPipelineIntegration:
    PHISHING = {
        "sender": "PayPal Security <security@paypa1-secure.com>",
        "reply_to": "noreply@evil-domain.ru",
        "subject": "URGENT: Your account is SUSPENDED",
        "body": (
            "Dear valued customer, your PayPal account has been suspended due to "
            "unauthorized access. Verify your credentials immediately or face permanent "
            "termination. Click: http://paypal-secure.xyz/verify?id=99"
        )
    }
    LEGIT = {
        "sender": "GitHub <noreply@github.com>",
        "subject": "Your pull request was merged",
        "body": "Your PR #42 was merged into main. View it at https://github.com/org/repo/pull/42"
    }

    @pytest.fixture(scope="class")
    def trained_detector(self, synthetic_data):
        X, y = synthetic_data
        df = X.copy()
        df["label"] = y
        df["sender"] = ""
        df["subject"] = ""
        df["body_text"] = "sample text " * 10
        # Train on synthetic data for speed — real performance requires real dataset
        detector = PhishingDetector()
        detector.engine.train(X, y, eval_split=0.2, cv_folds=3)
        return detector

    def test_analyse_returns_all_fields(self, trained_detector):
        from pipeline import AnalysisResult
        result = trained_detector.analyse(self.LEGIT)
        assert isinstance(result, AnalysisResult)
        assert result.features is not None
        assert result.prediction is not None
        assert result.threat_report is not None
        assert result.explanation is not None

    def test_risk_score_integer(self, trained_detector):
        result = trained_detector.analyse(self.PHISHING)
        assert isinstance(result.threat_report.risk_score, int)
        assert 0 <= result.threat_report.risk_score <= 100

    def test_feature_extraction_not_empty(self, trained_detector):
        result = trained_detector.analyse(self.PHISHING)
        assert len(result.features) > 0

    def test_model_votes_four_keys(self, trained_detector):
        result = trained_detector.analyse(self.PHISHING)
        assert len(result.prediction.model_votes) == 4
