"""
Integration test — Modules 3 & 4
Run with: python tests/test_engine.py
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from modules.detection_engine import DetectionEngine
from modules.threat_scorer import ThreatScorer, RiskLevel


def make_synthetic_data(n=600, seed=42):
    """
    Generate synthetic feature data for testing.
    Phishing emails (label=1) have higher values on threat features.
    """
    rng = np.random.default_rng(seed)
    n_phish = n // 2

    feature_cols = [
        "urgency_score","fear_score","credential_request_score","financial_request_score",
        "sender_brand_impersonation","reply_to_domain_mismatch","sender_free_email",
        "url_has_ip","url_brand_mismatch","url_suspicious_tld","url_shortener_count",
        "url_at_symbol","url_https_ratio","url_avg_length","url_count",
        "generic_greeting","subject_all_caps_ratio","subject_exclamation",
        "word_count","entropy","caps_ratio","type_token_ratio","stopword_ratio",
        "html_to_text_ratio","attachment_count","risky_attachment_count",
        "has_executable","has_macro_doc","missing_message_id","subject_contains_brand",
        "url_subdomain_depth_max","url_double_slash","subject_char_count",
        "exclamation_count","sentence_count","avg_word_length","url_numeric_ratio_avg",
        "unique_domain_count","subject_question_mark","subject_word_count",
        "sender_has_display_name","url_shortener_count","subject_has_re_fwd",
    ]

    # Phishing: high on threat features
    phish = {
        "urgency_score":              rng.beta(5, 2, n_phish),
        "fear_score":                 rng.beta(4, 2, n_phish),
        "credential_request_score":   rng.beta(5, 1, n_phish),
        "financial_request_score":    rng.beta(3, 2, n_phish),
        "sender_brand_impersonation": rng.binomial(1, 0.75, n_phish).astype(float),
        "reply_to_domain_mismatch":   rng.binomial(1, 0.70, n_phish).astype(float),
        "sender_free_email":          rng.binomial(1, 0.60, n_phish).astype(float),
        "url_has_ip":                 rng.binomial(1, 0.30, n_phish).astype(float),
        "url_brand_mismatch":         rng.binomial(1, 0.65, n_phish).astype(float),
        "url_suspicious_tld":         rng.binomial(1, 0.50, n_phish).astype(float),
        "url_shortener_count":        rng.poisson(1.5, n_phish).astype(float),
        "url_at_symbol":              rng.binomial(1, 0.20, n_phish).astype(float),
        "url_https_ratio":            rng.beta(2, 5, n_phish),
        "url_avg_length":             rng.normal(80, 20, n_phish),
        "url_count":                  rng.poisson(3, n_phish).astype(float),
        "generic_greeting":           rng.binomial(1, 0.80, n_phish).astype(float),
        "subject_all_caps_ratio":     rng.beta(4, 3, n_phish),
        "subject_exclamation":        rng.binomial(1, 0.70, n_phish).astype(float),
        "word_count":                 rng.normal(200, 80, n_phish),
        "entropy":                    rng.normal(4.2, 0.4, n_phish),
        "caps_ratio":                 rng.beta(3, 4, n_phish),
        "type_token_ratio":           rng.beta(3, 5, n_phish),
        "stopword_ratio":             rng.beta(2, 4, n_phish),
        "html_to_text_ratio":         rng.lognormal(1, 0.5, n_phish),
        "attachment_count":           rng.poisson(0.5, n_phish).astype(float),
        "risky_attachment_count":     rng.poisson(0.3, n_phish).astype(float),
        "has_executable":             rng.binomial(1, 0.10, n_phish).astype(float),
        "has_macro_doc":              rng.binomial(1, 0.15, n_phish).astype(float),
        "missing_message_id":         rng.binomial(1, 0.55, n_phish).astype(float),
        "subject_contains_brand":     rng.binomial(1, 0.65, n_phish).astype(float),
        "url_subdomain_depth_max":    rng.poisson(2, n_phish).astype(float),
        "url_double_slash":           rng.binomial(1, 0.25, n_phish).astype(float),
        "subject_char_count":         rng.normal(55, 15, n_phish),
        "exclamation_count":          rng.poisson(2, n_phish).astype(float),
        "sentence_count":             rng.poisson(8, n_phish).astype(float),
        "avg_word_length":            rng.normal(5, 1, n_phish),
        "url_numeric_ratio_avg":      rng.beta(3, 4, n_phish),
        "unique_domain_count":        rng.poisson(2, n_phish).astype(float),
        "subject_question_mark":      rng.binomial(1, 0.3, n_phish).astype(float),
        "subject_word_count":         rng.poisson(7, n_phish).astype(float),
        "sender_has_display_name":    rng.binomial(1, 0.85, n_phish).astype(float),
        "subject_has_re_fwd":         rng.binomial(1, 0.15, n_phish).astype(float),
    }

    # Legitimate: low on threat features
    legit = {
        "urgency_score":              rng.beta(1, 8, n_phish),
        "fear_score":                 rng.beta(1, 9, n_phish),
        "credential_request_score":   rng.beta(1, 8, n_phish),
        "financial_request_score":    rng.beta(1, 9, n_phish),
        "sender_brand_impersonation": rng.binomial(1, 0.02, n_phish).astype(float),
        "reply_to_domain_mismatch":   rng.binomial(1, 0.05, n_phish).astype(float),
        "sender_free_email":          rng.binomial(1, 0.40, n_phish).astype(float),
        "url_has_ip":                 rng.binomial(1, 0.01, n_phish).astype(float),
        "url_brand_mismatch":         rng.binomial(1, 0.03, n_phish).astype(float),
        "url_suspicious_tld":         rng.binomial(1, 0.02, n_phish).astype(float),
        "url_shortener_count":        rng.poisson(0.1, n_phish).astype(float),
        "url_at_symbol":              rng.binomial(1, 0.01, n_phish).astype(float),
        "url_https_ratio":            rng.beta(8, 2, n_phish),
        "url_avg_length":             rng.normal(50, 15, n_phish),
        "url_count":                  rng.poisson(1.5, n_phish).astype(float),
        "generic_greeting":           rng.binomial(1, 0.05, n_phish).astype(float),
        "subject_all_caps_ratio":     rng.beta(1, 8, n_phish),
        "subject_exclamation":        rng.binomial(1, 0.10, n_phish).astype(float),
        "word_count":                 rng.normal(150, 60, n_phish),
        "entropy":                    rng.normal(3.8, 0.3, n_phish),
        "caps_ratio":                 rng.beta(1, 6, n_phish),
        "type_token_ratio":           rng.beta(5, 3, n_phish),
        "stopword_ratio":             rng.beta(5, 3, n_phish),
        "html_to_text_ratio":         rng.lognormal(0.5, 0.3, n_phish),
        "attachment_count":           rng.poisson(0.2, n_phish).astype(float),
        "risky_attachment_count":     rng.poisson(0.02, n_phish).astype(float),
        "has_executable":             rng.binomial(1, 0.001, n_phish).astype(float),
        "has_macro_doc":              rng.binomial(1, 0.01, n_phish).astype(float),
        "missing_message_id":         rng.binomial(1, 0.05, n_phish).astype(float),
        "subject_contains_brand":     rng.binomial(1, 0.15, n_phish).astype(float),
        "url_subdomain_depth_max":    rng.poisson(0.5, n_phish).astype(float),
        "url_double_slash":           rng.binomial(1, 0.02, n_phish).astype(float),
        "subject_char_count":         rng.normal(45, 12, n_phish),
        "exclamation_count":          rng.poisson(0.2, n_phish).astype(float),
        "sentence_count":             rng.poisson(6, n_phish).astype(float),
        "avg_word_length":            rng.normal(5.5, 0.8, n_phish),
        "url_numeric_ratio_avg":      rng.beta(1, 6, n_phish),
        "unique_domain_count":        rng.poisson(1, n_phish).astype(float),
        "subject_question_mark":      rng.binomial(1, 0.1, n_phish).astype(float),
        "subject_word_count":         rng.poisson(5, n_phish).astype(float),
        "sender_has_display_name":    rng.binomial(1, 0.70, n_phish).astype(float),
        "subject_has_re_fwd":         rng.binomial(1, 0.30, n_phish).astype(float),
    }

    cols = list(phish.keys())
    phish_df = pd.DataFrame(phish)[cols].clip(lower=0)
    legit_df = pd.DataFrame(legit)[cols].clip(lower=0)

    X = pd.concat([phish_df, legit_df], ignore_index=True)
    y = pd.Series([1]*n_phish + [0]*n_phish)

    # Shuffle
    idx = rng.permutation(len(X))
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def test_engine_and_scorer():
    from sklearn.model_selection import train_test_split

    print("\n── Generating synthetic data ──────────────────────────")
    X, y = make_synthetic_data(n=800)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n── Training models ────────────────────────────────────")
    engine = DetectionEngine(random_state=42, cv_folds=3)
    engine.train(X_train, y_train, verbose=True)

    print("\n── Evaluating ─────────────────────────────────────────")
    results = engine.evaluate(X_test, y_test, verbose=True)

    # Assertions
    for name, result in results.items():
        assert result.f1 > 0.70, f"{name} F1={result.f1:.3f} below threshold"
        assert result.roc_auc > 0.80, f"{name} AUC={result.roc_auc:.3f} below threshold"
    print("\n  ✓ All models pass F1 > 0.70 and AUC > 0.80")

    print("\n── Testing threat scorer ──────────────────────────────")
    scorer = ThreatScorer()

    # Phishing sample
    phish_features = X_test[y_test == 1].iloc[0].to_dict()
    phish_probas   = engine.predict_proba(X_test[y_test==1].iloc[[0]])
    phish_report   = scorer.score(phish_probas, phish_features,
                                  email_subject="URGENT: Verify now!",
                                  email_sender="security@paypa1-secure.com")
    print(phish_report.summary())
    assert phish_report.is_phishing, "Should detect phishing sample"
    assert phish_report.risk_score >= 50

    # Legit sample
    legit_features = X_test[y_test == 0].iloc[0].to_dict()
    legit_probas   = engine.predict_proba(X_test[y_test==0].iloc[[0]])
    legit_report   = scorer.score(legit_probas, legit_features,
                                  email_subject="Your PR was merged",
                                  email_sender="noreply@github.com")
    print(legit_report.summary())
    assert not legit_report.is_phishing, "Should classify legit sample correctly"

    print("\n── Feature importance (XGBoost) ───────────────────────")
    imp = engine.feature_importance("xgboost")
    if imp is not None:
        print(imp.head(10).to_string())

    print("\n── Save / load round-trip ─────────────────────────────")
    engine.save("/tmp/phishing_models")
    engine2 = DetectionEngine()
    engine2.load("/tmp/phishing_models")
    p1 = engine.best_predict_proba(X_test.iloc[[0]])
    p2 = engine2.best_predict_proba(X_test.iloc[[0]])
    assert abs(p1 - p2) < 1e-6, "Save/load produced different predictions"
    print(f"  ✓ Save/load verified: prob={p1:.6f}")

    print("\n✅  All Module 3 & 4 tests passed.\n")


if __name__ == "__main__":
    test_engine_and_scorer()
