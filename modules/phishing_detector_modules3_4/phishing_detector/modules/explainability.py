"""
Module 5 — Explainable AI Engine
===================================
Uses SHAP to explain individual predictions and global feature importance.
This is the centerpiece of the project — it transforms raw model output
into human-readable reasons.

Outputs:
    - Per-prediction explanation with top contributing features
    - Signed contributions (what pushed score UP vs DOWN)
    - Global feature importance chart data
    - Natural-language explanation text for the dashboard
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class FeatureContribution:
    feature: str
    display_name: str
    value: float            # actual feature value
    shap_value: float       # SHAP contribution (signed)
    direction: str          # "increases_risk" | "decreases_risk" | "neutral"
    magnitude: str          # "high" | "medium" | "low"


@dataclass
class Explanation:
    prediction_label: str           # "Phishing" or "Legitimate"
    risk_score: int
    confidence: float
    contributions: list[FeatureContribution]
    top_risk_factors: list[FeatureContribution]
    top_safety_factors: list[FeatureContribution]
    natural_language: str
    base_value: float               # SHAP expected value


# ── Display name mapping ───────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    "urgency_score":              "Urgency language",
    "fear_score":                 "Fear-inducing language",
    "credential_request_score":   "Credential request",
    "financial_request_score":    "Financial language",
    "sender_brand_impersonation": "Brand impersonation",
    "reply_to_domain_mismatch":   "Reply-To domain mismatch",
    "missing_message_id":         "Missing Message-ID header",
    "sender_free_email":          "Free email provider",
    "generic_greeting":           "Generic salutation",
    "url_suspicious_tld":         "Suspicious domain extension",
    "url_brand_mismatch":         "Brand name in URL",
    "url_has_ip":                 "IP-based URL",
    "url_shortener_count":        "URL shortener used",
    "url_at_symbol":              "@ symbol in URL",
    "url_https_ratio":            "HTTPS usage ratio",
    "url_count":                  "Number of URLs",
    "url_avg_length":             "Average URL length",
    "subject_all_caps_ratio":     "Capitalisation in subject",
    "subject_exclamation":        "Exclamation marks in subject",
    "has_executable":             "Executable attachment",
    "has_macro_doc":              "Macro-enabled document",
    "attachment_count":           "Number of attachments",
    "entropy":                    "Text entropy (obfuscation)",
    "type_token_ratio":           "Vocabulary diversity",
    "stopword_ratio":             "Natural language ratio",
}


# ── Explainer ─────────────────────────────────────────────────────────────────

class ExplainabilityEngine:
    """
    Wraps a trained DetectionEngine and provides SHAP-based explanations.

    Usage:
        explainer = ExplainabilityEngine(detection_engine)
        explainer.fit(X_background)          # build SHAP background
        explanation = explainer.explain(features_dict)
    """

    def __init__(self, detection_engine):
        if not SHAP_AVAILABLE:
            raise ImportError("Install SHAP: pip install shap")
        self.engine = detection_engine
        self._explainer = None
        self._background: Optional[pd.DataFrame] = None

    # ── Setup ──────────────────────────────────────────────────────────────

    def fit(self, X_background: pd.DataFrame, n_background: int = 100) -> None:
        """
        Initialise the SHAP TreeExplainer on the XGBoost model.

        Args:
            X_background: representative sample of training data
            n_background: number of background samples for SHAP (100–200 is sufficient)
        """
        xgb_clf = self.engine._xgb_pipeline.named_steps["clf"]

        # Use a stratified sample for the background
        sample = X_background.sample(
            n=min(n_background, len(X_background)),
            random_state=42
        )
        self._background = sample[self.engine.feature_names_]
        self._explainer = shap.TreeExplainer(
            xgb_clf,
            data=self._background,
            feature_perturbation="interventional"
        )
        logger.info(f"SHAP explainer fitted on {len(self._background)} background samples")

    # ── Explain single ─────────────────────────────────────────────────────

    def explain(self, features: dict, risk_score: int, confidence: float) -> Explanation:
        """
        Generate a full explanation for a single prediction.

        Args:
            features: feature dict from FeatureExtractor
            risk_score: 0–100 integer score from DetectionEngine
            confidence: model probability (0.0–1.0)

        Returns:
            Explanation with per-feature SHAP contributions
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit(X_background) before explaining.")

        X = pd.DataFrame(
            [{k: features.get(k, 0.0) for k in self.engine.feature_names_}]
        )[self.engine.feature_names_]

        shap_values = self._explainer.shap_values(X)

        # shap_values shape: (1, n_features) for binary XGBoost
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # class 1 (phishing)
        else:
            sv = shap_values[0]

        base_value = float(self._explainer.expected_value
                           if not isinstance(self._explainer.expected_value, (list, np.ndarray))
                           else self._explainer.expected_value[1])

        contributions = self._build_contributions(X.iloc[0], sv)

        risk_factors   = [c for c in contributions if c.direction == "increases_risk"]
        safety_factors = [c for c in contributions if c.direction == "decreases_risk"]

        label = "Phishing" if risk_score > 30 else "Legitimate"
        nl = self._natural_language(label, risk_score, risk_factors[:5])

        return Explanation(
            prediction_label=label,
            risk_score=risk_score,
            confidence=confidence,
            contributions=contributions,
            top_risk_factors=risk_factors[:5],
            top_safety_factors=safety_factors[:3],
            natural_language=nl,
            base_value=base_value
        )

    # ── Global importance ──────────────────────────────────────────────────

    def global_importance(self, X: pd.DataFrame, max_evals: int = 500) -> pd.DataFrame:
        """
        Compute mean |SHAP| across a dataset for global feature ranking.

        Returns DataFrame with columns: feature, display_name, mean_shap
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit(X_background) first.")

        sample = X.sample(n=min(max_evals, len(X)), random_state=42)
        Xs = sample[self.engine.feature_names_]
        sv = self._explainer.shap_values(Xs)

        if isinstance(sv, list):
            sv = sv[1]

        mean_abs = np.abs(sv).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.engine.feature_names_,
            "display_name": [
                FEATURE_DISPLAY_NAMES.get(f, f) for f in self.engine.feature_names_
            ],
            "mean_shap": mean_abs
        }).sort_values("mean_shap", ascending=False).reset_index(drop=True)

        return df

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_contributions(
        self, row: pd.Series, shap_vals: np.ndarray
    ) -> list[FeatureContribution]:
        contributions = []
        for feat, sv in zip(self.engine.feature_names_, shap_vals):
            val = float(row[feat])
            sv_f = float(sv)
            direction = (
                "increases_risk"  if sv_f >  0.01 else
                "decreases_risk"  if sv_f < -0.01 else
                "neutral"
            )
            abs_sv = abs(sv_f)
            magnitude = (
                "high"   if abs_sv > 0.3 else
                "medium" if abs_sv > 0.1 else
                "low"
            )
            contributions.append(FeatureContribution(
                feature=feat,
                display_name=FEATURE_DISPLAY_NAMES.get(feat, feat),
                value=round(val, 4),
                shap_value=round(sv_f, 4),
                direction=direction,
                magnitude=magnitude
            ))

        return sorted(contributions, key=lambda c: abs(c.shap_value), reverse=True)

    @staticmethod
    def _natural_language(
        label: str, score: int, risk_factors: list[FeatureContribution]
    ) -> str:
        if label == "Legitimate":
            return (
                f"This email appears legitimate (risk score {score}/100). "
                "No significant phishing indicators were found."
            )

        if not risk_factors:
            return f"Phishing detected (risk score {score}/100) based on overall email patterns."

        # Build explanation from top factors
        lines = []
        for fc in risk_factors[:3]:
            name = fc.display_name.lower()
            if fc.value > 0:
                lines.append(f"{name} was detected")
            else:
                lines.append(name)

        reasons = "; ".join(lines)
        severity = (
            "very high" if score >= 80 else
            "high"      if score >= 60 else
            "moderate"
        )
        return (
            f"Phishing risk is {severity} (score {score}/100). "
            f"Key indicators: {reasons}. "
            f"Do not click any links or provide personal information."
        )


# ── Fallback explainer (no SHAP) ──────────────────────────────────────────────

class RuleBasedExplainer:
    """
    Lightweight fallback explainer using the ThreatScorer factor breakdown.
    Used when SHAP is unavailable or for quick inference.
    """

    def explain_from_report(self, report) -> str:
        """Generate natural language explanation from a ThreatReport."""
        if not report.is_phishing:
            return (
                f"This email appears safe (risk score: {report.risk_score}/100). "
                "No significant phishing indicators were found."
            )

        lines = []
        for f in report.triggered_factors[:5]:
            lines.append(f"• {f.label}")

        reasons = "\n".join(lines) if lines else "• Multiple suspicious patterns"

        return (
            f"⚠ Phishing detected — Risk Score: {report.risk_score}/100\n\n"
            f"Contributing factors:\n{reasons}\n\n"
            f"Recommendation: Do not interact with this email."
        )
