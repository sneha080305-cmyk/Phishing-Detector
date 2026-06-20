"""
PhishingDetector — Pipeline Orchestrator
==========================================
Single entry point that runs Modules 1–5 end-to-end.

Usage — analyse an email:
    detector = PhishingDetector.load("models/")
    result = detector.analyse(email_dict)
    print(result.threat_report.summary)
    print(result.explanation.natural_language)

Usage — train from scratch:
    detector = PhishingDetector()
    detector.fit(df)              # df from dataset_loader
    detector.save("models/")
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from modules.email_parser import EmailParser, EmailDocument
from modules.feature_extractor import FeatureExtractor
from modules.detection_engine import DetectionEngine, PredictionResult
from modules.threat_scorer import ThreatScorer, ThreatReport
from modules.explainability import ExplainabilityEngine, RuleBasedExplainer, Explanation


@dataclass
class AnalysisResult:
    """Complete analysis output for a single email."""
    email_doc: EmailDocument
    features: dict
    prediction: PredictionResult
    threat_report: ThreatReport
    explanation: Explanation | str   # Explanation if SHAP available, else plain text


class PhishingDetector:
    """
    End-to-end phishing detection pipeline.

    Modules:
        1. EmailParser        — parse raw email into EmailDocument
        2. FeatureExtractor   — extract 43 features
        3. DetectionEngine    — LR + RF + XGBoost ensemble prediction
        4. ThreatScorer       — structured risk report with factor breakdown
        5. ExplainabilityEngine — SHAP-based per-feature explanation
    """

    def __init__(self):
        self.parser     = EmailParser()
        self.extractor  = FeatureExtractor()
        self.engine     = DetectionEngine()
        self.scorer     = ThreatScorer()
        self._explainer = None
        self._fallback_explainer = RuleBasedExplainer()

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, build_explainer: bool = True) -> dict:
        """
        Train the full pipeline on a labelled email DataFrame.

        Args:
            df: columns [sender, subject, body_text, label]
            build_explainer: whether to fit SHAP explainer after training

        Returns:
            training metrics dict
        """
        from modules.dataset_loader import build_feature_matrix

        logger.info("Building feature matrix...")
        X, y = build_feature_matrix(df, self.extractor)

        logger.info("Training detection engine...")
        metrics = self.engine.train(X, y)

        if build_explainer:
            try:
                self._explainer = ExplainabilityEngine(self.engine)
                self._explainer.fit(X)
                logger.info("SHAP explainer ready.")
            except Exception as e:
                logger.warning(f"Could not build SHAP explainer: {e}")
                self._explainer = None

        return metrics

    # ── Inference ──────────────────────────────────────────────────────────

    def analyse(self, email_input) -> AnalysisResult:
        """
        Run the full pipeline on a single email.

        Args:
            email_input: dict, raw string, .eml path, or EmailDocument

        Returns:
            AnalysisResult with all module outputs
        """
        # Module 1 — Parse
        doc = self.parser.parse(email_input)

        # Module 2 — Features
        features = self.extractor.extract(doc)

        # Module 3 — Predict
        prediction = self.engine.predict_single(features)

        # Module 4 — Score
        threat_report = self.scorer.score(prediction)

        # Module 5 — Explain
        if self._explainer is not None:
            try:
                explanation = self._explainer.explain(
                    features, prediction.risk_score, prediction.confidence
                )
            except Exception as e:
                logger.warning(f"SHAP explanation failed, falling back: {e}")
                explanation = self._fallback_explainer.explain_from_report(threat_report)
        else:
            explanation = self._fallback_explainer.explain_from_report(threat_report)

        return AnalysisResult(
            email_doc=doc,
            features=features,
            prediction=prediction,
            threat_report=threat_report,
            explanation=explanation
        )

    # ── Save / Load ────────────────────────────────────────────────────────

    def save(self, model_dir: str | Path = "models/") -> None:
        self.engine.save(model_dir)
        logger.info(f"Pipeline saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: str | Path = "models/") -> "PhishingDetector":
        detector = cls()
        detector.engine.load(model_dir)
        logger.info("Pipeline loaded and ready.")
        return detector

    # ── Pretty print ───────────────────────────────────────────────────────

    def print_analysis(self, result: AnalysisResult) -> None:
        r = result.threat_report
        e = result.explanation
        p = result.prediction

        badge = {"green": "🟢", "amber": "🟡", "red": "🔴"}.get(r.badge_colour, "⚪")

        print(f"\n{'='*60}")
        print(f"  {badge}  THREAT ANALYSIS REPORT")
        print(f"{'='*60}")
        print(f"  Risk Score  : {r.risk_score}/100")
        print(f"  Threat Level: {r.threat_level}")
        print(f"  Verdict     : {'⚠ PHISHING' if r.is_phishing else '✓ LEGITIMATE'}")
        print()
        print("  ── Model votes ───────────────────────────────────────")
        for model, prob in p.model_votes.items():
            bar = "█" * int(prob * 20)
            print(f"  {model:<22} {prob:.2%}  {bar}")
        print()
        print("  ── Top risk factors ──────────────────────────────────")
        if r.triggered_factors:
            for f in r.triggered_factors[:6]:
                print(f"  ✗  {f.label}")
        else:
            print("  ✓  No significant risk factors detected")
        print()
        if isinstance(e, str):
            print(f"  {e}")
        else:
            print(f"  {e.natural_language}")
        print(f"{'='*60}\n")
