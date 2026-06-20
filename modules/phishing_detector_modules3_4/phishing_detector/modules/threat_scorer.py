"""
Module 4 — Threat Scoring Engine
===================================
Converts model probabilities + features into a structured ThreatReport
with risk score (0-100), category, and per-factor breakdown.

Risk levels:
    0-30   SAFE
    31-60  SUSPICIOUS
    61-85  HIGH RISK
    86-100 CRITICAL
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class RiskLevel(str, Enum):
    SAFE       = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK  = "HIGH RISK"
    CRITICAL   = "CRITICAL"


RISK_THRESHOLDS = [(0,30,RiskLevel.SAFE),(31,60,RiskLevel.SUSPICIOUS),(61,85,RiskLevel.HIGH_RISK),(86,100,RiskLevel.CRITICAL)]
RISK_COLORS = {RiskLevel.SAFE:"#22c55e",RiskLevel.SUSPICIOUS:"#f59e0b",RiskLevel.HIGH_RISK:"#f97316",RiskLevel.CRITICAL:"#ef4444"}
RISK_EMOJI  = {RiskLevel.SAFE:"🟢",RiskLevel.SUSPICIOUS:"🟡",RiskLevel.HIGH_RISK:"🟠",RiskLevel.CRITICAL:"🔴"}


@dataclass
class ThreatFactor:
    name: str
    feature_key: str
    value: float
    contribution: float
    triggered: bool

    def __str__(self):
        status = "✓" if self.triggered else "○"
        return f"  {status} {self.name:<42} +{self.contribution:.1f}%"


@dataclass
class ThreatReport:
    risk_score: int
    risk_level: RiskLevel
    is_phishing: bool
    model_probas: dict = field(default_factory=dict)
    factors: list = field(default_factory=list)
    email_subject: str = ""
    email_sender: str = ""

    @property
    def color(self): return RISK_COLORS[self.risk_level]

    @property
    def emoji(self): return RISK_EMOJI[self.risk_level]

    @property
    def triggered_factors(self): return [f for f in self.factors if f.triggered]

    def summary(self) -> str:
        lines = [
            f"\n{'='*56}",
            f"  {self.emoji} Risk Score  : {self.risk_score}/100  [{self.risk_level.value}]",
            f"  Verdict      : {'PHISHING' if self.is_phishing else 'LEGITIMATE'}",
            f"  Sender       : {self.email_sender or '(unknown)'}",
            f"  Subject      : {self.email_subject or '(none)'}",
            f"{'─'*56}",
            "  Model probabilities:",
        ]
        for model, prob in self.model_probas.items():
            bar = "█"*int(prob*20) + "░"*(20-int(prob*20))
            lines.append(f"    {model:<24} [{bar}] {prob*100:.1f}%")
        lines += [f"{'─'*56}", "  Top contributing factors:"]
        for f in self.factors[:8]:
            lines.append(str(f))
        lines.append(f"{'='*56}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "is_phishing": self.is_phishing,
            "color": self.color,
            "model_probas": self.model_probas,
            "factors": [{"name":f.name,"feature_key":f.feature_key,"value":round(f.value,4),"contribution":round(f.contribution,2),"triggered":f.triggered} for f in self.factors],
            "email_subject": self.email_subject,
            "email_sender": self.email_sender,
        }


FACTOR_REGISTRY = [
    ("sender_brand_impersonation",  "Sender impersonates a known brand",    22),
    ("reply_to_domain_mismatch",    "Reply-To domain differs from sender",  18),
    ("credential_request_score",    "Requests credentials or login",        20),
    ("url_has_ip",                  "URL contains raw IP address",          20),
    ("url_brand_mismatch",          "Brand name in URL but domain differs", 18),
    ("has_executable",              "Contains executable attachment",       25),
    ("has_macro_doc",               "Contains macro-enabled document",      20),
    ("url_suspicious_tld",          "URL uses suspicious TLD (.xyz, .tk)",  15),
    ("urgency_score",               "Urgent / action-required language",    15),
    ("financial_request_score",     "Requests financial information",       14),
    ("url_shortener_count",         "Email contains shortened URLs",        12),
    ("fear_score",                  "Fear-inducing language detected",      12),
    ("generic_greeting",            "Generic greeting (dear customer)",     10),
    ("url_at_symbol",               "URL contains @ symbol",               10),
    ("sender_free_email",           "Sent from free email provider",        8),
    ("subject_all_caps_ratio",      "Subject has excessive caps",           6),
    ("url_double_slash",            "URL contains double-slash redirect",   8),
    ("missing_message_id",          "Email has no Message-ID header",       7),
    ("url_subdomain_depth_max",     "Deep subdomain chain in URL",          7),
    ("risky_attachment_count",      "Has risky file attachment(s)",        15),
    ("subject_contains_brand",      "Subject mentions a known brand",       8),
    ("entropy",                     "High text entropy (obfuscation)",      5),
    ("subject_exclamation",         "Subject has exclamation mark",         4),
]


class ThreatScorer:
    ML_WEIGHT   = 0.60
    RULE_WEIGHT = 0.40

    def score(self, model_probas, features, email_subject="", email_sender="") -> ThreatReport:
        ml_prob    = float(np.mean(list(model_probas.values()))) if model_probas else 0.0
        rule_score, factors = self._compute_rule_score(features)
        blended    = (self.ML_WEIGHT * ml_prob * 100) + (self.RULE_WEIGHT * rule_score)
        risk_score = int(np.clip(round(blended), 0, 100))
        factors    = self._annotate_contributions(factors, rule_score)

        return ThreatReport(
            risk_score=risk_score,
            risk_level=self._classify(risk_score),
            is_phishing=risk_score >= 50,
            model_probas=model_probas,
            factors=sorted(factors, key=lambda f: abs(f.contribution), reverse=True),
            email_subject=email_subject,
            email_sender=email_sender,
        )

    def _compute_rule_score(self, features):
        total_weight = sum(w for _, _, w in FACTOR_REGISTRY)
        raw_score = 0.0
        factors = []
        for feat_key, label, weight in FACTOR_REGISTRY:
            val = float(features.get(feat_key, 0.0))
            if feat_key in ("urgency_score","fear_score","credential_request_score","financial_request_score","entropy","subject_all_caps_ratio"):
                intensity = min(val, 1.0)
                effective = weight * intensity
                triggered = val > 0.05
            elif feat_key == "url_subdomain_depth_max":
                effective = weight * min(val/3.0, 1.0)
                triggered = val >= 2
            elif feat_key == "url_shortener_count":
                effective = weight if val >= 1 else 0
                triggered = val >= 1
            else:
                effective = weight if val > 0 else 0
                triggered = val > 0
            raw_score += max(effective, 0)
            factors.append(ThreatFactor(name=label,feature_key=feat_key,value=val,contribution=0.0,triggered=triggered))
        return float(np.clip((raw_score/total_weight)*100, 0, 100)), factors

    def _annotate_contributions(self, factors, rule_score):
        safe_rule = max(rule_score, 1.0)
        total_pos = sum(abs(w) for k,_,w in FACTOR_REGISTRY if any(f.feature_key==k and f.triggered for f in factors))
        for f in factors:
            if f.triggered:
                weight = next((w for k,_,w in FACTOR_REGISTRY if k==f.feature_key), 0)
                f.contribution = (abs(weight)/max(total_pos,1)) * safe_rule * self.RULE_WEIGHT
        return factors

    @staticmethod
    def _classify(score):
        for lo,hi,level in RISK_THRESHOLDS:
            if lo<=score<=hi: return level
        return RiskLevel.CRITICAL
