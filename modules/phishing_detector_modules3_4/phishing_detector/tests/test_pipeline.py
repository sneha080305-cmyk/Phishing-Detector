"""
Tests for Module 1 (EmailParser) and Module 2 (FeatureExtractor)
Run with: pytest tests/test_pipeline.py -v
"""

import sys
sys.path.insert(0, str(__file__ and __import__('pathlib').Path(__file__).parent.parent))

import pytest
from modules.email_parser import EmailParser, EmailDocument
from modules.feature_extractor import FeatureExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────────

PHISHING_EMAIL_DICT = {
    "sender": "PayPal Security <security-alert@paypa1-secure.com>",
    "subject": "URGENT: Your account has been SUSPENDED!",
    "body": (
        "Dear valued customer,\n\n"
        "We have detected unauthorized access to your PayPal account. "
        "You must verify your password and credit card immediately or your account will be terminated.\n\n"
        "Click here to restore access: http://paypal-login-secure.xyz/verify?id=12345\n\n"
        "Failure to act within 24 hours will result in permanent suspension.\n\n"
        "PayPal Security Team"
    ),
    "reply_to": "noreply@another-domain.ru"
}

LEGIT_EMAIL_DICT = {
    "sender": "GitHub <noreply@github.com>",
    "subject": "Your pull request was merged",
    "body": (
        "Hi there,\n\n"
        "Your pull request #42 'Fix login bug' was successfully merged into main.\n\n"
        "View the pull request: https://github.com/your-org/your-repo/pull/42\n\n"
        "Thanks,\nThe GitHub Team"
    ),
    "reply_to": ""
}

RAW_EMAIL_STRING = """\
From: "Amazon Support" <support@amaz0n-help.net>
To: victim@example.com
Subject: Re: Your order needs confirmation
Reply-To: claims@foreign-domain.biz
Message-ID: 

Dear customer,

Your recent order has been flagged. Verify your account credentials immediately:
http://bit.ly/amzn-verify123

Regards,
Amazon Team
"""


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestEmailParser:
    def setup_method(self):
        self.parser = EmailParser()

    def test_parse_dict(self):
        doc = self.parser.parse(PHISHING_EMAIL_DICT)
        assert isinstance(doc, EmailDocument)
        assert doc.sender == PHISHING_EMAIL_DICT["sender"]
        assert doc.subject == PHISHING_EMAIL_DICT["subject"]
        assert "terminated" in doc.body_text

    def test_sender_domain_extracted(self):
        doc = self.parser.parse(PHISHING_EMAIL_DICT)
        assert doc.sender_domain == "paypa1-secure.com"

    def test_urls_extracted(self):
        doc = self.parser.parse(PHISHING_EMAIL_DICT)
        assert len(doc.urls) == 1
        assert "paypal-login-secure.xyz" in doc.urls[0]

    def test_domains_derived(self):
        doc = self.parser.parse(PHISHING_EMAIL_DICT)
        assert "paypal-login-secure.xyz" in doc.domains

    def test_legit_email_parse(self):
        doc = self.parser.parse(LEGIT_EMAIL_DICT)
        assert doc.sender_domain == "github.com"
        assert len(doc.urls) == 1

    def test_raw_string_parse(self):
        doc = self.parser.parse(RAW_EMAIL_STRING)
        assert "amaz0n-help.net" in doc.sender_domain
        assert doc.reply_to != ""
        assert len(doc.urls) >= 1

    def test_missing_message_id(self):
        doc = self.parser.parse(RAW_EMAIL_STRING)
        assert doc.message_id.strip() == "" or not doc.message_id

    def test_passthrough(self):
        doc = self.parser.parse(PHISHING_EMAIL_DICT)
        doc2 = self.parser.parse(doc)
        assert doc is doc2

    def test_empty_body(self):
        doc = self.parser.parse({"sender": "a@b.com", "subject": "hi", "body": ""})
        assert doc.body_text == ""
        assert doc.urls == []


# ── Feature extractor tests ───────────────────────────────────────────────────

class TestFeatureExtractor:
    def setup_method(self):
        self.parser = EmailParser()
        self.extractor = FeatureExtractor()

    def _features(self, email_dict):
        doc = self.parser.parse(email_dict)
        return self.extractor.extract(doc)

    def test_returns_dict(self):
        f = self._features(PHISHING_EMAIL_DICT)
        assert isinstance(f, dict)
        assert len(f) > 20, "Expected at least 20 features"

    def test_phishing_signals_high(self):
        f = self._features(PHISHING_EMAIL_DICT)
        assert f["urgency_score"] > 0
        assert f["credential_request_score"] > 0
        assert f["reply_to_domain_mismatch"] == 1
        assert f["sender_brand_impersonation"] == 1
        assert f["url_suspicious_tld"] == 1

    def test_legit_signals_low(self):
        f = self._features(LEGIT_EMAIL_DICT)
        assert f["urgency_score"] == 0 or f["urgency_score"] < 0.1
        assert f["reply_to_domain_mismatch"] == 0
        assert f["sender_brand_impersonation"] == 0

    def test_url_shortener_detected(self):
        doc = self.parser.parse(RAW_EMAIL_STRING)
        f = self.extractor.extract(doc)
        assert f["url_shortener_count"] >= 1

    def test_caps_ratio_for_shouting_subject(self):
        f = self._features(PHISHING_EMAIL_DICT)
        assert f["subject_all_caps_ratio"] > 0.3

    def test_all_features_numeric(self):
        f = self._features(PHISHING_EMAIL_DICT)
        for k, v in f.items():
            assert isinstance(v, (int, float)), f"Feature '{k}' is not numeric: {type(v)}"

    def test_no_nan_features(self):
        import math
        f = self._features(PHISHING_EMAIL_DICT)
        for k, v in f.items():
            assert not (isinstance(v, float) and math.isnan(v)), f"NaN in feature '{k}'"

    def test_missing_body_graceful(self):
        f = self._features({"sender": "x@y.com", "subject": "test", "body": ""})
        assert f["word_count"] == 0
        assert f["url_count"] == 0

    def test_attachment_features(self):
        from modules.email_parser import EmailDocument
        doc = EmailDocument()
        doc.attachments = [
            {"filename": "invoice.exe", "extension": ".exe", "is_risky": True, "size": 1024},
            {"filename": "report.pdf", "extension": ".pdf", "is_risky": False, "size": 512},
        ]
        f = self.extractor.extract(doc)
        assert f["attachment_count"] == 2
        assert f["risky_attachment_count"] == 1
        assert f["has_executable"] == 1


# ── Integration ───────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_phishing_vs_legit_feature_difference(self):
        parser = EmailParser()
        extractor = FeatureExtractor()

        phish_f = extractor.extract(parser.parse(PHISHING_EMAIL_DICT))
        legit_f = extractor.extract(parser.parse(LEGIT_EMAIL_DICT))

        # Phishing should score higher on these
        for key in ("urgency_score", "fear_score", "sender_brand_impersonation"):
            assert phish_f[key] >= legit_f[key], \
                f"{key}: phishing={phish_f[key]} should be >= legit={legit_f[key]}"

    def test_feature_keys_consistent(self):
        """Same keys regardless of email content."""
        parser = EmailParser()
        extractor = FeatureExtractor()

        keys1 = set(extractor.extract(parser.parse(PHISHING_EMAIL_DICT)).keys())
        keys2 = set(extractor.extract(parser.parse(LEGIT_EMAIL_DICT)).keys())
        assert keys1 == keys2, f"Key mismatch: {keys1.symmetric_difference(keys2)}"
