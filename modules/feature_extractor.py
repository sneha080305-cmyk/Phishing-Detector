"""
Module 2 — Feature Extraction
================================
Takes an EmailDocument and returns a feature dict (and optionally a
pandas Series) ready for ML model input.

Feature groups:
    A. Header / sender features
    B. URL features
    C. NLP / content features
    D. Attachment features
"""

import re
import math
import unicodedata
from collections import Counter
from urllib.parse import urlparse

import tldextract
import nltk
from loguru import logger

# Download NLTK resources silently on first use
for _resource, _path in [
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




URGENCY_WORDS = {
    "urgent", "immediately", "alert", "warning", "verify", "suspend",
    "suspended", "limited", "expire", "expires", "deadline", "act now",
    "action required", "confirm now", "update now", "failure to",
    "within 24 hours", "within 48 hours", "asap"
}

FEAR_WORDS = {
    "unauthorized", "unusual activity", "suspicious", "compromised",
    "fraud", "fraudulent", "hacked", "breach", "risk", "danger",
    "illegal", "violation", "penalty", "lawsuit", "legal action",
    "blocked", "terminated", "closed"
}

CREDENTIAL_WORDS = {
    "password", "username", "login", "sign in", "signin",
    "account", "credential", "verify your", "confirm your",
    "update your", "ssn", "social security", "bank account",
    "credit card", "debit card", "pin", "otp", "one-time password"
}

FINANCIAL_WORDS = {
    "payment", "invoice", "wire transfer", "bitcoin", "gift card",
    "itunes", "amazon card", "google play", "money", "fund",
    "transfer", "refund", "tax", "irs", "paypal", "bank"
}

GREETING_GENERIC = {
    "dear customer", "dear user", "dear account holder",
    "dear valued", "hello user", "dear sir", "dear madam"
}

# Known URL shorteners
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "is.gd", "buff.ly", "adf.ly", "rebrand.ly", "short.link",
    "tiny.cc", "cli.re", "cutt.ly"
}

# Legitimate brand domains often spoofed
SPOOFED_BRANDS = {
    "paypal", "apple", "amazon", "microsoft", "google", "netflix",
    "facebook", "instagram", "linkedin", "dropbox", "docusign",
    "chase", "wellsfargo", "bankofamerica", "citibank", "irs",
    "fedex", "ups", "dhl", "usps"
}

STOP_WORDS = set(stopwords.words("english"))



class FeatureExtractor:
    """
    Extracts a flat feature dictionary from an EmailDocument.

    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract(email_doc)
        # features is a dict[str, float | int]
    """

    def extract(self, doc) -> dict:
        features = {}
        features.update(self._header_features(doc))
        features.update(self._url_features(doc))
        features.update(self._nlp_features(doc))
        features.update(self._attachment_features(doc))
        return features


    def _header_features(self, doc) -> dict:
        f = {}

        sender = doc.sender.lower()
        subject = doc.subject.lower()
        reply_to = doc.reply_to.lower()

        # Sender has display name
        f["sender_has_display_name"] = int(bool(doc.sender_name))

        # Reply-To differs from From domain
        reply_domain = self._domain_from_email(reply_to)
        f["reply_to_domain_mismatch"] = int(
            bool(reply_domain) and reply_domain != doc.sender_domain
        )

        # Sender domain is free email provider
        free_providers = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"}
        f["sender_free_email"] = int(doc.sender_domain in free_providers)

        # Brand impersonation: brand name in sender but domain doesn't match
        impersonated = 0
        for brand in SPOOFED_BRANDS:
            if brand in sender and brand not in doc.sender_domain:
                impersonated = 1
                break
        f["sender_brand_impersonation"] = impersonated

        # Brand in subject
        brand_in_subject = int(any(b in subject for b in SPOOFED_BRANDS))
        f["subject_contains_brand"] = brand_in_subject

        # Subject features
        f["subject_has_re_fwd"] = int(bool(re.search(r'\b(re:|fwd:)', subject)))
        f["subject_all_caps_ratio"] = self._caps_ratio(doc.subject)
        f["subject_exclamation"] = int("!" in doc.subject)
        f["subject_question_mark"] = int("?" in doc.subject)
        f["subject_char_count"] = len(doc.subject)
        f["subject_word_count"] = len(doc.subject.split())

        # Generic greeting
        body_lower = doc.body_text.lower()
        f["generic_greeting"] = int(any(g in body_lower for g in GREETING_GENERIC))

        # No message-id (common in spam)
        f["missing_message_id"] = int(not doc.message_id)

        return f


    def _url_features(self, doc) -> dict:
        f = {}
        urls = doc.urls

        f["url_count"] = len(urls)
        f["unique_domain_count"] = len(doc.domains)

        if not urls:
            f.update({
                "url_has_ip": 0, "url_shortener_count": 0,
                "url_https_ratio": 0.0, "url_avg_length": 0.0,
                "url_suspicious_tld": 0, "url_brand_mismatch": 0,
                "url_at_symbol": 0, "url_double_slash": 0,
                "url_subdomain_depth_max": 0, "url_numeric_ratio_avg": 0.0,
            })
            return f

        ip_re = re.compile(r'https?://\d{1,3}(\.\d{1,3}){3}')
        suspicious_tlds = {".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".top", ".click"}

        has_ip, shortener_count, https_count = 0, 0, 0
        suspicious_tld, brand_mismatch = 0, 0
        at_sym, double_slash = 0, 0
        lengths, numeric_ratios, subdomain_depths = [], [], []

        for url in urls:
            if ip_re.match(url):
                has_ip = 1
            ext = tldextract.extract(url)
            domain_str = f"{ext.domain}.{ext.suffix}".lower()
            full_domain = f"{ext.domain}.{ext.suffix}".lower()
            if full_domain in URL_SHORTENERS:
                shortener_count += 1
            if url.startswith("https://"):
                https_count += 1

            # Suspicious TLD
            tld = "." + ext.suffix.split(".")[-1] if ext.suffix else ""
            if tld in suspicious_tlds:
                suspicious_tld = 1

            # Brand in URL but domain doesn't match brand
            for brand in SPOOFED_BRANDS:
                if brand in url.lower() and brand not in domain_str:
                    brand_mismatch = 1
                    break

            if "@" in url:
                at_sym = 1
            if "//" in url[8:]:
                double_slash = 1

            lengths.append(len(url))
            digits = sum(c.isdigit() for c in url)
            numeric_ratios.append(digits / max(len(url), 1))
            subdomain_depths.append(len(ext.subdomain.split(".")) if ext.subdomain else 0)

        f["url_has_ip"] = has_ip
        f["url_shortener_count"] = shortener_count
        f["url_https_ratio"] = https_count / len(urls)
        f["url_avg_length"] = sum(lengths) / len(lengths)
        f["url_suspicious_tld"] = suspicious_tld
        f["url_brand_mismatch"] = brand_mismatch
        f["url_at_symbol"] = at_sym
        f["url_double_slash"] = double_slash
        f["url_subdomain_depth_max"] = max(subdomain_depths, default=0)
        f["url_numeric_ratio_avg"] = sum(numeric_ratios) / len(numeric_ratios)

        return f


    def _nlp_features(self, doc) -> dict:
        f = {}
        text = doc.body_text.lower()
        raw_text = doc.body_text

        if not text.strip():
            return {k: 0 for k in [
                "urgency_score", "fear_score", "credential_request_score",
                "financial_request_score", "word_count", "avg_word_length",
                "type_token_ratio", "html_to_text_ratio", "caps_ratio",
                "exclamation_count", "question_count", "sentence_count",
                "stopword_ratio", "entropy"
            ]}

        # Lexicon scores (count of matching phrases / total words)
        words = word_tokenize(text)
        word_count = max(len(words), 1)

        def _lexicon_score(lexicon: set) -> float:
            hits = sum(1 for phrase in lexicon if phrase in text)
            return min(hits / max(word_count / 10, 1), 1.0)  # normalised 0-1

        f["urgency_score"] = _lexicon_score(URGENCY_WORDS)
        f["fear_score"] = _lexicon_score(FEAR_WORDS)
        f["credential_request_score"] = _lexicon_score(CREDENTIAL_WORDS)
        f["financial_request_score"] = _lexicon_score(FINANCIAL_WORDS)

        # Basic stats
        f["word_count"] = word_count
        f["avg_word_length"] = (
            sum(len(w) for w in words if w.isalpha()) / max(sum(w.isalpha() for w in words), 1)
        )

        # Type-token ratio (lexical diversity)
        unique_words = set(w for w in words if w.isalpha())
        f["type_token_ratio"] = len(unique_words) / word_count

        # HTML richness vs text (high ratio = image-heavy phish)
        f["html_to_text_ratio"] = (
            len(doc.body_html) / max(len(doc.body_text), 1) if doc.body_html else 0.0
        )

        # Stylistic
        f["caps_ratio"] = self._caps_ratio(raw_text)
        f["exclamation_count"] = raw_text.count("!")
        f["question_count"] = raw_text.count("?")
        f["sentence_count"] = len(re.split(r'[.!?]+', raw_text))

        # Stopword ratio (low = keyword-stuffed spam)
        alpha_words = [w for w in words if w.isalpha()]
        f["stopword_ratio"] = (
            sum(1 for w in alpha_words if w in STOP_WORDS) / max(len(alpha_words), 1)
        )

        # Shannon entropy of characters (high = obfuscated text)
        f["entropy"] = self._shannon_entropy(text)

        return f

    

    def _attachment_features(self, doc) -> dict:
        attachments = doc.attachments
        f = {
            "attachment_count": len(attachments),
            "risky_attachment_count": sum(1 for a in attachments if a.get("is_risky")),
            "has_executable": int(any(a.get("extension") == ".exe" for a in attachments)),
            "has_macro_doc": int(
                any(a.get("extension") in {".docm", ".xlsm", ".pptm"} for a in attachments)
            ),
        }
        return f


    @staticmethod
    def _caps_ratio(text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        return sum(1 for c in letters if c.isupper()) / max(len(letters), 1)

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        freq = Counter(text)
        length = len(text)
        return -sum((c / length) * math.log2(c / length) for c in freq.values())

    @staticmethod
    def _domain_from_email(address: str) -> str:
        match = re.search(r'@([\w.-]+)', address)
        return match.group(1).lower() if match else ""
