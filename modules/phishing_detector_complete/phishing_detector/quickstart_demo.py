"""
Quickstart Demo — Modules 1 & 2
================================
Run this to verify your pipeline works before training any models.

    python quickstart_demo.py

Expected output: feature dict printed for a phishing and a legit email,
with obvious score differences.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.email_parser import EmailParser
from modules.feature_extractor import FeatureExtractor


def demo(label: str, email_dict: dict):
    parser = EmailParser()
    extractor = FeatureExtractor()

    doc = parser.parse(email_dict)
    features = extractor.extract(doc)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Sender domain : {doc.sender_domain}")
    print(f"  URLs found    : {doc.urls}")
    print(f"  Reply-To      : {doc.reply_to}")
    print()
    print("  ── Key features ──────────────────────────────────────")

    highlight_keys = [
        "urgency_score", "fear_score", "credential_request_score",
        "financial_request_score", "sender_brand_impersonation",
        "reply_to_domain_mismatch", "url_suspicious_tld",
        "url_shortener_count", "url_has_ip", "url_brand_mismatch",
        "subject_all_caps_ratio", "generic_greeting",
    ]

    for k in highlight_keys:
        v = features.get(k, "—")
        bar = "🔴" if (isinstance(v, (int, float)) and v > 0) else "🟢"
        print(f"  {bar}  {k:<40} {v}")

    print(f"\n  Total features extracted: {len(features)}")


PHISHING = {
    "sender": "PayPal Security <security-alert@paypa1-secure.com>",
    "reply_to": "noreply@another-domain.ru",
    "subject": "URGENT: Your PayPal account has been SUSPENDED!",
    "body": (
        "Dear valued customer,\n\n"
        "We have detected unauthorized access to your PayPal account. "
        "You must verify your password and credit card immediately "
        "or your account will be permanently terminated.\n\n"
        "Click here: http://paypal-secure-login.xyz/verify?id=99\n"
        "Or: http://bit.ly/pp-restore\n\n"
        "Failure to act within 24 hours will result in account closure.\n\n"
        "PayPal Security Team"
    )
}

LEGIT = {
    "sender": "GitHub <noreply@github.com>",
    "reply_to": "",
    "subject": "Your pull request was merged",
    "body": (
        "Hi there,\n\n"
        "Your pull request #42 'Fix login bug' was successfully merged into main.\n\n"
        "View it at: https://github.com/your-org/your-repo/pull/42\n\n"
        "Thanks,\nThe GitHub Team"
    )
}

if __name__ == "__main__":
    demo("PHISHING EMAIL", PHISHING)
    demo("LEGITIMATE EMAIL", LEGIT)
    print("\n✓ Pipeline working correctly.\n")
