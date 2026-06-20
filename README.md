# PhishGuard
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A machine learning system for detecting phishing emails. It doesn't just flag an email as suspicious — it tells you exactly why, breaking down which signals triggered the alert and how much each one contributed to the final score.

Built as a full pipeline from raw email input to an interactive dashboard, with explainability at the center.

---

## Why I built this

Most phishing detectors are black boxes. They say "phishing" or "not phishing" and give you nothing else. That's not useful for someone who needs to make a real decision about whether to trust an email.

I wanted to build something that works more like a security analyst — one that looks at the sender, the language, the URLs, the attachments, and then explains its reasoning in plain terms. That's what PhishGuard does.

---

## What it does

You paste an email (or feed it programmatically), and the system gives you:

- A risk score from 0 to 100
- A threat level: Safe, Suspicious, or High Risk  
- A breakdown of every signal that fired — urgency language, brand impersonation, suspicious URLs, credential requests, etc.
- SHAP-based explanations showing how much each feature pushed the score up or down
- Per-model votes from three separate classifiers so you can see where they agree or disagree

```
Risk Score: 91/100 — PHISHING DETECTED

Signals detected:
  ✗ Sender impersonates PayPal but domain is paypa1-secure.com
  ✗ Reply-To goes to a completely different domain
  ✗ Urgency language: "suspended", "immediately", "24 hours"
  ✗ Credential request: "verify your password"
  ✗ URL uses suspicious .xyz TLD
```

---

## How it works

The pipeline has six stages:

**1. Email parsing** — Takes raw email text, .eml files, or structured dicts. Extracts sender, reply-to, subject, body, all URLs, domains, and attachments with automatic risk flagging for executable or macro-enabled files.

**2. Feature extraction** — Produces 43 named features across four groups: header/sender signals (brand impersonation, domain mismatch, free email provider), URL signals (IP-based URLs, shorteners, suspicious TLDs), NLP signals (urgency score, fear score, credential and financial language, Shannon entropy), and attachment signals.

**3. AI detection** — Three classifiers run in parallel: Logistic Regression as a fast baseline, Random Forest for non-linear patterns, and XGBoost as the primary model. Their outputs are combined through a soft-voting ensemble weighted 1:2:3 in favor of XGBoost.

**4. Threat scoring** — Converts the model's calibrated probability to a 0–100 risk score and maps 18 named signals to human-readable factor descriptions, sorted by contribution.

**5. Explainability (SHAP)** — Uses SHAP TreeExplainer on the XGBoost model to compute signed per-feature contributions. Every prediction comes with a waterfall chart showing exactly how the score was built, from the base rate to the final number.

**6. Dashboard** — A four-page Streamlit app with an overview page, email analysis page, URL scanner, and analytics with trend charts and model performance comparisons.

---

## Getting started

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the CEAS 2008 dataset from Kaggle and put it at `data/raw/ceas2008.csv`. Any CSV with email body text and labels works — see `modules/dataset_loader.py` for other supported formats.

Train the models:

```python
from modules.dataset_loader import load_ceas2008
from pipeline import PhishingDetector

df = load_ceas2008("data/raw/ceas2008.csv")
detector = PhishingDetector()
detector.fit(df)
detector.save("models/")
```

Analyse an email:

```python
detector = PhishingDetector.load("models/")

result = detector.analyse({
    "sender": "security@paypa1-secure.com",
    "subject": "URGENT: Your account has been suspended",
    "body": "Verify your credentials immediately or your account will be terminated..."
})

detector.print_analysis(result)
```

Launch the dashboard (no trained model required — runs in demo mode):

```bash
streamlit run dashboard/app.py
```

---

## Project layout

```
phishing_detector/
├── modules/
│   ├── email_parser.py        # Stage 1 — parse raw email into structured doc
│   ├── feature_extractor.py   # Stage 2 — extract 43 features
│   ├── detection_engine.py    # Stage 3 — LR + RF + XGBoost ensemble
│   ├── threat_scorer.py       # Stage 4 — risk score + factor breakdown
│   ├── explainability.py      # Stage 5 — SHAP explanations
│   ├── shap_visualiser.py     # Plotly charts for SHAP output
│   └── dataset_loader.py      # Dataset loaders (CEAS, Enron, custom CSV)
├── dashboard/
│   └── app.py                 # Stage 6 — Streamlit dashboard
├── tests/
│   ├── test_pipeline.py       # 20 tests for stages 1–2
│   ├── test_ml_pipeline.py    # Tests for stages 3–4
│   └── test_explainability.py # Tests for stage 5 + visualiser
├── pipeline.py                # Single entry point tying all stages together
├── quickstart_demo.py         # Run this first to verify setup
└── requirements.txt
```

---

## Running the tests

```bash
pytest tests/ -v
```

The test suite covers edge cases like empty email bodies, NaN feature values, type consistency across all 43 features, model save/load round-trips, SHAP value sign consistency, and end-to-end pipeline integration.

---

## Tech stack

Python 3.11, scikit-learn, XGBoost, NLTK, SHAP, tldextract, Streamlit, Plotly, joblib.

---

## Dataset sources

- CEAS 2008 — https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus
- Enron Spam — http://www.cs.cmu.edu/~enron/
- SpamAssassin public corpus — https://spamassassin.apache.org/old/publiccorpus/

---

## Notes

The dashboard runs in demo mode by default, simulating model output from rule-based heuristics. Once you train on a real dataset and call `detector.save("models/")`, swap the `_demo_analyse()` call in `dashboard/app.py` for `detector.analyse()` and everything goes live.

SHAP explanations require a trained XGBoost model. The `RuleBasedExplainer` in `explainability.py` serves as a fallback when SHAP isn't available or before training.
