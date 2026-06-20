# 🛡️ PhishGuard — AI-Powered Phishing Detection System

> An explainable AI threat analysis engine that doesn't just classify emails — it *reasons* about them.

---

## Overview

PhishGuard is a full-stack cybersecurity decision-support system for detecting phishing emails using machine learning and behavioural analysis. It goes beyond a simple classifier by providing **per-prediction explanations** through SHAP, giving analysts actionable insights into *why* an email was flagged.

---

## Architecture

```
Raw Email Input
      │
      ▼
┌─────────────────────┐
│  Module 1           │  EmailParser
│  Email Parsing      │  Handles raw strings, .eml files, dicts
└────────┬────────────┘
         │ EmailDocument
         ▼
┌─────────────────────┐
│  Module 2           │  FeatureExtractor
│  Feature Extraction │  43 features across 4 groups
└────────┬────────────┘
         │ Feature dict
         ▼
┌─────────────────────┐
│  Module 3           │  DetectionEngine
│  AI Detection       │  LR + RF + XGBoost ensemble
└────────┬────────────┘
         │ PredictionResult
         ▼
┌─────────────────────┐
│  Module 4           │  ThreatScorer
│  Threat Scoring     │  Risk score 0–100 + factor breakdown
└────────┬────────────┘
         │ ThreatReport
         ▼
┌─────────────────────┐
│  Module 5           │  ExplainabilityEngine (SHAP)
│  Explainable AI     │  Per-feature signed contributions
└────────┬────────────┘
         │ Explanation
         ▼
┌─────────────────────┐
│  Module 6           │  Streamlit Dashboard
│  Dashboard          │  4-page interactive web interface
└─────────────────────┘
```

---

## Feature Groups (Module 2)

### Header & Sender (12 features)
- Sender brand impersonation detection
- Reply-To domain mismatch
- Free email provider identification
- Generic salutation detection
- Subject capitalisation, punctuation analysis

### URL Features (10 features)
- IP-based URL detection
- URL shortener identification
- Suspicious TLD detection (`.xyz`, `.tk`, `.ml`, ...)
- Brand name in URL with domain mismatch
- Subdomain depth, URL length, numeric character ratio

### NLP / Content (14 features)
- Urgency language score (custom lexicon)
- Fear-based language score
- Credential request score
- Financial transaction language score
- Shannon entropy (obfuscation detection)
- Type-token ratio (lexical diversity)
- Stopword ratio

### Attachment Features (4 features)
- Executable attachment presence
- Macro-enabled document detection
- Risky attachment count

---

## Models (Module 3)

| Model | Role | Strength |
|---|---|---|
| Logistic Regression | Baseline | Fast, interpretable coefficients |
| Random Forest | Mid-tier | Non-linear patterns, robust |
| XGBoost | Primary | Best accuracy on tabular features |
| Ensemble (soft vote) | Production | Combines all three, weighted 1:2:3 |

All classifiers use class-weight balancing to handle dataset imbalance.

---

## Risk Scoring (Module 4)

```
0  ──────── 30 ──────── 60 ──────── 100
     SAFE      SUSPICIOUS   HIGH RISK
```

Score is the ensemble model's calibrated probability × 100.
Factor breakdown maps 18 named signals to human-readable explanations.

---

## Explainability (Module 5)

Uses **SHAP TreeExplainer** on the XGBoost model to compute signed per-feature contributions:

```
Risk Score: 91/100 — PHISHING DETECTED

Contributing factors:
  🔴 Credential request identified        +20%
  🔴 Known brand impersonated             +15%
  🔴 Urgency language detected            +18%
  🟡 URL shortener used                   +12%
  🟡 Reply-To domain mismatch             +15%
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download a dataset
Place the CEAS 2008 CSV at `data/raw/ceas2008.csv`.
Download: https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus

### 3. Train the model
```python
from modules.dataset_loader import load_ceas2008
from pipeline import PhishingDetector

df = load_ceas2008("data/raw/ceas2008.csv")
detector = PhishingDetector()
metrics = detector.fit(df)
detector.save("models/")
print(metrics["ensemble_cv"])
```

### 4. Analyse an email
```python
detector = PhishingDetector.load("models/")

result = detector.analyse({
    "sender": "security@paypa1-secure.com",
    "subject": "URGENT: Account suspended",
    "body": "Verify your password immediately or your account will be terminated..."
})

detector.print_analysis(result)
```

### 5. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
phishing_detector/
├── modules/
│   ├── email_parser.py       # Module 1 — Email parsing
│   ├── feature_extractor.py  # Module 2 — Feature extraction (43 features)
│   ├── detection_engine.py   # Module 3 — ML classifiers + ensemble
│   ├── threat_scorer.py      # Module 4 — Risk scoring + factor breakdown
│   ├── explainability.py     # Module 5 — SHAP explanations
│   └── dataset_loader.py     # Dataset ingestion utilities
├── dashboard/
│   └── app.py                # Module 6 — Streamlit dashboard
├── models/                   # Saved model artifacts (after training)
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Feature matrices
├── tests/
│   ├── test_pipeline.py      # Modules 1 & 2 tests (20 tests)
│   └── test_ml_pipeline.py   # Modules 3–5 tests
├── pipeline.py               # PhishingDetector orchestrator
├── quickstart_demo.py        # Demo script (no dataset needed)
└── requirements.txt
```

---

## Running Tests

```bash
# Modules 1 & 2
pytest tests/test_pipeline.py -v

# Modules 3–5
pytest tests/test_ml_pipeline.py -v

# All tests
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML | scikit-learn, XGBoost |
| NLP | NLTK |
| Explainability | SHAP |
| URL analysis | tldextract, dnspython |
| Dashboard | Streamlit, Plotly |
| Serialisation | joblib |

---

## Dataset Sources

- **CEAS 2008** — https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus
- **Enron Spam** — http://www.cs.cmu.edu/~enron/
- **SpamAssassin** — https://spamassassin.apache.org/old/publiccorpus/

---

## Scholarship Submission Notes

This project demonstrates:
1. **End-to-end ML system design** — not just a model, a complete pipeline
2. **Explainable AI** — SHAP-based reasoning, not black-box output
3. **Software engineering practices** — dataclasses, test suite, modular architecture
4. **Cybersecurity domain knowledge** — feature engineering grounded in real phishing patterns
5. **Production readiness** — model persistence, error handling, logging, graceful fallbacks
