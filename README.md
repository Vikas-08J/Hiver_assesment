# Hiver SDE Intern — AI Customer Support Agent

A production-grade, reproducible customer-support agent built on the **Customer Support on Twitter** dataset for `@AppleSupport`.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Benchmark Results Summary

Evaluated on a **hand-labelled golden test set of $N = 200$ human-annotated AppleSupport conversations** strictly held out from training and retrieval.

| System | Intent Accuracy | Intent Macro-F1 | Intent Weighted-F1 | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline 0 (Majority Class: `other`)** | 38.0% | 0.069 | 0.209 | Trivial |
| **Baseline 1 (TF-IDF + Logistic Regression)** | 80.5% | 0.708 | 0.812 | Supervised Baseline |
| **Proposed Hybrid (Char N-Gram + Calibrated Linear SVM)** | **88.5%** | **0.760** | **0.884** | **Production Architecture** |

### Reply Quality & Alignment Metrics ($N = 30$ Sample)
- **Human Mean Reply Score:** `3.365 / 5` ($N=200$), `3.333 / 5` ($N=30$)
- **LLM Judge Mean Overall Score:** `3.267 / 5`
- **Agreement Within $\pm 1$ Score:** `93.3%`
- **Exact Agreement:** `40.0%`
- **Weighted Cohen's Kappa ($\kappa$):** `0.100` (quadratic ordinal weights)
- **Spearman Rank Correlation ($\rho$):** `0.131` ($p = 0.490$)
- **Escalate Precision:** `100.0%` (when system escalates, human *always* agreed to escalate)
- **Hallucination Risk:** `1.000 / 5` (zero hallucination risk; strictly evidence-grounded)

Detailed breakdown and failure analysis can be found in [REPORT.md](REPORT.md) and [reports/final_report.md](reports/final_report.md).

---

## What This Project Does

For `@AppleSupport`, the pipeline:
1. **Extracts & Filters Conversations:** Reconstructs 20,000 customer $\rightarrow$ brand response pairs from the Twitter CS dataset.
2. **Defines an Operational Support-Intent Taxonomy:** 9 operational support classes (`account_access`, `billing_payment`, `order_purchase`, `device_issue`, `app_service_issue`, `subscription_cancel`, `technical_howto`, `status_followup`, `other`).
3. **Builds Intent Baselines & Calibrated Classifier:** Character n-gram ($3 \le n \le 5$) Linear SVM with 3-fold cross-validated Sigmoid calibration.
4. **Retrieves Historical Resolutions:** Sublinear TF-IDF cosine similarity index over historical resolutions.
5. **Enforces Multi-Layer Escalation:** Deterministic rules for high-risk signals (fraud, hack, legal), sensitive intents (account access), confidence thresholds ($< 0.68$), and retrieval similarity ($< 0.20$).
6. **Drafts Grounded Support Replies:** Constrained response generation adhering strictly to retrieved brand evidence, with resilient fallback to proven historical resolutions.
7. **Comprehensive Multi-Dimensional Evaluation:** Intent F1, LLM judge scoring (7 dimensions), human/judge agreement metrics, and 44 edge case tests.

---

## System Architecture

```text
Incoming Customer Message
           │
           ▼
┌───────────────────────────────┐
│ 1. Normalization & Preprocess │
│    Lowercasing, unicode norm  │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 2. Intent Classification      │──► Predicted Intent + Calibrated Confidence
│    Char n-gram Calibrated SVM │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 3. Historical Retriever       │──► Top-K Solved Support Resolutions (TF-IDF Cosine)
│    Customer -> Brand Pairs    │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 4. Multi-Layer Safety Policy  │
│    - High-risk keywords check │──► Auto-Handle: False (Escalate to Human Agent)
│    - Account-access check     │
│    - Confidence check (<0.68) │
│    - Evidence check (<0.20)   │
└──────────────┬────────────────┘
               │ Auto-Handle: True
               ▼
┌───────────────────────────────┐
│ 5. Grounded Reply Synthesis   │──► Output: Reply + Evidence + Decision + Reason
│    Evidence-constrained draft │
└───────────────────────────────┘
```

---

## Environment Setup

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/Vikas-08J/Hiver_assesment.git
cd Hiver_assesment

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Optional LLM API Configuration
For LLM-powered response rewriting and judging:
```powershell
copy .env.example .env
# Set GEMINI_API_KEY=your_key_here in .env
```
*(Note: The agent runs fully offline without an API key by using grounded template resolution and rubric evaluation).*

---

## Dataset

Primary source: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

Place the downloaded CSV at:
```text
data/raw/twcs.csv
```
*(Note: Per assignment instructions, `data/raw/twcs.csv` is excluded from git via `.gitignore`).*

---

## Reproduce the Pipeline

Execute the end-to-end evaluation pipeline with deterministic random seed `42`:

```powershell
# 1. Prepare 20,000 AppleSupport conversation pairs
python scripts/prepare_data.py --input data/raw/twcs.csv --brand AppleSupport --max-pairs 20000

# 2. Build golden candidate set (pre-annotated golden set already provided)
python scripts/build_golden.py --n 200

# 3. Train calibrated intent classifier
python scripts/train.py

# 4. Evaluate baselines vs proposed model on golden set
python scripts/evaluate.py

# 5. Run edge case test suite (44 edge cases: PII, slang, emojis, URLs)
python scripts/edge_case_test.py

# 6. Run 30-sample judge evaluation
python scripts/judge.py --n 30 --rubric

# 7. Compute human/judge agreement metrics
python scripts/agreement_analysis.py

# 8. Run failure mode extraction
python scripts/failure_analysis.py
```

---

## CLI Usage

### Interactive Agent Query
Run the customer support agent on any arbitrary message:

```powershell
# Example 1: Device issue (Auto-handled with historical troubleshooting)
python scripts/run_agent.py --text "my iPhone battery is dying really fast after the update"

# Example 2: Account security issue (Escalated to human agent)
python scripts/run_agent.py --text "I forgot my Apple ID password and my account is locked"

# Example 3: High-risk keyword (Escalated immediately)
python scripts/run_agent.py --text "someone hacked my account and made unauthorized charges"

# Run without LLM API dependency
python scripts/run_agent.py --text "how do I backup my photos to icloud" --no-llm
```

### CLI Output Example
```text
INTENT: device_issue
CONFIDENCE: 0.998
AUTO-HANDLE: True
REASON: Auto-handle: confident intent prediction and sufficiently similar historical evidence.

REPLY:
 Thanks for reaching out. Based on how similar cases were handled, the relevant support path is: @144549 We'd be glad to assist! What iOS version is currently running on your device?

EVIDENCE:
- 0.361 @AppleSupport my battery dies in 2 hours => @180074 Are you using a beta version? Let us know in DM: https://t.co/GDrqU22YpT
- 0.319 Okay but why has my iPhone 6 battery been dying 500x faster since installing IOS11?? => @134140 Send us a DM so we can look into your battery issue.
```

---

## Repository Structure

```text
hiver_sde_intern/
├── configs/
│   └── config.json               # System hyperparameters and threshold settings
├── data/
│   ├── golden/
│   │   ├── golden_set.csv        # Hand-labelled 200-example golden test set
│   │   └── README.txt            # Annotation metadata
│   ├── processed/
│   │   └── .gitkeep              # Placeholder (pairs.csv generated here)
│   └── raw/
│       └── .gitkeep              # Placeholder (place twcs.csv here)
├── reports/
│   ├── annotation_guidelines.md  # Annotation criteria for human raters
│   ├── report_template.md        # Filled assignment report template
│   └── final_report.md           # Comprehensive evaluation & architecture report
├── scripts/
│   ├── agreement_analysis.py     # Spearman, Cohen's kappa, auto-handle agreement
│   ├── build_golden.py           # Stratified candidate sampling
│   ├── edge_case_test.py         # 44 robustness edge case tests
│   ├── evaluate.py               # Macro-F1, accuracy, classification reports
│   ├── failure_analysis.py       # Top-5 failure modes with root-cause analysis
│   ├── judge.py                  # LLM-as-judge & rubric evaluation engine
│   ├── prepare_data.py           # Pair reconstruction & cleaning from TWCS
│   ├── run_agent.py              # Interactive CLI for testing customer queries
│   ├── smoke_test.py             # Quick sanity check
│   └── train.py                  # Trains calibrated Linear SVM classifier
├── src/
│   ├── agent.py                  # SupportAgent orchestrator
│   ├── data.py                   # Data parsing and conversation thread linking
│   ├── model.py                  # Char n-gram Pipeline & CalibratedClassifierCV
│   ├── policy.py                 # Multi-layer safety & escalation rules
│   ├── reply.py                  # Evidence-constrained reply generator
│   ├── retrieval.py              # Sublinear TF-IDF cosine retriever
│   └── taxonomy.py               # 9-intent taxonomy & keyword heuristics
├── artifacts/                    # Benchmark metrics, predictions, and judge outputs
├── CITATIONS.md                  # Dataset and framework references
├── DECISION_LOG.md               # 15 key architectural and design decisions
├── REPORT.md                     # Root copy of the comprehensive evaluation report
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Project documentation and reproduction guide
```

---

## Submission Checklist

- [x] Public/private GitHub repo (`https://github.com/Vikas-08J/Hiver_assesment.git`)
- [x] `README.md` with complete architecture and reproduction guide
- [x] Reproducible pipeline with random seed `42`
- [x] `data/golden/golden_set.csv` with 200 **human-labelled** examples
- [x] Evaluation metrics (Majority, TF-IDF+LR, Proposed Calibrated SVM)
- [x] Judge/human agreement metrics (Spearman $\rho$, quadratic weighted $\kappa$, MAE)
- [x] Comprehensive report $\le 6$ pages (`reports/final_report.md` & `REPORT.md`)
- [x] Top 5 real failure modes with root-cause hypotheses and mitigations
- [x] Empirical "What is misleading about the headline number?" section
- [x] Decision log with 15 architectural decisions (`DECISION_LOG.md`)
- [x] Citations (`CITATIONS.md`)
- [x] `.env` not committed (safeguarded via `.gitignore`)
- [x] No raw full dataset committed (`twcs.csv` ignored via `.gitignore`)
