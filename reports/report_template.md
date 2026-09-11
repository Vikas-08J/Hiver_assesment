# AI Support Agent — Evaluation Report

## 1. Executive summary

**Brand:** AppleSupport  
**Dataset:** Customer Support on Twitter  
**Golden set:** N = 200 human-labelled examples  

Headline result:

> Proposed system macro-F1: 0.760 vs TF-IDF+LR 0.708 vs majority 0.069. (Accuracy: 88.5% vs 80.5% vs 38.0%).

Reply quality:

> Human mean: 3.365 / 5 (N=200), 3.333 / 5 (N=30); LLM judge mean: 3.267 / 5; weighted κ: 0.100; Spearman ρ: 0.131 (within ±1 agreement: 93.3%, exact agreement: 40.0%).

Auto-handle precision:

> 4.3% of auto-handled examples were judged safe by humans (1/23). In contrast, Escalate Precision was 100.0% (7/7).

## 2. Problem framing

The goal is not to replace customer-support agents. The goal is to automate low-risk, repetitive requests while preserving an auditable path to a human.

### What good means

1. Correct intent.
2. Reply grounded in historical support behavior.
3. No unsupported policy/product/account claims.
4. Conservative escalation when confidence or evidence is weak.

### What we deliberately did not build

- no live Apple account access;
- no real order lookup;
- no payments/refunds execution;
- no autonomous policy invention;
- no production Twitter integration;
- no claim that historical responses are current policy.

## 3. Data and methodology

- Extracted 20,000 clean customer-to-response pairs for AppleSupport from `data/raw/twcs.csv`.
- Defined a 9-intent operational taxonomy (`account_access`, `billing_payment`, `order_purchase`, `device_issue`, `app_service_issue`, `subscription_cancel`, `technical_howto`, `status_followup`, `other`).
- Sampled 200 held-out examples via stratified weak-label allocation (`scripts/build_golden.py`).
- Hand-labelled `human_intent`, `human_auto_handle`, and `human_reply_score` independently of predictions. Golden examples were strictly excluded from training and retrieval indices to prevent data leakage.

## 4. Baselines and results

| System | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---|
| Majority | 0.380 | 0.069 | trivial majority class (`other`) |
| TF-IDF + Logistic Regression | 0.805 | 0.708 | word/char n-gram supervised baseline |
| Proposed hybrid | **0.885** | **0.760** | Char n-gram (3-5) + Calibrated Linear SVM + Cosine Retrieval + Escalation Policy |

### Proposed System Per-Intent Report

| Intent | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `account_access` | 1.00 | 0.80 | 0.89 | 5 |
| `app_service_issue` | 1.00 | 0.56 | 0.72 | 16 |
| `billing_payment` | 0.50 | 1.00 | 0.67 | 3 |
| `device_issue` | 0.97 | 0.91 | 0.94 | 65 |
| `order_purchase` | 0.45 | 0.83 | 0.59 | 6 |
| `other` | 0.90 | 1.00 | 0.95 | 76 |
| `status_followup` | 0.86 | 0.95 | 0.90 | 19 |
| `technical_howto` | 0.75 | 0.30 | 0.43 | 10 |
| **Macro Average** | **0.80** | **0.79** | **0.76** | **200** |
| **Weighted Average** | **0.90** | **0.89** | **0.88** | **200** |

## 5. Reply evaluation

Evaluated across N = 30 representative held-out golden instances:

- **Human reply mean:** 3.333 / 5
- **Judge overall score mean:** 3.267 / 5 (Correctness: 3.800, Grounding: 3.800, Actionability: 3.500, Tone: 4.000, Hallucination Risk: 1.000, Safety: 4.233)
- **Judge/Human weighted Cohen's kappa (quadratic):** 0.100
- **Spearman correlation:** 0.131 (p = 0.490)
- **Exact Agreement:** 40.0%
- **Within ±1 Agreement:** 93.3%
- **Mean Absolute Error (MAE):** 0.667 | **RMSE:** 0.894
- **Escalate Precision:** 100.0% (7/7 true escalations)
- **Auto-handle Precision:** 4.3% (1/23 true auto-handles; humans strongly preferred escalation for DM inquiries)

## 6. Top 5 failure modes

1. **Over-Triggering Escalation on Intent Ambiguity (Tweet 1590551):** Customer restored from iTunes but apps didn't download. Probability split across `app_service_issue`, `technical_howto`, and `device_issue`, causing confidence to drop to 0.566 (< 0.68).
2. **Keyword Hijacking Leading to Misclassification (Tweet 53553):** Customer mentions wiping backup and passcode; word "backup" triggers `order_purchase` feature weights instead of `device_issue`.
3. **Inquiry Complexity Disconnect (Tweet 2732930):** Customer asks whether to set up phone as new or restore. System auto-handled (conf=0.979, ret=0.276), but human rater marked escalate because contextual backup state is needed.
4. **Colloquial Slang Retrieval Mismatch (Tweet 1906030):** "yo iOS 11 is sooooo buggy... rendering it useless" failed retrieval similarity threshold (0.176 < 0.20) because historical responses used formal terms ("unexpected restart", "freezing").
5. **Non-English Language Leakage (Tweet 884993):** Spanish query about lost iPhone 6 auto-handled with standard Spanish link, but lost device mode requires human tier-2 security routing.

## 7. What is misleading about my headline number?

1. **Class Imbalance:** `other` (76) and `device_issue` (65) constitute 70.5% of the golden set. High accuracy (88.5%) masks low recall on `technical_howto` (30%) and low precision on `order_purchase` (45%).
2. **Lexical Leakage & Formulaic Twitter Cliches:** Repetitive Twitter support patterns ("Settings > General > About", "send us a DM") inflate retrieval similarity without testing real semantic reasoning.
3. **Classification Accuracy $\ne$ Operational Safety:** A model can predict intent with 99% accuracy but still give an unsafe auto-handle reply to a customer reporting account theft.
4. **Historical Corpus $\ne$ Current Policy:** Responses from 2017 link to deprecated iOS 11 patches and dead `t.co` URLs. Grounded retrieval on stale data yields functionally obsolete guidance.

## 8. One more week

1. **Dense Semantic Retrieval:** Integrate `all-MiniLM-L6-v2` dense vectors with BM25 via Reciprocal Rank Fusion to resolve colloquial slang mismatches.
2. **Temporal Holdout Split:** Split train and test strictly chronologically to evaluate true out-of-distribution drift.
3. **Conformal Prediction:** Replace static confidence thresholds (0.68) with conformal prediction guaranteeing statistical coverage bounds.
4. **Policy Freshness Engine:** Strip dead `t.co` links and obsolete iOS versions before synthesis.
5. **Operator Triage Dashboard:** Build an interactive human-in-the-loop review queue for flagged edge cases.
6. **Multi-Turn Context:** Assemble parent tweet threads for conversational continuity.
7. **Expanded Multi-Annotator Golden Set:** Scale to N = 500 with Fleiss' kappa inter-annotator validation.

## 9. Reproducibility

- **Commands:** `python scripts/evaluate.py`, `python scripts/judge.py --n 30 --rubric`, `python scripts/agreement_analysis.py`, `python scripts/failure_analysis.py`
- **Python Version:** 3.10.11
- **Random Seed:** 42 across all sampling, stratification, and model training
- **Dataset:** 20,000 pairs sampled from ThoughtVector `twcs.csv`
- **Models:** Character n-gram Calibrated LinearSVC; TF-IDF cosine similarity retriever
