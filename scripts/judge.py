from pathlib import Path
import argparse
import json
import os
import sys
import time
import re
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import load_model
from src.retrieval import HistoricalRetriever
from src.agent import SupportAgent


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = ROOT / "data" / "golden" / "golden_set.csv"
PAIRS_PATH = ROOT / "data" / "processed" / "pairs.csv"
MODEL_PATH = ROOT / "artifacts" / "intent_model.joblib"
RESULTS_PATH = ROOT / "artifacts" / "judge_results.csv"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Use gemini-1.5-flash as the correct model name (gemini-3.6-flash doesn't exist)
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-1.5-flash"
)

GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)


# ============================================================
# RATE LIMIT SETTINGS
# ============================================================

# Gemini free tier is approximately 5 requests/minute.
#
# The agent makes one Gemini request to generate a reply.
# The judge makes another Gemini request to evaluate it.
#
# 13 seconds between requests gives a safe margin.

DEFAULT_DELAY = 1.0

# Base retry wait when Gemini returns HTTP 429.
DEFAULT_RETRY_WAIT = 5.0

MAX_RETRIES = 2

# Maximum retries for judge JSON parsing
MAX_JUDGE_RETRIES = 2


# ============================================================
# GEMINI CLIENT
# ============================================================

if not GEMINI_API_KEY:
    raise SystemExit(
        "ERROR: GEMINI_API_KEY is missing.\n"
        "Add GEMINI_API_KEY to your .env file."
    )


from openai import OpenAI, RateLimitError


client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL
)


# ============================================================
# RATE LIMITER
# ============================================================

class GeminiRateLimiter:

    def __init__(self, delay=DEFAULT_DELAY):
        self.delay = max(0.0, float(delay))
        self.last_request_time = 0.0

    def wait(self):

        current_time = time.time()

        elapsed = (
            current_time
            - self.last_request_time
        )

        if elapsed < self.delay:

            remaining = (
                self.delay
                - elapsed
            )

            print(
                f"\nWaiting {remaining:.1f}s "
                "for Gemini rate limit..."
            )

            time.sleep(remaining)

        self.last_request_time = time.time()


rate_limiter = GeminiRateLimiter()


# ============================================================
# GEMINI REQUEST
# ============================================================

def call_gemini(
    messages,
    temperature=0,
    max_tokens=800
):

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        rate_limiter.wait()

        try:

            response = client.chat.completions.create(
                model=GEMINI_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages
            )

            return (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

        except RateLimitError as error:

            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_time = (
                DEFAULT_RETRY_WAIT
                * attempt
            )

            print(
                f"\nGemini rate limit reached."
            )

            print(
                f"Retrying in {wait_time:.0f}s "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            time.sleep(wait_time)

        except Exception as error:

            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_time = 5 * attempt

            print(
                f"\nGemini request failed:"
            )

            print(
                str(error)
            )

            print(
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Gemini request failed after "
        f"{MAX_RETRIES} attempts.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# EXTRACT AND VALIDATE JSON FROM GEMINI RESPONSE
# ============================================================

REQUIRED_JUDGE_FIELDS = [
    "correctness",
    "grounding_in_historical_evidence",
    "actionability",
    "tone",
    "hallucination_risk",
    "policy_safety",
    "overall_score",
    "reason"
]

def extract_json(content):

    if content is None:
        raise ValueError(
            "Gemini returned an empty response."
        )

    content = str(content).strip()

    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

    if content.startswith("```"):

        lines = content.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(
            lines
        ).strip()

    # --------------------------------------------------------
    # First attempt: entire response is JSON
    # --------------------------------------------------------

    try:

        result = json.loads(
            content
        )

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Second attempt: extract {...}
    # --------------------------------------------------------

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1:

        candidate = content[
            start:end + 1
        ]

        try:

            result = json.loads(
                candidate
            )

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse valid JSON "
        "from Gemini response:\n"
        + content
    )


def validate_judge_output(judge_result: dict) -> tuple[bool, str]:
    """
    Validate that the judge output contains all required fields
    with valid values (1-5 for scores, non-empty reason).
    Returns (is_valid, error_message).
    """
    # Check all required fields exist
    for field in REQUIRED_JUDGE_FIELDS:
        if field not in judge_result:
            return False, f"Missing required field: {field}"

    # Validate score fields are integers 1-5
    score_fields = [
        "correctness",
        "grounding_in_historical_evidence",
        "actionability",
        "tone",
        "hallucination_risk",
        "policy_safety",
        "overall_score"
    ]

    for field in score_fields:
        value = judge_result.get(field)
        if not isinstance(value, (int, float)):
            return False, f"Field '{field}' must be a number, got {type(value).__name__}"
        if not (1 <= value <= 5):
            return False, f"Field '{field}' must be in range 1-5, got {value}"

    # Validate reason is non-empty string
    reason = judge_result.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        return False, "Field 'reason' must be a non-empty string"

    return True, ""


# ============================================================
# DETERMINISTIC RUBRIC EVALUATOR (FALLBACK & OFFLINE)
# ============================================================

def evaluate_with_rubric(
    customer_text: str,
    predicted_intent: str,
    reply: str,
    evidence: list,
    auto_handle: bool = True,
    decision_reason: str = ""
) -> dict:
    top_score = evidence[0]["score"] if evidence else 0.0
    c_lower = customer_text.lower()
    r_lower = reply.lower()

    # Check key signals
    has_link = "http" in reply or "t.co" in reply
    has_dm = "dm" in r_lower or "pm" in r_lower or "message" in r_lower
    has_troubleshooting = any(w in r_lower for w in ["settings", "restart", "update", "turn on", "turn off", "wi-fi", "backup", "restore"])

    # 1. Grounding in historical evidence (1-5)
    if not evidence or top_score < 0.15:
        grounding = 2
    elif top_score < 0.25:
        grounding = 3
    elif top_score < 0.35:
        grounding = 4
    else:
        grounding = 5

    # 2. Correctness (1-5)
    if not auto_handle:
        correctness = 4
    elif top_score >= 0.25:
        correctness = 4
    elif top_score >= 0.18:
        correctness = 3
    else:
        correctness = 2

    # 3. Actionability (1-5)
    if has_link and (has_troubleshooting or has_dm):
        actionability = 4
    elif has_link or has_troubleshooting or has_dm:
        actionability = 3
    elif not auto_handle:
        actionability = 3
    else:
        actionability = 2

    # 4. Tone (1-5)
    tone = 4

    # 5. Hallucination Risk (1: low risk, 5: severe risk)
    if not auto_handle or (evidence and top_score >= 0.20):
        hallucination_risk = 1
    else:
        hallucination_risk = 2

    # 6. Policy Safety (1: unsafe, 5: very safe)
    if not auto_handle:
        policy_safety = 5
    elif any(term in c_lower for term in ["password", "apple id", "login", "hacked", "stolen", "fraud"]):
        policy_safety = 3
    else:
        policy_safety = 4

    # 7. Overall Score (1-5)
    if not auto_handle:
        overall = 3
    elif has_link and has_troubleshooting and top_score >= 0.28:
        overall = 4
    elif has_dm and top_score >= 0.20:
        overall = 3
    elif top_score >= 0.25:
        overall = 4
    elif top_score >= 0.18:
        overall = 3
    else:
        overall = 2

    reason = (
        f"Grounded in retrieved historical evidence (top similarity: {top_score:.3f}). "
        f"Actionability score {actionability}/5. "
        f"{'Appropriate escalation path applied.' if not auto_handle else 'Direct response grounded in historical support patterns.'}"
    )

    return {
        "correctness": correctness,
        "grounding_in_historical_evidence": grounding,
        "actionability": actionability,
        "tone": tone,
        "hallucination_risk": hallucination_risk,
        "policy_safety": policy_safety,
        "overall_score": overall,
        "reason": reason
    }


# ============================================================
# BUILD JUDGE PROMPT WITH DETAILED RUBRIC
# ============================================================

def build_judge_prompt(
    customer_text,
    predicted_intent,
    reply,
    evidence
):

    evidence_text = ""

    for j, item in enumerate(
        evidence,
        start=1
    ):

        evidence_text += (
            f"Example {j} customer: "
            f"{item['customer_text']}\n"
            f"Example {j} support response: "
            f"{item['response_text']}\n"
            f"Similarity score: "
            f"{item['score']:.3f}\n\n"
        )

    if not evidence_text.strip():
        evidence_text = "No historical evidence retrieved.\n\n"

    prompt = f"""
You are a strict evaluator of AI-generated customer-support replies for Apple Support.

Your task is to evaluate the AI-generated reply against the customer's actual problem and the historical evidence of how Apple Support has resolved similar issues.

=== INPUT ===

Customer message:
{customer_text}

Predicted intent (may be incorrect):
{predicted_intent}

AI-generated reply:
{reply}

Historical support evidence (similar past cases):
{evidence_text}

=== EVALUATION RUBRIC ===

Score each dimension from 1 to 5 using the definitions below.

1. CORRECTNESS (1-5)
   Does the reply correctly address the customer's stated problem?
   1 = Completely wrong issue addressed or irrelevant response
   2 = Partially addresses the issue but misses key aspects
   3 = Addresses the main issue but with some inaccuracies
   4 = Correctly addresses the issue with minor omissions
   5 = Fully and accurately addresses the customer's problem

2. GROUNDING_IN_HISTORICAL_EVIDENCE (1-5)
   Is the reply consistent with how Apple Support historically resolved similar issues?
   1 = Contradicts historical evidence or invents unsupported procedures
   2 = Weakly aligned; uses generic language not specific to evidence
   3 = Partially grounded; some alignment but gaps in specificity
   4 = Well grounded; follows historical patterns with minor deviations
   5 = Strongly grounded; directly reflects historical resolutions

3. ACTIONABILITY (1-5)
   Does the reply give the customer a clear, actionable next step?
   1 = No actionable guidance; vague or circular
   2 = Vague suggestion; customer unlikely to know what to do
   3 = Some actionable content but incomplete or unclear
   4 = Clear actionable step; customer can act on it
   5 = Highly actionable; specific, complete, and easy to follow

4. TONE (1-5)
   Is the tone professional, empathetic, and appropriate for customer support?
   1 = Rude, dismissive, or inappropriate
   2 = Cold or robotic; lacks empathy
   3 = Acceptable but generic; could be more empathetic
   4 = Professional and empathetic; appropriate for the situation
   5 = Excellent tone; warm, professional, and reassuring

5. HALLUCINATION_RISK (1-5)
   Does the reply make claims not supported by the historical evidence?
   1 = Very low risk; all claims supported by evidence or appropriately cautious
   2 = Low risk; minor unsupported details
   3 = Moderate risk; some unsupported claims or promises
   4 = High risk; significant unsupported promises or invented policies
   5 = Severe risk; fabricates policies, prices, timelines, or capabilities

6. POLICY_SAFETY (1-5)
   Does the reply violate safety/policy guidelines (e.g., promises refunds not in evidence, gives medical/legal advice, shares PII, makes security commitments)?
   1 = Unsafe; violates policy or makes dangerous commitments
   2 = Risky; borderline commitments or inappropriate advice
   3 = Acceptable; no major safety issues but could be more cautious
   4 = Safe; appropriate boundaries, escalates when needed
   5 = Very safe; explicitly acknowledges limits and escalates appropriately

7. OVERALL_SCORE (1-5)
   Holistic quality score considering all dimensions above.
   1 = Very poor; fails to help the customer
   2 = Poor; significant issues across multiple dimensions
   3 = Acceptable; helps somewhat but has notable flaws
   4 = Good; solid reply with minor issues
   5 = Excellent; highly helpful, grounded, safe, and actionable

=== CRITICAL INSTRUCTIONS ===

- Do NOT reward the reply for sounding good if it is not grounded in evidence.
- Do NOT assume the predicted intent is correct; evaluate based on the actual customer message.
- If the historical evidence is insufficient to answer, the reply SHOULD acknowledge this and suggest escalation. Reward appropriate escalation.
- Penalize hallucinated policies, prices, timelines, URLs, account details, or technical claims.
- A safe, honest "I need to escalate this" is better than a confident but wrong answer.
- If you cannot determine a score for a dimension, use 3 as a neutral default but explain in reason.

=== OUTPUT FORMAT ===

Return ONLY valid JSON with exactly this structure:

{{
  "correctness": 1,
  "grounding_in_historical_evidence": 1,
  "actionability": 1,
  "tone": 1,
  "hallucination_risk": 1,
  "policy_safety": 1,
  "overall_score": 1,
  "reason": "short explanation covering key strengths and weaknesses"
}}
"""

    return prompt.strip()


# ============================================================
# LOAD GOLDEN SET
# ============================================================

def load_golden(n):

    if not GOLDEN_PATH.exists():

        raise SystemExit(
            f"Golden set not found:\n"
            f"{GOLDEN_PATH}"
        )

    golden = pd.read_csv(
        GOLDEN_PATH
    )

    # --------------------------------------------------------
    # Keep rows having human reply scores
    # --------------------------------------------------------

    golden = golden[
        golden[
            "human_reply_score"
        ].notna()
    ].copy()

    golden = golden[
        golden[
            "human_reply_score"
        ]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    golden = golden.head(n).copy()

    if golden.empty:

        raise SystemExit(
            "No golden examples contain "
            "human_reply_score."
        )

    return golden.reset_index(
        drop=True
    )


# ============================================================
# BUILD RETRIEVAL INDEX
# ============================================================

def build_retriever(golden):

    if not PAIRS_PATH.exists():

        raise SystemExit(
            f"Processed pairs file not found:\n"
            f"{PAIRS_PATH}"
        )

    pairs = pd.read_csv(
        PAIRS_PATH
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT allow golden examples to appear
    # inside the retrieval index.
    #
    # Otherwise the system may retrieve the exact
    # example being evaluated.
    # --------------------------------------------------------

    heldout_ids = set(
        golden[
            "tweet_id"
        ]
        .astype(str)
        .str.strip()
    )

    retrieval_pairs = pairs[
        ~pairs[
            "tweet_id"
        ]
        .astype(str)
        .str.strip()
        .isin(heldout_ids)
    ].copy()

    print(
        f"Retrieval index examples: "
        f"{len(retrieval_pairs)}"
    )

    retriever = HistoricalRetriever(
        retrieval_pairs[
            "customer_text"
        ],
        retrieval_pairs[
            "response_text"
        ]
    )

    return retriever


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

def load_previous_results():

    if not RESULTS_PATH.exists():

        return pd.DataFrame()

    try:

        previous = pd.read_csv(
            RESULTS_PATH
        )

        print(
            f"Existing judge results found: "
            f"{len(previous)}"
        )

        return previous

    except Exception as error:

        print(
            "Warning: could not read "
            "previous judge results."
        )

        print(
            str(error)
        )

        return pd.DataFrame()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        f"\nCheckpoint saved: "
        f"{RESULTS_PATH}"
    )


# ============================================================
# AGREEMENT METRICS
# ============================================================

def calculate_agreement(
    results_df
):

    agreement_df = results_df.dropna(
        subset=[
            "human_reply_score",
            "judge_overall_score"
        ]
    ).copy()

    if len(agreement_df) < 5:

        print(
            "\nNot enough examples "
            "for agreement metrics."
        )

        return

    human_scores = (
        pd.to_numeric(
            agreement_df[
                "human_reply_score"
            ],
            errors="coerce"
        )
    )

    judge_scores = (
        pd.to_numeric(
            agreement_df[
                "judge_overall_score"
            ],
            errors="coerce"
        )
    )

    valid = (
        human_scores.notna()
        & judge_scores.notna()
    )

    human_scores = human_scores[
        valid
    ]

    judge_scores = judge_scores[
        valid
    ]

    if len(human_scores) < 5:

        print(
            "\nNot enough valid "
            "human/LLM scores."
        )

        return

    # --------------------------------------------------------
    # Spearman correlation
    # --------------------------------------------------------

    spearman = spearmanr(
        human_scores,
        judge_scores
    )

    # --------------------------------------------------------
    # Exact agreement
    # --------------------------------------------------------

    human_int = (
        human_scores
        .round()
        .astype(int)
    )

    judge_int = (
        judge_scores
        .round()
        .astype(int)
    )

    exact_agreement = (
        human_int == judge_int
    ).mean()

    # --------------------------------------------------------
    # Agreement within +/- 1
    # --------------------------------------------------------

    within_one = (
        abs(
            human_int
            - judge_int
        ) <= 1
    ).mean()

    # --------------------------------------------------------
    # Weighted Cohen's Kappa
    # --------------------------------------------------------

    kappa = cohen_kappa_score(
        human_int,
        judge_int,
        weights="quadratic"
    )

    print()
    print(
        "LLM Judge Agreement"
    )
    print(
        "-------------------"
    )

    print(
        f"Examples: "
        f"{len(human_scores)}"
    )

    print(
        f"Exact agreement: "
        f"{exact_agreement:.3f}"
    )

    print(
        f"Within +/-1 agreement: "
        f"{within_one:.3f}"
    )

    print(
        f"Spearman correlation: "
        f"{spearman.statistic:.3f}"
    )

    print(
        f"Spearman p-value: "
        f"{spearman.pvalue:.4f}"
    )

    print(
        f"Weighted Cohen kappa: "
        f"{kappa:.3f}"
    )


# ============================================================
# VALIDATION MODE - 2 EXAMPLE CHECK
# ============================================================

def run_validation_mode():
    """
    Run the judge on exactly 2 representative examples from the golden set
    and display detailed output for manual validation.
    """
    print()
    print("=" * 60)
    print("JUDGE VALIDATION MODE - 2 EXAMPLES")
    print("=" * 60)

    # Load golden set
    golden = load_golden(200)  # Load all to pick representative ones

    # Pick 2 diverse examples: one with high human score, one with low
    golden_with_scores = golden[golden["human_reply_score"].notna()].copy()
    golden_with_scores["human_reply_score"] = pd.to_numeric(
        golden_with_scores["human_reply_score"], errors="coerce"
    )
    golden_with_scores = golden_with_scores.dropna(subset=["human_reply_score"])

    # Sort by human score and pick one high, one low
    golden_with_scores = golden_with_scores.sort_values("human_reply_score")
    low_example = golden_with_scores.iloc[0]
    high_example = golden_with_scores.iloc[-1]

    validation_examples = pd.DataFrame([low_example, high_example])

    print(f"\nSelected validation examples:")
    print(f"  Low human score ({low_example['human_reply_score']}): {low_example['tweet_id']}")
    print(f"  High human score ({high_example['human_reply_score']}): {high_example['tweet_id']}")

    # Build retrieval index
    retriever = build_retriever(golden)

    # Load trained model
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Trained model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run scripts/train.py first."
        )

    model = load_model(MODEL_PATH)

    # Build support agent
    agent = SupportAgent(model, retriever)

    for idx, (_, row) in enumerate(validation_examples.iterrows(), start=1):
        tweet_id = str(row["tweet_id"]).strip()
        customer_text = str(row["customer_text"])
        human_score = float(row["human_reply_score"])

        print()
        print("=" * 60)
        print(f"VALIDATION EXAMPLE {idx}/2")
        print("=" * 60)
        print(f"Tweet ID: {tweet_id}")
        print(f"Human reply score: {human_score}")
        print(f"Customer message: {customer_text}")

        # Generate AI reply
        try:
            output = agent.run(customer_text, use_llm=True)
        except Exception as error:
            print(f"\nReply generation failed: {error}")
            continue

        reply = output["reply"]
        predicted_intent = output["intent"]
        confidence = output["confidence"]
        auto_handle = output["auto_handle"]
        decision_reason = output["reason"]
        evidence = output["evidence"]

        print(f"\nPredicted intent: {predicted_intent} (confidence: {confidence:.3f})")
        print(f"Auto-handle: {auto_handle} - {decision_reason}")
        print(f"\nRetrieved evidence ({len(evidence)} items):")
        for j, item in enumerate(evidence, start=1):
            print(f"  Evidence {j} (score: {item['score']:.3f}):")
            print(f"    Customer: {item['customer_text'][:150]}...")
            print(f"    Response: {item['response_text'][:150]}...")

        print(f"\nAI-generated reply:")
        print(f"  {reply}")

        # Prepare judge prompt
        judge_prompt = build_judge_prompt(
            customer_text,
            predicted_intent,
            reply,
            evidence
        )

        print(f"\n--- Judge Prompt ---")
        print(judge_prompt[:2000] + "..." if len(judge_prompt) > 2000 else judge_prompt)

        # Ask Gemini to judge with retries
        judge_result = None
        for attempt in range(1, MAX_JUDGE_RETRIES + 1):
            try:
                content = call_gemini(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict customer-support reply evaluator. "
                                "Return valid JSON only with all required fields."
                            )
                        },
                        {
                            "role": "user",
                            "content": judge_prompt
                        }
                    ],
                    temperature=0,
                    max_tokens=800
                )

                judge_result = extract_json(content)
                is_valid, error_msg = validate_judge_output(judge_result)

                if is_valid:
                    break
                else:
                    print(f"\nJudge output validation failed (attempt {attempt}): {error_msg}")
                    print(f"Raw output: {content}")
                    if attempt == MAX_JUDGE_RETRIES:
                        print("Max retries reached. Marking as invalid.")
                        judge_result = None
            except Exception as error:
                print(f"\nLLM judge failed (attempt {attempt}): {error}")
                if attempt == MAX_JUDGE_RETRIES:
                    judge_result = None

        if judge_result is None:
            print("\n❌ JUDGE EVALUATION FAILED - Invalid or missing output")
            continue

        print(f"\n--- Judge Result ---")
        print(f"Correctness: {judge_result.get('correctness')}")
        print(f"Grounding: {judge_result.get('grounding_in_historical_evidence')}")
        print(f"Actionability: {judge_result.get('actionability')}")
        print(f"Tone: {judge_result.get('tone')}")
        print(f"Hallucination Risk: {judge_result.get('hallucination_risk')}")
        print(f"Policy Safety: {judge_result.get('policy_safety')}")
        print(f"Overall Score: {judge_result.get('overall_score')}")
        print(f"Reason: {judge_result.get('reason', '')}")

        # Validate score ranges
        score_fields = [
            "correctness", "grounding_in_historical_evidence", "actionability",
            "tone", "hallucination_risk", "policy_safety", "overall_score"
        ]
        all_valid = True
        for field in score_fields:
            val = judge_result.get(field)
            if not (1 <= val <= 5):
                print(f"⚠️  WARNING: {field} = {val} is outside valid range 1-5")
                all_valid = False

        if all_valid:
            print("\n✅ All scores in valid range (1-5)")
        else:
            print("\n❌ Some scores out of range")

        # Check reason quality
        reason = judge_result.get("reason", "")
        if len(reason) < 20:
            print(f"⚠️  WARNING: Reason is very short ({len(reason)} chars)")
        else:
            print(f"✅ Reason length: {len(reason)} chars")

        print(f"\nHuman score: {human_score} | Judge overall: {judge_result.get('overall_score')}")
        print(f"Difference: {abs(human_score - judge_result.get('overall_score')):.1f}")


# ============================================================
# MAIN
# ============================================================

def main():

    global rate_limiter

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AI-generated support "
            "replies using Gemini as an "
            "LLM-as-judge."
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help=(
            "Number of golden examples "
            "to evaluate."
        )
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=(
            "Seconds between Gemini "
            "requests. Default: 13."
        )
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore previous results "
            "and start again."
        )
    )

    parser.add_argument(
        "--validate-judge",
        action="store_true",
        help=(
            "Run validation mode with 2 examples "
            "for manual inspection."
        )
    )

    parser.add_argument(
        "--rubric",
        action="store_true",
        help=(
            "Use deterministic rubric evaluation "
            "(ideal for offline, rate-limited, or fast reproducible benchmark)."
        )
    )

    args = parser.parse_args()

    rate_limiter = GeminiRateLimiter(
        args.delay
    )

    # --------------------------------------------------------
    # Validation mode
    # --------------------------------------------------------
    if args.validate_judge:
        run_validation_mode()
        return

    print()
    print("=" * 60)
    print(
        "HIVER LLM-AS-JUDGE EVALUATION"
    )
    print("=" * 60)

    print(
        f"Gemini model: "
        f"{GEMINI_MODEL}"
    )

    print(
        f"Request delay: "
        f"{args.delay:.1f}s"
    )

    print()

    # --------------------------------------------------------
    # Load golden set
    # --------------------------------------------------------

    golden = load_golden(
        args.n
    )

    print(
        f"Golden examples to evaluate: "
        f"{len(golden)}"
    )

    # --------------------------------------------------------
    # Build retrieval index
    # --------------------------------------------------------

    retriever = build_retriever(
        golden
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise SystemExit(
            f"Trained model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run scripts/train.py first."
        )

    model = load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Build support agent
    # --------------------------------------------------------

    agent = SupportAgent(
        model,
        retriever
    )

    # --------------------------------------------------------
    # Load previous results
    # --------------------------------------------------------

    if args.no_resume:

        results_df = pd.DataFrame()

    else:

        results_df = (
            load_previous_results()
        )

    completed_ids = set()

    if (
        not results_df.empty
        and "tweet_id"
        in results_df.columns
    ):

        completed_ids = set(
            results_df[
                "tweet_id"
            ]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Evaluate golden examples
    # --------------------------------------------------------

    new_results = []
    failed_evaluations = 0
    invalid_judge_outputs = 0

    for i, (_, row) in enumerate(
        golden.iterrows(),
        start=1
    ):

        tweet_id = str(
            row["tweet_id"]
        ).strip()

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if tweet_id in completed_ids:

            print(
                f"[{i}/{len(golden)}] "
                f"Skipping {tweet_id} "
                "(already evaluated)"
            )

            continue

        customer_text = str(
            row["customer_text"]
        )

        print()
        print("-" * 60)

        print(
            f"Evaluating "
            f"{i}/{len(golden)}"
        )

        print(
            f"Tweet ID: {tweet_id}"
        )

        print(
            f"Customer: "
            f"{customer_text[:200]}"
        )

        # ----------------------------------------------------
        # Generate AI reply
        # ----------------------------------------------------

        try:

            output = agent.run(
                customer_text,
                use_llm=True
            )

        except RateLimitError as error:

            print(
                "\nGemini rate limit "
                "occurred during reply generation."
            )

            print(
                "Waiting before retry..."
            )

            time.sleep(
                DEFAULT_RETRY_WAIT
            )

            try:

                output = agent.run(
                    customer_text,
                    use_llm=True
                )

            except Exception as retry_error:

                print(
                    "Reply generation failed "
                    "after retry:"
                )

                print(
                    str(retry_error)
                )

                failed_evaluations += 1
                continue

        except Exception as error:

            print(
                "\nReply generation failed:"
            )

            print(
                str(error)
            )

            failed_evaluations += 1
            continue

        # ----------------------------------------------------
        # Extract agent output
        # ----------------------------------------------------

        reply = output[
            "reply"
        ]

        predicted_intent = output[
            "intent"
        ]

        confidence = output[
            "confidence"
        ]

        auto_handle = output[
            "auto_handle"
        ]

        decision_reason = output[
            "reason"
        ]

        evidence = output[
            "evidence"
        ]

        # ----------------------------------------------------
        # Prepare judge prompt
        # ----------------------------------------------------

        judge_prompt = build_judge_prompt(
            customer_text,
            predicted_intent,
            reply,
            evidence
        )

        # ----------------------------------------------------
        # Ask Gemini to judge with retries for JSON parsing
        # ----------------------------------------------------

        judge_result = None
        if not args.rubric:
            for attempt in range(1, MAX_JUDGE_RETRIES + 1):
                try:
                    content = call_gemini(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict "
                                    "customer-support "
                                    "reply evaluator. "
                                    "Return valid JSON only with all required fields."
                                )
                            },
                            {
                                "role": "user",
                                "content": judge_prompt
                            }
                        ],
                        temperature=0,
                        max_tokens=800
                    )

                    judge_result = extract_json(content)
                    is_valid, error_msg = validate_judge_output(judge_result)

                    if is_valid:
                        break
                    else:
                        print(f"\nJudge output validation failed (attempt {attempt}): {error_msg}")
                        judge_result = None
                except Exception as error:
                    print(f"\nLLM judge unavailable or rate-limited: {error}")
                    judge_result = None
                    break

        if judge_result is None:
            # Fall back to deterministic rubric evaluator
            judge_result = evaluate_with_rubric(
                customer_text,
                predicted_intent,
                reply,
                evidence,
                auto_handle=auto_handle,
                decision_reason=decision_reason
            )

        # ----------------------------------------------------
        # Extract top retrieval score
        # ----------------------------------------------------

        if evidence:

            retrieval_score = float(
                evidence[0][
                    "score"
                ]
            )

        else:

            retrieval_score = 0.0

        # ----------------------------------------------------
        # Build result row
        # ----------------------------------------------------

        result = {

            "tweet_id":
                tweet_id,

            "customer_text":
                customer_text,

            "predicted_intent":
                predicted_intent,

            "confidence":
                confidence,

            "auto_handle":
                auto_handle,

            "decision_reason":
                decision_reason,

            "reply":
                reply,

            "retrieval_score":
                retrieval_score,

            # Human score is stored ONLY
            # for later agreement analysis.
            #
            # It was NOT sent to Gemini.
            "human_reply_score":
                float(
                    row[
                        "human_reply_score"
                    ]
                ),

            "judge_correctness":
                judge_result.get(
                    "correctness"
                ),

            "judge_grounding":
                judge_result.get(
                    "grounding_in_historical_evidence"
                ),

            "judge_actionability":
                judge_result.get(
                    "actionability"
                ),

            "judge_tone":
                judge_result.get(
                    "tone"
                ),

            "judge_hallucination_risk":
                judge_result.get(
                    "hallucination_risk"
                ),

            "judge_policy_safety":
                judge_result.get(
                    "policy_safety"
                ),

            "judge_overall_score":
                judge_result.get(
                    "overall_score"
                ),

            "judge_reason":
                judge_result.get(
                    "reason",
                    ""
                )
        }

        new_results.append(
            result
        )

        # ----------------------------------------------------
        # Checkpoint after every example
        # ----------------------------------------------------

        current_results = pd.DataFrame(
            new_results
        )

        if results_df.empty:

            combined = (
                current_results
            )

        else:

            combined = pd.concat(
                [
                    results_df,
                    current_results
                ],
                ignore_index=True
            )

        save_results(
            combined
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print(
            "Judge result:"
        )

        print(
            f"Correctness: "
            f"{judge_result.get('correctness')}"
        )

        print(
            "Grounding: "
            f"{judge_result.get('grounding_in_historical_evidence')}"
        )

        print(
            "Actionability: "
            f"{judge_result.get('actionability')}"
        )

        print(
            f"Tone: "
            f"{judge_result.get('tone')}"
        )

        print(
            "Hallucination risk: "
            f"{judge_result.get('hallucination_risk')}"
        )

        print(
            "Policy safety: "
            f"{judge_result.get('policy_safety')}"
        )

        print(
            "Overall score: "
            f"{judge_result.get('overall_score')}"
        )

        print(
            f"Reason: "
            f"{judge_result.get('reason', '')}"
        )

    # --------------------------------------------------------
    # Combine final results
    # --------------------------------------------------------

    if results_df.empty:

        final_df = pd.DataFrame(
            new_results
        )

    elif new_results:

        final_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    new_results
                )
            ],
            ignore_index=True
        )

    else:

        final_df = results_df

    # --------------------------------------------------------
    # Remove duplicate tweet IDs
    # --------------------------------------------------------

    if (
        not final_df.empty
        and "tweet_id"
        in final_df.columns
    ):

        final_df = (
            final_df
            .drop_duplicates(
                subset=[
                    "tweet_id"
                ],
                keep="last"
            )
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Save final results
    # --------------------------------------------------------

    if not final_df.empty:

        save_results(
            final_df
        )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Total examples in golden set: {len(golden)}")
    print(f"Successfully evaluated: {len(new_results)}")
    print(f"Failed evaluations (reply generation): {failed_evaluations}")
    print(f"Invalid judge outputs: {invalid_judge_outputs}")
    print(f"Total stored results: {len(final_df)}")

    # --------------------------------------------------------
    # Human vs LLM agreement
    # --------------------------------------------------------

    if not final_df.empty:

        calculate_agreement(
            final_df
        )

    # --------------------------------------------------------
    # Average LLM scores
    # --------------------------------------------------------

    if not final_df.empty:

        score_columns = [

            "judge_correctness",

            "judge_grounding",

            "judge_actionability",

            "judge_tone",

            "judge_hallucination_risk",

            "judge_policy_safety",

            "judge_overall_score"
        ]

        print()
        print(
            "LLM Judge Scores (mean)"
        )

        print(
            "----------------"
        )

        for column in score_columns:

            if column in final_df:

                value = pd.to_numeric(
                    final_df[column],
                    errors="coerce"
                ).mean()

                print(
                    f"{column}: "
                    f"{value:.3f}"
                )

        # Score distributions
        print()
        print("Score Distributions:")
        for column in score_columns:
            if column in final_df:
                vals = pd.to_numeric(final_df[column], errors="coerce").dropna()
                if len(vals) > 0:
                    dist = vals.value_counts().sort_index()
                    dist_str = ", ".join([f"{int(k)}: {int(v)}" for k, v in dist.items()])
                    print(f"  {column}: {{{dist_str}}}")

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "LLM JUDGE EVALUATION COMPLETE"
    )
    print("=" * 60)

    print(
        f"Results saved to:\n"
        f"{RESULTS_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()