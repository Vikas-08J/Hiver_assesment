"""
Failure Analysis Script (Task 21)
Identifies top 5 failure modes from the final evaluation.
"""
from pathlib import Path
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))

JUDGE_RESULTS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "judge_results.csv"
GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set.csv"
INTENT_PREDICTIONS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "intent_predictions.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "failure_analysis.json"


def load_data():
    """Load all evaluation data."""
    judge_results = pd.read_csv(JUDGE_RESULTS_PATH) if JUDGE_RESULTS_PATH.exists() else pd.DataFrame()
    golden = pd.read_csv(GOLDEN_PATH) if GOLDEN_PATH.exists() else pd.DataFrame()
    intent_preds = pd.read_csv(INTENT_PREDICTIONS_PATH) if INTENT_PREDICTIONS_PATH.exists() else pd.DataFrame()
    
    return judge_results, golden, intent_preds


def analyze_intent_failures(golden, intent_preds):
    """Analyze intent classification failures."""
    if intent_preds.empty or golden.empty:
        return []
    
    merged = golden.merge(intent_preds[["tweet_id", "pred_intent"]], on="tweet_id", how="left")
    merged = merged.dropna(subset=["human_intent", "pred_intent"])
    
    failures = merged[merged["human_intent"] != merged["pred_intent"]].copy()
    
    failure_modes = []
    
    # Group by true intent and predicted intent
    confusion = failures.groupby(["human_intent", "pred_intent"]).size().reset_index(name="count")
    confusion = confusion.sort_values("count", ascending=False)
    
    for _, row in confusion.head(10).iterrows():
        examples = failures[
            (failures["human_intent"] == row["human_intent"]) & 
            (failures["pred_intent"] == row["pred_intent"])
        ]
        
        failure_modes.append({
            "type": "intent_misclassification",
            "name": f"Misclassified {row['human_intent']} as {row['pred_intent']}",
            "count": int(row["count"]),
            "true_intent": row["human_intent"],
            "predicted_intent": row["pred_intent"],
            "examples": examples[["tweet_id", "customer_text", "human_intent", "pred_intent"]].head(3).to_dict("records")
        })
    
    return failure_modes


def analyze_reply_failures(judge_results, golden):
    """Analyze reply quality failures."""
    if judge_results.empty:
        return []
    
    # Merge with golden for missing human scores/labels
    cols_to_merge = [c for c in ["human_reply_score", "human_intent", "human_auto_handle"] if c not in judge_results.columns]
    if cols_to_merge:
        merged = judge_results.merge(
            golden[["tweet_id"] + cols_to_merge],
            on="tweet_id",
            how="left"
        )
    else:
        merged = judge_results.copy()
    
    merged = merged.dropna(subset=["judge_overall_score", "human_reply_score"])
    
    failure_modes = []
    
    # 1. Low judge score (poor reply quality)
    low_quality = merged[merged["judge_overall_score"] <= 2].copy()
    if len(low_quality) > 0:
        failure_modes.append({
            "type": "low_reply_quality",
            "name": "Low reply quality (judge score <= 2)",
            "count": len(low_quality),
            "examples": low_quality.nsmallest(3, "judge_overall_score")[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_overall_score", "judge_reason", "human_reply_score"]
            ].to_dict("records")
        })
    
    # 2. High hallucination risk
    high_hallucination = merged[merged["judge_hallucination_risk"] >= 4].copy()
    if len(high_hallucination) > 0:
        failure_modes.append({
            "type": "hallucination",
            "name": "High hallucination risk (score >= 4)",
            "count": len(high_hallucination),
            "examples": high_hallucination.nlargest(3, "judge_hallucination_risk")[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_hallucination_risk", "judge_reason"]
            ].to_dict("records")
        })
    
    # 3. Poor grounding
    poor_grounding = merged[merged["judge_grounding"] <= 2].copy()
    if len(poor_grounding) > 0:
        failure_modes.append({
            "type": "poor_grounding",
            "name": "Poor historical grounding (score <= 2)",
            "count": len(poor_grounding),
            "examples": poor_grounding.nsmallest(3, "judge_grounding")[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_grounding", "judge_reason", "retrieval_score"]
            ].to_dict("records")
        })
    
    # 4. Policy/safety violations
    policy_violations = merged[merged["judge_policy_safety"] <= 2].copy()
    if len(policy_violations) > 0:
        failure_modes.append({
            "type": "policy_safety",
            "name": "Policy/safety violation (score <= 2)",
            "count": len(policy_violations),
            "examples": policy_violations.nsmallest(3, "judge_policy_safety")[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_policy_safety", "judge_reason"]
            ].to_dict("records")
        })
    
    # 5. Judge-human disagreement (judge much higher than human)
    merged["score_diff"] = merged["judge_overall_score"] - merged["human_reply_score"]
    judge_overestimates = merged[merged["score_diff"] >= 2].copy()
    if len(judge_overestimates) > 0:
        failure_modes.append({
            "type": "judge_overestimation",
            "name": "Judge overestimates quality (judge >= human + 2)",
            "count": len(judge_overestimates),
            "examples": judge_overestimates.nlargest(3, "score_diff")[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "human_reply_score", "judge_overall_score", "judge_reason"]
            ].to_dict("records")
        })
    
    # 6. Judge-human disagreement (judge much lower than human)
    judge_underestimates = merged[merged["score_diff"] <= -2].copy()
    if len(judge_underestimates) > 0:
        failure_modes.append({
            "type": "judge_underestimation",
            "name": "Judge underestimates quality (judge <= human - 2)",
            "count": len(judge_underestimates),
            "examples": judge_underestimates.nsmallest(3, "score_diff")[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "human_reply_score", "judge_overall_score", "judge_reason"]
            ].to_dict("records")
        })
    
    return failure_modes


def analyze_auto_handle_failures(judge_results, golden):
    """Analyze auto-handle decision failures."""
    if judge_results.empty or golden.empty:
        return []
    
    merged = judge_results.merge(
        golden[["tweet_id", "human_auto_handle"]],
        on="tweet_id",
        how="left"
    )
    
    merged = merged.dropna(subset=["human_auto_handle", "auto_handle"])
    merged["human_auto_handle"] = merged["human_auto_handle"].map(
        lambda x: 1 if str(x).strip().lower() in ("yes", "1", "true") else 0
    )
    merged["auto_handle"] = merged["auto_handle"].map(
        lambda x: 1 if str(x).strip().lower() in ("yes", "1", "true") else 0
    )
    
    failure_modes = []
    
    # False auto-handle (UNSAFE - system says auto, human says escalate)
    false_auto = merged[(merged["human_auto_handle"] == 0) & (merged["auto_handle"] == 1)].copy()
    if len(false_auto) > 0:
        failure_modes.append({
            "type": "false_auto_handle",
            "name": "False auto-handle (UNSAFE: system auto, human escalate)",
            "count": len(false_auto),
            "examples": false_auto[
                ["tweet_id", "customer_text", "predicted_intent", "confidence", "decision_reason", "judge_overall_score", "judge_reason", "retrieval_score"]
            ].head(5).to_dict("records")
        })
    
    # False escalate (system says escalate, human says auto)
    false_escalate = merged[(merged["human_auto_handle"] == 1) & (merged["auto_handle"] == 0)].copy()
    if len(false_escalate) > 0:
        failure_modes.append({
            "type": "false_escalate",
            "name": "False escalate (system escalate, human auto)",
            "count": len(false_escalate),
            "examples": false_escalate[
                ["tweet_id", "customer_text", "predicted_intent", "confidence", "decision_reason", "judge_overall_score", "judge_reason", "retrieval_score"]
            ].head(5).to_dict("records")
        })
    
    return failure_modes


def analyze_edge_cases(judge_results, golden):
    """Analyze edge cases."""
    if judge_results.empty or golden.empty:
        return []
    
    merged = judge_results.merge(golden, on="tweet_id", how="left", suffixes=("", "_golden"))
    
    failure_modes = []
    
    # Short messages (< 20 chars)
    merged["msg_len"] = merged["customer_text"].str.len()
    short_msgs = merged[merged["msg_len"] < 20].copy()
    if len(short_msgs) > 0:
        failure_modes.append({
            "type": "short_message",
            "name": "Very short messages (< 20 chars)",
            "count": len(short_msgs),
            "examples": short_msgs.nsmallest(3, "msg_len")[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_overall_score", "auto_handle"]
            ].to_dict("records")
        })
    
    # Messages with URLs
    has_url = merged[merged["customer_text"].str.contains(r"https?://|t\.co/", na=False)].copy()
    if len(has_url) > 0:
        failure_modes.append({
            "type": "contains_url",
            "name": "Messages containing URLs",
            "count": len(has_url),
            "examples": has_url.head(3)[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_overall_score", "auto_handle"]
            ].to_dict("records")
        })
    
    # Messages with mentions
    has_mention = merged[merged["customer_text"].str.contains(r"@\w+", na=False)].copy()
    if len(has_mention) > 0:
        failure_modes.append({
            "type": "contains_mention",
            "name": "Messages containing @mentions",
            "count": len(has_mention),
            "examples": has_mention.head(3)[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_overall_score", "auto_handle"]
            ].to_dict("records")
        })
    
    # Messages with emojis
    has_emoji = merged[merged["customer_text"].str.contains(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", na=False)].copy()
    if len(has_emoji) > 0:
        failure_modes.append({
            "type": "contains_emoji",
            "name": "Messages containing emojis",
            "count": len(has_emoji),
            "examples": has_emoji.head(3)[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_overall_score", "auto_handle"]
            ].to_dict("records")
        })
    
    # Non-English messages (simple heuristic)
    non_english = merged[merged["customer_text"].str.contains(r"[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]", na=False)].copy()
    if len(non_english) > 0:
        failure_modes.append({
            "type": "non_english",
            "name": "Non-English messages",
            "count": len(non_english),
            "examples": non_english.head(3)[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_overall_score", "auto_handle"]
            ].to_dict("records")
        })
    
    # Low retrieval score (< 0.3)
    low_retrieval = merged[merged["retrieval_score"] < 0.3].copy()
    if len(low_retrieval) > 0:
        failure_modes.append({
            "type": "low_retrieval",
            "name": "Low retrieval score (< 0.3)",
            "count": len(low_retrieval),
            "examples": low_retrieval.nsmallest(3, "retrieval_score")[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_overall_score", "retrieval_score", "auto_handle"]
            ].to_dict("records")
        })
    
    # No evidence retrieved
    no_evidence = merged[merged["retrieval_score"] == 0].copy()
    if len(no_evidence) > 0:
        failure_modes.append({
            "type": "no_evidence",
            "name": "No historical evidence retrieved",
            "count": len(no_evidence),
            "examples": no_evidence.head(3)[
                ["tweet_id", "customer_text", "predicted_intent", "reply", "judge_overall_score", "auto_handle"]
            ].to_dict("records")
        })
    
    return failure_modes


def main():
    print("=" * 60)
    print("FAILURE ANALYSIS (Task 21)")
    print("=" * 60)
    
    judge_results, golden, intent_preds = load_data()
    print(f"Loaded {len(judge_results)} judge results")
    print(f"Loaded {len(golden)} golden examples")
    print(f"Loaded {len(intent_preds)} intent predictions")
    
    all_failures = []
    
    # Intent classification failures
    print("\n--- Intent Classification Failures ---")
    intent_failures = analyze_intent_failures(golden, intent_preds)
    for fm in intent_failures:
        print(f"  {fm['name']}: {fm['count']} examples")
        all_failures.append(fm)
    
    # Reply quality failures
    print("\n--- Reply Quality Failures ---")
    reply_failures = analyze_reply_failures(judge_results, golden)
    for fm in reply_failures:
        print(f"  {fm['name']}: {fm['count']} examples")
        all_failures.append(fm)
    
    # Auto-handle failures
    print("\n--- Auto-Handle Failures ---")
    auto_failures = analyze_auto_handle_failures(judge_results, golden)
    for fm in auto_failures:
        print(f"  {fm['name']}: {fm['count']} examples")
        all_failures.append(fm)
    
    # Edge cases
    print("\n--- Edge Cases ---")
    edge_failures = analyze_edge_cases(judge_results, golden)
    for fm in edge_failures:
        print(f"  {fm['name']}: {fm['count']} examples")
        all_failures.append(fm)
    
    # Sort by count and get top 5
    all_failures.sort(key=lambda x: x["count"], reverse=True)
    top_5 = all_failures[:5]
    
    print("\n" + "=" * 60)
    print("TOP 5 FAILURE MODES")
    print("=" * 60)
    
    for i, fm in enumerate(top_5, 1):
        print(f"\n{i}. {fm['name']} ({fm['type']})")
        print(f"   Count: {fm['count']}")
        print(f"   Examples:")
        for ex in fm["examples"][:2]:
            print(f"     - {ex.get('tweet_id', 'N/A')}: {ex.get('customer_text', '')[:80]}...")
            if "judge_reason" in ex:
                print(f"       Judge: {ex['judge_reason'][:80]}...")
            if "decision_reason" in ex:
                print(f"       Decision: {ex['decision_reason'][:80]}...")
    
    # Save detailed results
    output = {
        "top_5_failure_modes": top_5,
        "all_failure_modes": all_failures,
        "summary": {
            "total_failure_modes_identified": len(all_failures),
            "top_5_types": [fm["type"] for fm in top_5]
        }
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()