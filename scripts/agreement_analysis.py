"""
Agreement Analysis Script (Task 19)
Calculates agreement metrics between human labels and LLM judge scores.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

sys.path.append(str(Path(__file__).resolve().parents[1]))

JUDGE_RESULTS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "judge_results.csv"
GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "agreement_analysis.json"


def load_data():
    """Load judge results and golden set."""
    if not JUDGE_RESULTS_PATH.exists():
        raise SystemExit(f"Judge results not found: {JUDGE_RESULTS_PATH}")
    
    judge_results = pd.read_csv(JUDGE_RESULTS_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    
    return judge_results, golden


def calculate_agreement_metrics(judge_results):
    """Calculate comprehensive agreement metrics."""
    
    # Filter rows with both human and judge scores
    agreement_df = judge_results.dropna(
        subset=["human_reply_score", "judge_overall_score"]
    ).copy()
    
    if len(agreement_df) < 5:
        print("Not enough examples for agreement metrics.")
        return None
    
    human_scores = pd.to_numeric(agreement_df["human_reply_score"], errors="coerce")
    judge_scores = pd.to_numeric(agreement_df["judge_overall_score"], errors="coerce")
    
    valid = human_scores.notna() & judge_scores.notna()
    human_scores = human_scores[valid]
    judge_scores = judge_scores[valid]
    
    if len(human_scores) < 5:
        print("Not enough valid human/LLM scores.")
        return None
    
    # Convert to integers for categorical metrics
    human_int = human_scores.round().astype(int).clip(1, 5)
    judge_int = judge_scores.round().astype(int).clip(1, 5)
    
    results = {}
    
    # 1. Spearman correlation (ordinal)
    spearman = spearmanr(human_scores, judge_scores)
    results["spearman_correlation"] = float(spearman.statistic)
    results["spearman_pvalue"] = float(spearman.pvalue)
    
    # 2. Pearson correlation (for reference)
    pearson = pearsonr(human_scores, judge_scores)
    results["pearson_correlation"] = float(pearson.statistic)
    results["pearson_pvalue"] = float(pearson.pvalue)
    
    # 3. Exact agreement
    exact_agreement = (human_int == judge_int).mean()
    results["exact_agreement"] = float(exact_agreement)
    
    # 4. Agreement within +/- 1
    within_one = (abs(human_int - judge_int) <= 1).mean()
    results["within_one_agreement"] = float(within_one)
    
    # 5. Weighted Cohen's Kappa (quadratic weights for ordinal)
    kappa = cohen_kappa_score(human_int, judge_int, weights="quadratic")
    results["weighted_cohen_kappa"] = float(kappa)
    
    # 6. Unweighted Cohen's Kappa
    kappa_unweighted = cohen_kappa_score(human_int, judge_int)
    results["unweighted_cohen_kappa"] = float(kappa_unweighted)
    
    # 7. Mean absolute error
    mae = abs(human_scores - judge_scores).mean()
    results["mean_absolute_error"] = float(mae)
    
    # 8. Root mean squared error
    rmse = np.sqrt(((human_scores - judge_scores) ** 2).mean())
    results["rmse"] = float(rmse)
    
    # 9. Confusion matrix
    cm = confusion_matrix(human_int, judge_int, labels=[1, 2, 3, 4, 5])
    results["confusion_matrix"] = cm.tolist()
    results["confusion_matrix_labels"] = [1, 2, 3, 4, 5]
    
    # 10. Per-score agreement
    per_score_agreement = {}
    for score in [1, 2, 3, 4, 5]:
        human_mask = human_int == score
        if human_mask.sum() > 0:
            judge_scores_for_human = judge_int[human_mask]
            agreement = (judge_scores_for_human == score).mean()
            per_score_agreement[int(score)] = {
                "n_examples": int(human_mask.sum()),
                "agreement": float(agreement),
                "mean_judge_score": float(judge_scores_for_human.mean())
            }
    results["per_score_agreement"] = per_score_agreement
    
    # 11. Disagreement analysis
    disagreements = agreement_df[human_int != judge_int].copy()
    disagreements["score_diff"] = (judge_int - human_int).abs()
    results["n_disagreements"] = len(disagreements)
    results["disagreement_examples"] = disagreements[
        ["tweet_id", "customer_text", "human_reply_score", "judge_overall_score", "judge_reason"]
    ].to_dict("records")
    
    return results


def calculate_auto_handle_agreement(judge_results, golden):
    """Calculate agreement on auto-handle decisions."""
    
    # Merge judge results with golden set for human_auto_handle
    merged = judge_results.merge(
        golden[["tweet_id", "human_auto_handle"]],
        on="tweet_id",
        how="left"
    )
    
    # Filter rows with both human and system auto-handle decisions
    merged = merged.dropna(subset=["human_auto_handle", "auto_handle"])
    
    if len(merged) < 5:
        print("Not enough examples for auto-handle agreement.")
        return None
    
    human_auto = merged["human_auto_handle"].map(
        lambda x: 1 if str(x).strip().lower() in ("yes", "1", "true") else 0
    )
    system_auto = merged["auto_handle"].map(
        lambda x: 1 if str(x).strip().lower() in ("yes", "1", "true") else 0
    )
    
    results = {}
    
    # Confusion matrix
    # human=1 (auto), system=1 (auto) -> True Positive (correct auto-handle)
    # human=0 (escalate), system=0 (escalate) -> True Negative (correct escalate)
    # human=0 (escalate), system=1 (auto) -> False Positive (unsafe auto-handle)
    # human=1 (auto), system=0 (escalate) -> False Negative (over-escalation)
    
    tp = ((human_auto == 1) & (system_auto == 1)).sum()
    tn = ((human_auto == 0) & (system_auto == 0)).sum()
    fp = ((human_auto == 0) & (system_auto == 1)).sum()
    fn = ((human_auto == 1) & (system_auto == 0)).sum()
    
    results["confusion_matrix"] = {
        "true_auto_handle": int(tp),
        "true_escalate": int(tn),
        "false_auto_handle": int(fp),  # System says auto, human says escalate (UNSAFE)
        "false_escalate": int(fn)      # System says escalate, human says auto (over-cautious)
    }
    
    # Precision, Recall, F1 for auto-handle class
    auto_handle_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    auto_handle_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    auto_handle_f1 = 2 * auto_handle_precision * auto_handle_recall / (auto_handle_precision + auto_handle_recall) if (auto_handle_precision + auto_handle_recall) > 0 else 0
    
    results["auto_handle_precision"] = float(auto_handle_precision)
    results["auto_handle_recall"] = float(auto_handle_recall)
    results["auto_handle_f1"] = float(auto_handle_f1)
    
    # Precision, Recall, F1 for escalate class
    escalate_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    escalate_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    escalate_f1 = 2 * escalate_precision * escalate_recall / (escalate_precision + escalate_recall) if (escalate_precision + escalate_recall) > 0 else 0
    
    results["escalate_precision"] = float(escalate_precision)
    results["escalate_recall"] = float(escalate_recall)
    results["escalate_f1"] = float(escalate_f1)
    
    # Coverage (percentage auto-handled)
    coverage = system_auto.mean()
    results["auto_handle_coverage"] = float(coverage)
    
    # Human coverage
    human_coverage = human_auto.mean()
    results["human_auto_handle_coverage"] = float(human_coverage)
    
    # Unsafe auto-handle rate (false auto-handle / total auto-handled by system)
    unsafe_rate = fp / (tp + fp) if (tp + fp) > 0 else 0
    results["unsafe_auto_handle_rate"] = float(unsafe_rate)
    
    # False auto-handle examples (critical for safety)
    false_auto_examples = merged[(human_auto == 0) & (system_auto == 1)]
    results["false_auto_handle_examples"] = false_auto_examples[
        ["tweet_id", "customer_text", "predicted_intent", "confidence", "decision_reason", "judge_overall_score", "judge_reason"]
    ].to_dict("records")
    
    return results


def main():
    print("=" * 60)
    print("AGREEMENT ANALYSIS (Task 19)")
    print("=" * 60)
    
    judge_results, golden = load_data()
    print(f"Loaded {len(judge_results)} judge results")
    print(f"Loaded {len(golden)} golden examples")
    
    # Calculate reply quality agreement
    print("\n--- Reply Quality Agreement ---")
    reply_agreement = calculate_agreement_metrics(judge_results)
    
    if reply_agreement:
        print(f"Examples: {len(judge_results.dropna(subset=['human_reply_score', 'judge_overall_score']))}")
        print(f"Spearman correlation: {reply_agreement['spearman_correlation']:.3f} (p={reply_agreement['spearman_pvalue']:.4f})")
        print(f"Pearson correlation: {reply_agreement['pearson_correlation']:.3f} (p={reply_agreement['pearson_pvalue']:.4f})")
        print(f"Exact agreement: {reply_agreement['exact_agreement']:.3f}")
        print(f"Within +/-1 agreement: {reply_agreement['within_one_agreement']:.3f}")
        print(f"Weighted Cohen's kappa: {reply_agreement['weighted_cohen_kappa']:.3f}")
        print(f"Unweighted Cohen's kappa: {reply_agreement['unweighted_cohen_kappa']:.3f}")
        print(f"MAE: {reply_agreement['mean_absolute_error']:.3f}")
        print(f"RMSE: {reply_agreement['rmse']:.3f}")
        
        print("\nPer-score agreement:")
        for score, data in reply_agreement["per_score_agreement"].items():
            print(f"  Human={score}: n={data['n_examples']}, agreement={data['agreement']:.3f}, mean_judge={data['mean_judge_score']:.2f}")
        
        print(f"\nDisagreements: {reply_agreement['n_disagreements']}")
        for ex in reply_agreement["disagreement_examples"][:5]:
            print(f"  {ex['tweet_id']}: human={ex['human_reply_score']}, judge={ex['judge_overall_score']}, diff={abs(ex['human_reply_score'] - ex['judge_overall_score']):.1f}")
            print(f"    Judge reason: {ex['judge_reason'][:100]}...")
    
    # Calculate auto-handle agreement
    print("\n--- Auto-Handle Agreement ---")
    auto_agreement = calculate_auto_handle_agreement(judge_results, golden)
    
    if auto_agreement:
        cm = auto_agreement["confusion_matrix"]
        print(f"Confusion Matrix:")
        print(f"  True Auto-handle: {cm['true_auto_handle']}")
        print(f"  True Escalate: {cm['true_escalate']}")
        print(f"  False Auto-handle (UNSAFE): {cm['false_auto_handle']}")
        print(f"  False Escalate (over-cautious): {cm['false_escalate']}")
        print(f"Auto-handle Precision: {auto_agreement['auto_handle_precision']:.3f}")
        print(f"Auto-handle Recall: {auto_agreement['auto_handle_recall']:.3f}")
        print(f"Auto-handle F1: {auto_agreement['auto_handle_f1']:.3f}")
        print(f"Escalate Precision: {auto_agreement['escalate_precision']:.3f}")
        print(f"Escalate Recall: {auto_agreement['escalate_recall']:.3f}")
        print(f"Escalate F1: {auto_agreement['escalate_f1']:.3f}")
        print(f"Auto-handle Coverage (system): {auto_agreement['auto_handle_coverage']:.3f}")
        print(f"Auto-handle Coverage (human): {auto_agreement['human_auto_handle_coverage']:.3f}")
        print(f"Unsafe Auto-handle Rate: {auto_agreement['unsafe_auto_handle_rate']:.3f}")
        
        print(f"\nFalse Auto-handle Examples (UNSAFE): {len(auto_agreement['false_auto_handle_examples'])}")
        for ex in auto_agreement["false_auto_handle_examples"][:5]:
            print(f"  {ex['tweet_id']}: intent={ex['predicted_intent']}, conf={ex['confidence']:.2f}")
            print(f"    Reason: {ex['decision_reason']}")
            print(f"    Judge score: {ex['judge_overall_score']}, Judge reason: {ex['judge_reason'][:80]}...")
    
    # Save results
    output = {
        "reply_quality_agreement": reply_agreement,
        "auto_handle_agreement": auto_agreement
    }
    
    import json
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()