"""
Edge Case Testing Script (Task 22)
Tests the system on various edge cases to understand behavior.
"""
from pathlib import Path
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import load_model
from src.retrieval import HistoricalRetriever
from src.agent import SupportAgent

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "intent_model.joblib"
PAIRS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "pairs.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "edge_case_analysis.json"


# Define edge case test inputs
EDGE_CASES = [
    # Empty/near-empty messages
    {"category": "empty_message", "text": "", "expected": "escalate"},
    {"category": "whitespace_only", "text": "   ", "expected": "escalate"},
    {"category": "single_char", "text": "?", "expected": "escalate"},
    
    # Extremely short tweets
    {"category": "very_short", "text": "Help", "expected": "escalate"},
    {"category": "very_short_2", "text": "Hi", "expected": "escalate"},
    {"category": "very_short_3", "text": "Why", "expected": "escalate"},
    
    # Messages containing URLs
    {"category": "url_only", "text": "https://t.co/abc123", "expected": "escalate"},
    {"category": "url_with_text", "text": "Check this https://t.co/abc123 please help", "expected": "escalate"},
    {"category": "multiple_urls", "text": "See https://t.co/abc123 and https://t.co/def456", "expected": "escalate"},
    
    # Messages containing usernames/mentions
    {"category": "mention_only", "text": "@AppleSupport", "expected": "escalate"},
    {"category": "multiple_mentions", "text": "@AppleSupport @115858 help me", "expected": "escalate"},
    {"category": "mention_with_text", "text": "@AppleSupport my phone is broken", "expected": "auto_handle"},
    
    # Messages containing emojis
    {"category": "emoji_only", "text": "😭😭😭", "expected": "escalate"},
    {"category": "emoji_with_text", "text": "My phone is broken 😭😭", "expected": "auto_handle"},
    {"category": "mixed_emojis", "text": "😡😡😡 fix this @AppleSupport", "expected": "auto_handle"},
    
    # Spelling mistakes
    {"category": "typos", "text": "my iphon is brokn help", "expected": "auto_handle"},
    {"category": "severe_typos", "text": "iphne not wurking plz hlp", "expected": "auto_handle"},
    
    # Slang
    {"category": "slang", "text": "my phone is whack fam", "expected": "auto_handle"},
    {"category": "internet_slang", "text": "ios 11 is trash smh", "expected": "auto_handle"},
    
    # Multiple questions in one tweet
    {"category": "multi_question", "text": "How do I backup? And how do I restore?", "expected": "auto_handle"},
    {"category": "multi_question_2", "text": "Why is my battery draining? How to fix? When is update?", "expected": "auto_handle"},
    
    # Multiple intents
    {"category": "multi_intent", "text": "My phone is broken and I want a refund", "expected": "auto_handle"},
    {"category": "multi_intent_2", "text": "Can't login and need to cancel subscription", "expected": "auto_handle"},
    
    # Unclear intent
    {"category": "unclear", "text": "This is weird", "expected": "escalate"},
    {"category": "vague", "text": "Something is wrong with my device", "expected": "escalate"},
    {"category": "ambiguous", "text": "I have a problem", "expected": "escalate"},
    
    # No relevant historical examples (out of domain)
    {"category": "out_of_domain", "text": "How do I cook pasta?", "expected": "escalate"},
    {"category": "out_of_domain_2", "text": "What's the weather like?", "expected": "escalate"},
    
    # Conflicting historical examples (would need specific test)
    # Unusual customer sentiment
    {"category": "angry", "text": "This is the worst product ever! I hate Apple!", "expected": "escalate"},
    {"category": "sarcastic", "text": "Oh great, another amazing update that breaks everything", "expected": "escalate"},
    {"category": "polite", "text": "Hello, could you please help me with my iPhone?", "expected": "auto_handle"},
    
    # Potentially sensitive/private information
    {"category": "pii_email", "text": "My email is john@example.com and I can't login", "expected": "escalate"},
    {"category": "pii_phone", "text": "Call me at 555-123-4567 to fix my account", "expected": "escalate"},
    {"category": "pii_address", "text": "Ship my order to 123 Main St, New York", "expected": "escalate"},
    
    # Duplicate messages
    {"category": "duplicate", "text": "My iPhone 7 battery drains fast", "expected": "auto_handle"},
    
    # Very long messages
    {"category": "very_long", "text": " ".join(["This is a very long message about my iPhone 7 battery draining really fast and I've tried everything including restarting and updating and nothing works and I'm really frustrated and need help please"] * 5), "expected": "auto_handle"},
    
    # Unusual punctuation
    {"category": "excessive_punctuation", "text": "HELP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "expected": "escalate"},
    {"category": "no_punctuation", "text": "my iphone is broken help me please", "expected": "auto_handle"},
    
    # High-risk signals (should escalate)
    {"category": "fraud", "text": "I think my account was hacked and someone stole my money", "expected": "escalate"},
    {"category": "legal", "text": "I'm going to sue Apple for this defective product", "expected": "escalate"},
    {"category": "security", "text": "There's a security breach in my iCloud account", "expected": "escalate"},
    
    # Account access (should escalate per policy)
    {"category": "account_access", "text": "I forgot my Apple ID password and can't reset it", "expected": "escalate"},
    {"category": "account_locked", "text": "My Apple ID is locked for security reasons", "expected": "escalate"},
]


def load_system():
    """Load the trained model and build the agent."""
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    
    model = load_model(MODEL_PATH)
    
    pairs = pd.read_csv(PAIRS_PATH)
    retriever = HistoricalRetriever(pairs["customer_text"], pairs["response_text"])
    
    agent = SupportAgent(model, retriever)
    return agent


def test_edge_case(agent, case):
    """Test a single edge case."""
    try:
        output = agent.run(case["text"], use_llm=False)  # Use template reply for speed
        return {
            "category": case["category"],
            "input_text": case["text"],
            "expected": case["expected"],
            "predicted_intent": output["intent"],
            "confidence": output["confidence"],
            "auto_handle": output["auto_handle"],
            "decision_reason": output["reason"],
            "reply": output["reply"],
            "retrieval_score": output["evidence"][0]["score"] if output["evidence"] else 0.0,
            "error": None
        }
    except Exception as e:
        return {
            "category": case["category"],
            "input_text": case["text"],
            "expected": case["expected"],
            "error": str(e)
        }


def main():
    print("=" * 60)
    print("EDGE CASE TESTING (Task 22)")
    print("=" * 60)
    
    agent = load_system()
    print(f"Loaded model and retriever ({len(agent.retriever.texts)} historical examples)")
    
    results = []
    for case in EDGE_CASES:
        print(f"\nTesting: {case['category']}")
        print(f"  Input: {case['text'][:60]}...")
        result = test_edge_case(agent, case)
        results.append(result)
        
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Intent: {result['predicted_intent']} (conf: {result['confidence']:.3f})")
            print(f"  Auto-handle: {result['auto_handle']} - {result['decision_reason']}")
            print(f"  Retrieval score: {result['retrieval_score']:.3f}")
            print(f"  Reply: {result['reply'][:80]}...")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("EDGE CASE ANALYSIS")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    
    # By category
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        auto_rate = cat_df["auto_handle"].mean() if "auto_handle" in cat_df.columns else 0
        avg_conf = cat_df["confidence"].mean() if "confidence" in cat_df.columns else 0
        avg_retrieval = cat_df["retrieval_score"].mean() if "retrieval_score" in cat_df.columns else 0
        print(f"\n{category}:")
        print(f"  Count: {len(cat_df)}")
        print(f"  Auto-handle rate: {auto_rate:.2f}")
        print(f"  Avg confidence: {avg_conf:.3f}")
        print(f"  Avg retrieval score: {avg_retrieval:.3f}")
    
    # Check alignment with expectations
    df_valid = df.dropna(subset=["auto_handle"]).copy()
    if len(df_valid) > 0:
        df_valid["expected_auto"] = df_valid["expected"].map({"auto_handle": 1, "escalate": 0})
        df_valid["aligned"] = df_valid["auto_handle"] == df_valid["expected_auto"]
        alignment_rate = df_valid["aligned"].mean()
        print(f"\nOverall alignment with expectations: {alignment_rate:.2f}")
        
        misaligned = df_valid[~df_valid["aligned"]]
        if len(misaligned) > 0:
            print(f"\nMisaligned cases ({len(misaligned)}):")
            for _, row in misaligned.iterrows():
                print(f"  {row['category']}: expected={row['expected']}, got={'auto' if row['auto_handle'] else 'escalate'}")
                print(f"    Reason: {row['decision_reason']}")
    
    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()