from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.taxonomy import weak_label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/pairs.csv")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["weak_suggestion"] = df["customer_text"].map(weak_label)
    # Stratify over weak suggestions only to ensure coverage; the final labels
    # MUST be independently supplied by a human.
    parts = []
    for label, g in df.groupby("weak_suggestion"):
        take = max(1, round(args.n * len(g) / len(df)))
        parts.append(g.sample(min(take, len(g)), random_state=42))
    out = pd.concat(parts).drop_duplicates("tweet_id").sample(frac=1, random_state=42)
    out = out.head(args.n).copy()

    out["human_intent"] = ""
    out["human_auto_handle"] = ""
    out["human_reply_score"] = ""
    out["human_notes"] = ""
    out.to_csv("data/golden/golden_set.csv", index=False)
    print("Created data/golden/golden_set.csv")
    print("IMPORTANT: fill human_* columns manually before evaluation.")

if __name__ == "__main__":
    main()
