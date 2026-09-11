from pathlib import Path
import sys
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import train_classifier, save_model
from src.taxonomy import weak_label

def main():
    df = pd.read_csv("data/processed/pairs.csv")
    golden = pd.read_csv("data/golden/golden_set.csv")
    heldout_ids = set(golden["tweet_id"].astype(str))
    train_df = df[~df["tweet_id"].astype(str).isin(heldout_ids)].copy()

    # Weak labels are used only to bootstrap the classifier. The golden set
    # remains held out and is hand-labelled independently for evaluation.
    train_df["label"] = train_df["customer_text"].map(weak_label)

    if len(train_df) < 100:
        raise SystemExit("Too few weakly-labelled training examples.")

    model = train_classifier(train_df["customer_text"], train_df["label"])
    Path("artifacts").mkdir(exist_ok=True)
    save_model(model, "artifacts/intent_model.joblib")
    print(f"trained on {len(train_df)} weakly-labelled examples")
    print("Golden set was excluded from training.")

if __name__ == "__main__":
    main()