from pathlib import Path
import sys
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import load_model, predict
from src.taxonomy import weak_label


def main():

    # -----------------------------
    # 1. Load golden set
    # -----------------------------
    golden = pd.read_csv("data/golden/golden_set.csv")

    golden = golden[
        golden["human_intent"].astype(str).str.strip().ne("")
    ].copy()

    if len(golden) < 50:
        raise SystemExit("Not enough hand-labelled examples.")

    # -----------------------------
    # 2. Load proposed model
    # -----------------------------
    model = load_model("artifacts/intent_model.joblib")

    proposed_preds = [
        predict(model, x)[0]
        for x in golden["customer_text"]
    ]

    # -----------------------------
    # 3. Majority baseline
    # -----------------------------
    majority = golden["human_intent"].value_counts().idxmax()

    majority_preds = [majority] * len(golden)

    # -----------------------------
    # 4. Load historical data
    # -----------------------------
    df = pd.read_csv("data/processed/pairs.csv")

    # Golden examples must NEVER be used
    # for baseline training.
    heldout_ids = set(
        golden["tweet_id"].astype(str)
    )

    train_df = df[
        ~df["tweet_id"].astype(str).isin(heldout_ids)
    ].copy()

    # -----------------------------
    # 5. Generate weak labels
    # -----------------------------
    train_df["label"] = train_df[
        "customer_text"
    ].map(weak_label)

    # Keep "other" because the proposed
    # model is also trained with "other".
    train_df = train_df[
        train_df["label"].notna()
    ].copy()

    if len(train_df) < 100:
        raise SystemExit(
            "Too few weakly-labelled training examples."
        )

    print(
        f"Baseline training examples: {len(train_df)}"
    )

    # -----------------------------
    # 6. TF-IDF + Logistic Regression
    # -----------------------------
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    X_train = vec.fit_transform(
        train_df["customer_text"]
    )

    X_gold = vec.transform(
        golden["customer_text"]
    )

    lr = LogisticRegression(
        max_iter=1500,
        class_weight="balanced"
    )

    lr.fit(
        X_train,
        train_df["label"]
    )

    simple_preds = lr.predict(X_gold)

    # -----------------------------
    # 7. Calculate metrics
    # -----------------------------
    metrics = {
        "n_gold": len(golden),

        "majority_accuracy": accuracy_score(
            golden["human_intent"],
            majority_preds
        ),

        "majority_macro_f1": f1_score(
            golden["human_intent"],
            majority_preds,
            average="macro"
        ),

        "tfidf_lr_accuracy": accuracy_score(
            golden["human_intent"],
            simple_preds
        ),

        "tfidf_lr_macro_f1": f1_score(
            golden["human_intent"],
            simple_preds,
            average="macro"
        ),

        "proposed_accuracy": accuracy_score(
            golden["human_intent"],
            proposed_preds
        ),

        "proposed_macro_f1": f1_score(
            golden["human_intent"],
            proposed_preds,
            average="macro"
        ),
    }

    # -----------------------------
    # 8. Print results
    # -----------------------------
    print(pd.Series(metrics))

    print("\nProposed per-intent report:")

    print(
        classification_report(
            golden["human_intent"],
            proposed_preds,
            zero_division=0
        )
    )

    # -----------------------------
    # 9. Save metrics
    # -----------------------------
    Path("artifacts").mkdir(exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        "artifacts/metrics.csv",
        index=False
    )

    pd.DataFrame({
        "tweet_id": golden["tweet_id"],
        "text": golden["customer_text"],
        "gold_intent": golden["human_intent"],
        "pred_intent": proposed_preds
    }).to_csv(
        "artifacts/intent_predictions.csv",
        index=False
    )


if __name__ == "__main__":
    main()