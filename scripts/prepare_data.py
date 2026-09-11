from pathlib import Path
import argparse
import pandas as pd


def normalize_id(x):
    if pd.isna(x):
        return ""

    s = str(x).strip()

    # Pandas may read integer Twitter IDs containing NaN in the
    # same column as floats, producing values such as "698.0".
    if s.endswith(".0"):
        s = s[:-2]

    return s


def parse_ids(x):
    if pd.isna(x):
        return []

    return [
        normalize_id(z)
        for z in str(x).split(",")
        if normalize_id(z)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--brand", default="AppleSupport")
    ap.add_argument("--max-pairs", type=int, default=20000)
    args = ap.parse_args()

    print("Reading dataset...")

    df = pd.read_csv(args.input, low_memory=False)

    required = {
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    # Normalize identifiers.
    df["tweet_id"] = df["tweet_id"].apply(normalize_id)
    df["author_id"] = df["author_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Robust handling of True/False values.
    df["inbound"] = (
        df["inbound"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    # AppleSupport's outbound tweets.
    support = df[
        (~df["inbound"]) &
        (df["author_id"].eq(args.brand))
    ].copy()

    support["parent_id"] = support[
        "in_response_to_tweet_id"
    ].apply(
        lambda x: parse_ids(x)[0] if parse_ids(x) else ""
    )

    # Customer tweets.
    customer = df[df["inbound"]].copy()

    customer = customer.rename(
        columns={"text": "customer_text"}
    )

    # Join:
    #
    # AppleSupport reply:
    #     in_response_to_tweet_id
    #
    #            ↓
    #
    # Customer:
    #     tweet_id
    #
    pairs = support.merge(
        customer[
            ["tweet_id", "customer_text", "created_at"]
        ],
        left_on="parent_id",
        right_on="tweet_id",
        how="inner",
        suffixes=("_support", "_customer"),
    )

    pairs = pairs.rename(
        columns={"text": "response_text"}
    )

    pairs = pairs[
        pairs["customer_text"].str.strip().ne("")
        &
        pairs["response_text"].str.strip().ne("")
    ][
        [
            "tweet_id_customer",
            "created_at_customer",
            "customer_text",
            "response_text",
        ]
    ]

    pairs = pairs.rename(
        columns={
            "tweet_id_customer": "tweet_id",
            "created_at_customer": "created_at",
        }
    )

    pairs = pairs.drop_duplicates("tweet_id")

    if len(pairs) > args.max_pairs:
        pairs = pairs.sample(
            args.max_pairs,
            random_state=42
        )

    out = Path("data/processed/pairs.csv")

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pairs.to_csv(
        out,
        index=False
    )

    print(f"Brand: {args.brand}")
    print(f"Support tweets: {len(support)}")
    print(f"Customer-support pairs: {len(pairs)}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()