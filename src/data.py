from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED = {
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id"
}

def load_twcs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df

def parse_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]

def build_pairs(df: pd.DataFrame, brand: str, max_pairs: int | None = None) -> pd.DataFrame:
    # Brand is inferred from the author IDs represented by outbound support tweets.
    # The dataset anonymizes handles, so the practical way to select a brand is
    # to use a brand author-id mapping produced by this preparation script.
    # If a mapping exists, select by that mapping; otherwise brand may be a
    # normalized label assigned during discovery.
    d = df.copy()
    d["tweet_id"] = d["tweet_id"].astype(str)
    d["in_response_to_tweet_id"] = d["in_response_to_tweet_id"].astype("string")
    d["inbound"] = d["inbound"].astype(bool)

    # Brand discovery uses the conventional public handle mapping bundled in
    # scripts/prepare_data.py. The resulting filtered rows carry `brand`.
    if "brand" not in d.columns:
        raise ValueError("Prepared dataframe needs a 'brand' column.")

    d = d[d["brand"].eq(brand)].copy()
    lookup = d.set_index("tweet_id")["text"].to_dict()
    inbound = d[d["inbound"]].copy()
    inbound["response_text"] = inbound["response_tweet_id"].map(
        lambda x: lookup.get(parse_ids(x)[0], "") if parse_ids(x) else ""
    )
    inbound = inbound[
        inbound["text"].notna()
        & inbound["response_text"].notna()
        & inbound["response_text"].astype(str).str.len().gt(0)
    ].copy()
    inbound = inbound.rename(columns={"text": "customer_text"})
    inbound["response_text"] = inbound["response_text"].astype(str)
    inbound["customer_text"] = inbound["customer_text"].astype(str)
    inbound = inbound.drop_duplicates("tweet_id")
    if max_pairs:
        inbound = inbound.sample(min(max_pairs, len(inbound)), random_state=42)
    return inbound[["tweet_id", "created_at", "customer_text", "response_text"]].reset_index(drop=True)
