from pathlib import Path
import argparse, pandas as pd, sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import load_model
from src.retrieval import HistoricalRetriever
from src.agent import SupportAgent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    pairs = pd.read_csv("data/processed/pairs.csv")
    model = load_model("artifacts/intent_model.joblib")
    retriever = HistoricalRetriever(pairs["customer_text"], pairs["response_text"])
    agent = SupportAgent(model, retriever)
    result = agent.run(args.text, use_llm=not args.no_llm)
    print("\nINTENT:", result["intent"])
    print("CONFIDENCE:", round(result["confidence"], 3))
    print("AUTO-HANDLE:", result["auto_handle"])
    print("REASON:", result["reason"])
    print("\nREPLY:\n", result["reply"])
    print("\nEVIDENCE:")
    for e in result["evidence"]:
        print("-", round(e["score"],3), e["customer_text"], "=>", e["response_text"])

if __name__ == "__main__":
    main()
