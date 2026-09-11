from __future__ import annotations
from .model import predict
from .policy import decide
from .reply import llm_reply


class SupportAgent:
    def __init__(self, model, retriever, thresholds=None):
        self.model = model
        self.retriever = retriever
        self.thresholds = thresholds or {}

    def run(self, text, use_llm=True):
        intent, confidence = predict(self.model, text)

        evidence = self.retriever.search(text, k=3)

        top_score = evidence[0]["score"] if evidence else 0.0

        decision = decide(
            intent,
            confidence,
            top_score,
            text,
            threshold=self.thresholds.get("auto_handle_threshold", 0.68),
            retrieval_threshold=self.thresholds.get("retrieval_threshold", 0.20),
        )

        if decision["auto_handle"]:
            reply = llm_reply(text, intent, evidence) if use_llm else (
                evidence[0]["response_text"] if evidence else
                "Thanks for reaching out. We need a little more information to help with this request."
            )
        else:
            reply = (
                "Thanks for reaching out. "
                "A support specialist should review this request and assist you further."
            )

        return {
            "intent": intent,
            "confidence": confidence,
            "reply": reply,
            "evidence": evidence,
            **decision,
        }