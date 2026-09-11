from __future__ import annotations

HIGH_RISK = [
    "fraud", "scam", "stolen", "hack", "hacked", "unauthorized",
    "legal", "lawsuit", "police", "chargeback", "identity theft",
    "security breach", "data breach"
]

ALWAYS_ESCALATE = {
    "account_access",
}

def decide(intent, confidence, retrieval_score, text, threshold=0.68, retrieval_threshold=0.20):
    t = text.lower()
    risk = [w for w in HIGH_RISK if w in t]
    if risk:
        return {
            "auto_handle": False,
            "reason": f"Escalate: high-risk signal ({risk[0]})."
        }
    if intent in ALWAYS_ESCALATE:
        return {
            "auto_handle": False,
            "reason": "Escalate: account-access issues may require identity/account verification."
        }
    if confidence < threshold:
        return {
            "auto_handle": False,
            "reason": f"Escalate: classifier confidence {confidence:.2f} is below {threshold:.2f}."
        }
    if retrieval_score < retrieval_threshold:
        return {
            "auto_handle": False,
            "reason": f"Escalate: no sufficiently similar historical resolution (score {retrieval_score:.2f})."
        }
    return {
        "auto_handle": True,
        "reason": "Auto-handle: confident intent prediction and sufficiently similar historical evidence."
    }
