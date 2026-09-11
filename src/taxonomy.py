INTENTS = [
    "account_access",
    "billing_payment",
    "order_purchase",
    "device_issue",
    "app_service_issue",
    "subscription_cancel",
    "technical_howto",
    "status_followup",
    "other",
]

KEYWORDS = {
    "account_access": [
        "apple id", "password", "login", "log in", "locked", "account", "sign in",
        "verification", "verify", "icloud account", "forgot password"
    ],
    "billing_payment": [
        "charged", "charge", "billing", "bill", "payment", "paid", "invoice",
        "refund", "money", "card", "subscription charge"
    ],
    "order_purchase": [
        "order", "purchase", "bought", "buy", "delivery", "deliver", "shipping",
        "shipment", "tracking", "package", "store", "receipt"
    ],
    "device_issue": [
        "iphone", "ipad", "macbook", "mac", "watch", "airpods", "battery",
        "screen", "broken", "crash", "restart", "won't turn", "not turning",
        "overheating", "hardware"
    ],
    "app_service_issue": [
        "icloud", "imessage", "facetime", "itunes", "app store", "apple music",
        "apple tv", "service", "server", "not working", "down", "sync"
    ],
    "subscription_cancel": [
        "cancel", "cancellation", "unsubscribe", "renew", "renewal", "trial",
        "membership"
    ],
    "technical_howto": [
        "how do i", "how to", "where can i", "steps", "enable", "disable",
        "set up", "setup", "change", "turn on", "turn off", "connect"
    ],
    "status_followup": [
        "update", "status", "still waiting", "any update", "follow up",
        "follow-up", "heard back", "case", "ticket", "response"
    ],
}

def weak_label(text: str) -> str:
    t = str(text).lower()
    scores = {intent: 0 for intent in INTENTS}
    for intent, words in KEYWORDS.items():
        for word in words:
            if word in t:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"
