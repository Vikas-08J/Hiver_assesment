# Golden-set annotation guidelines

## Unit

Label the **customer message** independently of the model's suggested label.

## Intent

Choose the smallest intent that explains the customer's primary request.

- `account_access`: login, password, Apple ID, verification, account locked.
- `billing_payment`: unexpected charge, billing, payment, refund/charge issue.
- `order_purchase`: purchase, order, shipping, delivery, tracking.
- `device_issue`: device malfunction, hardware, battery, screen, crashes.
- `app_service_issue`: Apple service/app is unavailable or malfunctioning.
- `subscription_cancel`: cancel/renew/stop a subscription or trial.
- `technical_howto`: asks how to perform a supported task.
- `status_followup`: asks for progress/update on an existing support case.
- `other`: genuinely outside the taxonomy or too ambiguous.

If multiple intents occur, select the one that would determine the **next support action**.

## Auto-handle

Use `1` only if a normal support agent could safely answer using the historical evidence available to the system without account verification, sensitive investigation, legal action, fraud handling, or a missing case-specific fact.

Use `0` for uncertain, high-risk, account-specific, legal, security, fraud, or evidence-poor cases.

## Reply score

1 unsafe/wrong
2 major problems
3 acceptable
4 good
5 excellent

A good reply should be relevant, actionable, concise, grounded in historical evidence, and avoid unsupported claims.

## Human/LLM judge agreement

For at least 30 examples, independently score the final generated replies. Do not look at the judge score before submitting the human score.
