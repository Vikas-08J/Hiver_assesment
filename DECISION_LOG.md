# Decision log

1. **One brand only:** AppleSupport. This keeps the taxonomy and historical resolution style coherent.
2. **Small taxonomy:** 9 intents instead of dozens. The agent should make operational decisions, not reproduce every topic.
3. **Historical retrieval:** replies are grounded in real brand responses rather than generated from generic knowledge.
4. **TF-IDF retrieval first:** fast, transparent, CPU-friendly, and easy to audit.
5. **Character n-grams for classification:** Twitter contains misspellings, abbreviations, URLs, usernames, and short messages.
6. **Linear SVM with calibration:** strong sparse-text baseline while exposing a usable confidence score.
7. **Conservative escalation:** low confidence or weak retrieval evidence should not be auto-handled.
8. **Account-access escalation:** account-specific support commonly requires information unavailable in a public tweet.
9. **High-risk keyword escalation:** fraud, legal, security, and similar cases should be routed to humans.
10. **LLM is a rewriting layer, not the source of truth:** retrieved historical evidence constrains the draft.
11. **No synthetic golden labels:** automatically produced labels can accelerate annotation but do not satisfy the assignment's hand-labelled requirement.
12. **Judge validation:** an LLM judge is only evidence after measuring agreement with a human rater.
13. **No full dataset in Git:** the dataset is large and has licensing constraints; the repo contains code, not the raw corpus.
14. **Reproducible random seed:** all sampling uses seed 42.
15. **Report the misleading metric:** a single F1 number can hide leakage, imbalance, retrieval memorization, unsafe automation, and policy staleness.
