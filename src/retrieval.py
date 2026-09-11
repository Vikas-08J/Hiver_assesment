from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HistoricalRetriever:
    def __init__(self, texts, responses):
        self.texts = list(texts)
        self.responses = list(responses)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=100000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, k=3):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        order = np.argsort(-scores)[:k]
        return [
            {
                "customer_text": self.texts[i],
                "response_text": self.responses[i],
                "score": float(scores[i]),
            }
            for i in order
        ]
