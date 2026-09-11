from __future__ import annotations
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def build_classifier():
    base = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=120000,
            sublinear_tf=True,
        )),
        ("svc", LinearSVC(C=1.5, class_weight="balanced"))
    ])
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)

def train_classifier(texts, labels):
    model = build_classifier()
    model.fit(texts, labels)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def predict(model, text):
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    idx = int(np.argmax(probs))
    return str(classes[idx]), float(probs[idx])
