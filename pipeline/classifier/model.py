"""Intent classifier: TF-IDF features + logistic regression, with calibrated-ish confidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


@dataclass(frozen=True)
class Prediction:
    intent: str
    confidence: float


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ]
    )
    return Pipeline([("features", features), ("clf", LogisticRegression(C=10, max_iter=1000))])


class IntentModel:
    def __init__(self, pipeline: Pipeline, version: str) -> None:
        self.pipeline = pipeline
        self.version = version

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> IntentModel:
        pipe = build_pipeline().fit(texts, labels)
        digest = hashlib.sha256(np.asarray(pipe.named_steps["clf"].coef_).tobytes()).hexdigest()[:8]
        return cls(pipe, f"tfidf-lr-{digest}")

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        proba = self.pipeline.predict_proba(texts)
        idx = proba.argmax(axis=1)
        classes = self.classes
        return [Prediction(classes[i], float(proba[n, i])) for n, i in enumerate(idx)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "version": self.version}, path)

    @classmethod
    def load(cls, path: str | Path) -> IntentModel:
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["version"])
