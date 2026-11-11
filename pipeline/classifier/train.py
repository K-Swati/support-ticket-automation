"""Train the intent model and write an honest evaluation report.

Two numbers are reported:

* ``in_distribution`` - fresh noisy tickets from the *training* templates
* ``unseen_templates`` - tickets from templates the model never saw

The second is the one to quote. Both come from synthetic data (see ``data.py``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

from pipeline.classifier.data import generate
from pipeline.classifier.model import IntentModel
from pipeline.config import Settings
from pipeline.schemas import INTENTS


def evaluate(model: IntentModel, n_per_intent: int, seed: int, held_out: bool) -> dict:
    data = generate(n_per_intent, seed, held_out=held_out)
    preds = model.predict_many([e.text for e in data])
    y_true = [e.intent for e in data]
    y_pred = [p.intent for p in preds]
    report = classification_report(y_true, y_pred, labels=list(INTENTS), output_dict=True, zero_division=0)
    return {
        "n": len(data),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "mean_confidence": round(sum(p.confidence for p in preds) / len(preds), 4),
        "per_class_f1": {k: round(report[k]["f1-score"], 4) for k in INTENTS},
        "top_confusions": [
            {"true": t, "predicted": p, "count": c}
            for (t, p), c in Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p).most_common(5)
        ],
    }


def train_and_report(settings: Settings, n_train: int = 300, n_eval: int = 200, seed: int = 7) -> dict:
    train = generate(n_train, seed)
    model = IntentModel.train([e.text for e in train], [e.intent for e in train])
    model.save(settings.model_path)
    report = {
        "model_version": model.version,
        "train_examples": len(train),
        "data": "synthetic, template-generated (see pipeline/classifier/data.py)",
        "in_distribution": evaluate(model, n_eval, seed + 1, held_out=False),
        "unseen_templates": evaluate(model, n_eval, seed + 2, held_out=True),
    }
    out = Path(settings.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classifier_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
