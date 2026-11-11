"""Run the whole pipeline in one process against a mocked AWS (no Docker, no credentials).

    python scripts/demo.py

It trains the model, submits tickets through the real API, injects a few
malformed messages, runs both workers until the queues drain, and prints where
every message ended up.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.update(AWS_ACCESS_KEY_ID="test", AWS_SECRET_ACCESS_KEY="test", AWS_DEFAULT_REGION="us-east-1")

import boto3  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from moto import mock_aws  # noqa: E402

from pipeline.api import create_app  # noqa: E402
from pipeline.classifier.data import generate  # noqa: E402
from pipeline.classifier.model import IntentModel  # noqa: E402
from pipeline.classifier.worker import ClassifierWorker  # noqa: E402
from pipeline.config import Settings  # noqa: E402
from pipeline.metrics import Metrics  # noqa: E402
from pipeline.queues import provision_all, queue_depth, send_message  # noqa: E402
from pipeline.responder.generator import TemplateResponder  # noqa: E402
from pipeline.responder.worker import ResponderWorker  # noqa: E402
from pipeline.store import Store  # noqa: E402


def main() -> None:
    tmp = tempfile.mkdtemp()
    settings = dataclasses.replace(
        Settings(), backoff_base_seconds=0, wait_time_seconds=0, db_path=f"{tmp}/demo.db", model_path=f"{tmp}/m.joblib"
    )
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-east-1")
        urls = provision_all(sqs, settings)
        store, metrics = Store(settings.db_path), Metrics()

        train = generate(200, seed=7)
        model = IntentModel.train([e.text for e in train], [e.intent for e in train])
        print(f"trained {model.version} on {len(train)} synthetic tickets")

        tickets = generate(8, seed=123, held_out=True)  # phrasings the model has never seen
        ids: dict[str, str] = {}
        with TestClient(create_app(settings, sqs=sqs, store=store)) as api:
            for n, t in enumerate(tickets):
                r = api.post("/messages", json={"customer_id": f"cust-{n}", "text": t.text, "idempotency_key": f"t{n}"})
                ids[r.json()["message_id"]] = t.intent
            dup = api.post("/messages", json={"customer_id": "cust-0", "text": tickets[0].text, "idempotency_key": "t0"})
            print(f"resubmitting key t0 -> duplicate={dup.json()['duplicate']} (nothing enqueued)")

            send_message(sqs, urls.incoming, "{ this is not json")
            send_message(sqs, urls.incoming, json.dumps({"schema_version": 7, "message_id": "x", "correlation_id": "y"}))

            classifier = ClassifierWorker(
                settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review, model, store, metrics
            )
            responder = ResponderWorker(
                settings, sqs, urls.classified, urls.classified_dlq, TemplateResponder(), store, metrics
            )
            while classifier.run_once(0):
                pass
            while responder.run_once(0):
                pass

            statuses = Counter(api.get(f"/messages/{mid}").json()["status"] for mid in ids)
            correct = sum(api.get(f"/messages/{mid}").json()["intent"] == want for mid, want in ids.items())

        print("\nwhere the messages ended up")
        print(f"  tickets submitted : {len(ids)}  ->  {dict(statuses)}")
        print(f"  intent correct    : {correct}/{len(ids)} (unseen phrasings, synthetic data)")
        for name, url in urls.names.items():
            depth = queue_depth(sqs, url)["visible"]
            if depth:
                print(f"  queue {name:<16}: {depth} message(s)")
        print("  counters          :", dict(sorted(metrics.snapshot().items())))


if __name__ == "__main__":
    main()
