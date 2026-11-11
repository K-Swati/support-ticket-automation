import pytest
from fastapi.testclient import TestClient

from pipeline.api import create_app
from pipeline.classifier.worker import ClassifierWorker
from pipeline.responder.generator import TemplateResponder
from pipeline.responder.worker import ResponderWorker


@pytest.fixture
def client(settings, sqs, store):
    with TestClient(create_app(settings, sqs=sqs, store=store)) as c:
        yield c


def test_health_and_validation(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/messages", json={"customer_id": "c", "text": ""}).status_code == 422
    assert client.post("/messages", json={"text": "hi"}).status_code == 422
    assert client.get("/messages/does-not-exist").status_code == 404


def test_idempotency_key_prevents_duplicate_enqueue(client):
    body = {"customer_id": "c1", "text": "charged twice", "idempotency_key": "order-77"}
    first = client.post("/messages", json=body)
    second = client.post("/messages", json=body)
    assert first.status_code == second.status_code == 202
    assert first.json()["duplicate"] is False and second.json()["duplicate"] is True
    assert first.json()["message_id"] == second.json()["message_id"]
    assert client.get("/stats").json()["queues"]["incoming"]["visible"] == 1


def test_enqueue_failure_returns_503_and_leaves_no_orphan(settings, sqs, store):
    class DownSQS:
        def __getattr__(self, name):
            return getattr(sqs, name)

        def send_message(self, **kw):
            raise RuntimeError("sqs unavailable")

    with TestClient(create_app(settings, sqs=DownSQS(), store=store)) as c:
        resp = c.post("/messages", json={"customer_id": "c1", "text": "hello there"})
    assert resp.status_code == 503
    assert store.status_counts() == {}


def test_end_to_end_message_flow(client, settings, sqs, store, urls, metrics, trained_model):
    sent = client.post("/messages", json={"customer_id": "c9", "text": "I want to cancel my Pro subscription"}).json()
    mid = sent["message_id"]
    assert client.get(f"/messages/{mid}").json()["status"] == "received"

    ClassifierWorker(
        settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review, trained_model, store, metrics
    ).run_once(0)
    mid_state = client.get(f"/messages/{mid}").json()
    assert mid_state["intent"] == "cancellation" and mid_state["status"] in ("classified", "needs_human_review")

    ResponderWorker(settings, sqs, urls.classified, urls.classified_dlq, TemplateResponder(), store, metrics).run_once(0)
    final = client.get(f"/messages/{mid}").json()
    assert final["status"] == "responded" and "cancel" in final["response"].lower()
    assert final["correlation_id"] == sent["correlation_id"]
