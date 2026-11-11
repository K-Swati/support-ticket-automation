import json

from pipeline.classifier.worker import ClassifierWorker
from pipeline.queues import queue_depth, send_message
from pipeline.responder.generator import TemplateResponder
from pipeline.responder.worker import ResponderWorker
from pipeline.schemas import ClassifiedMessage, IncomingMessage, dump_message, new_id
from tests.conftest import StubModel, drain


def incoming(text="I was charged twice", **kw) -> IncomingMessage:
    return IncomingMessage(message_id=kw.pop("message_id", new_id()), correlation_id=new_id(), customer_id="c1", text=text)


def make_classifier(settings, sqs, urls, store, metrics, model):
    return ClassifierWorker(
        settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review, model, store, metrics
    )


def enqueue(sqs, urls, msg: IncomingMessage) -> None:
    send_message(sqs, urls.incoming, dump_message(msg), {"correlation_id": msg.correlation_id})


def test_confident_message_goes_to_classified_queue(settings, sqs, urls, store, metrics):
    msg = incoming()
    store.create_message(msg.message_id, "c1", msg.text, msg.correlation_id)
    enqueue(sqs, urls, msg)
    worker = make_classifier(settings, sqs, urls, store, metrics, StubModel("billing", 0.9))
    assert worker.run_once(wait_time=0) == 1
    (out,) = drain(sqs, urls.classified)
    body = json.loads(out["Body"])
    assert body["intent"] == "billing" and body["model_version"] == "stub-1" and body["message_id"] == msg.message_id
    assert out["MessageAttributes"]["correlation_id"]["StringValue"] == msg.correlation_id
    assert store.get_message(msg.message_id)["status"] == "classified"
    assert queue_depth(sqs, urls.incoming)["visible"] == 0
    assert metrics.snapshot()["classifier.processed"] == 1


def test_low_confidence_is_routed_to_human_review(settings, sqs, urls, store, metrics):
    msg = incoming("something vague")
    store.create_message(msg.message_id, "c1", msg.text, msg.correlation_id)
    enqueue(sqs, urls, msg)
    make_classifier(settings, sqs, urls, store, metrics, StubModel("general_inquiry", 0.3)).run_once(0)
    assert drain(sqs, urls.classified) == []
    (review,) = drain(sqs, urls.human_review)
    assert json.loads(review["Body"])["reason"] == "low_confidence"
    assert store.get_message(msg.message_id)["status"] == "needs_human_review"
    assert metrics.snapshot()["classifier.routed_to_human"] == 1


def test_invalid_json_goes_straight_to_dlq_with_reason(settings, sqs, urls, store, metrics):
    send_message(sqs, urls.incoming, "{not json", {"correlation_id": "abc"})
    make_classifier(settings, sqs, urls, store, metrics, StubModel()).run_once(0)
    (dead,) = drain(sqs, urls.incoming_dlq)
    attrs = {k: v["StringValue"] for k, v in dead["MessageAttributes"].items()}
    assert attrs["error_reason"] == "invalid_json" and attrs["failed_stage"] == "classifier"
    assert attrs["correlation_id"] == "abc" and dead["Body"] == "{not json"
    assert queue_depth(sqs, urls.incoming)["visible"] == 0  # not retried
    assert metrics.snapshot()["classifier.poison"] == 1


def test_unsupported_schema_version_is_poison(settings, sqs, urls, store, metrics):
    body = json.loads(dump_message(incoming()))
    body["schema_version"] = 99
    send_message(sqs, urls.incoming, json.dumps(body))
    make_classifier(settings, sqs, urls, store, metrics, StubModel()).run_once(0)
    (dead,) = drain(sqs, urls.incoming_dlq)
    assert dead["MessageAttributes"]["error_reason"]["StringValue"] == "unsupported_schema_version"


def test_missing_field_is_poison(settings, sqs, urls, store, metrics):
    send_message(sqs, urls.incoming, json.dumps({"schema_version": 1, "message_id": "m", "correlation_id": "c"}))
    make_classifier(settings, sqs, urls, store, metrics, StubModel()).run_once(0)
    (dead,) = drain(sqs, urls.incoming_dlq)
    assert dead["MessageAttributes"]["error_reason"]["StringValue"] == "schema_validation_failed"


def test_transient_failure_is_retried_then_dead_lettered(settings, sqs, urls, store, metrics):
    msg = incoming()
    enqueue(sqs, urls, msg)
    model = StubModel(error=RuntimeError("model backend down"))
    worker = make_classifier(settings, sqs, urls, store, metrics, model)
    for _ in range(4):  # 3 failed attempts, then SQS redrives on the next receive
        worker.run_once(0)
    assert model.calls == 3
    assert metrics.snapshot()["classifier.retried"] == 3
    (dead,) = drain(sqs, urls.incoming_dlq)
    assert json.loads(dead["Body"])["message_id"] == msg.message_id
    assert store.is_processed("classifier", msg.message_id) is False


def test_failure_in_one_message_does_not_block_the_batch(settings, sqs, urls, store, metrics):
    bad, good = incoming("bad"), incoming("good")
    for m in (bad, good):
        store.create_message(m.message_id, "c1", m.text, m.correlation_id)
        enqueue(sqs, urls, m)

    class PickyModel(StubModel):
        def predict(self, text):
            if text == "bad":
                raise RuntimeError("boom")
            return super().predict(text)

    make_classifier(settings, sqs, urls, store, metrics, PickyModel()).run_once(0)
    assert [json.loads(m["Body"])["message_id"] for m in drain(sqs, urls.classified)] == [good.message_id]
    assert metrics.snapshot()["classifier.retried"] == 1


def test_duplicate_delivery_is_processed_once(settings, sqs, urls, store, metrics):
    msg = incoming()
    store.create_message(msg.message_id, "c1", msg.text, msg.correlation_id)
    enqueue(sqs, urls, msg)
    enqueue(sqs, urls, msg)  # at-least-once delivery: the same logical message arrives twice
    worker = make_classifier(settings, sqs, urls, store, metrics, StubModel())
    worker.run_once(0)
    assert len(drain(sqs, urls.classified)) == 1
    assert metrics.snapshot()["classifier.duplicates"] == 1


def test_responder_stores_reply_once(settings, sqs, urls, store, metrics):
    msg = incoming()
    store.create_message(msg.message_id, "c1", msg.text, msg.correlation_id)
    classified = ClassifiedMessage(
        message_id=msg.message_id, correlation_id=msg.correlation_id, customer_id="c1", text=msg.text,
        intent="billing", confidence=0.9, model_version="v",
    )
    for _ in range(2):
        send_message(sqs, urls.classified, dump_message(classified))
    worker = ResponderWorker(settings, sqs, urls.classified, urls.classified_dlq, TemplateResponder(), store, metrics)
    worker.run_once(0)
    record = store.get_message(msg.message_id)
    assert record["status"] == "responded" and "billing" in record["response"].lower()
    assert metrics.snapshot()["responder.duplicates"] == 1
