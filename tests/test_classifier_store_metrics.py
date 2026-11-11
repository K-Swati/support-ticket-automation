import json

from pipeline.classifier.data import HELD_OUT, TEMPLATES, generate
from pipeline.classifier.model import IntentModel
from pipeline.classifier.train import train_and_report
from pipeline.metrics import Metrics, correlation_context, current_correlation_id, log_event
from pipeline.responder.generator import OpenAIResponder, get_responder
from pipeline.schemas import INTENTS


def test_data_generation_is_deterministic_and_balanced():
    a, b = generate(20, seed=1), generate(20, seed=1)
    assert [e.text for e in a] == [e.text for e in b]
    assert {i: sum(e.intent == i for e in a) for i in INTENTS} == dict.fromkeys(INTENTS, 20)


def test_held_out_templates_are_disjoint_from_training():
    for templates in TEMPLATES.values():
        assert len(templates) > HELD_OUT + 3
        assert not set(templates[:-HELD_OUT]) & set(templates[-HELD_OUT:])


def test_classifier_generalises_to_unseen_templates(trained_model):
    data = generate(60, seed=99, held_out=True)
    preds = trained_model.predict_many([e.text for e in data])
    accuracy = sum(p.intent == e.intent for p, e in zip(preds, data, strict=True)) / len(data)
    assert accuracy >= 0.7, accuracy  # well above the 0.2 chance level; synthetic data, see README


def test_confidence_is_a_probability(trained_model):
    p = trained_model.predict("why was I billed twice")
    assert 0 <= p.confidence <= 1 and p.intent in INTENTS


def test_model_save_load_roundtrip(trained_model, tmp_path):
    path = tmp_path / "m.joblib"
    trained_model.save(path)
    loaded = IntentModel.load(path)
    assert loaded.version == trained_model.version
    assert loaded.predict("reset my password") == trained_model.predict("reset my password")


def test_train_and_report_writes_metrics(settings):
    report = train_and_report(settings, n_train=60, n_eval=30)
    saved = json.loads(open(f"{settings.reports_dir}/classifier_metrics.json").read())
    assert saved["model_version"] == report["model_version"]
    assert report["in_distribution"]["accuracy"] >= report["unseen_templates"]["accuracy"]


def test_store_is_idempotent_and_never_moves_backwards(store):
    rec, created = store.create_message("m1", "c", "hi", "corr", "key-1")
    assert created
    again, created2 = store.create_message("m2", "c", "hi", "corr2", "key-1")
    assert not created2 and again["message_id"] == "m1"
    store.save_classification("m1", "billing", 0.9, "v", "auto")
    store.save_response("m1", "billing", 0.9, "first reply", "template")
    store.save_response("m1", "billing", 0.9, "second reply", "template")
    store.save_classification("m1", "billing", 0.9, "v", "auto")  # late duplicate
    final = store.get_message("m1")
    assert final["response"] == "first reply" and final["status"] == "responded"
    assert store.mark_processed("classifier", "m1") is True
    assert store.mark_processed("classifier", "m1") is False


def test_metrics_flush_batches_by_twenty_and_resets():
    sent = []

    class CW:
        def put_metric_data(self, Namespace, MetricData):  # noqa: N803
            sent.append(len(MetricData))

    m = Metrics("ns", CW())
    for i in range(45):
        m.inc(f"metric{i}", i + 1)
    assert m.flush() == 45 and sent == [20, 20, 5]
    assert m.flush() == 0 and m.snapshot() == {}


def test_correlation_id_is_attached_to_log_lines(caplog):
    import logging

    with correlation_context("abc123"), caplog.at_level(logging.INFO):
        log_event(logging.getLogger("t"), logging.INFO, "hello", x=1)
    assert current_correlation_id() is None
    assert json.loads(caplog.records[0].message) == {"event": "hello", "correlation_id": "abc123", "x": 1}


def test_openai_responder_uses_model_and_falls_back():
    class Fake:
        def __init__(self, content):
            self.content = content
            self.chat = self
            self.completions = self

        def create(self, **kw):
            self.kw = kw
            msg = type("M", (), {"content": self.content})
            return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    ok = OpenAIResponder("m", client=Fake("  Hello!  "))
    assert ok.respond("billing", "x") == "Hello!" and ok._client.kw["model"] == "m"
    assert "billing" in ok._client.kw["messages"][0]["content"]
    empty = OpenAIResponder("m", client=Fake(""))
    assert "billing" in empty.respond("billing", "x").lower()
    assert get_responder("template").name == "template"
