from __future__ import annotations

import dataclasses

import boto3
import pytest
from moto import mock_aws

from pipeline.classifier.data import generate
from pipeline.classifier.model import IntentModel, Prediction
from pipeline.config import Settings
from pipeline.metrics import Metrics
from pipeline.queues import provision_all
from pipeline.store import Store


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return dataclasses.replace(
        Settings(),
        backoff_base_seconds=0,  # retries become visible immediately so tests do not sleep
        max_receive_count=3,
        wait_time_seconds=0,
        db_path=str(tmp_path / "test.db"),
        model_path=str(tmp_path / "model.joblib"),
        reports_dir=str(tmp_path / "reports"),
    )


@pytest.fixture
def sqs():
    with mock_aws():
        yield boto3.client("sqs", region_name="us-east-1")


@pytest.fixture
def urls(sqs, settings):
    return provision_all(sqs, settings)


@pytest.fixture
def store(settings) -> Store:
    return Store(settings.db_path)


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


@pytest.fixture(scope="session")
def trained_model() -> IntentModel:
    data = generate(120, seed=11)
    return IntentModel.train([e.text for e in data], [e.intent for e in data])


class StubModel:
    """Model double with a fixed prediction, optionally raising."""

    version = "stub-1"

    def __init__(self, intent="billing", confidence=0.95, error: Exception | None = None):
        self.intent, self.confidence, self.error, self.calls = intent, confidence, error, 0

    def predict(self, text: str) -> Prediction:
        self.calls += 1
        if self.error:
            raise self.error
        return Prediction(self.intent, self.confidence)


def drain(sqs, url: str) -> list[dict]:
    """Receive every visible message on ``url`` (and delete it)."""
    out = []
    while True:
        resp = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=10, MessageAttributeNames=["All"])
        msgs = resp.get("Messages", [])
        if not msgs:
            return out
        for m in msgs:
            out.append(m)
            sqs.delete_message(QueueUrl=url, ReceiptHandle=m["ReceiptHandle"])
