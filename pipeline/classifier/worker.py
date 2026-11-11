"""Classifier worker: incoming -> (classified | human_review)."""

from __future__ import annotations

import logging
from typing import Any

from pipeline.classifier.model import IntentModel
from pipeline.config import Settings
from pipeline.metrics import Metrics, log_event
from pipeline.queues import ReceivedMessage, send_message
from pipeline.schemas import ClassifiedMessage, HumanReviewMessage, IncomingMessage, dump_message
from pipeline.store import Store
from pipeline.worker_base import Outcome, Worker

logger = logging.getLogger(__name__)


class ClassifierWorker(Worker):
    stage = "classifier"
    message_model = IncomingMessage

    def __init__(
        self,
        settings: Settings,
        sqs: Any,
        source_url: str,
        dlq_url: str,
        classified_url: str,
        human_review_url: str,
        model: IntentModel,
        store: Store,
        metrics: Metrics | None = None,
    ) -> None:
        super().__init__(settings, sqs, source_url, dlq_url, metrics)
        self.classified_url = classified_url
        self.human_review_url = human_review_url
        self.model = model
        self.store = store

    def handle(self, message: IncomingMessage, received: ReceivedMessage) -> Outcome:
        if self.store.is_processed(self.stage, message.message_id):
            return Outcome.DUPLICATE
        prediction = self.model.predict(message.text)
        confident = prediction.confidence >= self.settings.confidence_threshold
        fields = {
            "message_id": message.message_id,
            "correlation_id": message.correlation_id,
            "customer_id": message.customer_id,
            "text": message.text,
            "intent": prediction.intent,
            "confidence": round(prediction.confidence, 4),
            "model_version": self.model.version,
        }
        attrs = {"correlation_id": message.correlation_id}
        if confident:
            send_message(self.sqs, self.classified_url, dump_message(ClassifiedMessage(**fields)), attrs)
        else:
            send_message(self.sqs, self.human_review_url, dump_message(HumanReviewMessage(**fields)), attrs)
            self._count("routed_to_human")
        self.store.save_classification(
            message.message_id, prediction.intent, prediction.confidence, self.model.version,
            "auto" if confident else "human_review",
        )
        # Recorded last: if anything above failed the message is retried and replays are harmless.
        self.store.mark_processed(self.stage, message.message_id)
        log_event(logger, logging.INFO, "classified", intent=prediction.intent, confidence=round(prediction.confidence, 3))
        return Outcome.PROCESSED
