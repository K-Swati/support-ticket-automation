"""Responder worker: classified -> stored response."""

from __future__ import annotations

from typing import Any

from pipeline.config import Settings
from pipeline.metrics import Metrics
from pipeline.queues import ReceivedMessage
from pipeline.responder.generator import Responder
from pipeline.schemas import ClassifiedMessage
from pipeline.store import Store
from pipeline.worker_base import Outcome, Worker


class ResponderWorker(Worker):
    stage = "responder"
    message_model = ClassifiedMessage

    def __init__(
        self,
        settings: Settings,
        sqs: Any,
        source_url: str,
        dlq_url: str,
        responder: Responder,
        store: Store,
        metrics: Metrics | None = None,
    ) -> None:
        super().__init__(settings, sqs, source_url, dlq_url, metrics)
        self.responder = responder
        self.store = store

    def handle(self, message: ClassifiedMessage, received: ReceivedMessage) -> Outcome:
        if self.store.is_processed(self.stage, message.message_id):
            return Outcome.DUPLICATE
        reply = self.responder.respond(message.intent, message.text)
        self.store.save_response(message.message_id, message.intent, message.confidence, reply, self.responder.name)
        self.store.mark_processed(self.stage, message.message_id)
        return Outcome.PROCESSED
