"""Ingest API: accepts messages, enqueues them, and reports their processing status."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from pipeline.config import Settings
from pipeline.metrics import correlation_context, log_event
from pipeline.metrics import metrics as default_metrics
from pipeline.queues import QueueUrls, create_client, provision_all, queue_depth, resolve_all, send_message
from pipeline.schemas import IncomingMessage, MessageAccepted, MessageIn, dump_message, new_id
from pipeline.store import Store

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, sqs: Any | None = None, store: Store | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.sqs = sqs or create_client(settings)
        app.state.store = store or Store(settings.db_path)
        app.state.urls = (
            provision_all(app.state.sqs, settings) if settings.provision_on_start else resolve_all(app.state.sqs, settings)
        )
        yield

    app = FastAPI(title="SQS ML pipeline - ingest API", version="1.0.0", lifespan=lifespan)

    @app.post("/messages", status_code=202, response_model=MessageAccepted)
    def submit(body: MessageIn) -> MessageAccepted:
        st: Store = app.state.store
        urls: QueueUrls = app.state.urls
        message_id, correlation_id = new_id(), new_id()
        record, created = st.create_message(message_id, body.customer_id, body.text, correlation_id, body.idempotency_key)
        if not created:  # same idempotency key: return the original, enqueue nothing
            return MessageAccepted(
                message_id=record["message_id"], correlation_id=record["correlation_id"],
                status=record["status"], duplicate=True,
            )
        envelope = IncomingMessage(
            message_id=message_id, correlation_id=correlation_id, customer_id=body.customer_id, text=body.text
        )
        try:
            send_message(app.state.sqs, urls.incoming, dump_message(envelope), {"correlation_id": correlation_id})
        except Exception as exc:
            st.delete_message(message_id)  # do not leave a record that will never be processed
            with correlation_context(correlation_id):
                log_event(logger, logging.ERROR, "enqueue_failed", error=repr(exc))
            raise HTTPException(503, "queue unavailable, please retry") from exc
        default_metrics.inc("api.accepted")
        return MessageAccepted(message_id=message_id, correlation_id=correlation_id, status=record["status"])

    @app.get("/messages/{message_id}")
    def get_message(message_id: str) -> dict:
        record = app.state.store.get_message(message_id)
        if record is None:
            raise HTTPException(404, "message not found")
        return record

    @app.get("/stats")
    def stats() -> dict:
        urls: QueueUrls = app.state.urls
        return {
            "statuses": app.state.store.status_counts(),
            "queues": {name: queue_depth(app.state.sqs, url) for name, url in urls.names.items()},
            "counters": default_metrics.snapshot(),
        }

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
