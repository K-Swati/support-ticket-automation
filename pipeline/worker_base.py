"""Shared consume loop with retry/backoff, poison handling and idempotency hooks."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar

from pipeline.config import Settings
from pipeline.metrics import Metrics, correlation_context, log_event
from pipeline.metrics import metrics as default_metrics
from pipeline.queues import (
    QueueError,
    ReceivedMessage,
    backoff_seconds,
    change_visibility,
    delete_message,
    receive_messages,
    send_message,
)
from pipeline.schemas import Envelope, PoisonMessageError, parse_message

logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    PROCESSED = "processed"
    DUPLICATE = "duplicate"


class Worker(ABC):
    """Base class: subclasses declare ``stage``/``message_model`` and implement ``handle``.

    Per-message failure policy (one bad message never blocks its neighbours):

    * unparseable or unsupported schema -> copied to the DLQ with error attributes, then deleted
    * ``handle`` raises ``PoisonMessageError`` -> same as above
    * ``handle`` raises any other error -> visibility timeout is set to an exponential backoff and
      the message is left for SQS to redeliver; once ``maxReceiveCount`` is exceeded SQS itself
      moves it to the DLQ
    * duplicates (``Outcome.DUPLICATE``) are acknowledged without side effects
    """

    stage: ClassVar[str]
    message_model: ClassVar[type[Envelope]]

    def __init__(
        self, settings: Settings, sqs: Any, source_url: str, dlq_url: str, metrics: Metrics | None = None
    ) -> None:
        self.settings = settings
        self.sqs = sqs
        self.source_url = source_url
        self.dlq_url = dlq_url
        self.metrics = metrics or default_metrics

    @abstractmethod
    def handle(self, message: Any, received: ReceivedMessage) -> Outcome:
        """Process one validated message. Must be safe to call more than once."""

    def _count(self, name: str, value: int = 1) -> None:
        self.metrics.inc(f"{self.stage}.{name}", value)

    def _dead_letter(self, received: ReceivedMessage, exc: PoisonMessageError) -> None:
        attrs = {
            "error_reason": exc.reason,
            "error_detail": exc.detail[:200],
            "source_queue": self.source_url.rsplit("/", 1)[-1],
            "failed_stage": self.stage,
            "original_message_id": received.message_id,
        }
        if "correlation_id" in received.attributes:
            attrs["correlation_id"] = received.attributes["correlation_id"]
        send_message(self.sqs, self.dlq_url, received.body, attrs)
        delete_message(self.sqs, self.source_url, received.receipt_handle)
        self._count("poison")
        log_event(logger, logging.WARNING, "message_dead_lettered", reason=exc.reason, detail=exc.detail)

    def _retry_later(self, received: ReceivedMessage, exc: Exception) -> None:
        delay = backoff_seconds(received.receive_count, self.settings.backoff_base_seconds, self.settings.backoff_max_seconds)
        change_visibility(self.sqs, self.source_url, received.receipt_handle, delay)
        self._count("retried")
        exhausted = received.receive_count >= self.settings.max_receive_count
        log_event(
            logger,
            logging.WARNING,
            "processing_failed_retries_exhausted" if exhausted else "processing_failed_will_retry",
            error=repr(exc),
            receive_count=received.receive_count,
            retry_in_seconds=delay,
        )

    def _process(self, received: ReceivedMessage) -> None:
        cid = received.attributes.get("correlation_id")
        try:
            with correlation_context(cid):
                message = parse_message(received.body, self.message_model)
            with correlation_context(message.correlation_id):
                outcome = self.handle(message, received)
                delete_message(self.sqs, self.source_url, received.receipt_handle)
                self._count("duplicates" if outcome is Outcome.DUPLICATE else "processed")
                log_event(logger, logging.INFO, f"message_{outcome.value}", message_id=message.message_id)
        except PoisonMessageError as exc:
            with correlation_context(cid):
                self._dead_letter(received, exc)
        except Exception as exc:  # noqa: BLE001 - any handler failure is retried
            with correlation_context(cid):
                self._retry_later(received, exc)

    def run_once(self, wait_time: int | None = None) -> int:
        """Poll once and process everything received. Returns how many were received."""
        batch = receive_messages(
            self.sqs,
            self.source_url,
            max_messages=self.settings.batch_size,
            wait_time=self.settings.wait_time_seconds if wait_time is None else wait_time,
        )
        for received in batch:
            self._count("received")
            self._process(received)
        return len(batch)

    def run_forever(self, stop: threading.Event, flush_every: float = 60.0) -> None:
        log_event(logger, logging.INFO, "worker_started", stage=self.stage, queue=self.source_url)
        last_flush = time.monotonic()
        while not stop.is_set():
            try:
                self.run_once()
            except QueueError:
                logger.exception("queue error")
                stop.wait(2)
            except Exception:  # noqa: BLE001 - keep the worker alive on transient AWS errors
                logger.exception("poll failed")
                stop.wait(5)
            if time.monotonic() - last_flush >= flush_every:
                self._flush_metrics()
                last_flush = time.monotonic()
        self._flush_metrics()
        log_event(logger, logging.INFO, "worker_stopped", stage=self.stage)

    def _flush_metrics(self) -> None:
        try:
            self.metrics.flush()
        except Exception:  # noqa: BLE001 - metrics must never break processing
            logger.exception("metrics flush failed")
