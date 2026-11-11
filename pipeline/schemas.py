"""Versioned message schemas shared by every service.

Each queue message is a JSON document with an explicit ``schema_version``.
Consumers validate before doing any work; anything they cannot understand is a
*poison message* (see :class:`PoisonMessageError`) and is dead-lettered
immediately instead of being retried pointlessly.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

INTENTS: tuple[str, ...] = ("billing", "technical_issue", "account_access", "cancellation", "general_inquiry")
Intent = Literal["billing", "technical_issue", "account_access", "cancellation", "general_inquiry"]


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class PoisonMessageError(Exception):
    """The message can never be processed successfully; retrying is pointless."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class Envelope(BaseModel):
    """Fields common to every message on every queue."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = SCHEMA_VERSION
    message_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=utcnow_iso)


class IncomingMessage(Envelope):
    """Ingest API -> classifier worker."""

    customer_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=5000)


class ClassifiedMessage(Envelope):
    """Classifier worker -> responder worker."""

    customer_id: str
    text: str
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str


class HumanReviewMessage(ClassifiedMessage):
    """Classifier worker -> human review queue (confidence below threshold)."""

    reason: str = "low_confidence"


def parse_message[M: Envelope](raw: str, model: type[M]) -> M:
    """Parse and validate ``raw``; raise :class:`PoisonMessageError` if impossible."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PoisonMessageError("invalid_json", str(exc)) from exc
    if not isinstance(data, dict):
        raise PoisonMessageError("invalid_json", "top-level JSON value is not an object")
    version = data.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PoisonMessageError("unsupported_schema_version", f"got {version!r}")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise PoisonMessageError("schema_validation_failed", exc.errors()[0]["msg"]) from exc


def dump_message(message: BaseModel) -> str:
    return message.model_dump_json()


# ---- API request/response models ---------------------------------------
class MessageIn(BaseModel):
    customer_id: str = Field(min_length=1, max_length=128, examples=["cust-1042"])
    text: str = Field(min_length=1, max_length=5000, examples=["I was charged twice this month"])
    idempotency_key: str | None = Field(default=None, max_length=128)


class MessageAccepted(BaseModel):
    message_id: str
    correlation_id: str
    status: str
    duplicate: bool = False
