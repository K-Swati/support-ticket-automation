"""Thin boto3 helpers around SQS: provisioning, batching, long polling."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import boto3

from pipeline.config import Settings

SQS_BATCH_LIMIT = 10
DLQ_RETENTION_SECONDS = 14 * 24 * 3600
QUEUE_RETENTION_SECONDS = 4 * 24 * 3600


class QueueError(RuntimeError):
    """An SQS operation partially or fully failed."""


def create_client(settings: Settings, service: str = "sqs") -> Any:
    """boto3 client honouring ``AWS_ENDPOINT_URL`` (LocalStack)."""
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            kwargs["aws_access_key_id"] = "test"  # LocalStack accepts any credentials
            kwargs["aws_secret_access_key"] = "test"
    return boto3.client(service, **kwargs)


@dataclass(frozen=True)
class QueuePair:
    url: str
    arn: str
    dlq_url: str
    dlq_arn: str


@dataclass(frozen=True)
class QueueUrls:
    incoming: str
    incoming_dlq: str
    classified: str
    classified_dlq: str
    human_review: str
    names: dict[str, str] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class ReceivedMessage:
    message_id: str
    receipt_handle: str
    body: str
    receive_count: int
    attributes: dict[str, str]


def get_queue_url(sqs: Any, name: str) -> str:
    return str(sqs.get_queue_url(QueueName=name)["QueueUrl"])


def get_queue_arn(sqs: Any, url: str) -> str:
    return str(sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"])


def create_queue_with_dlq(
    sqs: Any, name: str, dlq_name: str, *, max_receive_count: int = 3, visibility_timeout: int = 30
) -> QueuePair:
    """Create ``name`` plus its dead-letter queue and wire up the redrive policy (idempotent)."""
    dlq_url = sqs.create_queue(
        QueueName=dlq_name, Attributes={"MessageRetentionPeriod": str(DLQ_RETENTION_SECONDS)}
    )["QueueUrl"]
    dlq_arn = get_queue_arn(sqs, dlq_url)
    redrive = json.dumps({"deadLetterTargetArn": dlq_arn, "maxReceiveCount": str(max_receive_count)})
    url = sqs.create_queue(
        QueueName=name,
        Attributes={
            "VisibilityTimeout": str(visibility_timeout),
            "MessageRetentionPeriod": str(QUEUE_RETENTION_SECONDS),
            "RedrivePolicy": redrive,
        },
    )["QueueUrl"]
    return QueuePair(url=url, arn=get_queue_arn(sqs, url), dlq_url=dlq_url, dlq_arn=dlq_arn)


def create_plain_queue(sqs: Any, name: str, visibility_timeout: int = 30) -> str:
    return str(sqs.create_queue(QueueName=name, Attributes={"VisibilityTimeout": str(visibility_timeout)})["QueueUrl"])


def _urls(inc: str, inc_dlq: str, cls: str, cls_dlq: str, review: str, s: Settings) -> QueueUrls:
    return QueueUrls(
        incoming=inc, incoming_dlq=inc_dlq, classified=cls, classified_dlq=cls_dlq, human_review=review,
        names={
            s.incoming_queue: inc, s.incoming_dlq: inc_dlq, s.classified_queue: cls,
            s.classified_dlq: cls_dlq, s.human_review_queue: review,
        },
    )


def provision_all(sqs: Any, settings: Settings) -> QueueUrls:
    """Create every queue the pipeline needs (idempotent)."""
    kw = {"max_receive_count": settings.max_receive_count, "visibility_timeout": settings.visibility_timeout}
    inc = create_queue_with_dlq(sqs, settings.incoming_queue, settings.incoming_dlq, **kw)
    cls = create_queue_with_dlq(sqs, settings.classified_queue, settings.classified_dlq, **kw)
    review = create_plain_queue(sqs, settings.human_review_queue, settings.visibility_timeout)
    return _urls(inc.url, inc.dlq_url, cls.url, cls.dlq_url, review, settings)


def resolve_all(sqs: Any, settings: Settings) -> QueueUrls:
    """Look up URLs of already-existing queues."""
    return _urls(
        get_queue_url(sqs, settings.incoming_queue), get_queue_url(sqs, settings.incoming_dlq),
        get_queue_url(sqs, settings.classified_queue), get_queue_url(sqs, settings.classified_dlq),
        get_queue_url(sqs, settings.human_review_queue), settings,
    )


def _to_attrs(attributes: dict[str, str] | None) -> dict[str, dict[str, str]]:
    return {k: {"DataType": "String", "StringValue": str(v)[:256]} for k, v in (attributes or {}).items()}


def send_message(sqs: Any, url: str, body: str, attributes: dict[str, str] | None = None) -> str:
    kwargs: dict[str, Any] = {"QueueUrl": url, "MessageBody": body}
    if attributes:
        kwargs["MessageAttributes"] = _to_attrs(attributes)
    return str(sqs.send_message(**kwargs)["MessageId"])


def send_batch(sqs: Any, url: str, bodies: Iterable[str], max_attempts: int = 3) -> int:
    """Send many messages in 10-entry batches, retrying only the entries that failed."""
    pending = list(bodies)
    total = len(pending)
    for _ in range(max_attempts):
        failed: list[str] = []
        for i in range(0, len(pending), SQS_BATCH_LIMIT):
            chunk = pending[i : i + SQS_BATCH_LIMIT]
            resp = sqs.send_message_batch(
                QueueUrl=url, Entries=[{"Id": str(n), "MessageBody": b} for n, b in enumerate(chunk)]
            )
            failed.extend(chunk[int(f["Id"])] for f in resp.get("Failed", []))
        pending = failed
        if not pending:
            return total
    raise QueueError(f"{len(pending)} of {total} messages failed to send after {max_attempts} attempts")


def receive_messages(
    sqs: Any, url: str, *, max_messages: int = SQS_BATCH_LIMIT, wait_time: int = 0, visibility_timeout: int | None = None
) -> list[ReceivedMessage]:
    """Long-poll for up to ``max_messages`` messages."""
    kwargs: dict[str, Any] = {
        "QueueUrl": url,
        "MaxNumberOfMessages": max(1, min(max_messages, SQS_BATCH_LIMIT)),
        "WaitTimeSeconds": wait_time,
        "MessageSystemAttributeNames": ["ApproximateReceiveCount"],
        "MessageAttributeNames": ["All"],
    }
    if visibility_timeout is not None:
        kwargs["VisibilityTimeout"] = visibility_timeout
    resp = sqs.receive_message(**kwargs)
    out: list[ReceivedMessage] = []
    for m in resp.get("Messages", []):
        attrs = {k: v.get("StringValue", "") for k, v in m.get("MessageAttributes", {}).items()}
        out.append(
            ReceivedMessage(
                message_id=m["MessageId"],
                receipt_handle=m["ReceiptHandle"],
                body=m["Body"],
                receive_count=int(m.get("Attributes", {}).get("ApproximateReceiveCount", "1")),
                attributes=attrs,
            )
        )
    return out


def delete_message(sqs: Any, url: str, receipt_handle: str) -> None:
    sqs.delete_message(QueueUrl=url, ReceiptHandle=receipt_handle)


def delete_batch(sqs: Any, url: str, receipt_handles: Sequence[str]) -> None:
    for i in range(0, len(receipt_handles), SQS_BATCH_LIMIT):
        chunk = receipt_handles[i : i + SQS_BATCH_LIMIT]
        resp = sqs.delete_message_batch(
            QueueUrl=url, Entries=[{"Id": str(n), "ReceiptHandle": h} for n, h in enumerate(chunk)]
        )
        if resp.get("Failed"):
            raise QueueError(f"failed to delete {len(resp['Failed'])} messages")


def change_visibility(sqs: Any, url: str, receipt_handle: str, timeout: int) -> None:
    sqs.change_message_visibility(QueueUrl=url, ReceiptHandle=receipt_handle, VisibilityTimeout=timeout)


def backoff_seconds(receive_count: int, base: int, cap: int) -> int:
    """Exponential backoff: base, 2*base, 4*base ... capped (SQS maximum is 12 hours)."""
    if base <= 0:
        return 0
    return int(min(cap, 43200, base * 2 ** max(receive_count - 1, 0)))


def queue_depth(sqs: Any, url: str) -> dict[str, int]:
    attrs = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["All"])["Attributes"]
    return {
        "visible": int(attrs.get("ApproximateNumberOfMessages", 0)),
        "in_flight": int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)),
        "delayed": int(attrs.get("ApproximateNumberOfMessagesDelayed", 0)),
    }
