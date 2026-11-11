"""Environment-driven configuration.

Every knob is an environment variable so the same image runs as any service.
Set ``AWS_ENDPOINT_URL`` (for example ``http://localstack:4566``) to talk to
LocalStack instead of real AWS.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    aws_region: str = "us-east-1"
    aws_endpoint_url: str | None = None
    queue_prefix: str = ""

    # Reliability
    max_receive_count: int = 3  # receives before SQS moves a message to the DLQ
    visibility_timeout: int = 30  # seconds a received message stays hidden
    backoff_base_seconds: int = 2  # retry delay = base * 2 ** (receive_count - 1)
    backoff_max_seconds: int = 300
    wait_time_seconds: int = 10  # long-poll duration
    batch_size: int = 10  # SQS maximum per receive

    # ML routing
    confidence_threshold: float = 0.6  # below this, a human reviews the ticket
    model_path: str = "models/intent_model.joblib"
    reports_dir: str = "reports"

    # Storage
    db_path: str = "data/pipeline.db"

    # Observability
    cloudwatch_enabled: bool = False
    cloudwatch_namespace: str = "SqsMlPipeline"
    log_level: str = "INFO"

    # Behaviour
    provision_on_start: bool = True
    responder: str = "template"  # template | openai
    llm_model: str = "gpt-4o-mini"

    @property
    def incoming_queue(self) -> str:
        return f"{self.queue_prefix}incoming"

    @property
    def incoming_dlq(self) -> str:
        return f"{self.queue_prefix}incoming-dlq"

    @property
    def classified_queue(self) -> str:
        return f"{self.queue_prefix}classified"

    @property
    def classified_dlq(self) -> str:
        return f"{self.queue_prefix}classified-dlq"

    @property
    def human_review_queue(self) -> str:
        return f"{self.queue_prefix}human_review"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        d = cls()
        return cls(
            aws_region=e.get("AWS_REGION", e.get("AWS_DEFAULT_REGION", d.aws_region)),
            aws_endpoint_url=e.get("AWS_ENDPOINT_URL") or None,
            queue_prefix=e.get("QUEUE_PREFIX", d.queue_prefix),
            max_receive_count=int(e.get("MAX_RECEIVE_COUNT", d.max_receive_count)),
            visibility_timeout=int(e.get("VISIBILITY_TIMEOUT", d.visibility_timeout)),
            backoff_base_seconds=int(e.get("BACKOFF_BASE_SECONDS", d.backoff_base_seconds)),
            backoff_max_seconds=int(e.get("BACKOFF_MAX_SECONDS", d.backoff_max_seconds)),
            wait_time_seconds=int(e.get("WAIT_TIME_SECONDS", d.wait_time_seconds)),
            batch_size=int(e.get("BATCH_SIZE", d.batch_size)),
            confidence_threshold=float(e.get("CONFIDENCE_THRESHOLD", d.confidence_threshold)),
            model_path=e.get("MODEL_PATH", d.model_path),
            reports_dir=e.get("REPORTS_DIR", d.reports_dir),
            db_path=e.get("DB_PATH", d.db_path),
            cloudwatch_enabled=_bool(e.get("CLOUDWATCH_METRICS_ENABLED"), d.cloudwatch_enabled),
            cloudwatch_namespace=e.get("CLOUDWATCH_NAMESPACE", d.cloudwatch_namespace),
            log_level=e.get("LOG_LEVEL", d.log_level),
            provision_on_start=_bool(e.get("PROVISION_ON_START"), d.provision_on_start),
            responder=e.get("RESPONDER", d.responder),
            llm_model=e.get("LLM_MODEL", d.llm_model),
        )
