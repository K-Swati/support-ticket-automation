"""Command line: ``python -m pipeline train|provision|api|classifier|responder``."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from pipeline.classifier.model import IntentModel
from pipeline.classifier.train import train_and_report
from pipeline.classifier.worker import ClassifierWorker
from pipeline.config import Settings
from pipeline.metrics import Metrics
from pipeline.queues import create_client, provision_all, resolve_all
from pipeline.responder.generator import get_responder
from pipeline.responder.worker import ResponderWorker
from pipeline.store import Store


def _stop_event() -> threading.Event:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    return stop


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pipeline")
    p.add_argument("command", choices=["train", "provision", "api", "classifier", "responder"])
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.command == "train":
        report = train_and_report(settings)
        print(f"model {report['model_version']}  unseen-template accuracy: {report['unseen_templates']['accuracy']}")
        return 0

    if args.command == "api":
        import uvicorn

        from pipeline.api import create_app

        uvicorn.run(create_app(settings), host="0.0.0.0", port=args.port)
        return 0

    sqs = create_client(settings)
    urls = provision_all(sqs, settings) if settings.provision_on_start else resolve_all(sqs, settings)
    if args.command == "provision":
        print("queues ready:", ", ".join(urls.names))
        return 0

    cw = create_client(settings, "cloudwatch") if settings.cloudwatch_enabled else None
    metrics = Metrics(settings.cloudwatch_namespace, cw)
    store = Store(settings.db_path)
    if args.command == "classifier":
        worker = ClassifierWorker(
            settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review,
            IntentModel.load(settings.model_path), store, metrics,
        )
    else:
        worker = ResponderWorker(
            settings, sqs, urls.classified, urls.classified_dlq, get_responder(settings.responder, settings.llm_model),
            store, metrics,
        )
    worker.run_forever(_stop_event())
    return 0


if __name__ == "__main__":
    sys.exit(main())
