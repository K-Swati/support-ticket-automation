# Support Ticket Automation (event-driven ML on AWS SQS)

Automates customer support tickets: a pipeline built from decoupled services that talk only through queues. Each stage can fail, retry and scale on its own, and one bad message never blocks the rest.

```
 client ──> FastAPI ingest ──> [incoming] ──> classifier worker ──┬──> [classified] ──> responder worker ──> result store
              │  idempotency       │ DLQ                         │         │ DLQ
              └─ status API        ▼                             │         ▼
                              [incoming-dlq]                     └──> [human_review]   (low confidence)
                                                                        [classified-dlq]
```

**What it demonstrates**

- **Failure isolation**: malformed messages go straight to a dead-letter queue (DLQ) with the reason attached; transient errors retry with exponential backoff, then SQS moves them to the DLQ
- **Idempotency**: duplicate API calls (idempotency key) and duplicate queue deliveries (SQS is at-least-once) are processed once
- **ML in the loop**: a scikit-learn intent classifier, with low-confidence tickets routed to a human review queue instead of guessed
- **Observability**: JSON logs carrying a correlation id from API to final response, counters, optional CloudWatch metrics
- **Local AWS**: runs on LocalStack with Docker Compose, and tests run against `moto` with no credentials
- Versioned message schemas, FastAPI, Docker, GitHub Actions CI

## Quick start (no Docker, no AWS)

```bash
pip install -r requirements-dev.txt
pytest -q                 # 32 tests against mocked SQS
python scripts/demo.py    # whole pipeline in one process
```

Demo output (40 tickets in phrasings the model has never seen, plus 2 malformed messages):

```
resubmitting key t0 -> duplicate=True (nothing enqueued)
  tickets submitted : 40  ->  {'responded': 28, 'needs_human_review': 12}
  intent correct    : 34/40 (unseen phrasings, synthetic data)
  queue incoming-dlq: 2 message(s)
  queue human_review: 12 message(s)
```

## Run the full system with LocalStack

```bash
docker compose up --build
curl -s localhost:8000/messages -H 'content-type: application/json' \
  -d '{"customer_id":"cust-1","text":"I was charged twice for my Pro plan","idempotency_key":"abc-1"}'
curl -s localhost:8000/messages/<message_id>     # received -> classified -> responded
curl -s localhost:8000/stats                      # queue depths, status counts, counters
```

Pointing at real AWS needs only IAM credentials and no `AWS_ENDPOINT_URL`. Every setting is an environment variable (see `pipeline/config.py`).

## Failure handling

| Situation | What happens |
|---|---|
| Invalid JSON, unsupported `schema_version`, missing field | Copied to the DLQ with `error_reason`, `failed_stage` and `correlation_id` attributes, then deleted. **Not retried**, because it can never succeed |
| Handler raises (model backend down, timeout) | Visibility timeout is set to `base x 2^(attempt-1)` seconds. After `maxReceiveCount` receives, SQS moves it to the DLQ |
| One bad message in a batch of 10 | Only that message is retried or dead-lettered; the other nine complete normally |
| Same message delivered twice | Second delivery is acknowledged with no side effects (idempotency ledger) |
| Same API request sent twice | Same `idempotency_key` returns the original `message_id`, nothing enqueued |
| Enqueue fails in the API | The stored record is removed and the client gets `503`, so no orphan is left |
| Confidence below threshold (default 0.6) | Routed to `human_review` instead of auto-responding |

All of these are covered by tests in `tests/test_workers.py` and `tests/test_api_e2e.py`.

## Classifier results (read this before quoting a number)

The tickets are **synthetic**: generated from phrase templates with random slot values and typo noise (no public dataset is bundled). To avoid flattering numbers, the **last two templates of each intent are held out** and used only for evaluation.

| Evaluation set | Accuracy | Macro F1 |
|---|---|---|
| Fresh noisy tickets from the *training* templates | 1.000 | 1.000 |
| Tickets from **templates never seen in training** | **0.833** | 0.832 |

- The first row is memorisation of templates and means little. The **second row is the honest figure**.
- The biggest confusions are `billing -> general_inquiry` and `technical_issue -> general_inquiry`, the vaguest category.
- Mean confidence drops from 0.99 to 0.72 on unseen phrasing, which is why the human-review route matters.
- These numbers describe synthetic data only and say nothing about real customer text. The value of this project is the pipeline design, and the model is a swappable component.

Reproduce: `python -m pipeline train` writes `reports/classifier_metrics.json`.

## Project layout

```
pipeline/
  api.py              ingest API: POST /messages, GET /messages/{id}, /stats, /health
  worker_base.py      consume loop, retry/backoff, poison handling, idempotency hook
  classifier/         synthetic data, TF-IDF + logistic regression, training report, worker
  responder/          template responder (offline) and optional OpenAI responder, worker
  queues.py           boto3 SQS helpers: provisioning, batch send, long poll, backoff
  schemas.py          versioned pydantic messages, poison-message detection
  store.py            SQLite store with idempotency ledger (swap for DynamoDB/Postgres)
  metrics.py          counters, correlation-id logging, CloudWatch export
scripts/demo.py       one-process end-to-end demo
tests/                32 tests (moto-mocked SQS)
docker-compose.yml    LocalStack + API + two workers
```

## Honest limitations

- `docker-compose.yml` and the Docker image were written but **not run** in the environment this was developed in (no Docker daemon); the tests and demo run against `moto`. Run `docker compose up --build` once on your machine before relying on it.
- SQLite is fine for a demo and a single host. Use DynamoDB or Postgres for several hosts.
- Standard SQS queues are at-least-once and unordered; the design assumes that and does not need FIFO queues. Retries use visibility timeouts rather than SQS delay queues.
- The OpenAI responder is optional (`RESPONDER=openai`) and unit-tested with a fake client, not against the live API.
- Not included: authentication on the API, rate limiting, autoscaling policy, Terraform.

## Ideas for extension

- Replace SQLite with DynamoDB and add Terraform for the queues
- Add a redrive tool that replays DLQ messages after a fix
- Track classifier drift using the stored confidence values
