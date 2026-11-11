"""SQLite persistence: messages, results and the idempotency ledger.

A connection is opened per operation (WAL mode), which keeps the store safe to
share across FastAPI worker threads and across processes that mount the same
file. For a real deployment swap this module for DynamoDB or Postgres; the
surface is deliberately small.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    text            TEXT NOT NULL,
    correlation_id  TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    message_id    TEXT PRIMARY KEY,
    intent        TEXT,
    confidence    REAL,
    model_version TEXT,
    route         TEXT,
    response      TEXT,
    generator     TEXT,
    classified_at TEXT,
    responded_at  TEXT
);
CREATE TABLE IF NOT EXISTS processed (
    stage        TEXT NOT NULL,
    key          TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    PRIMARY KEY (stage, key)
);
"""

STATUS_RECEIVED = "received"
STATUS_CLASSIFIED = "classified"
STATUS_RESPONDED = "responded"
STATUS_NEEDS_REVIEW = "needs_human_review"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- messages ------------------------------------------------------
    def create_message(
        self, message_id: str, customer_id: str, text: str, correlation_id: str, idempotency_key: str | None = None
    ) -> tuple[dict[str, Any], bool]:
        """Insert a message. Returns ``(record, created)``.

        If ``idempotency_key`` was seen before, the original record is returned
        with ``created=False`` and nothing is inserted.
        """
        now = _now()
        with self._conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO messages VALUES (?,?,?,?,?,?,?,?)",
                (message_id, customer_id, text, correlation_id, idempotency_key, STATUS_RECEIVED, now, now),
            )
            created = cur.rowcount == 1
            if created:
                key = message_id
            else:
                row = c.execute(
                    "SELECT message_id FROM messages WHERE idempotency_key = ? OR message_id = ?",
                    (idempotency_key, message_id),
                ).fetchone()
                key = row["message_id"]
        record = self.get_message(key)
        assert record is not None
        return record, created

    def delete_message(self, message_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                """SELECT m.message_id, m.customer_id, m.text, m.correlation_id, m.status,
                          m.created_at, m.updated_at,
                          r.intent, r.confidence, r.model_version, r.route,
                          r.response, r.generator, r.classified_at, r.responded_at
                   FROM messages m LEFT JOIN results r USING (message_id)
                   WHERE m.message_id = ?""",
                (message_id,),
            ).fetchone()
        return dict(row) if row else None

    def status_counts(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) AS n FROM messages GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    # ---- results -------------------------------------------------------
    def save_classification(self, message_id: str, intent: str, confidence: float, model_version: str, route: str) -> None:
        """Record the classifier's decision. ``route`` is ``auto`` or ``human_review``."""
        now = _now()
        status = STATUS_CLASSIFIED if route == "auto" else STATUS_NEEDS_REVIEW
        with self._conn() as c:
            c.execute(
                """INSERT INTO results (message_id, intent, confidence, model_version, route, classified_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(message_id) DO UPDATE SET
                     intent=excluded.intent, confidence=excluded.confidence,
                     model_version=excluded.model_version, route=excluded.route,
                     classified_at=excluded.classified_at""",
                (message_id, intent, confidence, model_version, route, now),
            )
            # Never move a message backwards (a late duplicate after it was answered).
            c.execute(
                "UPDATE messages SET status=?, updated_at=? WHERE message_id=? AND status IN (?,?)",
                (status, now, message_id, STATUS_RECEIVED, STATUS_CLASSIFIED),
            )

    def save_response(self, message_id: str, intent: str, confidence: float, response: str, generator: str) -> None:
        """Record the generated response. First writer wins; replays are no-ops."""
        now = _now()
        with self._conn() as c:
            c.execute(
                """INSERT INTO results (message_id, intent, confidence, route, response, generator, responded_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(message_id) DO UPDATE SET
                     response=excluded.response, generator=excluded.generator,
                     responded_at=excluded.responded_at
                   WHERE results.response IS NULL""",
                (message_id, intent, confidence, "auto", response, generator, now),
            )
            c.execute(
                "UPDATE messages SET status=?, updated_at=? WHERE message_id=?",
                (STATUS_RESPONDED, now, message_id),
            )

    # ---- idempotency ledger -------------------------------------------
    def is_processed(self, stage: str, key: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT 1 FROM processed WHERE stage=? AND key=?", (stage, key)).fetchone()
        return row is not None

    def mark_processed(self, stage: str, key: str) -> bool:
        """Atomically record completion. Returns False if it was already recorded."""
        with self._conn() as c:
            cur = c.execute("INSERT OR IGNORE INTO processed VALUES (?,?,?)", (stage, key, _now()))
            return cur.rowcount == 1
