import json

import pytest

from pipeline import queues as q
from tests.conftest import drain


def test_provision_is_idempotent(sqs, settings, urls):
    again = q.provision_all(sqs, settings)
    assert again.incoming == urls.incoming and again.incoming_dlq == urls.incoming_dlq
    assert q.resolve_all(sqs, settings).classified == urls.classified


def test_redrive_policy_points_at_dlq(sqs, urls):
    attrs = sqs.get_queue_attributes(QueueUrl=urls.incoming, AttributeNames=["RedrivePolicy"])["Attributes"]
    policy = json.loads(attrs["RedrivePolicy"])
    assert policy["deadLetterTargetArn"].endswith("incoming-dlq") and int(policy["maxReceiveCount"]) == 3


def test_send_receive_delete_roundtrip(sqs, urls):
    q.send_message(sqs, urls.incoming, "hello", {"correlation_id": "c1"})
    (msg,) = q.receive_messages(sqs, urls.incoming)
    assert msg.body == "hello" and msg.receive_count == 1 and msg.attributes["correlation_id"] == "c1"
    q.delete_message(sqs, urls.incoming, msg.receipt_handle)
    assert q.receive_messages(sqs, urls.incoming) == []


def test_send_batch_splits_into_tens(sqs, urls):
    assert q.send_batch(sqs, urls.incoming, [f"m{i}" for i in range(25)]) == 25
    assert len(drain(sqs, urls.incoming)) == 25


def test_send_batch_retries_only_failed_entries():
    calls = []

    class Flaky:
        def send_message_batch(self, QueueUrl, Entries):  # noqa: N803
            calls.append([e["MessageBody"] for e in Entries])
            failed = [{"Id": "1"}] if len(calls) == 1 else []
            return {"Failed": failed}

    assert q.send_batch(Flaky(), "u", ["a", "b", "c"]) == 3
    assert calls == [["a", "b", "c"], ["b"]]


def test_send_batch_raises_when_entries_keep_failing():
    class Broken:
        def send_message_batch(self, QueueUrl, Entries):  # noqa: N803
            return {"Failed": [{"Id": "0"}]}

    with pytest.raises(q.QueueError):
        q.send_batch(Broken(), "u", ["a"], max_attempts=2)


def test_backoff_is_exponential_and_capped():
    assert [q.backoff_seconds(n, 2, 300) for n in (1, 2, 3, 4)] == [2, 4, 8, 16]
    assert q.backoff_seconds(20, 2, 300) == 300
    assert q.backoff_seconds(3, 0, 300) == 0


def test_message_moves_to_dlq_after_max_receives(sqs, urls):
    q.send_message(sqs, urls.incoming, "never acked")
    for _ in range(4):  # 3 receives allowed; the 4th triggers the redrive
        for m in q.receive_messages(sqs, urls.incoming):
            q.change_visibility(sqs, urls.incoming, m.receipt_handle, 0)
    assert q.queue_depth(sqs, urls.incoming_dlq)["visible"] == 1


def test_queue_depth_reports_counts(sqs, urls):
    q.send_batch(sqs, urls.human_review, ["a", "b"])
    assert q.queue_depth(sqs, urls.human_review)["visible"] == 2
