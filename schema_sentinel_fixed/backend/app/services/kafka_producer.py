from __future__ import annotations

import json
import os
from typing import Any

from confluent_kafka import Producer

def _producer() -> Producer:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    return Producer({"bootstrap.servers": bootstrap})


def publish_rows(
    *,
    topic: str,
    dataset: str,
    batch_id: str,
    rows: list[dict[str, Any]],
    on_delivery=None,
) -> int:
    """Publish rows into Kafka in the same shape the worker expects."""

    p = _producer()
    sent = 0

    for row in rows:
        msg = {
            "dataset": dataset,
            "batch_id": batch_id,
            "payload": row,
        }
        p.produce(topic, json.dumps(msg).encode("utf-8"), callback=on_delivery)
        sent += 1

        # Let producer send in the background
        p.poll(0)

    p.flush(10)
    return sent
