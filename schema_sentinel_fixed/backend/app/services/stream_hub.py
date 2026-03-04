from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, Deque


class StreamHub:
    """A tiny in-memory pub/sub for the "Stream Console".

    - Producers (e.g., simulate batch) publish JSON events.
    - Frontend connects via Server-Sent Events (SSE) and receives events live.

    Notes:
    - This is DEV-friendly (in-memory). Restarting the backend clears history.
    - Works best with a single backend process (uvicorn --reload is fine).
    """

    def __init__(self, max_history: int = 500):
        self._history: Deque[str] = deque(maxlen=max_history)
        self._subs: set[asyncio.Queue[str]] = set()
        self._lock = asyncio.Lock()

    async def publish(self, event: dict[str, Any]) -> None:
        msg = json.dumps(event, ensure_ascii=False, default=str)
        self._history.append(msg)
        async with self._lock:
            for q in list(self._subs):
                # best-effort (don't block on slow clients)
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

    async def subscribe(self) -> tuple[asyncio.Queue[str], list[str]]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._subs.add(q)
            history = list(self._history)
        return q, history

    async def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        async with self._lock:
            self._subs.discard(q)


# Global singleton (simple for dev)
STREAM_HUB = StreamHub()
