"""Rate limiting for the Beasty Bar web UI."""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from _04_ui.core.config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW,
    ):
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()
        self._last_sweep = 0.0

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        with self._lock:
            # Periodic sweep of stale entries to prevent unbounded memory growth
            if now - self._last_sweep > self._window:
                self._last_sweep = now
                stale = [
                    k
                    for k, ts in self._requests.items()
                    if not ts or now - ts[-1] >= self._window
                ]
                for k in stale:
                    del self._requests[k]

            # Clean old entries for this client
            self._requests[client_id] = [
                t for t in self._requests[client_id] if now - t < self._window
            ]
            if len(self._requests[client_id]) >= self._max_requests:
                return False
            self._requests[client_id].append(now)
            return True
