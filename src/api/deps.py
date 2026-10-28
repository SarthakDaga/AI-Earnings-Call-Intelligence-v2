"""Shared services, auth and rate limiting."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request

from src import config


@dataclass
class Services:
    retriever: object
    llm: object
    agent: object
    summary_jobs: dict = field(default_factory=dict)


def get_services(request: Request) -> Services:
    return request.app.state.services


def check_admin(x_api_key: str | None) -> None:
    """Write endpoints are disabled unless ADMIN_API_KEY is configured."""
    if not config.ADMIN_API_KEY:
        raise HTTPException(403, "Write endpoints are disabled on this deployment")
    if not x_api_key or not hmac.compare_digest(x_api_key, config.ADMIN_API_KEY):
        raise HTTPException(401, "Missing or invalid X-API-Key")


def require_admin(x_api_key: str | None = Header(default=None)) -> None:
    check_admin(x_api_key)


class RateLimiter:
    """Sliding-window per-IP limiter. `expensive` paths (LLM calls) get a third of the budget."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[tuple[str, str], deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, ip: str, bucket: str) -> tuple[bool, int]:
        if self.per_minute <= 0:
            return True, 0
        limit = max(3, self.per_minute // 3) if bucket == "expensive" else self.per_minute
        now = time.monotonic()
        with self.lock:
            q = self.hits[(ip, bucket)]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= limit:
                return False, int(60 - (now - q[0])) + 1
            q.append(now)
        return True, 0
