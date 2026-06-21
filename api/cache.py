"""경로 결과 캐시 — TTL 기반 인메모리 딕셔너리."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class TTLCache:
    """단순 TTL 캐시. 스레드 안전 불필요 (단일 worker 가정)."""

    def __init__(self, maxsize: int = 2048, ttl_seconds: float = 300.0) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            del self._store[k]
        if len(self._store) >= self.maxsize:
            oldest = sorted(self._store, key=lambda k: self._store[k][0])
            for k in oldest[: len(self._store) - self.maxsize + 1]:
                del self._store[k]

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        ts, value = entry
        if time.monotonic() - ts > self.ttl:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._evict()
        self._store[key] = (time.monotonic(), value)

    def make_key(self, *parts: Any) -> str:
        payload = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, int]:
        return {"size": self.size, "hits": self.hits, "misses": self.misses}


route_cache = TTLCache(maxsize=2048, ttl_seconds=300.0)
search_cache = TTLCache(maxsize=512, ttl_seconds=3600.0)
