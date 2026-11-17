"""Small, cached, rate-limited client for the public Codeforces API."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import requests

LOG = logging.getLogger(__name__)


class CodeforcesAPIError(RuntimeError):
    pass


class CodeforcesAPI:
    def __init__(
        self,
        cache_dir: str | Path = "data/raw/api_cache",
        min_interval_seconds: float = 2.1,
        max_retries: int = 6,
        timeout_seconds: int = 60,
    ):
        self.base_url = "https://codeforces.com/api"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = float(min_interval_seconds)
        self.max_retries = int(max_retries)
        self.timeout = int(timeout_seconds)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CF-FirstAttemptResearch/1.0 (educational project)"
        })
        self._last_request_start = 0.0

    def _cache_path(self, method: str, params: dict[str, Any]) -> Path:
        payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{method.replace('.', '_')}_{digest}.json"

    def _wait_for_slot(self) -> None:
        elapsed = time.monotonic() - self._last_request_start
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_start = time.monotonic()

    def get(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> Any:
        params = params or {}
        cache_path = self._cache_path(method, params)

        if use_cache and not force_refresh and cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        url = f"{self.base_url}/{method}"
        last_error = None

        for attempt in range(self.max_retries):
            self._wait_for_slot()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in (429, 503, 502, 504):
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(60, 2 ** attempt + random.random())
                    LOG.warning("%s returned HTTP %s; sleeping %.1fs", method, response.status_code, delay)
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                payload = response.json()

                if payload.get("status") != "OK":
                    comment = payload.get("comment", "Unknown API error")
                    # The API sometimes reports call-limit issues in a successful HTTP response.
                    if "limit" in comment.lower() or "too many" in comment.lower():
                        delay = min(90, 3 * (attempt + 1) + random.random())
                        LOG.warning("%s API limit response; sleeping %.1fs", method, delay)
                        time.sleep(delay)
                        last_error = CodeforcesAPIError(comment)
                        continue
                    raise CodeforcesAPIError(f"{method}: {comment}")

                result = payload.get("result")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = cache_path.with_suffix(".tmp")
                with temp_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False)
                temp_path.replace(cache_path)
                return result

            except (requests.RequestException, ValueError, CodeforcesAPIError) as exc:
                last_error = exc
                if isinstance(exc, CodeforcesAPIError) and "limit" not in str(exc).lower():
                    raise
                delay = min(60, 2 ** attempt + random.random())
                LOG.warning("Request failed (%s), retry %d/%d in %.1fs",
                            exc, attempt + 1, self.max_retries, delay)
                time.sleep(delay)

        raise CodeforcesAPIError(f"{method} failed after retries: {last_error}")
