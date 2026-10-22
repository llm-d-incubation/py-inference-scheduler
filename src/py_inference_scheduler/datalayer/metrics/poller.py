# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Sequence

import aiohttp

from py_inference_scheduler.framework import Endpoint

logger = logging.getLogger(__name__)

FetchMetrics = Callable[[Endpoint, aiohttp.ClientSession], Awaitable[None]]

_LOG_CADENCE = 300


class MetricsPoller:
    """Polls worker metrics in the background at a set interval.

    Runs in its own thread with a private event loop, so the serving loop can
    neither starve the polling nor receive any of its I/O.

    WARNING: the thread shares the GIL, so the poll must stay I/O-bound.
    CPU-bound work here can delay routing decisions.
    staleness() returns the snapshot's age.
    """

    def __init__(
        self,
        list_endpoints: Callable[[], Sequence[Endpoint]],
        fetch_metrics: FetchMetrics,
        interval_ms: int = 100,
    ) -> None:
        self._list_endpoints = list_endpoints
        self._fetch = fetch_metrics
        self._interval = interval_ms / 1000.0
        self._last_refresh = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="metrics-poller", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def staleness(self) -> float:
        """How old the last metrics poll was."""
        return time.monotonic() - self._last_refresh if self._last_refresh else float("inf")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        session = loop.run_until_complete(self._make_session())
        last_log = 0.0
        log_interval = _LOG_CADENCE * self._interval
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                endpoints = self._list_endpoints()
                if endpoints:
                    results = loop.run_until_complete(
                        asyncio.gather(
                            *[self._fetch(ep, session) for ep in endpoints],
                            return_exceptions=True,
                        )
                    )
                    failures = [r for r in results if isinstance(r, BaseException)]
                    if len(failures) < len(endpoints):
                        self._last_refresh = time.monotonic()
                    now = time.monotonic()
                    if failures and now - last_log > log_interval:
                        logger.warning(
                            "metrics-poller: %d/%d fetches failed (first: %r)",
                            len(failures), len(endpoints), failures[0],
                        )
                        last_log = now
                    elif now - last_log > log_interval:
                        logger.info(
                            "metrics-poller: %d endpoints, scrape %.0fms, interval %.0fms",
                            len(endpoints), (now - started) * 1000, self._interval * 1000,
                        )
                        last_log = now
                self._stop.wait(max(0.0, self._interval - (time.monotonic() - started)))
        finally:
            loop.run_until_complete(session.close())
            loop.close()

    @staticmethod
    async def _make_session() -> aiohttp.ClientSession:
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2))
