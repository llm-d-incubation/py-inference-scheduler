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

import logging
import threading
import time
from typing import Any, Sequence
from uuid import uuid4

from py_inference_scheduler.core.scheduler import Scheduler
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.poller import FetchMetrics, MetricsPoller
from py_inference_scheduler.datalayer.metrics.vime.vllm import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)


class _RouterLoadView(InflightStore):
    """Inflight counts sourced from the router's own per-worker load counters.

    The vllm-router fork tracks inflight requests itself (increment at
    selection, decrement at stream end), so instead of bookkeeping our own
    InflightStore we mirror the loads it hands us on every select() call.
    increment/decrement stay unused no-ops from the parent.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loads: dict[str, int] = {}
        self._loads_lock = threading.Lock()

    def merge(self, loads: dict[str, int]) -> None:
        with self._loads_lock:
            self._loads.update(loads)

    def prune(self, urls: Sequence[str]) -> None:
        with self._loads_lock:
            for url in urls:
                self._loads.pop(url, None)

    def get(self, endpoint_name: str) -> int:
        with self._loads_lock:
            return self._loads.get(endpoint_name, 0)

    def get_all(self) -> dict[str, int]:
        with self._loads_lock:
            return dict(self._loads)


class VllmRouterSchedulerAdapter:
    """Bridges the vllm-router fork's external-policy callable to Scheduler.

    The router calls select() from multiple tokio threads while holding the
    GIL; the adapter lock serializes decisions because scheduler plugins are
    not thread-safe (the GIL alone does not make multi-step selection atomic).
    Workers register with the router, not with us, so the Endpoint registry
    is built lazily from the worker dicts of each call and TTL-pruned.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        *,
        metrics_interval_ms: int = 100,
        endpoint_ttl_s: float = 60.0,
        fetch_metrics: FetchMetrics = fetch_worker_metrics,
    ) -> None:
        self._scheduler = scheduler
        self._lock = threading.Lock()
        self._endpoints: dict[str, Endpoint] = {}
        self._last_seen: dict[str, float] = {}
        self._ttl = endpoint_ttl_s
        self._loads = _RouterLoadView()
        self._poller = MetricsPoller(
            self.list_endpoints, self._loads, fetch_metrics, interval_ms=metrics_interval_ms
        )

    def start(self) -> None:
        self._poller.start()

    def stop(self) -> None:
        self._poller.stop()

    def staleness(self) -> float:
        return self._poller.staleness()

    def list_endpoints(self) -> list[Endpoint]:
        with self._lock:
            return list(self._endpoints.values())

    def seed_workers(self, urls: Sequence[str]) -> None:
        """Pre-register statically known workers for the poller.

        Without this, requests in the first poll interval of a cold boot are
        scheduled on empty stats. Unreachable seeds age out via TTL.
        """
        now = time.monotonic()
        with self._lock:
            for url in urls:
                if url not in self._endpoints:
                    self._endpoints[url] = Endpoint(
                        name=url,
                        attributes={
                            "url": url,
                            "worker_type": "regular",
                            "queue_len": 0,
                            "routing_stats": {},
                        },
                    )
                    self._last_seen[url] = now
                    logger.info("Seeded worker %s", url)

    def select(
        self,
        workers: Sequence[dict[str, Any]],
        request_text: str | None,
        headers: dict[str, str] | None,
    ) -> int | None:
        """External-policy callable: index into workers, or None for fallback.

        Never raises: any failure defers to the router's built-in fallback.
        """
        if not workers:
            return None
        try:
            with self._lock:
                candidates = self._sync_endpoints(workers)
                request = LLMRequest(
                    request_id=uuid4().hex,
                    target_model=str(workers[0].get("model_id") or "") or None,
                    headers=headers or {},
                    body=request_text,
                )
                scored = self._scheduler.run(request, candidates)
        except Exception:
            logger.exception("Selection failed, deferring to router fallback")
            return None
        if not scored:
            return None
        winner_url = scored[0].endpoint.name
        for idx, worker in enumerate(workers):
            if worker["url"] == winner_url:
                return idx
        logger.warning("Winner %s not in offered worker set", winner_url)
        return None

    def _sync_endpoints(self, workers: Sequence[dict[str, Any]]) -> list[Endpoint]:
        """Upsert endpoints from this call's worker dicts; prune the unseen.

        queue_len is written here from the router's load counter and kept
        fresh between requests by the poller via _RouterLoadView.
        """
        now = time.monotonic()
        loads: dict[str, int] = {}
        candidates: list[Endpoint] = []
        for worker in workers:
            url = str(worker["url"])
            load = int(worker.get("load") or 0)
            loads[url] = load
            endpoint = self._endpoints.get(url)
            if endpoint is None:
                endpoint = Endpoint(
                    name=url,
                    attributes={
                        "url": url,
                        "worker_type": str(worker.get("worker_type") or "regular"),
                        "queue_len": 0,
                        "routing_stats": {},
                    },
                )
                self._endpoints[url] = endpoint
                logger.info("Tracking worker %s", url)
            endpoint.attributes["queue_len"] = load
            self._last_seen[url] = now
            candidates.append(endpoint)
        self._loads.merge(loads)
        # TTL substitutes for the deregistration signal we no longer get:
        # the router just stops offering removed workers, and without expiry
        # the poller would scrape dead URLs forever
        expired = [url for url, seen in self._last_seen.items() if now - seen > self._ttl]
        for url in expired:
            self._endpoints.pop(url, None)
            self._last_seen.pop(url, None)
            logger.info("Dropping worker %s (unseen for %.0fs)", url, self._ttl)
        if expired:
            self._loads.prune(expired)
        return candidates
