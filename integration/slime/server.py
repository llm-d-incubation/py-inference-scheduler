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
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from py_inference_scheduler import Scheduler
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.poller import FetchMetrics, MetricsPoller
from py_inference_scheduler.datalayer.metrics.slime.sglang import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)

# Headers that must not be forwarded verbatim when proxying to a worker.
_HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}
_METRIC_STALENESS_CADENCE = 10

# Returns the prompt the scheduler routes on
RoutingBody = Callable[[dict], object]


class WorkerRegistry:
    """In-memory directory of registered SGLang workers."""

    def __init__(self) -> None:
        self._by_id: dict[str, Endpoint] = {}
        self._url_to_id: dict[str, str] = {}

    def add(self, url: str, worker_type: str = "regular") -> str:
        existing = self._url_to_id.get(url)
        if existing is not None:
            return existing
        worker_id = uuid.uuid4().hex
        self._by_id[worker_id] = Endpoint(
            name=worker_id,
            attributes={"url": url, "worker_type": worker_type, "routing_stats": {}},
        )
        self._url_to_id[url] = worker_id
        return worker_id

    def remove(self, worker_id: str) -> bool:
        ep = self._by_id.pop(worker_id, None)
        if ep is None:
            return False
        self._url_to_id.pop(str(ep.attributes.get("url")), None)
        return True

    def list_workers_as_dicts(self) -> list[dict[str, str]]:
        return [{"url": str(ep.attributes["url"]), "id": ep.name} for ep in self._by_id.values()]

    def endpoints(self) -> list[Endpoint]:
        return list(self._by_id.values())


def _safe_json(raw: bytes) -> dict:
    """Parse request bytes to a dict; {} if empty/invalid/non-dict (never raises)."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Shared aiohttp session: unbounded connections + no per-request timeout.

    vime uses this directly as its app lifespan; slime wraps it (create_app's
    _lifespan) to add the MetricsPoller start/stop around it.
    The latter will be vime's default mode in the future.
    """
    # limit=0 removes aiohttp's default 100-connection cap so requests fan out across engines.
    connector = aiohttp.TCPConnector(limit=0)
    # total=None lifts aiohttp's default 300s cap per generation
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        app.state.http = session
        yield


async def schedule_and_proxy(  # noqa: PLR0913
    request: Request,
    *,
    registry: WorkerRegistry,
    inflight: InflightStore,
    scheduling_lock: asyncio.Lock,
    scheduler: Scheduler,
    fetch_metrics: FetchMetrics | None,
    routing_body: RoutingBody,
    generate_path: str,
) -> Response:
    """Pick a worker under the scheduling lock and proxy the request to it.

    If fetch_metrics is given, every endpoint's metrics are scraped inside this
    request before scheduling (vime's mode). If it is None, scheduling reads
    the routing_stats a background MetricsPoller keeps fresh (slime's mode).
    Vime will be changed to follow slime's request path soon.
    """
    raw = await request.body()
    llm_req = LLMRequest(request_id=uuid.uuid4().hex, body=routing_body(_safe_json(raw)))

    endpoints = registry.endpoints()
    if not endpoints:
        return JSONResponse(status_code=503, content={"error": "no workers registered"})

    session: aiohttp.ClientSession = request.app.state.http

    async with scheduling_lock:
        if fetch_metrics is not None:
            await asyncio.gather(*[fetch_metrics(ep, session) for ep in endpoints])
        # Counted here, not by the poller: decisions between two polls must see each other.
        for ep in endpoints:
            ep.attributes["queue_len"] = inflight.get(ep.name)
        selected = scheduler.run(llm_req, candidates=endpoints)
        if not selected:
            return JSONResponse(status_code=503, content={"error": "no worker selected"})
        winner = selected[0].endpoint
        inflight.increment(winner.name)

    worker_url = str(winner.attributes["url"])
    # Forward client headers to the worker, minus hop-by-hop ones aiohttp resets itself.
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    try:
        async with session.post(
            f"{worker_url}{generate_path}", data=raw, headers=fwd_headers
        ) as resp:
            content = await resp.read()
            media_type = resp.headers.get("content-type", "application/json")
            return Response(content=content, status_code=resp.status, media_type=media_type)
    except Exception:
        logger.exception("Proxy to worker %s failed", worker_url)
        return JSONResponse(status_code=502, content={"error": "worker request failed"})
    finally:
        inflight.decrement(winner.name)


def _routing_body(body: dict) -> object:
    """The content the scorers route on: prefer token ids, fall back to text."""
    ids = body.get("input_ids")
    if ids:
        return ids
    return body.get("text", "")


def register_worker_routes(app: FastAPI, registry: WorkerRegistry) -> None:
    @app.post("/workers")
    async def add_worker(request: Request) -> JSONResponse:
        body = _safe_json(await request.body())
        url = body.get("url")
        if not url:
            return JSONResponse(status_code=400, content={"error": "missing 'url'"})
        worker_id = registry.add(str(url), str(body.get("worker_type", "regular")))
        logger.info("Registered worker %s -> %s", worker_id, url)
        return JSONResponse(content={"status": "success", "id": worker_id, "url": url})

    @app.get("/workers")
    async def list_workers() -> JSONResponse:
        return JSONResponse(content={"workers": registry.list_workers_as_dicts()})


def create_app(scheduler: Scheduler, metrics_refresh_ms: int = 100) -> FastAPI:
    """Build the slime router FastAPI app around a configured Scheduler."""
    registry = WorkerRegistry()
    inflight = InflightStore()
    scheduling_lock = asyncio.Lock()
    poller = MetricsPoller(registry.endpoints, fetch_worker_metrics, interval_ms=metrics_refresh_ms)

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Run the poller for the app's lifetime, around the shared-session lifespan."""
        poller.start()
        try:
            async with lifespan(app):
                yield
        finally:
            poller.stop()

    app = FastAPI(title="slime sampling router", lifespan=_lifespan)
    app.state.poller = poller
    app.state.last_stale_warn = 0.0
    stale_after = max(0.5, 5 * metrics_refresh_ms / 1000.0)

    register_worker_routes(app, registry)

    @app.delete("/workers/{worker_id}")
    async def delete_worker(worker_id: str) -> JSONResponse:
        if registry.remove(worker_id):
            logger.info("Deregistered worker %s", worker_id)
            return JSONResponse(content={"status": "success"})
        return JSONResponse(status_code=404, content={"status": "not_found"})

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        staleness = poller.staleness()
        if staleness > stale_after:
            now = time.monotonic()
            if now - app.state.last_stale_warn > _METRIC_STALENESS_CADENCE * stale_after:
                logger.warning("routing on stale metrics: snapshot %.2fs old", staleness)
                app.state.last_stale_warn = now
        return await schedule_and_proxy(
            request,
            registry=registry,
            inflight=inflight,
            scheduling_lock=scheduling_lock,
            scheduler=scheduler,
            fetch_metrics=None,
            routing_body=_routing_body,
            generate_path="/generate",
        )

    return app
