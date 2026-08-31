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
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import integration.tunix.plugins  # noqa: F401  registers the worker_state filter
from integration.slime.server import _safe_json
from py_inference_scheduler import Scheduler
from py_inference_scheduler.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)


def create_app(scheduler: Scheduler) -> FastAPI:
    """Build the tunix rollout decision-service FastAPI app.

    Unlike the slime/vime routers this app never proxies requests or scrapes
    worker metrics: tunix workers speak tunix's own cloudpickle-gRPC transport,
    so the orchestrator ships each candidate's stats inline with the request
    and dispatches to the winner itself. The sidecar only decides.
    """
    scheduling_lock = asyncio.Lock()
    app = FastAPI(title="tunix rollout scheduler")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    @app.post("/schedule")
    async def schedule(request: Request) -> JSONResponse:
        body = _safe_json(await request.body())
        raw_candidates = body.get("candidates")
        if not raw_candidates or not isinstance(raw_candidates, list):
            return JSONResponse(
                status_code=400, content={"error": "missing 'candidates'"}
            )
        try:
            candidates = [
                Endpoint(name=str(c["name"]), attributes=dict(c.get("attributes") or {}))
                for c in raw_candidates
            ]
        except (KeyError, TypeError):
            return JSONResponse(
                status_code=400, content={"error": "each candidate needs a 'name'"}
            )

        llm_req = LLMRequest(
            request_id=str(body.get("request_id") or uuid.uuid4().hex),
            target_model=body.get("target_model"),
            body=body.get("prompt") or "",
        )

        # Serialize decisions: stateful scorers (prefix indexer) mutate on
        # pre_request, mirroring slime's scheduling_lock.
        async with scheduling_lock:
            selected = scheduler.run(llm_req, candidates=candidates)

        if not selected:
            return JSONResponse(content={"picked": None, "fallback": True, "scores": {}})
        winner = selected[0]
        return JSONResponse(
            content={
                "picked": winner.endpoint.name,
                "fallback": False,
                "scores": {winner.endpoint.name: float(winner.score)},
            }
        )

    @app.post("/reset")
    async def reset() -> JSONResponse:
        """Drop stateful scorer caches (e.g. prefix affinity after weight sync)."""
        reset_count = 0
        async with scheduling_lock:
            for profile in scheduler.profiles.values():
                for w in profile.scorers:
                    reset_fn = getattr(w.scorer, "reset", None)
                    if callable(reset_fn):
                        reset_fn()
                        reset_count += 1
        logger.info("Reset %d stateful scorers", reset_count)
        return JSONResponse(content={"status": "ok", "reset_scorers": reset_count})

    return app
