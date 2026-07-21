"""LLMServerClient that routes via py-inference-scheduler (upstream main)
on verl commit 334d9f8b.

Routing semantics mirror integration/verl/verl_hook.py (the v0.7.1 hook):
per-request metrics refresh under a lock, Scheduler.run() pick, a local
InflightStore feeding the queue_len attribute, and fallback to verl's
global load balancer when the scheduler returns no endpoint.

One deliberate departure: the v0.7.1 hook monkey-patches vLLMHttpServer with
a get_routing_stats() RPC method. On this verl commit the server actors are
created before any hook module is imported, so a class patch can never reach
them. Instead we scrape each server's HTTP /metrics endpoint directly with
upstream's own get_vllm_routing_stats(), via an adapter that supplies
get_server_address() from the load-balancer address string. Same metrics,
same parsing, no patch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

from datalayer.metrics.datastore import InflightStore
from datalayer.metrics.verl.vllm import get_vllm_routing_stats
from scheduling import Scheduler
from scheduling.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)


class _AddrServer:
    """Adapter exposing get_server_address() so upstream's
    get_vllm_routing_stats() can scrape an endpoint known only by address."""

    def __init__(self, addr: str):
        host, port = addr.rsplit(":", 1)
        self._addr = (host, int(port))

    def get_server_address(self):
        return self._addr


class PyInferenceLLMClient(LLMServerClient):
    """Scheduler-routed client; per-process state is rebuilt after unpickling
    into each AgentLoopWorker actor (locks and the Scheduler must not cross
    process boundaries)."""

    def __init__(self, config, load_balancer_handle=None, *, address_to_handle, **kwargs):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._address_to_handle = dict(address_to_handle)
        self._init_local_state()

    def _init_local_state(self) -> None:
        # Worker processes have no logging handler configured, so INFO-level
        # pick logs would be dropped by logging.lastResort (WARNING+ only).
        # Attach a stderr handler once per process; Ray forwards actor stderr
        # to the driver log, giving positive routing evidence in train.log.
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        self._scheduler = Scheduler()  # reads ROUTER_CONFIG_PATH, hot-reloads on mtime
        self._inflight = InflightStore()
        self._lock = asyncio.Lock()
        self._lb_acquired: set[str] = set()
        self._pick_count = 0
        self._scrapers = {addr: _AddrServer(addr) for addr in self._address_to_handle}
        self._endpoints = [
            Endpoint(name=addr, attributes={"replica_obj": handle, "routing_stats": {}})
            for addr, handle in self._address_to_handle.items()
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_scheduler", "_inflight", "_lock", "_lb_acquired",
                    "_pick_count", "_scrapers", "_endpoints"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_local_state()

    async def _refresh_endpoint(self, ep: Endpoint) -> None:
        stats = await get_vllm_routing_stats(self._scrapers[ep.name])
        if stats.get("error"):
            logger.warning("[PyInferenceLLMClient] metrics scrape error for %s: %s",
                           ep.name, stats["error"])
        ep.attributes["queue_len"] = self._inflight.get(ep.name)
        ep.attributes["routing_stats"] = {
            "num_waiting_reqs": stats.get("num_waiting_reqs", 0),
            "num_running_reqs": stats.get("num_running_reqs", 0),
            "kv": stats.get("kv", 0.0),
            "error": stats.get("error"),
        }

    async def _acquire_server(self, request_id: str, prompt_ids: Optional[list[int]] = None):
        if prompt_ids is None:
            # Direct base-class callers bypass the scheduler.
            return await super()._acquire_server(request_id)

        # Same rationale as the v0.7.1 hook: metrics refresh is part of the
        # scheduling critical section so picks always see fresh stats.
        async with self._lock:
            await asyncio.gather(*(self._refresh_endpoint(ep) for ep in self._endpoints))
            req = LLMRequest(request_id=request_id, body=prompt_ids)
            selected = self._scheduler.run(req, candidates=self._endpoints)
            if selected:
                ep = selected[0].endpoint
                self._inflight.increment(ep.name)
                self._pick_count += 1
                if self._pick_count <= 20 or self._pick_count % 500 == 0:
                    logger.info("[PyInferenceLLMClient] pick #%d: %s for request %s "
                                "(inflight=%d)", self._pick_count, ep.name, request_id,
                                self._inflight.get(ep.name))
                return ep.name, ep.attributes["replica_obj"]

        logger.warning("[PyInferenceLLMClient] scheduler returned no endpoints; "
                       "falling back to verl global LB for request %s", request_id)
        self._lb_acquired.add(request_id)
        server_id, handle = await super()._acquire_server(request_id)
        self._inflight.increment(server_id)
        return server_id, handle

    def _release_server(self, server_id: str, request_id: Optional[str] = None) -> None:
        self._inflight.decrement(server_id)
        if request_id is not None and request_id in self._lb_acquired:
            super()._release_server(server_id)
            self._lb_acquired.discard(request_id)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            return await server.generate.remote(
                request_id=uuid4().hex,  # fresh id per turn, as base class does
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **kwargs,
            )
        finally:
            self._release_server(server_id, request_id=request_id)
