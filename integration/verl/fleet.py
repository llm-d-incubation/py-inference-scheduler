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
import os
import uuid
from dataclasses import dataclass
from typing import cast

import aiohttp
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from integration.verl.admission import Admission
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.poller import MetricsPoller
from py_inference_scheduler.datalayer.metrics.verl.fetch_metrics import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint, LLMRequest

_ACTOR_NAME = "rls_fleet"
# A queued admit holds a call slot until it returns, so finish must always find a free one.
_MAX_CALLS = 1_000_000


@dataclass
class FleetSnapshot:
    """Engine metrics and fleet-wide in-flight counts, keyed by engine name."""

    stats: dict[str, dict[str, object]]
    inflight: dict[str, int]


class Fleet(InflightStore):
    """State shared by every AgentLoopWorker, held in one named Ray actor.

    - Recovers the engine handles from verl's balancer once for the whole fleet.
    - Polls engine metrics in the background, so no decision scrapes an engine.
    - Counts in-flight requests per engine across all workers.
    """

    def __init__(self, interval_ms: int) -> None:
        super().__init__()
        self._endpoints: dict[str, Endpoint] = {}
        self._poller = MetricsPoller(
            lambda: list(self._endpoints.values()), self, _fetch, interval_ms=interval_ms
        )
        self._poller.start()

    def watch(self, handles: dict[str, ray.actor.ActorHandle]) -> None:
        """Add these engines to the background poll."""
        for name, handle in handles.items():
            if name not in self._endpoints:
                self._endpoints[name] = Endpoint(
                    name=name, attributes={"replica_obj": handle, "routing_stats": {}}
                )

    def discover(
        self, balancer: ray.actor.ActorHandle, expected: int
    ) -> dict[str, ray.actor.ActorHandle]:
        """Recover the engine handles from verl's balancer, which enumerates only ids.

        - Acquires with unique request ids until every engine is visited, then releases them.
        - Runs in this actor, so concurrent workers never drain the balancer at the same time.
        """
        handles = asyncio.run(_walk(balancer, expected, self._handles()))
        self.watch(handles)
        return handles

    def _handles(self) -> dict[str, ray.actor.ActorHandle]:
        return {name: ep.attributes["replica_obj"] for name, ep in self._endpoints.items()}

    def snapshot(self) -> FleetSnapshot:
        stats = {
            name: ep.attributes.get("routing_stats", {}) for name, ep in self._endpoints.items()
        }
        return FleetSnapshot(cast("dict[str, dict[str, object]]", stats), self.get_all())


class AdmissionFleet(Fleet):
    """The fleet for a profile with flow control, which also admits every request.

    - Admits each request to an engine with room and queues those that fit nowhere.
    - Runs as an async actor, so a queued admit waits without blocking the finishes that free room.
    """

    def __init__(self, interval_ms: int) -> None:
        super().__init__(interval_ms)
        self._retry_interval_s = interval_ms / 1000
        self._admission: Admission | None = None
        self._discovery = asyncio.Lock()

    async def discover(  # type: ignore[override]  # Ray calls either kind the same way
        self, balancer: ray.actor.ActorHandle, expected: int
    ) -> dict[str, ray.actor.ActorHandle]:
        """Fleet's walk, awaited one at a time, so the event loop keeps serving meanwhile."""
        async with self._discovery:
            handles = await _walk(balancer, expected, self._handles())
            self.watch(handles)
            return handles

    async def admit(self, request: LLMRequest) -> Endpoint:
        """Admit a request to an engine with room, waiting in line until one has it."""
        return await self._get_admission().admit(request)

    def finish(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        """Release an admitted request's reservation and retry the requests waiting for one."""
        self._get_admission().finish(request_id, endpoint_name, output_tokens)

    def _get_admission(self) -> Admission:
        if self._admission is None:
            # Loaded once, because a hot reload would replace the plugin and drop its reservations.
            config = SchedulerConfig.from_file(os.environ["ROUTER_CONFIG_PATH"])
            scheduler = Scheduler.new_with_config(config)
            plugins = scheduler.get_flow_control_plugins()
            if len(plugins) != 1:
                raise ValueError("verl admission needs exactly one flow_control plugin.")
            self._admission = Admission(
                scheduler,
                plugins[0],
                self,
                lambda: list(self._endpoints.values()),
                self._retry_interval_s,
            )
        return self._admission


async def _walk(
    balancer: ray.actor.ActorHandle, expected: int, known: dict[str, ray.actor.ActorHandle]
) -> dict[str, ray.actor.ActorHandle]:
    """Acquire engines from verl's balancer until `expected` are known, then release them."""
    handles = dict(known)
    acquired: list[str] = []
    # Live traffic skews the balancer's counters, so an engine can be visited twice.
    for _ in range(expected * 3):
        if len(handles) >= expected:
            break
        server_id, handle = await balancer.acquire_server.remote(
            request_id=f"rls-discover-{uuid.uuid4().hex}"
        )
        acquired.append(server_id)
        handles[server_id] = handle
    for server_id in acquired:
        balancer.release_server.remote(server_id=server_id)
    return handles


async def _fetch(ep: Endpoint, inflight: InflightStore, session: aiohttp.ClientSession) -> None:
    # verl serves engine stats over a Ray call to the server actor, so the HTTP session is unused.
    await fetch_worker_metrics(ep, inflight)


_FleetActor = ray.remote(Fleet)
_AdmissionFleetActor = ray.remote(AdmissionFleet)


def _worker_node() -> str:
    """The caller's node if it advertises CPUs, otherwise the first live node that does.

    verl places agent-loop workers only on nodes with CPUs, so a head node without them runs none.
    """
    here: str = ray.get_runtime_context().get_node_id()
    with_cpus: list[str] = [
        n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU")
    ]
    return here if here in with_cpus or not with_cpus else with_cpus[0]


def fleet_actor(*, admits: bool = False) -> ray.actor.ActorHandle:
    """The job's fleet actor, created by the first caller and shared with the rest by name.

    A profile with flow control gets an AdmissionFleet, any other profile the plain Fleet.
    """
    interval_ms = int(os.environ.get("RLS_METRICS_INTERVAL_MS", "100"))
    # num_cpus=0: an actor pending behind verl's CPU reservations would stall every worker. Zero
    # CPUs also lets Ray pick a head node that runs no workers, so pin it to a node that runs them.
    # The pin is hard because Ray often ignores a soft one for an actor that asks for no CPUs.
    near_workers = NodeAffinitySchedulingStrategy(_worker_node(), soft=False)
    shared = {
        "name": _ACTOR_NAME,
        "get_if_exists": True,
        "num_cpus": 0,
        "scheduling_strategy": near_workers,
    }
    if admits:
        actor = _AdmissionFleetActor.options(max_concurrency=_MAX_CALLS, **shared)
    else:
        actor = _FleetActor.options(**shared)
    return actor.remote(interval_ms)


async def admit(
    fleet: ray.actor.ActorHandle, request: LLMRequest
) -> Endpoint:
    """Ask the fleet actor to admit a request.

    If the caller is cancelled first, an admission that completes later is handed back, since Ray
    keeps the call running.
    """
    admitting: asyncio.Future[Endpoint] = asyncio.ensure_future(
        fleet.admit.remote(request)
    )
    try:
        return await asyncio.shield(admitting)
    except asyncio.CancelledError:
        admitting.add_done_callback(lambda done: _hand_back(fleet, request.request_id, done))
        raise


def _hand_back(
    fleet: ray.actor.ActorHandle,
    request_id: str,
    admitting: asyncio.Future[Endpoint],
) -> None:
    if not admitting.cancelled() and admitting.exception() is None:
        fleet.finish.remote(request_id, admitting.result().name, None)


class PendingFinishes:
    """Sends each turn's finish to the fleet and holds the trajectory's next turn until it has run.

    The AdmissionFleet is async, so Ray does not keep one caller's calls in order. Without the wait,
    the previous turn's finish could run after the next turn's admit and free the new reservation.
    Unlike holding verl's output until the finish lands, this lets the finish overlap the tool call.
    """

    def __init__(self, fleet: ray.actor.ActorHandle) -> None:
        self._fleet = fleet
        self._running: dict[str, asyncio.Task[None]] = {}

    def start(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        self._running[request_id] = asyncio.get_running_loop().create_task(
            self._finish(request_id, endpoint_name, output_tokens)
        )

    async def wait(self, request_id: str) -> None:
        running = self._running.get(request_id)
        if running is not None:
            await running

    async def _finish(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        try:
            await self._fleet.finish.remote(request_id, endpoint_name, output_tokens)
        finally:
            # A newer finish for the same trajectory keeps its entry.
            if self._running.get(request_id) is asyncio.current_task():
                del self._running[request_id]
