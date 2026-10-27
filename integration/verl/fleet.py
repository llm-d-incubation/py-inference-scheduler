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
import threading
import time
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
        handles = asyncio.run(_collect_engine_handles(balancer, expected, self._handles()))
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

    - Admits each request to the highest-ranked engine its worker sent that has room, or queues it
      in one queue for every worker and calls the worker back once an engine has room.
    - Runs as a sync actor like Fleet, so Ray keeps each worker's calls in order: a turn's finish
      always runs before the trajectory's next admit.
    """

    def __init__(self, interval_ms: int) -> None:
        super().__init__(interval_ms)
        self._admission: Admission | None = None
        self._workers: dict[str, ray.actor.ActorHandle] = {}
        retry_interval_s = interval_ms / 1000
        threading.Thread(target=self._retry_on_polls, args=(retry_interval_s,), daemon=True).start()

    def register(self, worker_id: str, worker: ray.actor.ActorHandle) -> None:
        """Remember how to call a worker back when its queued request is admitted."""
        self._workers[worker_id] = worker

    def admit(
        self, request: LLMRequest, ranking: list[str], worker_id: str, ticket: str
    ) -> str | None:
        """Admit a request and return its engine's name, or queue it and return None."""
        winner = self._get_admission().admit(
            request,
            ranking,
            ticket,
            lambda result: self._deliver(worker_id, ticket, request.request_id, result),
        )
        return winner.name if winner is not None else None

    def cancel(self, ticket: str) -> None:
        """Drop a queued request whose caller gave up."""
        self._get_admission().cancel(ticket)

    def finish(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        """Release an admitted request's reservation and retry the requests waiting for one."""
        self._get_admission().finish(request_id, endpoint_name, output_tokens)

    def retry(self) -> None:
        """Admit the queued requests that now fit."""
        self._get_admission().retry()

    def _deliver(
        self, worker_id: str, ticket: str, request_id: str, result: Endpoint | Exception
    ) -> None:
        name = result.name if isinstance(result, Endpoint) else None
        error = None if name is not None else repr(result)
        self._workers[worker_id].rls_admitted.remote(ticket, request_id, name, error)

    def _retry_on_polls(self, interval_s: float) -> None:
        # Capacity also arrives with metrics polls, when no finish is coming to retry the queue. The
        # retry runs as a call on this actor, so it never races the actor's other calls.
        actor = None
        while True:
            time.sleep(interval_s)
            if self._admission is not None and self._admission.has_waiting():
                actor = actor or ray.get_runtime_context().current_actor
                actor.retry.remote()

    def _get_admission(self) -> Admission:
        if self._admission is None:
            # Loaded once, because a hot reload would replace the plugin and drop its reservations.
            config = SchedulerConfig.from_file(os.environ["ROUTER_CONFIG_PATH"])
            plugins = Scheduler.new_with_config(config).get_flow_control_plugins()
            if len(plugins) != 1:
                raise ValueError("verl admission needs exactly one flow_control plugin.")
            self._admission = Admission(
                plugins[0],
                self,
                lambda: list(self._endpoints.values()),
            )
        return self._admission


async def _collect_engine_handles(
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
    actor = (_AdmissionFleetActor if admits else _FleetActor).options(**shared)
    return actor.remote(interval_ms)


async def request_admission(
    fleet: ray.actor.ActorHandle,
    request: LLMRequest,
    ranking: list[str],
    worker_id: str,
    ticket: str,
) -> str | None:
    """Ask the fleet to admit a request to its top-ranked engine with room, or None if queued.

    If the caller is cancelled first, an admission that completes later is undone and a queued
    request leaves the line, since Ray keeps the call running.
    """
    admitting: asyncio.Future[str | None] = asyncio.ensure_future(
        fleet.admit.remote(request, ranking, worker_id, ticket)
    )
    try:
        return await asyncio.shield(admitting)
    except asyncio.CancelledError:
        admitting.add_done_callback(
            lambda done: _undo_admission(fleet, request.request_id, ticket, done)
        )
        raise


def _undo_admission(
    fleet: ray.actor.ActorHandle,
    request_id: str,
    ticket: str,
    admitting: asyncio.Future[str | None],
) -> None:
    if admitting.cancelled() or admitting.exception() is not None:
        return
    name = admitting.result()
    if name is None:
        fleet.cancel.remote(ticket)
    else:
        fleet.finish.remote(request_id, name, None)


class FleetView:
    """One worker's view of the fleet: shared snapshots plus the worker's own sends.

    - One snapshot request serves every decision waiting at the same moment, so a decision holds
      the worker's lock only while it scores.
    - Adds what this worker sent after a snapshot was requested, which the snapshot misses.
    """

    def __init__(self, fleet: ray.actor.ActorHandle) -> None:
        self._fleet = fleet
        # This worker's in-flight requests per engine as sent, and the snapshot request in flight.
        self._local: dict[str, int] = {}
        self._pending: tuple[asyncio.Future[FleetSnapshot], dict[str, int]] | None = None

    async def fetch(self) -> tuple[FleetSnapshot, dict[str, int]]:
        """The fleet's latest snapshot, and this worker's counts when it was requested."""
        if self._pending is None:
            requested = asyncio.ensure_future(self._fleet.snapshot.remote())
            self._pending = (requested, dict(self._local))
        future, base = self._pending
        try:
            snapshot = await asyncio.shield(future)
        finally:
            if self._pending is not None and self._pending[0] is future:
                self._pending = None
        return snapshot, base

    def apply(
        self, snapshot: FleetSnapshot, base: dict[str, int], endpoints: list[Endpoint]
    ) -> list[Endpoint]:
        """The endpoints, carrying the snapshot's stats and the in-flight counts it missed."""
        for ep in endpoints:
            ep.attributes["routing_stats"] = snapshot.stats.get(ep.name, {})
            unseen = self._local.get(ep.name, 0) - base.get(ep.name, 0)
            ep.attributes["queue_len"] = snapshot.inflight.get(ep.name, 0) + unseen
        return endpoints

    def count(self, name: str, delta: int) -> None:
        self._local[name] = self._local.get(name, 0) + delta

    async def request_admission(
        self, request: LLMRequest, ranking: list[str], worker_id: str, ticket: str
    ) -> str | None:
        """Ask the fleet to admit the request, counting the top-ranked engine until it answers.

        The count goes up before this first yields, so this worker's next decisions see it, and the
        admit call goes out before it too, ahead of this worker's next snapshot request.
        """
        guess = ranking[0] if ranking else None
        if guess:
            self.count(guess, 1)
        try:
            return await request_admission(self._fleet, request, ranking, worker_id, ticket)
        finally:
            if guess:
                self.count(guess, -1)
