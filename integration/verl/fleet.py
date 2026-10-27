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

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.poller import MetricsPoller
from py_inference_scheduler.datalayer.metrics.verl.fetch_metrics import fetch_worker_metrics
from py_inference_scheduler.framework import Endpoint

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


def _worker_node() -> str:
    """The caller's node if it advertises CPUs, otherwise the first live node that does.

    verl places agent-loop workers only on nodes with CPUs, so a head node without them runs none.
    """
    here: str = ray.get_runtime_context().get_node_id()
    with_cpus: list[str] = [
        n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU")
    ]
    return here if here in with_cpus or not with_cpus else with_cpus[0]


def fleet_actor() -> ray.actor.ActorHandle:
    """The fleet actor, created by the first caller and shared with the rest by name."""
    interval_ms = int(os.environ.get("RLS_METRICS_INTERVAL_MS", "100"))
    # num_cpus=0: an actor pending behind verl's CPU reservations would stall every worker. Zero
    # CPUs also lets Ray pick a head node that runs no workers, so pin it to a node that runs them.
    # The pin is hard because Ray often ignores a soft one for an actor that asks for no CPUs.
    near_workers = NodeAffinitySchedulingStrategy(_worker_node(), soft=False)
    return _FleetActor.options(
        name=_ACTOR_NAME, get_if_exists=True, num_cpus=0, scheduling_strategy=near_workers
    ).remote(interval_ms)
