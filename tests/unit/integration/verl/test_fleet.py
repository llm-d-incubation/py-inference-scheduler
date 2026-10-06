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
import time
from collections.abc import Awaitable, Callable, Iterator

import pytest
import ray

from integration.verl.fleet import Fleet, FleetSnapshot, fleet_actor

_FleetActor = ray.remote(Fleet)


@ray.remote(num_cpus=0)
class _Engine:
    def __init__(self, kv: float) -> None:
        self.kv = kv

    def get_routing_stats(self) -> dict[str, float]:
        return {"num_running_reqs": 1, "num_waiting_reqs": 0, "kv": self.kv}


@ray.remote(num_cpus=0)
class _Balancer:
    """verl's GlobalRequestLoadBalancer reduced to a least-in-flight acquire."""

    def __init__(self, servers: dict[str, object]) -> None:
        self.servers = servers
        self.inflight = dict.fromkeys(servers, 0)
        self.acquires = 0

    def acquire_server(self, request_id: str) -> tuple[str, object]:
        server_id = min(self.inflight, key=lambda name: self.inflight[name])
        self.inflight[server_id] += 1
        self.acquires += 1
        return server_id, self.servers[server_id]

    def release_server(self, server_id: str) -> None:
        self.inflight[server_id] -= 1

    def counters(self) -> tuple[int, int]:
        return self.acquires, sum(self.inflight.values())


@pytest.fixture(scope="module")
def ray_session() -> Iterator[None]:
    ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


async def _eventually(read: Callable[[], Awaitable[object]], expected: object) -> object:
    """Fire-and-forget calls from another caller land later, so poll until they have."""
    deadline = time.monotonic() + 10
    value = await read()
    while value != expected and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        value = await read()
    return value


def _engines() -> dict[str, object]:
    return {f"e{i}": _Engine.remote(kv=i / 10) for i in range(3)}  # type: ignore[attr-defined]


async def test_workers_share_one_fleet_by_name(ray_session: None) -> None:
    worker_a, worker_b = fleet_actor(), fleet_actor()
    worker_a.increment.remote("shared-e1")
    worker_b.increment.remote("shared-e1")
    worker_b.increment.remote("shared-e2")
    worker_b.decrement.remote("shared-e2")

    snapshot: FleetSnapshot = await worker_a.snapshot.remote()

    assert snapshot.inflight["shared-e1"] == 2
    assert snapshot.inflight["shared-e2"] == 0


async def test_discover_visits_every_engine_and_releases_the_balancer(ray_session: None) -> None:
    engines = _engines()
    balancer = _Balancer.remote(engines)  # type: ignore[attr-defined]
    fleet = _FleetActor.remote(50)

    handles = await fleet.discover.remote(balancer, len(engines))

    assert set(handles) == set(engines)
    assert await _eventually(balancer.counters.remote, (3, 0)) == (3, 0)


async def test_discover_drains_once_per_fleet(ray_session: None) -> None:
    engines = _engines()
    balancer = _Balancer.remote(engines)  # type: ignore[attr-defined]
    fleet = _FleetActor.remote(50)

    await fleet.discover.remote(balancer, len(engines))
    await fleet.discover.remote(balancer, len(engines))

    acquires, _ = await balancer.counters.remote()
    assert acquires == len(engines)


async def test_watched_engines_are_polled_in_the_background(ray_session: None) -> None:
    fleet = _FleetActor.remote(50)
    await fleet.watch.remote(_engines())

    async def polled_kv() -> dict[str, object]:
        snapshot: FleetSnapshot = await fleet.snapshot.remote()
        return {name: stats.get("kv") for name, stats in snapshot.stats.items()}

    expected = {"e0": 0.0, "e1": 0.1, "e2": 0.2}
    assert await _eventually(polled_kv, expected) == expected
