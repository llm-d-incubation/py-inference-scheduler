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

"""Hook compat check for the modern (verl v0.9.x) layout, GPU-free.

Runs on the Ray head pod: spins up fake rollout-server actors plus a REAL
verl GlobalRequestLoadBalancer, then drives two InferenceSchedulerServerClients,
standing in for two AgentLoopWorkers, through discovery, routing, generate and
release. Verifies:

- Both clients see every engine.
- Same-prefix turns stick to one engine.
- Polled engine load reaches decisions, so a loaded engine is avoided.
- A burst of concurrent first turns spreads evenly over idle engines.
- Fleet in-flight counts and the balancer's counters return to zero.

Usage:
    ROUTER_CONFIG_PATH=integration/verl/examples/scheduler.yaml \
    PYTHONPATH=/tmp/swe_repo:/tmp/swe_repo/src \
        python3 -m integration.verl.hook_compat_check
"""

from __future__ import annotations

import asyncio
import collections
import time
from types import SimpleNamespace
from typing import NamedTuple

import ray
from omegaconf import OmegaConf


class Load(NamedTuple):
    running: int
    waiting: int
    kv: float


_IDLE = Load(running=0, waiting=0, kv=0.0)


def _stats(load: Load) -> dict[str, float]:
    """The routing stats a vLLM engine under this load reports."""
    return {"num_running_reqs": load.running, "num_waiting_reqs": load.waiting, "kv": load.kv}


@ray.remote(num_cpus=0)
class FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.load = _IDLE
        self.hold_s = 0.0

    async def generate(self, **kwargs):  # noqa: ANN201
        if self.hold_s:
            await asyncio.sleep(self.hold_s)
        # Shaped like verl's TokenOutput, whose token_ids the hook counts.
        return SimpleNamespace(token_ids=[1, 2, 3], server=self.name)

    def configure(self, load: Load, hold_s: float) -> None:
        self.load, self.hold_s = load, hold_s

    def get_routing_stats(self):  # noqa: ANN201
        return _stats(self.load)


async def _set_loads(fleet, servers, loads: dict[str, Load], hold_s: float = 0.0) -> None:
    """Give each fake engine a load, then wait until the fleet's poll reports it.

    The fleet only polls engines it has discovered, so call this after the first routed turn.
    """
    await asyncio.gather(*(servers[name].configure.remote(loads[name], hold_s) for name in servers))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        polled = (await fleet.snapshot.remote()).stats
        if all(_reports(polled.get(name, {}), loads[name]) for name in servers):
            return
        await asyncio.sleep(0.05)
    msg = "the fleet's poll never reported the new engine loads"
    raise TimeoutError(msg)


def _reports(stats: dict[str, object], load: Load) -> bool:
    return all(stats.get(key) == value for key, value in _stats(load).items())


async def _turn(client, request_id: str, prompt_ids: list[int]) -> str:
    # generate passes the fake engine's reply through, so it names the engine that served the turn.
    out = await client.generate(request_id=request_id, prompt_ids=prompt_ids, sampling_params={})
    return str(out.server)


async def _prefix_sticky(client) -> bool:
    # The fake engines start idle, so only the shared prefix decides where the turns go.
    turns = 4
    prefix = list(range(400))
    routed = collections.Counter([
        await _turn(client, f"sticky-{i}", prefix + list(range(1000 + i * 50, 1050 + i * 50)))
        for i in range(turns)
    ])
    print(f"same-prefix turns: {dict(routed)}")
    return max(routed.values()) == turns


async def _load_aware(clients, servers) -> bool:
    loads = {
        "srv-0": Load(running=12, waiting=20, kv=0.97),
        "srv-1": Load(running=0, waiting=0, kv=0.05),
        "srv-2": Load(running=6, waiting=3, kv=0.55),
    }
    await _set_loads(clients[0].core.fleet, servers, loads)
    prefix = list(range(400))
    routed = collections.Counter([
        await _turn(clients[i % 2], f"load-{i}", [*prefix, i]) for i in range(12)
    ])
    print(f"turns over loaded srv-0 / idle srv-1 / middling srv-2: {dict(routed)}")
    return routed["srv-0"] == 0 and routed["srv-1"] >= routed["srv-2"]


async def _burst(client, servers) -> bool:
    turns = 48
    max_gap = 6  # between the busiest and the quietest engine
    # Each turn holds its engine for a second, so every turn is in flight at the same time.
    await _set_loads(client.core.fleet, servers, dict.fromkeys(servers, _IDLE), hold_s=1.0)
    routed = collections.Counter(
        await asyncio.gather(*(_turn(client, f"burst-{i}", [7, i]) for i in range(turns)))
    )
    print(f"burst of {turns} first turns over idle engines: {dict(routed)}")
    return len(routed) == len(servers) and max(routed.values()) - min(routed.values()) <= max_gap


async def main() -> int:
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)

    from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

    from integration.verl import verl_hook

    print("layout:", verl_hook._VERL_LAYOUT)
    assert verl_hook._VERL_LAYOUT == "modern", "expected modern layout on this verl build"  # noqa: S101

    servers = {f"srv-{i}": FakeServer.remote(f"srv-{i}") for i in range(3)}  # type: ignore[attr-defined]
    # verl 0.9.x ships the balancer as a plain class, so wrap it as an actor here.
    lb = ray.remote(GlobalRequestLoadBalancer).options(num_cpus=0).remote(servers)

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"ignore_eos": False}}})
    clients = [
        verl_hook.InferenceSchedulerServerClient(config, load_balancer_handle=lb) for _ in range(2)
    ]

    sticky_ok = await _prefix_sticky(clients[0])
    load_ok = await _load_aware(clients, servers)
    burst_ok = await _burst(clients[0], servers)

    views = [len(c.core.endpoints) for c in clients]
    snapshot = await clients[0].core.fleet.snapshot.remote()
    residual = sum(snapshot.inflight.values())
    lb_status = await lb.get_status.remote()
    print(f"engines seen per client: {views}")
    print(f"fleet in-flight residual: {residual}")
    print(f"LB total_inflight after run: {lb_status['total_inflight']}")

    ok = (
        sticky_ok
        and load_ok
        and burst_ok
        and views == [len(servers)] * len(clients)
        and residual == 0
        and lb_status["total_inflight"] == 0
    )
    print("HOOK COMPAT CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
