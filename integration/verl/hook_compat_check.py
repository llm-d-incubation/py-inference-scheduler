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

- both clients see every engine;
- same-prefix turns stick to one engine;
- polled engine load reaches decisions, so a loaded engine is avoided;
- a burst of concurrent first turns spreads evenly over idle engines;
- fleet in-flight counts and the balancer's counters return to zero.

Usage:
    ROUTER_CONFIG_PATH=integration/verl/examples/scheduler.yaml \
    PYTHONPATH=/tmp/swe_repo:/tmp/swe_repo/src \
        python3 -m integration.verl.hook_compat_check
"""

from __future__ import annotations

import asyncio
import collections
from typing import cast

import ray
from omegaconf import OmegaConf

# (running, waiting, kv) per fake engine.
_IDLE = (0, 0, 0.0)
_SKEWED = {"srv-0": (12, 20, 0.97), "srv-1": (0, 0, 0.05), "srv-2": (6, 3, 0.55)}
# Long enough for the fleet poller to pick up a load change.
_SETTLE_S = 0.5
_BURST_N = 48
_BURST_SPREAD_MAX = 6


@ray.remote(num_cpus=0)
class FakeServer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.load = _IDLE
        self.hold_s = 0.0

    async def generate(self, **kwargs):  # noqa: ANN201
        if self.hold_s:
            await asyncio.sleep(self.hold_s)
        return {"token_ids": [1, 2, 3], "server": self.name}

    def configure(self, load: tuple[int, int, float], hold_s: float) -> None:
        self.load, self.hold_s = load, hold_s

    def get_routing_stats(self):  # noqa: ANN201
        running, waiting, kv = self.load
        return {"num_running_reqs": running, "num_waiting_reqs": waiting, "kv": kv}


async def _configure(servers, loads, hold_s: float = 0.0) -> None:
    await asyncio.gather(*(servers[name].configure.remote(loads[name], hold_s) for name in servers))
    await asyncio.sleep(_SETTLE_S)


async def _turn(client, request_id: str, prompt_ids: list[int]) -> str:
    # generate passes through the rollout server's reply: a dict from FakeServer.
    out = cast(
        dict,
        await client.generate(request_id=request_id, prompt_ids=prompt_ids, sampling_params={}),
    )
    return str(out["server"])


async def _prefix_sticky(client, servers) -> bool:
    await _configure(servers, dict.fromkeys(servers, _IDLE))
    prefix = list(range(400))
    routed = collections.Counter([
        await _turn(client, f"sticky-{i}", prefix + list(range(1000 + i * 50, 1050 + i * 50)))
        for i in range(4)
    ])
    print(f"same-prefix turns: {dict(routed)}")
    return max(routed.values()) == 4  # noqa: PLR2004


async def _load_aware(clients, servers) -> bool:
    await _configure(servers, _SKEWED)
    prefix = list(range(400))
    routed = collections.Counter([
        await _turn(clients[i % 2], f"load-{i}", [*prefix, i]) for i in range(12)
    ])
    print(f"turns over loaded srv-0 / idle srv-1 / middling srv-2: {dict(routed)}")
    return routed["srv-0"] == 0 and routed["srv-1"] >= routed["srv-2"]


async def _burst(client, servers) -> bool:
    await _configure(servers, dict.fromkeys(servers, _IDLE), hold_s=1.0)
    routed = collections.Counter(
        await asyncio.gather(*(_turn(client, f"burst-{i}", [7, i]) for i in range(_BURST_N)))
    )
    print(f"burst of {_BURST_N} first turns over idle engines: {dict(routed)}")
    return (
        len(routed) == len(servers)
        and max(routed.values()) - min(routed.values()) <= _BURST_SPREAD_MAX
    )


async def main() -> int:
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)

    from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

    from integration.verl import verl_hook

    print("layout:", verl_hook._VERL_LAYOUT)
    assert verl_hook._VERL_LAYOUT == "modern", "expected modern layout on this verl build"  # noqa: S101

    servers = {f"srv-{i}": FakeServer.remote(f"srv-{i}") for i in range(3)}  # type: ignore[attr-defined]
    # Some verl builds ship the balancer as a Ray actor class, others as a plain class.
    lb_cls = (
        GlobalRequestLoadBalancer
        if hasattr(GlobalRequestLoadBalancer, "remote")
        else ray.remote(GlobalRequestLoadBalancer)
    )
    lb = lb_cls.options(num_cpus=0).remote(servers)

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"ignore_eos": False}}})
    clients = [
        verl_hook.InferenceSchedulerServerClient(config, load_balancer_handle=lb) for _ in range(2)
    ]

    sticky_ok = await _prefix_sticky(clients[0], servers)
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
