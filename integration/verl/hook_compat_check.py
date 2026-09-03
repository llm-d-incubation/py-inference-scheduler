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
verl GlobalRequestLoadBalancer, then drives InferenceSchedulerServerClient
through bootstrap -> scheduler routing -> generate -> release. Verifies:
endpoint bootstrap drain, prefix-sticky routing via the scheduler engine,
inflight accounting returning to zero, and LB counters untouched.

Usage:
    ROUTER_CONFIG_PATH=integration/verl/examples/scheduler.yaml \
    PYTHONPATH=/tmp/swe_repo:/tmp/swe_repo/src \
        python3 -m integration.verl.hook_compat_check
"""

from __future__ import annotations

import asyncio
import collections

import ray
from omegaconf import OmegaConf


@ray.remote
class FakeServer:
    def __init__(self, name: str):  # noqa: ANN204
        self.name = name
        self.calls = 0

    async def generate(self, **kwargs):  # noqa: ANN201
        self.calls += 1
        return {"token_ids": [1, 2, 3], "server": self.name}

    def get_routing_stats(self):  # noqa: ANN201
        return {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.1}

    def get_calls(self):  # noqa: ANN201
        return self.calls


async def main() -> int:
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)

    from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

    from integration.verl import verl_hook

    print("layout:", verl_hook._VERL_LAYOUT)
    assert verl_hook._VERL_LAYOUT == "modern", "expected modern layout on this verl build"  # noqa: S101

    servers = {f"srv-{i}": FakeServer.remote(f"srv-{i}") for i in range(3)}
    lb = GlobalRequestLoadBalancer.remote(servers)

    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"ignore_eos": False}}})
    client = verl_hook.InferenceSchedulerServerClient(config, load_balancer_handle=lb)

    shared_prefix = list(range(400))
    routed = collections.defaultdict(int)
    for i in range(4):  # same prefix, growing tail: multi-turn shape
        out = await client.generate(
            request_id=f"traj-{i}",
            prompt_ids=shared_prefix + list(range(1000 + i * 50, 1000 + (i + 1) * 50)),
            sampling_params={"temperature": 1.0},
        )
        routed[out["server"]] += 1

    n_endpoints = len(client.core.endpoints)
    residual_inflight = sum(client.core.inflight_store.get(ep.name) for ep in client.core.endpoints)
    lb_status = await lb.get_status.remote()

    print(f"endpoints bootstrapped: {n_endpoints}")
    print(f"routing distribution (same-prefix requests): {dict(routed)}")
    print(f"scheduler inflight residual: {residual_inflight}")
    print(f"LB total_inflight after run: {lb_status['total_inflight']}")

    ok = (
        n_endpoints == 3  # noqa: PLR2004
        and max(routed.values()) == 4  # prefix-sticky: all four hit one server  # noqa: PLR2004
        and residual_inflight == 0
        and lb_status["total_inflight"] == 0
    )
    print("HOOK COMPAT CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
