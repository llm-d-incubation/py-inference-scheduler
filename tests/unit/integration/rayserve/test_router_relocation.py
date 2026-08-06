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


import asyncio
from collections import OrderedDict
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from py_inference_scheduler.datalayer.rayserve.relocation import (
    PULL_OF_HEADER,
    TARGET_HEADER,
)


def _router(registry=None, *, relocation_on=True):
    from integration.rayserve.router import IGWRouter

    router = object.__new__(IGWRouter)
    router._loop = None
    router.deployment_name = "m"
    router._relocation_on = relocation_on
    router._request_replica = OrderedDict(registry or {})
    router._request_replica_cap = 4096
    router.scheduler = SimpleNamespace(has_flow_control=lambda: False)
    return router


def _pending(headers):
    return SimpleNamespace(
        metadata=SimpleNamespace(request_id="serve-id", is_streaming=False),
        args=[
            SimpleNamespace(prompt="hello", model="m"),
            SimpleNamespace(headers=headers),
        ],
    )


def _replica(name):
    return SimpleNamespace(replica_id=name)


def test_target_header_short_circuits_routing():
    router = _router()
    picked = asyncio.run(
        router.choose_replicas(
            [_replica("r1"), _replica("r2")],
            _pending({TARGET_HEADER: "r2"}),
        )
    )
    assert [str(r.replica_id) for r in picked[0]] == ["r2"]


def test_pull_of_header_excludes_pushed_replica():
    router = _router(registry={"push-1": "r1"})
    # build_endpoints is unavailable on this bare router, so the flow falls
    # back to a random choice over the (already filtered) candidates.
    picked = asyncio.run(
        router.choose_replicas(
            [_replica("r1"), _replica("r2")],
            _pending({PULL_OF_HEADER: "push-1"}),
        )
    )
    assert [str(r.replica_id) for r in picked[0]] == ["r2"]


def test_on_request_routed_records_ingress_request_id():
    router = _router()
    router.on_request_routed(_pending({"x-request-id": "push-9"}), "r3", result=None)
    assert router._request_replica["push-9"] == "r3"


def test_target_header_inert_when_relocation_off(monkeypatch):
    import integration.rayserve.router as router_mod

    # Pin the fallback's random pick to index 0 (r1): if the gate leaked, the
    # target header would force r2 instead.
    monkeypatch.setattr(router_mod.random, "randint", lambda a, b: 0)
    router = _router(relocation_on=False)
    picked = asyncio.run(
        router.choose_replicas(
            [_replica("r1"), _replica("r2")],
            _pending({TARGET_HEADER: "r2"}),
        )
    )
    assert [str(r.replica_id) for r in picked[0]] == ["r1"]


def test_registry_not_written_when_relocation_off():
    router = _router(relocation_on=False)
    router.on_request_routed(_pending({"x-request-id": "push-9"}), "r3", result=None)
    assert not router._request_replica


def test_registry_evicts_oldest_beyond_cap():
    router = _router()
    router._request_replica_cap = 2
    for i in range(3):
        router.on_request_routed(_pending({"x-request-id": f"push-{i}"}), f"r{i}", result=None)
    assert list(router._request_replica) == ["push-1", "push-2"]


def test_target_pin_cannot_name_the_pushed_replica():
    router = _router(registry={"push-1": "r1"})
    picked = asyncio.run(
        router.choose_replicas(
            [_replica("r1"), _replica("r2")],
            _pending({PULL_OF_HEADER: "push-1", TARGET_HEADER: "r1"}),
        )
    )
    # r1 is excluded before the pin is considered; fallback picks from [r2].
    assert [str(r.replica_id) for r in picked[0]] == ["r2"]
