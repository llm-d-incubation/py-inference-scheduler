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
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import ray

from integration.verl import fleet as fleet_module
from integration.verl.admission import Admission
from integration.verl.fleet import request_admission
from py_inference_scheduler import Scheduler
from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.framework import Endpoint, LLMRequest


def _admission(engines: list[Endpoint]) -> tuple[Admission, InflightStore]:
    config = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "p": {
                "flow_control": {"type": "kv_saturation", "default_osl": 0},
                "scorers": [{"type": "least_queue", "weight": 1.0}],
                "picker": {"type": "max_score"},
            }
        },
    }
    scheduler = Scheduler.new_with_config(SchedulerConfig.from_dict(config))
    plugin = scheduler.get_flow_control_plugins()[0]
    counts = InflightStore()
    return Admission(scheduler, plugin, counts, lambda: engines), counts


def _engine(name: str, capacity: int | None = None) -> Endpoint:
    attributes: dict[str, object] = {"routing_stats": {}}
    if capacity is not None:
        attributes["kv_cache_size"] = capacity
    return Endpoint(name=name, attributes=attributes)


def _turn(trajectory: str, prompt_tokens: int) -> LLMRequest:
    return LLMRequest(request_id=trajectory, body=list(range(prompt_tokens)))


class _Inbox:
    """What a queued request's caller receives."""

    def __init__(self) -> None:
        self.received: list[Endpoint | Exception] = []

    def __call__(self, result: Endpoint | Exception) -> None:
        self.received.append(result)


def _admitted(admission: Admission, request: LLMRequest) -> Endpoint | None:
    return admission.admit(request, request.request_id, _Inbox())


def _queued(admission: Admission, request: LLMRequest) -> _Inbox:
    inbox = _Inbox()
    assert admission.admit(request, request.request_id, inbox) is None
    return inbox


def test_scorers_pick_among_engines_with_room():
    full, busy, idle = _engine("full", 100), _engine("busy", 100), _engine("idle", 100)
    admission, counts = _admission([full, busy, idle])
    for name in ("busy", "busy", "busy", "idle", "idle"):
        counts.increment(name)
    assert _admitted(admission, _turn("filler", 60)) is full

    # full still has the shortest queue but only 40 tokens free.
    assert _admitted(admission, _turn("t1", 50)) is idle
    assert counts.get("idle") == 3


def test_a_request_waits_until_space_frees():
    engine = _engine("e1", 100)
    admission, counts = _admission([engine])
    _admitted(admission, _turn("a", 60))
    waiter = _queued(admission, _turn("b", 60))

    admission.finish("a", "e1", None)

    assert waiter.received == [engine]
    assert counts.get("e1") == 1


def test_waiting_requests_go_longest_prompt_first():
    engine = _engine("e1", 1000)
    admission, _ = _admission([engine])
    _admitted(admission, _turn("a", 900))
    shorter = _queued(admission, _turn("b", 400))
    longer = _queued(admission, _turn("c", 700))

    admission.finish("a", "e1", None)
    assert longer.received == [engine]
    assert shorter.received == []

    admission.finish("c", "e1", None)
    assert shorter.received == [engine]


def test_a_shorter_request_fills_room_too_small_for_the_longest():
    engine = _engine("e1", 100)
    admission, _ = _admission([engine])
    _admitted(admission, _turn("a", 50))
    _admitted(admission, _turn("b", 40))
    longest = _queued(admission, _turn("c", 60))
    shorter = _queued(admission, _turn("d", 45))

    # b's finish frees 40 tokens, leaving 50: too few for c, enough for d.
    admission.finish("b", "e1", None)

    assert longest.received == []
    assert shorter.received == [engine]


def test_an_oversize_request_waits_for_an_idle_engine():
    engine = _engine("e1", 100)
    admission, _ = _admission([engine])
    _admitted(admission, _turn("a", 10))
    oversize = _queued(admission, _turn("big", 500))

    admission.finish("a", "e1", None)

    assert oversize.received == [engine]


def test_a_trajectory_returns_to_its_engine_over_a_less_loaded_one():
    home, other = _engine("home", 100), _engine("other", 100)
    admission, counts = _admission([home, other])
    counts.increment("other")
    assert _admitted(admission, _turn("t1", 10)) is home
    admission.finish("t1", "home", 5)
    for _ in range(3):
        counts.increment("home")

    assert _admitted(admission, _turn("t1", 30)) is home


def test_a_cancelled_request_leaves_the_line():
    engine = _engine("e1", 100)
    admission, counts = _admission([engine])
    _admitted(admission, _turn("a", 100))
    waiter = _queued(admission, _turn("b", 10))
    admission.cancel("b")

    admission.finish("a", "e1", None)

    assert waiter.received == []
    assert counts.get("e1") == 0
    assert _admitted(admission, _turn("c", 100)) is engine


def test_a_retry_admits_waiting_requests_once_the_first_poll_brings_capacity():
    engine = _engine("e1")
    admission, _ = _admission([engine])
    waiter = _queued(admission, _turn("t1", 10))

    engine.attributes["kv_cache_size"] = 100
    admission.retry()

    assert waiter.received == [engine]


def test_a_waiting_caller_sees_why_admission_failed():
    engine = _engine("e1")
    admission, _ = _admission([engine])
    waiter = _queued(admission, _turn("t1", 10))

    engine.attributes["routing_stats"] = {"num_running_reqs": 0}
    admission.retry()

    [error] = waiter.received
    assert isinstance(error, RuntimeError)
    assert "KV capacity" in str(error)


def _admitting_fleet() -> SimpleNamespace:
    admitting: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
    finished: list[tuple[object, ...]] = []
    cancelled: list[str] = []
    return SimpleNamespace(
        admitting=admitting,
        finished=finished,
        cancelled=cancelled,
        admit=SimpleNamespace(remote=lambda request, worker_id, ticket: admitting),
        finish=SimpleNamespace(remote=lambda *args: finished.append(args)),
        cancel=SimpleNamespace(remote=cancelled.append),
    )


async def _cancelled_caller(fleet: SimpleNamespace) -> None:
    caller = asyncio.create_task(
        request_admission(fleet, LLMRequest(request_id="t1"), "w1", "ticket")
    )
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller


async def test_returns_the_engine_the_fleet_admitted_to():
    fleet = _admitting_fleet()
    fleet.admitting.set_result("e1")

    assert await request_admission(fleet, LLMRequest(request_id="t1"), "w1", "ticket") == "e1"
    assert fleet.finished == []


async def test_returns_none_for_a_queued_request():
    fleet = _admitting_fleet()
    fleet.admitting.set_result(None)

    assert await request_admission(fleet, LLMRequest(request_id="t1"), "w1", "ticket") is None


async def test_undoes_an_admission_made_after_the_caller_gave_up():
    fleet = _admitting_fleet()
    await _cancelled_caller(fleet)

    fleet.admitting.set_result("e1")
    await asyncio.sleep(0)

    assert fleet.finished == [("t1", "e1", None)]


async def test_takes_a_queued_request_out_of_line_when_its_caller_gave_up():
    fleet = _admitting_fleet()
    await _cancelled_caller(fleet)

    fleet.admitting.set_result(None)
    await asyncio.sleep(0)

    assert fleet.cancelled == ["ticket"]
    assert fleet.finished == []


async def test_undoes_nothing_when_admission_failed():
    fleet = _admitting_fleet()
    await _cancelled_caller(fleet)

    fleet.admitting.set_exception(RuntimeError("no capacity"))
    await asyncio.sleep(0)

    assert fleet.finished == []
    assert fleet.cancelled == []


def _recording(kind: str, created: list[tuple[str, dict[str, object]]]) -> SimpleNamespace:
    def options(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(remote=lambda interval_ms: created.append((kind, kwargs)))

    return SimpleNamespace(options=options)


@pytest.mark.parametrize(("admits", "kind"), [(True, "admission"), (False, "plain")])
def test_the_profile_decides_which_fleet_the_job_gets(monkeypatch, admits, kind):
    created: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(fleet_module, "_AdmissionFleetActor", _recording("admission", created))
    monkeypatch.setattr(fleet_module, "_FleetActor", _recording("plain", created))
    monkeypatch.setattr(fleet_module, "_worker_node", lambda: "0" * 56)  # a node id's hex length

    fleet_module.fleet_actor(admits=admits)

    assert [k for k, _ in created] == [kind]
    # Concurrent calls would make Ray run a fleet's calls out of order, so neither kind allows them.
    assert "max_concurrency" not in created[0][1]


@pytest.fixture
def ray_session() -> Iterator[None]:
    ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


@pytest.mark.parametrize("admits", [True, False])
def test_a_worker_sees_the_fleet_the_job_already_has(ray_session, admits):
    # Held like the manager holds it: Ray ends an actor once its creator drops every handle.
    job_fleet = fleet_module.fleet_actor(admits=admits)

    # A worker that read a different profile still joins the job's fleet, and can tell its kind.
    joined = fleet_module.fleet_actor(admits=not admits)

    assert hasattr(job_fleet, "admit") is admits
    assert hasattr(joined, "admit") is admits
