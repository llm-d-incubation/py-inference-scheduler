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
from integration.verl.fleet import PendingFinishes, admit
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
    return Admission(scheduler, plugin, counts, lambda: engines, retry_interval_s=0.01), counts


def _engine(name: str, capacity: int | None = None) -> Endpoint:
    attributes: dict[str, object] = {"routing_stats": {}}
    if capacity is not None:
        attributes["kv_cache_size"] = capacity
    return Endpoint(name=name, attributes=attributes)


def _turn(trajectory: str, prompt_tokens: int) -> LLMRequest:
    return LLMRequest(request_id=trajectory, body=list(range(prompt_tokens)))


async def _queued(admission: Admission, request: LLMRequest) -> asyncio.Task[Endpoint]:
    task = asyncio.create_task(admission.admit(request))
    await asyncio.sleep(0)
    assert not task.done()
    return task


async def test_scorers_pick_among_engines_with_room():
    full, busy, idle = _engine("full", 100), _engine("busy", 100), _engine("idle", 100)
    admission, counts = _admission([full, busy, idle])
    for name in ("busy", "busy", "busy", "idle", "idle"):
        counts.increment(name)
    assert await admission.admit(_turn("filler", 60)) is full

    # full still has the shortest queue but only 40 tokens free.
    assert await admission.admit(_turn("t1", 50)) is idle
    assert counts.get("idle") == 3


async def test_a_request_waits_until_space_frees():
    engine = _engine("e1", 100)
    admission, counts = _admission([engine])
    await admission.admit(_turn("a", 60))
    waiter = await _queued(admission, _turn("b", 60))

    admission.finish("a", "e1", None)

    assert await waiter is engine
    assert counts.get("e1") == 1


async def test_waiting_requests_go_longest_prompt_first():
    engine = _engine("e1", 1000)
    admission, _ = _admission([engine])
    await admission.admit(_turn("a", 900))
    shorter = await _queued(admission, _turn("b", 400))
    longer = await _queued(admission, _turn("c", 700))

    admission.finish("a", "e1", None)
    assert await longer is engine
    assert not shorter.done()

    admission.finish("c", "e1", None)
    assert await shorter is engine


async def test_a_trajectory_returns_to_its_engine_over_a_less_loaded_one():
    home, other = _engine("home", 100), _engine("other", 100)
    admission, counts = _admission([home, other])
    counts.increment("other")
    assert await admission.admit(_turn("t1", 10)) is home
    admission.finish("t1", "home", 5)
    for _ in range(3):
        counts.increment("home")

    assert await admission.admit(_turn("t1", 30)) is home


async def test_a_cancelled_waiter_leaves_the_line():
    engine = _engine("e1", 100)
    admission, counts = _admission([engine])
    await admission.admit(_turn("a", 100))
    waiter = await _queued(admission, _turn("b", 10))
    waiter.cancel()

    admission.finish("a", "e1", None)

    assert counts.get("e1") == 0
    assert await admission.admit(_turn("c", 100)) is engine


async def test_capacity_from_the_first_poll_admits_waiting_requests():
    engine = _engine("e1")
    admission, _ = _admission([engine])
    waiter = await _queued(admission, _turn("t1", 10))

    engine.attributes["kv_cache_size"] = 100

    assert await asyncio.wait_for(waiter, timeout=1) is engine


async def test_a_waiting_caller_sees_why_admission_failed():
    engine = _engine("e1")
    admission, _ = _admission([engine])
    waiter = await _queued(admission, _turn("t1", 10))

    engine.attributes["routing_stats"] = {"num_running_reqs": 0}

    with pytest.raises(RuntimeError, match="KV capacity"):
        await asyncio.wait_for(waiter, timeout=1)


ENGINE = Endpoint(name="e1", attributes={"replica_obj": "handle"})


def _admitting_fleet() -> SimpleNamespace:
    admitting: asyncio.Future[Endpoint] = asyncio.get_running_loop().create_future()
    finished: list[tuple[object, ...]] = []
    return SimpleNamespace(
        admitting=admitting,
        finished=finished,
        admit=SimpleNamespace(remote=lambda request: admitting),
        finish=SimpleNamespace(remote=lambda *args: finished.append(args)),
    )


async def _cancelled_caller(fleet: SimpleNamespace) -> None:
    caller = asyncio.create_task(admit(fleet, LLMRequest(request_id="t1")))
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller


async def test_returns_the_engine_the_fleet_admitted_to():
    fleet = _admitting_fleet()
    fleet.admitting.set_result(ENGINE)

    assert await admit(fleet, LLMRequest(request_id="t1")) is ENGINE
    assert fleet.finished == []


async def test_hands_back_an_admission_made_after_the_caller_gave_up():
    fleet = _admitting_fleet()
    await _cancelled_caller(fleet)

    fleet.admitting.set_result(ENGINE)
    await asyncio.sleep(0)

    assert fleet.finished == [("t1", "e1", None)]


async def test_hands_back_nothing_when_admission_failed():
    fleet = _admitting_fleet()
    await _cancelled_caller(fleet)

    fleet.admitting.set_exception(RuntimeError("no capacity"))
    await asyncio.sleep(0)

    assert fleet.finished == []


def _finishing_fleet(*results: asyncio.Future[None]) -> SimpleNamespace:
    """A fleet whose finish calls return the given futures in order."""
    calls: list[tuple[object, ...]] = []
    pending = iter(results)

    def finish(*args: object) -> asyncio.Future[None]:
        calls.append(args)
        return next(pending)

    return SimpleNamespace(calls=calls, finish=SimpleNamespace(remote=finish))


def _future() -> asyncio.Future[None]:
    return asyncio.get_running_loop().create_future()


async def test_next_turn_waits_until_the_last_finish_has_run():
    done = _future()
    fleet = _finishing_fleet(done)
    finishes = PendingFinishes(fleet)
    finishes.start("t1", "e1", 32)
    next_turn = asyncio.create_task(finishes.wait("t1"))
    await asyncio.sleep(0)

    assert not next_turn.done()
    done.set_result(None)
    await next_turn
    assert fleet.calls == [("t1", "e1", 32)]


async def test_does_not_wait_without_a_running_finish():
    finishes = PendingFinishes(_finishing_fleet())

    await asyncio.wait_for(finishes.wait("t1"), timeout=1)


async def test_forgets_a_finish_once_it_has_run():
    done = _future()
    finishes = PendingFinishes(_finishing_fleet(done))
    finishes.start("t1", "e1", 32)
    done.set_result(None)
    await finishes.wait("t1")

    assert finishes._running == {}


async def test_an_older_finish_does_not_drop_a_newer_one():
    older, newer = _future(), _future()
    finishes = PendingFinishes(_finishing_fleet(older, newer))
    finishes.start("t1", "e1", 32)
    finishes.start("t1", "e2", 16)
    older.set_result(None)
    await asyncio.sleep(0)
    next_turn = asyncio.create_task(finishes.wait("t1"))
    await asyncio.sleep(0)

    assert not next_turn.done()
    newer.set_result(None)
    await next_turn


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
    assert ("max_concurrency" in created[0][1]) is admits


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
