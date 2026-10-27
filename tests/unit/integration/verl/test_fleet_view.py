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
from types import SimpleNamespace

import pytest

from integration.verl.fleet import FleetSnapshot, FleetView
from py_inference_scheduler.framework import Endpoint, LLMRequest


def _fleet() -> SimpleNamespace:
    """A fleet whose snapshot and admit calls answer only when a test resolves them."""
    snapshots: list[asyncio.Future[FleetSnapshot]] = []
    admitting: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()

    def snapshot_call() -> asyncio.Future[FleetSnapshot]:
        snapshots.append(asyncio.get_running_loop().create_future())
        return snapshots[-1]

    return SimpleNamespace(
        snapshots=snapshots,
        admitting=admitting,
        snapshot=SimpleNamespace(remote=snapshot_call),
        admit=SimpleNamespace(remote=lambda request, ranking, worker_id, ticket: admitting),
    )


def _queue_len(view: FleetView, name: str) -> object:
    """The in-flight count the view adds for an engine to an empty snapshot."""
    [engine] = view.apply(FleetSnapshot(stats={}, inflight={}), {}, [Endpoint(name=name)])
    return engine.attributes["queue_len"]


async def test_decisions_waiting_together_share_one_snapshot_request():
    fleet = _fleet()
    view = FleetView(fleet)
    first, second = asyncio.ensure_future(view.fetch()), asyncio.ensure_future(view.fetch())
    await asyncio.sleep(0)

    snapshot = FleetSnapshot(stats={}, inflight={"e1": 2})
    fleet.snapshots[0].set_result(snapshot)

    assert len(fleet.snapshots) == 1
    assert (await first)[0] is snapshot
    assert (await second)[0] is snapshot


async def test_a_decision_after_the_answer_asks_again():
    fleet = _fleet()
    view = FleetView(fleet)
    first = asyncio.ensure_future(view.fetch())
    await asyncio.sleep(0)
    fleet.snapshots[0].set_result(FleetSnapshot(stats={}, inflight={}))
    await first

    second = asyncio.ensure_future(view.fetch())
    await asyncio.sleep(0)

    assert len(fleet.snapshots) == 2
    second.cancel()


async def test_the_view_adds_only_the_sends_a_snapshot_missed():
    fleet = _fleet()
    view = FleetView(fleet)
    view.count("e1", 1)
    fetching = asyncio.ensure_future(view.fetch())
    await asyncio.sleep(0)
    view.count("e1", 1)

    # The fleet counted the first send before the snapshot, but not the second.
    fleet.snapshots[0].set_result(FleetSnapshot(stats={}, inflight={"e1": 5}))
    snapshot, base = await fetching
    [engine] = view.apply(snapshot, base, [Endpoint(name="e1")])

    assert engine.attributes["queue_len"] == 6


async def test_the_top_ranked_engine_counts_only_until_the_fleet_answers():
    fleet = _fleet()
    view = FleetView(fleet)
    answer = asyncio.ensure_future(
        view.request_admission(LLMRequest(request_id="t1"), ["e2", "e1"], "w", "k")
    )
    await asyncio.sleep(0)
    assert _queue_len(view, "e2") == 1

    fleet.admitting.set_result(None)

    assert await answer is None
    assert _queue_len(view, "e2") == 0


async def test_a_cancelled_admission_request_leaves_no_count():
    fleet = _fleet()
    view = FleetView(fleet)
    answer = asyncio.ensure_future(
        view.request_admission(LLMRequest(request_id="t1"), ["e2"], "w", "k")
    )
    await asyncio.sleep(0)

    answer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await answer

    assert _queue_len(view, "e2") == 0
