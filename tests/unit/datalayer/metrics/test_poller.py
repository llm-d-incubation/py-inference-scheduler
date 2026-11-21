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

import itertools
import time

from py_inference_scheduler.datalayer.metrics.poller import MetricsPoller
from py_inference_scheduler.framework import Endpoint


def _wait_for(cond, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _counting_fetch(counter):
    async def fetch(ep, session):
        ep.attributes["routing_stats"] = {"tick": next(counter)}

    return fetch


def _failing_fetch():
    async def fetch(ep, session):
        raise RuntimeError("scrape failed")

    return fetch


def test_staleness_is_infinite_before_first_fetch():
    p = MetricsPoller(list, _counting_fetch(itertools.count()))
    assert p.staleness() == float("inf")


def test_refreshes_stats_and_becomes_fresh():
    ep = Endpoint(name="a", attributes={})
    p = MetricsPoller(lambda: [ep], _counting_fetch(itertools.count()), interval_ms=10)
    p.start()
    try:
        assert _wait_for(lambda: "routing_stats" in ep.attributes)
        assert _wait_for(lambda: p.staleness() < 1.0)
    finally:
        p.stop()


def test_all_failures_never_mark_fresh():
    ep = Endpoint(name="a", attributes={})
    p = MetricsPoller(lambda: [ep], _failing_fetch(), interval_ms=10)
    p.start()
    try:
        time.sleep(0.1)
        assert p.staleness() == float("inf")
    finally:
        p.stop()


def test_partial_failure_still_counts_as_fresh():
    ok, bad = Endpoint(name="ok", attributes={}), Endpoint(name="bad", attributes={})
    counting = _counting_fetch(itertools.count())
    failing = _failing_fetch()

    async def fetch(ep, session):
        if ep.name == "bad":
            await failing(ep, session)
        else:
            await counting(ep, session)

    p = MetricsPoller(lambda: [ok, bad], fetch, interval_ms=10)
    p.start()
    try:
        assert _wait_for(lambda: p.staleness() < 1.0)
        assert "routing_stats" not in bad.attributes
    finally:
        p.stop()


def test_stop_halts_polling():
    ep = Endpoint(name="a", attributes={})
    p = MetricsPoller(lambda: [ep], _counting_fetch(itertools.count()), interval_ms=10)
    p.start()
    assert _wait_for(lambda: ep.attributes.get("routing_stats"))
    p.stop()
    assert _wait_for(lambda: not p._thread.is_alive())
    tick = ep.attributes["routing_stats"]["tick"]
    time.sleep(0.05)
    assert ep.attributes["routing_stats"]["tick"] == tick


def test_new_endpoints_are_picked_up_between_cycles():
    eps: list[Endpoint] = [Endpoint(name="a", attributes={})]
    p = MetricsPoller(lambda: list(eps), _counting_fetch(itertools.count()), interval_ms=10)
    p.start()
    try:
        assert _wait_for(lambda: "routing_stats" in eps[0].attributes)
        late = Endpoint(name="late", attributes={})
        eps.append(late)
        assert _wait_for(lambda: "routing_stats" in late.attributes)
    finally:
        p.stop()
