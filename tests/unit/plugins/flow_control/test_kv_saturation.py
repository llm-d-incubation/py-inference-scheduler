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

import pytest

from py_inference_scheduler.framework import Endpoint, LLMRequest
from py_inference_scheduler.framework.registry import build_flow_control
from py_inference_scheduler.plugins.flow_control import kv_saturation
from py_inference_scheduler.plugins.flow_control.kv_saturation import KVSaturationPlugin


def _engine(name: str, capacity: int) -> Endpoint:
    return Endpoint(name=name, attributes={"kv_cache_size": capacity})


def _turn(trajectory: str, prompt_tokens: int) -> LLMRequest:
    return LLMRequest(request_id=trajectory, body=list(range(prompt_tokens)))


def _offered(plugin: KVSaturationPlugin, request: LLMRequest, engines: list[Endpoint]) -> list[str]:
    return [ep.name for ep in plugin.get_allowed_candidates(request, engines)]


def test_is_registered():
    assert isinstance(build_flow_control("kv_saturation", default_osl=8), KVSaturationPlugin)


def test_rejects_a_negative_default_osl():
    with pytest.raises(ValueError, match="default_osl"):
        KVSaturationPlugin(default_osl=-1)


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_rejects_a_budget_fraction_outside_zero_to_one(fraction: float):
    with pytest.raises(ValueError, match="budget_fraction"):
        KVSaturationPlugin(budget_fraction=fraction)


def test_budget_fraction_keeps_headroom_unreserved():
    plugin = KVSaturationPlugin(default_osl=0, budget_fraction=0.5)
    engine = _engine("e1", 200)
    plugin.reserve(_turn("t1", 100), engine)

    # Half of 200 is the budget, and 100 of it is held.
    assert _offered(plugin, _turn("t2", 1), [engine]) == []


def test_a_first_turn_needs_its_prompt_plus_default_osl():
    plugin = KVSaturationPlugin(default_osl=10)
    fits, short = _engine("fits", 200), _engine("short", 200)
    plugin.reserve(_turn("a", 90), fits)
    plugin.reserve(_turn("b", 91), short)

    # 90 prompt + 10 output = 100, against 100 and 99 tokens still free.
    assert _offered(plugin, _turn("t1", 90), [fits, short]) == ["fits"]


def test_waits_for_engines_without_a_successful_poll():
    plugin = KVSaturationPlugin(default_osl=0)
    unpolled = Endpoint(name="unpolled", attributes={"routing_stats": {}})
    failed = Endpoint(name="failed", attributes={"routing_stats": {"error": "timeout"}})

    assert _offered(plugin, _turn("t1", 1), [unpolled, failed]) == []


def test_fails_when_polled_engines_report_no_capacity():
    plugin = KVSaturationPlugin(default_osl=0)
    engine = Endpoint(name="e1", attributes={"routing_stats": {"num_running_reqs": 0}})

    with pytest.raises(RuntimeError, match="KV capacity"):
        plugin.get_allowed_candidates(_turn("t1", 1), [engine])


def test_reservations_consume_the_budget_until_released():
    plugin = KVSaturationPlugin(default_osl=0)
    engine = _engine("e1", 100)
    first, second = _turn("t1", 60), _turn("t2", 60)

    plugin.reserve(first, engine)
    assert _offered(plugin, second, [engine]) == []

    plugin.release(first, "e1")
    assert _offered(plugin, second, [engine]) == ["e1"]


def test_reserving_a_request_again_moves_its_reservation():
    plugin = KVSaturationPlugin(default_osl=0)
    e1, e2 = _engine("e1", 100), _engine("e2", 100)
    plugin.reserve(_turn("t1", 60), e1)

    assert _offered(plugin, _turn("t1", 60), [e1]) == ["e1"]
    plugin.reserve(_turn("t1", 60), e2)

    assert _offered(plugin, _turn("t2", 100), [e1, e2]) == ["e1"]
    plugin.release(_turn("t1", 60), "e2")
    assert _offered(plugin, _turn("t2", 100), [e1, e2]) == ["e1", "e2"]


def test_offers_the_last_engine_alone_when_it_fits():
    plugin = KVSaturationPlugin(default_osl=0)
    engines = [_engine("e1", 100), _engine("e2", 100)]
    plugin.reserve(_turn("t1", 10), engines[1])
    plugin.release(_turn("t1", 10), "e2", output_tokens=5)

    assert _offered(plugin, _turn("t1", 20), engines) == ["e2"]


def test_offers_every_fitting_engine_when_the_last_engine_is_full():
    plugin = KVSaturationPlugin(default_osl=0)
    engines = [_engine("e1", 100), _engine("e2", 100), _engine("e3", 100)]
    plugin.reserve(_turn("t1", 10), engines[1])
    plugin.release(_turn("t1", 10), "e2", output_tokens=5)
    plugin.reserve(_turn("other", 95), engines[1])

    assert _offered(plugin, _turn("t1", 20), engines) == ["e1", "e3"]


def test_next_turn_assumes_the_average_output():
    plugin = KVSaturationPlugin(default_osl=500)
    engine = _engine("e1", 700)
    plugin.reserve(_turn("t1", 10), engine)
    plugin.release(_turn("t1", 10), "e1", output_tokens=40)
    plugin.reserve(_turn("other", 100), engine)

    # other books 100 + 500 default, leaving 100: 60 prompt + 40 average output fits, 61 + 40 not.
    assert _offered(plugin, _turn("t1", 60), [engine]) == ["e1"]
    assert _offered(plugin, _turn("t1", 61), [engine]) == []


def test_an_oversize_request_waits_for_an_idle_engine_and_fills_it():
    plugin = KVSaturationPlugin(default_osl=0)
    engine = _engine("e1", 100)
    big, small = _turn("big", 150), _turn("small", 1)

    plugin.reserve(small, engine)
    assert _offered(plugin, big, [engine]) == []
    plugin.release(small, "e1")
    assert _offered(plugin, big, [engine]) == ["e1"]

    plugin.reserve(big, engine)
    assert _offered(plugin, small, [engine]) == []


def test_forgets_the_oldest_trajectories(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(kv_saturation, "_MAX_REMEMBERED_TRAJECTORIES", 1)
    plugin = KVSaturationPlugin(default_osl=0)
    engines = [_engine("e1", 100), _engine("e2", 100)]
    for trajectory in ("t1", "t2"):
        plugin.reserve(_turn(trajectory, 10), engines[1])
        plugin.release(_turn(trajectory, 10), "e2", output_tokens=5)

    assert _offered(plugin, _turn("t1", 10), engines) == ["e1", "e2"]
    assert _offered(plugin, _turn("t2", 10), engines) == ["e2"]


def test_releasing_an_unknown_request_changes_nothing():
    plugin = KVSaturationPlugin(default_osl=0)
    plugin.release(_turn("never-reserved", 10), "e1")
    assert _offered(plugin, _turn("t1", 100), [_engine("e1", 100)]) == ["e1"]


def test_max_admissible_kv_tokens_is_the_budget_minus_reservations():
    plugin = KVSaturationPlugin(default_osl=0, budget_fraction=0.5)
    engine = _engine("e1", 200)
    plugin.reserve(_turn("t1", 30), engine)

    assert plugin.max_admissible_kv_tokens(engine) == 70


def test_an_engine_without_capacity_has_no_room():
    assert KVSaturationPlugin().max_admissible_kv_tokens(Endpoint(name="e1", attributes={})) == 0


@pytest.mark.parametrize(("held", "prompt"), [(0, 500), (30, 70), (30, 71), (100, 1)])
def test_the_size_check_agrees_with_the_offer(held: int, prompt: int):
    plugin = KVSaturationPlugin(default_osl=0)
    engine = _engine("e1", 100)
    if held:
        plugin.reserve(_turn("held", held), engine)

    # A queue may skip a request by size alone only when the full check turns it away too.
    request = _turn("t1", prompt)
    fits_by_size = plugin.kv_tokens_needed(request) <= plugin.max_admissible_kv_tokens(engine)
    assert fits_by_size == (_offered(plugin, request, [engine]) == ["e1"])
