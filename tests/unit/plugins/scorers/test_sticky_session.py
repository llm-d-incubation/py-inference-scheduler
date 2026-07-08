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

from py_inference_scheduler.framework import CycleState, Endpoint, LLMRequest, build_scorer
from py_inference_scheduler.plugins import StickySessionScorer

_HEADER_NAME = "x-sticky-session"


def _pods(*names: str) -> dict[str, Endpoint]:
    return {name: Endpoint(name=name) for name in names}


def _request(value: str | None) -> LLMRequest:
    headers = {} if value is None else {_HEADER_NAME: value}
    return LLMRequest(request_id="request-1", headers=headers)


def _selected(scores: dict[str, float]) -> str:
    selected = [name for name, score in scores.items() if score == 1.0]
    assert len(selected) == 1
    return selected[0]


def test_sticky_session_requires_header_name():
    with pytest.raises(ValueError, match="header name"):
        StickySessionScorer(header_name="")


def test_sticky_session_scores_empty_candidates():
    scorer = StickySessionScorer(header_name=_HEADER_NAME)

    assert scorer.score(CycleState(), _request("tenant-a"), {}) == {}


def test_sticky_session_missing_or_empty_header_returns_zero_scores():
    scorer = StickySessionScorer(header_name=_HEADER_NAME)
    pods = _pods("ep1", "ep2")

    assert scorer.score(CycleState(), _request(None), pods) == {"ep1": 0.0, "ep2": 0.0}
    assert scorer.score(CycleState(), _request("   "), pods) == {"ep1": 0.0, "ep2": 0.0}


def test_sticky_session_uses_case_insensitive_header_lookup():
    scorer = StickySessionScorer(header_name="X-Sticky-Session")
    pods = _pods("ep1", "ep2", "ep3")
    lower = LLMRequest(request_id="request-1", headers={_HEADER_NAME: "tenant-a"})
    mixed = LLMRequest(
        request_id="request-1",
        headers={"X-Sticky-Session": "tenant-a"},
    )

    lower_scores = scorer.score(CycleState(), lower, pods)
    mixed_scores = scorer.score(CycleState(), mixed, pods)

    assert lower_scores == mixed_scores


def test_sticky_session_accepts_opaque_identifier_formats():
    scorer = StickySessionScorer(header_name=_HEADER_NAME)
    pods = _pods("ep1", "ep2", "ep3")
    request = _request("tenant/a:chat#42")

    first = scorer.score(CycleState(), request, pods)
    second = scorer.score(CycleState(), request, pods)

    assert first == second
    assert set(first.values()) == {0.0, 1.0}


def test_sticky_session_remaps_only_when_winner_is_removed():
    scorer = StickySessionScorer(header_name=_HEADER_NAME)
    pods = _pods("ep1", "ep2", "ep3", "ep4")
    request = _request("tenant-a")

    original = _selected(scorer.score(CycleState(), request, pods))
    remaining = {name: endpoint for name, endpoint in pods.items() if name != original}
    remapped = _selected(scorer.score(CycleState(), request, remaining))
    restored = _selected(scorer.score(CycleState(), request, pods))

    assert remapped != original
    assert restored == original


def test_sticky_session_can_be_built_from_registry():
    scorer = build_scorer("sticky_session", header_name=_HEADER_NAME)

    assert isinstance(scorer, StickySessionScorer)
