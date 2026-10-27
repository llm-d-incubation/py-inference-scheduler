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

import os
import textwrap

import pytest

from py_inference_scheduler import Scheduler, SchedulerConfig
from py_inference_scheduler.framework import Endpoint, LLMRequest, SchedulerProfile, WeightedScorer
from py_inference_scheduler.plugins import (
    ConstantScorer,
    MaxScorePicker,
    QueueLengthScorer,
    SimpleFilter,
    SingleProfileHandler,
)


class RandomPicker:
    def pick(self, cycle_state, request, scored_endpoints):
        if not scored_endpoints:
            return None
        # pick the first scored endpoint
        return scored_endpoints[0]


def make_scheduler_with_profile(profile: SchedulerProfile) -> Scheduler:
    ph = SingleProfileHandler()
    cfg = SchedulerConfig(profile_handler=ph, profiles={profile.name: profile})
    return Scheduler.new_with_config(cfg)


def test_no_candidate_pods_raises():
    profile = SchedulerProfile(name="default")
    s = make_scheduler_with_profile(profile)

    with pytest.raises(ValueError, match="no scheduling candidates provided"):
        s.schedule(LLMRequest(request_id="1", target_model="m"), [])


def test_finds_highest_score_pod():
    # create two pods
    p1 = Endpoint(name="pod1")
    p2 = Endpoint(name="pod2")

    # create scorer that gives pod1 score 1, pod2 score 2
    class MapScorer:
        def score(self, cycle_state, request, pods):
            # `pods` is a mapping name->Endpoint
            if hasattr(pods, "items"):
                return {name: (2.0 if name == "pod2" else 1.0) for name in pods}
            # fallback for sequences
            return {p.name: (2.0 if p.name == "pod2" else 1.0) for p in pods}

    profile = (
        SchedulerProfile(name="default")
        .with_scorers(WeightedScorer(MapScorer(), 1.0))
        .with_picker(RandomPicker())
    )
    s = make_scheduler_with_profile(profile)

    res = s.schedule(LLMRequest(request_id="r", target_model="m"), [p1, p2])
    assert res is not None
    pr = res.profile_results.get("default")
    assert pr is not None
    assert pr.endpoint_list, "expected at least one chosen pod"
    chosen = pr.endpoint_list[0].endpoint
    assert chosen.name == "pod2"


def test_filter_removes_pods():
    p1 = Endpoint(name="pod1", attributes={"zone": "a"})
    p2 = Endpoint(name="pod2", attributes={"zone": "b"})

    f = SimpleFilter(key="zone", value="a")
    profile = (
        SchedulerProfile(name="default")
        .with_filters(f)
        .with_scorers(WeightedScorer(ConstantScorer(1.0), 1.0))
        .with_picker(RandomPicker())
    )
    s = make_scheduler_with_profile(profile)

    res = s.schedule(LLMRequest(request_id="r", target_model=None), [p1, p2])
    pr = res.profile_results.get("default")
    assert pr is not None
    assert len(pr.endpoint_list) == 1
    assert pr.endpoint_list[0].endpoint.name == "pod1"


def test_filter_scorer_picker_combined():
    p1 = Endpoint(name="pod1", attributes={"zone": "a", "waiting_queue_size": 5})
    p2 = Endpoint(name="pod2", attributes={"zone": "b", "waiting_queue_size": 2})
    p3 = Endpoint(name="pod3", attributes={"zone": "a", "waiting_queue_size": 10})

    qls = QueueLengthScorer(attribute_key="waiting_queue_size")
    f = SimpleFilter(key="zone", value="a")
    profile = (
        SchedulerProfile(name="default")
        .with_filters(f)
        .with_scorers(WeightedScorer(qls, 1.0))
        .with_picker(MaxScorePicker())
    )
    s = make_scheduler_with_profile(profile)

    res = s.schedule(LLMRequest(request_id="r", target_model=None), [p1, p2, p3])
    pr = res.profile_results.get("default")
    assert pr is not None
    assert pr.endpoint_list, "expected at least one chosen pod"
    chosen = pr.endpoint_list[0].endpoint
    # pod2 has the smallest queue size, but is in the wrong zone, so pod 1 should be chosen
    assert chosen.name == "pod1"
    assert [scored.endpoint.name for scored in pr.ranked] == ["pod1", "pod3"]


WITH_FLOW_CONTROL = textwrap.dedent(
    """
    profile_handler:
      type: single_profile
    profiles:
      default:
        flow_control:
          type: kv_saturation
        scorers:
          - type: least_queue
            weight: 1.0
        picker:
          type: max_score
    """
)


def test_sees_flow_control_before_the_first_decision(tmp_path):
    config_file = tmp_path / "scheduler.yaml"
    config_file.write_text(WITH_FLOW_CONTROL, encoding="utf-8")

    assert Scheduler(config_path=str(config_file)).has_flow_control()


def test_listing_plugins_leaves_reloads_to_schedule(tmp_path):
    config_file = tmp_path / "scheduler.yaml"
    config_file.write_text(WITH_FLOW_CONTROL, encoding="utf-8")
    scheduler = Scheduler(config_path=str(config_file))
    assert scheduler.has_flow_control()

    without = WITH_FLOW_CONTROL.replace("    flow_control:\n      type: kv_saturation\n", "")
    config_file.write_text(without, encoding="utf-8")
    os.utime(config_file, (scheduler.last_mtime + 2, scheduler.last_mtime + 2))

    assert scheduler.has_flow_control()


class LearningScorer:
    """Scores every pod the same and remembers where requests went."""

    def __init__(self) -> None:
        self.learned: list[str] = []

    def score(self, cycle_state, request, pods):
        return dict.fromkeys(pods, 1.0)

    def pre_request(self, cycle_state, request, endpoint):
        self.learned.append(endpoint.name)


def _learning_scheduler() -> tuple[Scheduler, LearningScorer]:
    scorer = LearningScorer()
    profile = SchedulerProfile(name="default").with_scorers(WeightedScorer(scorer, 1.0))
    return make_scheduler_with_profile(profile), scorer


def test_scorers_learn_the_pick_by_default():
    scheduler, scorer = _learning_scheduler()
    scheduler.schedule(LLMRequest(request_id="r"), [Endpoint(name="pod1"), Endpoint(name="pod2")])

    assert scorer.learned == ["pod1"]


def test_scorers_learn_only_the_endpoint_recorded_later():
    scheduler, scorer = _learning_scheduler()
    request, other = LLMRequest(request_id="r"), Endpoint(name="pod2")
    result = scheduler.schedule(request, [Endpoint(name="pod1"), other], record=False)
    assert scorer.learned == []

    scheduler.record(result, request, other)
    assert scorer.learned == ["pod2"]
