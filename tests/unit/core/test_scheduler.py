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


import pytest
from scheduling import SchedulerConfig, Scheduler
from scheduling.plugins import SingleProfileHandler, SimpleFilter, ConstantScorer, QueueLengthScorer, MaxScorePicker
from scheduling.framework import SchedulerProfile, WeightedScorer
from scheduling.framework import Endpoint, LLMRequest

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

    with pytest.raises(ValueError):
        s.schedule(LLMRequest(request_id="1", target_model="m"), [])


def test_finds_highest_score_pod():
    # create two pods
    p1 = Endpoint(name="pod1")
    p2 = Endpoint(name="pod2")

    # create scorer that gives pod1 score 1, pod2 score 2
    class MapScorer:
        def score(self, cycle_state, request, pods):
            # `pods` is a mapping name->Endpoint
            if hasattr(pods, 'items'):
                return {name: (2.0 if name == "pod2" else 1.0) for name in pods.keys()}
            # fallback for sequences
            return {p.name: (2.0 if p.name == "pod2" else 1.0) for p in pods}

    profile = SchedulerProfile(name="default").with_scorers(WeightedScorer(MapScorer(), 1.0)).with_picker(RandomPicker())   
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
    profile = SchedulerProfile(name="default").with_filters(f).with_scorers(WeightedScorer(ConstantScorer(1.0), 1.0)).with_picker(RandomPicker())
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
    profile = SchedulerProfile(name="default").with_filters(f).with_scorers(WeightedScorer(qls, 1.0)).with_picker((MaxScorePicker()))
    s = make_scheduler_with_profile(profile)

    res = s.schedule(LLMRequest(request_id="r", target_model=None), [p1, p2, p3])
    pr = res.profile_results.get("default")
    assert pr is not None
    assert pr.endpoint_list, "expected at least one chosen pod"
    chosen = pr.endpoint_list[0].endpoint
    # pod2 has the smallest queue size, but is in the wrong zone, so pod 1 should be chosen
    assert chosen.name == "pod1"