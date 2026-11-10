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


import os
import textwrap
from pathlib import Path

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


@pytest.mark.parametrize("replacement", [None, "profiles: [", "profiles: {}"])
def test_reload_error_keeps_last_valid_config(tmp_path, caplog, replacement):
    config_path = tmp_path / "scheduler.yaml"
    config = textwrap.dedent(
        """
        profile_handler:
          type: single_profile
        profiles:
          default:
            filters:
              - type: simple
                key: zone
                value: a
        """
    )
    config_path.write_text(config, encoding="utf-8")
    scheduler = Scheduler(config_path=str(config_path))
    request = LLMRequest(request_id="r", target_model=None)
    candidates = [
        Endpoint(name="a", attributes={"zone": "a"}),
        Endpoint(name="b", attributes={"zone": "b"}),
    ]
    assert scheduler.run(request, candidates)[0].endpoint.name == "a"

    if replacement is None:
        config_path.unlink()
    else:
        config_path.write_text(replacement, encoding="utf-8")
        os.utime(config_path, (scheduler.last_mtime + 1, scheduler.last_mtime + 1))
    assert scheduler.run(request, candidates)[0].endpoint.name == "a"
    assert "keeping the last valid configuration" in caplog.text

    if replacement is not None:
        caplog.clear()
        assert scheduler.run(request, candidates)[0].endpoint.name == "a"
        assert "keeping the last valid configuration" not in caplog.text

    config_path.write_text(config.replace("value: a", "value: b"), encoding="utf-8")
    os.utime(config_path, (scheduler.last_mtime + 2, scheduler.last_mtime + 2))
    assert scheduler.run(request, candidates)[0].endpoint.name == "b"


@pytest.mark.parametrize("content", [None, "profiles: {}"])
def test_initial_config_error_is_not_suppressed(tmp_path, content):
    config_path = tmp_path / "scheduler.yaml"
    if content is not None:
        config_path.write_text(content, encoding="utf-8")
    scheduler = Scheduler(config_path=str(config_path))
    for _ in range(2):
        with pytest.raises((FileNotFoundError, ValueError)):
            scheduler.run(LLMRequest(request_id="r", target_model=None), [Endpoint(name="a")])


def test_unreadable_reload_retries_without_new_mtime(tmp_path, monkeypatch, caplog):
    config_path = tmp_path / "scheduler.yaml"
    config = textwrap.dedent(
        """
        profile_handler:
          type: single_profile
        profiles:
          default:
            filters:
              - type: simple
                key: zone
                value: a
        """
    )
    config_path.write_text(config, encoding="utf-8")
    scheduler = Scheduler(config_path=str(config_path))
    request = LLMRequest(request_id="r", target_model=None)
    candidates = [
        Endpoint(name="a", attributes={"zone": "a"}),
        Endpoint(name="b", attributes={"zone": "b"}),
    ]
    assert scheduler.run(request, candidates)[0].endpoint.name == "a"

    config_path.write_text(config.replace("value: a", "value: b"), encoding="utf-8")
    os.utime(config_path, (scheduler.last_mtime + 1, scheduler.last_mtime + 1))

    def fail_open(*args, **kwargs):
        raise PermissionError("configuration is temporarily unreadable")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_open)
        assert scheduler.run(request, candidates)[0].endpoint.name == "a"
    assert "keeping the last valid configuration" in caplog.text
    assert scheduler.run(request, candidates)[0].endpoint.name == "b"
