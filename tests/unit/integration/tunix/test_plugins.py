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

from integration.tunix.plugins import WorkerStateFilter
from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.framework import CycleState, Endpoint, LLMRequest


def _pods(**states):
    return {
        name: Endpoint(name=name, attributes={"state": state}) for name, state in states.items()
    }


def _request():
    return LLMRequest(request_id="r1", body="hello")


def test_drops_non_ready_states():
    f = WorkerStateFilter()
    pods = _pods(a="READY", b="SYNCING", c="ERROR")
    assert set(f.filter(CycleState(), _request(), pods)) == {"a"}


def test_missing_state_treated_as_unknown_and_dropped():
    f = WorkerStateFilter()
    pods = {
        "a": Endpoint(name="a", attributes={"state": "READY"}),
        "b": Endpoint(name="b", attributes={}),
    }
    assert set(f.filter(CycleState(), _request(), pods)) == {"a"}


def test_all_dropped_returns_full_ballot():
    f = WorkerStateFilter()
    pods = _pods(a="SYNCING", b="SYNCING")
    assert set(f.filter(CycleState(), _request(), pods)) == {"a", "b"}


def test_custom_allowed_states():
    f = WorkerStateFilter(allowed_states=["READY", "RUNNING"])
    pods = _pods(a="RUNNING", b="SYNCING")
    assert set(f.filter(CycleState(), _request(), pods)) == {"a"}


def test_registered_in_config_registry():
    config = SchedulerConfig.from_dict(
        {
            "profile_handler": {"type": "single_profile"},
            "profiles": {
                "p": {
                    "filters": [{"type": "worker_state", "allowed_states": ["READY"]}],
                    "scorers": [{"type": "least_queue", "weight": 1.0}],
                    "picker": {"type": "max_score"},
                }
            },
        }
    )
    (profile,) = config.profiles.values()
    assert isinstance(profile.filters[0], WorkerStateFilter)
