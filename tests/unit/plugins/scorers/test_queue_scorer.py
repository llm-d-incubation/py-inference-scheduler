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


from scheduling.framework import Endpoint, LLMRequest
from scheduling.framework import SchedulerProfile, WeightedScorer
from scheduling.plugins import QueueLengthScorer, RandomPicker
from scheduling.plugins import SingleProfileHandler
from scheduling.core.config import SchedulerConfig
from scheduling.core.scheduler import Scheduler


def make_scheduler_with_profile(profile: SchedulerProfile) -> Scheduler:
    ph = SingleProfileHandler()
    cfg = SchedulerConfig(profile_handler=ph, profiles={profile.name: profile})
    return Scheduler.new_with_config(cfg)


def test_queue_length_scorer_prefers_lower_queue():
    p1 = Endpoint(name="pod1", attributes={"waiting_queue_size": 5})
    p2 = Endpoint(name="pod2", attributes={"waiting_queue_size": 1})

    scorer = QueueLengthScorer()
    profile = SchedulerProfile(name="default").with_scorers(WeightedScorer(scorer, 1.0)).with_picker(RandomPicker())
    s = make_scheduler_with_profile(profile)

    res = s.schedule(LLMRequest(request_id="r", target_model=None), [p1, p2])
    pr = res.profile_results.get("default")
    assert pr is not None
    assert pr.endpoint_list, "expected chosen endpoint"
    chosen = pr.endpoint_list[0].endpoint
    assert chosen.name == "pod2"
