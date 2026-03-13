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
import concurrent.futures
from scheduling.plugins import RoundRobinScorer
from scheduling.framework import Endpoint, CycleState, LLMRequest

def test_round_robin_cycling():
    scorer = RoundRobinScorer()
    pods = {
        "ep1": Endpoint(name="ep1"),
        "ep2": Endpoint(name="ep2"),
    }
    request = LLMRequest(request_id="test")
    state = CycleState()

    # Verify 3 cycles through the endpoints
    for _ in range(3):
        for expected in ["ep1", "ep2"]:
            scores = scorer.score(state, request, pods)
            assert scores == {expected: 1.0}

def test_round_robin_scaling():
    scorer = RoundRobinScorer()
    request = LLMRequest(request_id="test")
    state = CycleState()

    scores1 = scorer.score(state, request, {"ep1": Endpoint(name="ep1"), "ep2": Endpoint(name="ep2")})
    assert scores1 == {"ep1": 1.0}

    # index doesn't change when number of pods change
    scores2 = scorer.score(state, request, {"ep1": Endpoint(name="ep1"), "ep2": Endpoint(name="ep2"), "ep3": Endpoint(name="ep3")})
    assert scores2 == {"ep2": 1.0}

def test_round_robin_concurrency():
    scorer = RoundRobinScorer()
    pods = {f"ep{i}": Endpoint(name=f"ep{i}") for i in range(10)}
    request = LLMRequest(request_id="test")
    state = CycleState()

    num_threads = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(scorer.score, state, request, pods) for _ in range(num_threads)]
        results = [f.result() for f in futures]

    # With 20 requests and 10 pods, each pod should be selected exactly twice
    selection_counts = {}
    for res in results:
        assert len(res) == 1
        name = list(res.keys())[0]
        selection_counts[name] = selection_counts.get(name, 0) + 1

    for i in range(10):
        assert selection_counts[f"ep{i}"] == 2

def test_round_robin_empty_endpoints():
    scorer = RoundRobinScorer()
    request = LLMRequest(request_id="test")
    state = CycleState()
    assert scorer.score(state, request, {}) == {}
