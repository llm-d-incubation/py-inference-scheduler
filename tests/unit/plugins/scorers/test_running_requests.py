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
from typing import Dict, Any
from scheduling.framework import Endpoint, LLMRequest, CycleState
from scheduling.plugins import RunningQueueScorer

class TestRunningQueueScorer:
    def setup_method(self):
        self.scorer = RunningQueueScorer()
        self.cycle_state = CycleState()
        self.request = LLMRequest(request_id="test_req")

    def test_score_basic(self):
        """Test scoring with varied running requests."""
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={"routing_stats": {"num_running_reqs": 10}}),
            "ep2": Endpoint(name="ep2", attributes={"routing_stats": {"num_running_reqs": 50}}),
            "ep3": Endpoint(name="ep3", attributes={"routing_stats": {"num_running_reqs": 100}}),
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        # Max = 100, Min = 10
        # Formula: (100 - val) / (100 - 10)
        
        # ep1: (100 - 10) / 90 = 1.0
        assert scores["ep1"] == 1.0
        
        # ep2: (100 - 50) / 90 = 50/90 = 0.555...
        assert pytest.approx(scores["ep2"], 0.01) == 0.555
        
        # ep3: (100 - 100) / 90 = 0.0
        assert scores["ep3"] == 0.0

    def test_score_all_equal(self):
        """Test scoring when all endpoints have equal running requests."""
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={"routing_stats": {"num_running_reqs": 20}}),
            "ep2": Endpoint(name="ep2", attributes={"routing_stats": {"num_running_reqs": 20}}),
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        assert scores["ep1"] == 1.0
        assert scores["ep2"] == 1.0

    def test_score_empty_stats(self):
        """Test with missing routing_stats (default 0)."""
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={}), # Defaults to 0 running
            "ep2": Endpoint(name="ep2", attributes={"routing_stats": {"num_running_reqs": 10}}),
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        # Min = 0 (ep1), Max = 10 (ep2)
        # ep1: (10 - 0) / 10 = 1.0
        assert scores["ep1"] == 1.0
        # ep2: (10 - 10) / 10 = 0.0
        assert scores["ep2"] == 0.0

    if __name__ == "__main__":
        pytest.main([__file__])
