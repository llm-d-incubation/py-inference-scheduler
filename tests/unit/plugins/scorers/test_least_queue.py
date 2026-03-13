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
from scheduling.plugins import LeastQueueScorer

class TestLeastQueueScorer:
    def setup_method(self):
        self.scorer = LeastQueueScorer()
        self.cycle_state = CycleState()
        self.request = LLMRequest(request_id="test_req")

    def test_score_normalization(self):
        """Test that LeastQueueScorer normalizes scores to [0,1] correctly based on queue_len."""
        endpoints = {
            "ep_best": Endpoint(name="ep_best", attributes={"queue_len": 5}),
            "ep_mid": Endpoint(name="ep_mid", attributes={"queue_len": 15}),
            "ep_worst": Endpoint(name="ep_worst", attributes={"queue_len": 25}),
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        # Max = 25, Min = 5
        # Formula: (max_queue - queue_len) / (max_queue - min_queue)
        
        # ep_best: (25 - 5) / 20 = 1.0
        assert scores["ep_best"] == 1.0
        
        # ep_mid: (25 - 15) / 20 = 10/20 = 0.5
        assert scores["ep_mid"] == 0.5
        
        # ep_worst: (25 - 25) / 20 = 0.0
        assert scores["ep_worst"] == 0.0

    def test_score_all_equal(self):
        """Test scoring when all endpoints have identical queue_len."""
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={"queue_len": 12}),
            "ep2": Endpoint(name="ep2", attributes={"queue_len": 12}),
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        # If all are equal, everyone gets 1.0 to avoid division by zero
        assert scores["ep1"] == 1.0
        assert scores["ep2"] == 1.0

if __name__ == "__main__":
    pytest.main([__file__])
