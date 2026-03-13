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
from scheduling.plugins import KVCacheScorer

class TestKVCacheScorer:
    def setup_method(self):
        self.scorer = KVCacheScorer()
        self.cycle_state = CycleState()
        self.request = LLMRequest(request_id="test_req")

    def test_score_relative_logic(self):
        """Test relative scoring logic for KV cache usage."""
        # Using relative logic: Score = (max - cur) / (max - min)
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={"routing_stats": {"kv": 0.2}}), # 20%
            "ep2": Endpoint(name="ep2", attributes={"routing_stats": {"kv": 0.5}}), # 50%
            "ep3": Endpoint(name="ep3", attributes={"routing_stats": {"kv": 0.8}}), # 80%
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        # Min = 0.2, Max = 0.8. Range = 0.6
        
        # ep1: (0.8 - 0.2) / 0.6 = 1.0
        assert scores["ep1"] == 1.0
        
        # ep2: (0.8 - 0.5) / 0.6 = 0.3 / 0.6 = 0.5
        assert scores["ep2"] == 0.5
        
        # ep3: (0.8 - 0.8) / 0.6 = 0.0
        assert scores["ep3"] == 0.0

    def test_score_all_equal(self):
        """Test scoring when all endpoints have same usage."""
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={"routing_stats": {"kv": 0.5}}),
            "ep2": Endpoint(name="ep2", attributes={"routing_stats": {"kv": 0.5}}),
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        assert scores["ep1"] == 1.0
        assert scores["ep2"] == 1.0

    def test_score_missing_stats(self):
        """Test with missing KV stats (default 0.0)."""
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={}), # Defaults to 0.0 usage
            "ep2": Endpoint(name="ep2", attributes={"routing_stats": {"kv": 0.5}}),
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        # Min = 0.0, Max = 0.5
        # ep1: (0.5 - 0.0) / 0.5 = 1.0
        assert scores["ep1"] == 1.0
        # ep2: (0.5 - 0.5) / 0.5 = 0.0
        assert scores["ep2"] == 0.0

    def test_score_clamping(self):
        """Confirm that extremely high values don't break logic (though physically impossible)."""
        endpoints = {
            "ep1": Endpoint(name="ep1", attributes={"routing_stats": {"kv": 1.2}}), # 120% utilized??
            "ep2": Endpoint(name="ep2", attributes={"routing_stats": {"kv": 0.2}}),
        }
        
        scores = self.scorer.score(self.cycle_state, self.request, endpoints)
        
        # Min = 0.2, Max = 1.2. Range = 1.0
        # ep1: (1.2 - 1.2) / 1.0 = 0.0
        assert scores["ep1"] == 0.0
        # ep2: (1.2 - 0.2) / 1.0 = 1.0
        assert scores["ep2"] == 1.0

    if __name__ == "__main__":
        pytest.main([__file__])
