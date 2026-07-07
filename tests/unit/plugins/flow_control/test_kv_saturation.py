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

from py_inference_scheduler.framework import CycleState, Endpoint, LLMRequest
from py_inference_scheduler.plugins.flow_control.kv_saturation import KVSaturationPlugin


class TestKVSaturationPlugin:
    def setup_method(self):
        self.config = {
            "enable_drip": True,
            "drip_threshold_kv": 0.2,
            "drip_interval_s": 0.1,
        }
        self.plugin = KVSaturationPlugin(**self.config)
        self.cycle_state = CycleState()
        self.request = LLMRequest(request_id="test_req", body="test prompt")

    def test_get_allowed_candidates_budget_ok(self):
        """Test that candidates are allowed when budget is not exceeded."""
        endpoints = [
            Endpoint(name="ep1", attributes={"kv_cache_size": 2000}),
            Endpoint(name="ep2", attributes={"kv_cache_size": 2000}),
        ]

        # No usage yet, should allow all
        allowed = self.plugin.get_allowed_candidates(self.request, endpoints)
        assert len(allowed) == 2
        assert allowed[0].name == "ep1"
        assert allowed[1].name == "ep2"

    def test_get_allowed_candidates_budget_full(self):
        """Test that candidates are blocked when budget is exceeded."""
        endpoints = [
            Endpoint(name="ep1", attributes={"kv_cache_size": 1000}),
        ]

        # Simulate usage full
        self.plugin._replica_token_usage["ep1"] = 1000

        # Request needs tokens (estimate > 0)
        allowed = self.plugin.get_allowed_candidates(self.request, endpoints)
        assert len(allowed) == 0

    def test_reserve_and_release(self):
        """Test that reserve and release update state correctly."""
        endpoint = Endpoint(name="ep1", attributes={"kv_cache_size": 1000})

        # Reserve
        self.plugin.reserve(self.request, endpoint)
        assert "ep1" in self.plugin._replica_token_usage
        assert self.plugin._replica_token_usage["ep1"] > 0
        assert self.request.request_id in self.plugin._budgeted_requests

        # Release
        self.plugin.release(self.request, "ep1")
        assert self.plugin._replica_token_usage["ep1"] == 0
        assert self.request.request_id not in self.plugin._budgeted_requests

    def test_drip_admission(self):
        """Test that drip admission allows a candidate even if budget is full."""
        endpoints = [
            Endpoint(name="ep1", attributes={"kv_cache_size": 1000, "routing_stats": {"kv": 0.1}}),
        ]

        # Simulate usage full
        self.plugin._replica_token_usage["ep1"] = 1000

        # Drip should allow it because physical KV (0.1) < threshold (0.2)
        allowed = self.plugin.get_allowed_candidates(self.request, endpoints)
        assert len(allowed) == 1
        assert allowed[0].name == "ep1"

    def test_drip_admission_blocked_by_threshold(self):
        """Test that drip is blocked if physical KV is above threshold."""
        endpoints = [
            Endpoint(name="ep1", attributes={"kv_cache_size": 1000, "routing_stats": {"kv": 0.3}}),
        ]

        # Simulate usage full
        self.plugin._replica_token_usage["ep1"] = 1000

        # Drip should NOT allow it because physical KV (0.3) >= threshold (0.2)
        allowed = self.plugin.get_allowed_candidates(self.request, endpoints)
        assert len(allowed) == 0

    def test_drip_admission_blocked_by_interval(self):
        """Test that drip is blocked if called too frequently."""
        endpoints = [
            Endpoint(name="ep1", attributes={"kv_cache_size": 1000, "routing_stats": {"kv": 0.1}}),
        ]

        # Simulate usage full
        self.plugin._replica_token_usage["ep1"] = 1000

        # First drip should succeed
        allowed = self.plugin.get_allowed_candidates(self.request, endpoints)
        assert len(allowed) == 1

        # Second drip immediately after should fail due to interval (0.1s)
        allowed = self.plugin.get_allowed_candidates(self.request, endpoints)
        assert len(allowed) == 0


if __name__ == "__main__":
    pytest.main([__file__])
