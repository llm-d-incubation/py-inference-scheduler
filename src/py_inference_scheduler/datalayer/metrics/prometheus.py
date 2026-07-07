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

from typing import Any

from prometheus_client.parser import text_string_to_metric_families

# SGLang gauge name -> the routing-stats key our scorers expect.
_SGLANG_METRICS = {
    "sglang:num_queue_reqs": "num_waiting_reqs",
    "sglang:num_running_reqs": "num_running_reqs",
    "sglang:token_usage": "kv",
}


def empty_sglang_stats() -> dict[str, Any]:
    return {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0, "error": None}


def parse_sglang(text: str) -> dict[str, Any]:
    """Parse an SGLang Prometheus ``/metrics`` payload into routing stats.

    Returns num_waiting_reqs, num_running_reqs (ints) and kv (float
    KV-cache utilisation in [0, 1]).
    """
    stats = empty_sglang_stats()
    for family in text_string_to_metric_families(text):
        key = _SGLANG_METRICS.get(family.name)
        if key is None:
            continue
        values = [s.value for s in family.samples if s.name == family.name]
        if not values:
            continue
        # A gauge may appear as several samples (multiprocess mode emits one per
        # PID). For a single replica max() picks the real value over idle 0s.
        value = max(values)
        stats[key] = float(value) if key == "kv" else int(value)
    return stats


# vLLM gauge name -> the routing-stats key our scorers expect.
_VLLM_METRICS = {
    "vllm:num_requests_waiting": "num_waiting_reqs",
    "vllm:num_requests_running": "num_running_reqs",
    "vllm:kv_cache_usage_perc": "kv",
}


def empty_vllm_stats() -> dict[str, Any]:
    return {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0, "error": None}


def parse_vllm(text: str) -> dict[str, Any]:
    """Parse a vLLM Prometheus ``/metrics`` payload into routing stats.

    Returns num_waiting_reqs, num_running_reqs (ints) and kv (float KV-cache
    utilisation in [0, 1]).
    """
    stats = empty_vllm_stats()
    for family in text_string_to_metric_families(text):
        key = _VLLM_METRICS.get(family.name)
        if key is None:
            continue
        values = [s.value for s in family.samples if s.name == family.name]
        if not values:
            continue
        # A gauge may appear as several samples (multiprocess mode emits one per
        # PID). For a single replica max() picks the real value over idle 0s.
        value = max(values)
        stats[key] = float(value) if key == "kv" else int(value)
    return stats
