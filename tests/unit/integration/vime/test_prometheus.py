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

from py_inference_scheduler.datalayer.metrics.prometheus import parse_vllm

# Metric names match vLLM v0.22.0 (vllm/v1/metrics/loggers.py).
_VLLM_METRICS = (
    "# HELP vllm:num_requests_running running\n"
    "# TYPE vllm:num_requests_running gauge\n"
    'vllm:num_requests_running{model_name="qwen"} 12.0\n'
    "# TYPE vllm:num_requests_waiting gauge\n"
    'vllm:num_requests_waiting{model_name="qwen"} 3.0\n'
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="qwen"} 0.74\n'
    "# TYPE vllm:num_preemptions_total counter\n"
    "vllm:num_preemptions_total 99.0\n"
)


def test_parse_vllm_extracts_labeled_gauges():
    stats = parse_vllm(_VLLM_METRICS)
    assert stats["num_running_reqs"] == 12
    assert stats["num_waiting_reqs"] == 3
    assert stats["kv"] == 0.74
    assert stats["error"] is None
    # counts are ints, kv is a float
    assert isinstance(stats["num_running_reqs"], int)
    assert isinstance(stats["kv"], float)


def test_parse_vllm_ignores_by_reason_sibling():
    # vLLM v0.22.0 also emits num_requests_waiting_by_reason (a distinct family
    # that sums to num_requests_waiting). It must NOT be double-counted into
    # num_waiting_reqs — matching is by exact family name.
    text = (
        "# TYPE vllm:num_requests_waiting gauge\n"
        "vllm:num_requests_waiting 4.0\n"
        "# TYPE vllm:num_requests_waiting_by_reason gauge\n"
        'vllm:num_requests_waiting_by_reason{reason="capacity"} 3.0\n'
        'vllm:num_requests_waiting_by_reason{reason="deferred"} 1.0\n'
    )
    stats = parse_vllm(text)
    assert stats["num_waiting_reqs"] == 4


def test_parse_vllm_missing_metrics_defaults_to_zero():
    stats = parse_vllm("# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 5.0\n")
    assert stats["num_running_reqs"] == 5
    assert stats["num_waiting_reqs"] == 0
    assert stats["kv"] == 0.0


def test_parse_vllm_empty_payload():
    stats = parse_vllm("")
    assert stats == {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0, "error": None}


def test_parse_vllm_multiproc_takes_max_across_samples():
    # Prometheus multiprocess mode (e.g. TP>1) exposes one sample per PID; the
    # scheduler process reports the real value while others report 0.
    text = (
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{pid="1"} 0.0\n'
        'vllm:num_requests_running{pid="2"} 7.0\n'
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        'vllm:kv_cache_usage_perc{pid="1"} 0.2\n'
        'vllm:kv_cache_usage_perc{pid="2"} 0.8\n'
    )
    stats = parse_vllm(text)
    assert stats["num_running_reqs"] == 7
    assert stats["kv"] == 0.8
