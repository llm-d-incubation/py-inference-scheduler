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

from py_inference_scheduler.datalayer.metrics.prometheus import parse_sglang

_SGLANG_METRICS = (
    "# HELP sglang:num_running_reqs running\n"
    "# TYPE sglang:num_running_reqs gauge\n"
    'sglang:num_running_reqs{model_name="qwen",engine="0"} 12.0\n'
    "# TYPE sglang:num_queue_reqs gauge\n"
    'sglang:num_queue_reqs{model_name="qwen",engine="0"} 3.0\n'
    "# TYPE sglang:token_usage gauge\n"
    'sglang:token_usage{model_name="qwen",engine="0"} 0.74\n'
    "# TYPE sglang:other_metric gauge\n"
    "sglang:other_metric 99.0\n"
)


def test_parse_sglang_extracts_labeled_gauges():
    stats = parse_sglang(_SGLANG_METRICS)
    assert stats["num_running_reqs"] == 12
    assert stats["num_waiting_reqs"] == 3
    assert stats["kv"] == 0.74
    assert stats["error"] is None
    # counts are ints, kv is a float
    assert isinstance(stats["num_running_reqs"], int)
    assert isinstance(stats["kv"], float)


def test_parse_sglang_missing_metrics_defaults_to_zero():
    stats = parse_sglang("# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs 5.0\n")
    assert stats["num_running_reqs"] == 5
    assert stats["num_waiting_reqs"] == 0
    assert stats["kv"] == 0.0


def test_parse_sglang_empty_payload():
    stats = parse_sglang("")
    assert stats == {"num_waiting_reqs": 0, "num_running_reqs": 0, "kv": 0.0, "error": None}


def test_parse_sglang_multiproc_takes_max_across_samples():
    # Prometheus multiprocess mode (e.g. TP>1) exposes one sample per PID; the
    # scheduler process reports the real value while others report 0.
    text = (
        "# TYPE sglang:num_running_reqs gauge\n"
        'sglang:num_running_reqs{pid="1"} 0.0\n'
        'sglang:num_running_reqs{pid="2"} 7.0\n'
        "# TYPE sglang:token_usage gauge\n"
        'sglang:token_usage{pid="1"} 0.2\n'
        'sglang:token_usage{pid="2"} 0.8\n'
    )
    stats = parse_sglang(text)
    assert stats["num_running_reqs"] == 7
    assert stats["kv"] == 0.8
