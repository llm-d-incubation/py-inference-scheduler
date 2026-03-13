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
from typing import Any, Dict, Sequence, Mapping
from ...framework import (
    ScorerPlugin, 
    Endpoint, 
    LLMRequest, 
    score_by_metric,
    register_scorer
)


@register_scorer("queue_length")
class QueueLengthScorer(ScorerPlugin):
    """Scores endpoints based on a 'waiting_queue_size' attribute.
    Lower queue size is better. 
    """

    def __init__(self, attribute_key: str = "waiting_queue_size") -> None:
        self.attribute_key = attribute_key

    def score(self, cycle_state: Any, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        if isinstance(endpoints, Mapping):
            result: Dict[str, float] = {}
            for name, ep in endpoints.items():
                raw = ep.attributes.get(self.attribute_key, 0)
                try:
                    size = int(raw)
                except Exception:
                    size = 0
                result[name] = float(-size)
            return result

        return {ep.name: float(-int(ep.attributes.get(self.attribute_key, 0))) for ep in endpoints}


@register_scorer("least_queue")
class LeastQueueScorer(ScorerPlugin):
    """Scores endpoints based on their real-time Ray Serve actor queue length."""

    def score(self, cycle_state: Any, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        return score_by_metric(
            endpoints,
            metric_extractor=lambda ep: float(ep.attributes.get("queue_len", 0)),
            lower_is_better=True
        )

@register_scorer("waiting_queue")
class WaitingQueueScorer(ScorerPlugin):
    """Scores candidate endpoints based on the number of waiting requests inside the vLLM engine."""
    def score(self, cycle_state: Any, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        return score_by_metric(
            endpoints,
            metric_extractor=lambda ep: float(ep.attributes.get("routing_stats", {}).get("num_waiting_reqs", 0)),
            lower_is_better=True
        )


@register_scorer("running_queue")
class RunningQueueScorer(ScorerPlugin):
    """Scores candidate endpoints based on the number of running requests inside the vLLM engine."""
    def score(self, cycle_state: Any, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        return score_by_metric(
            endpoints,
            metric_extractor=lambda ep: float(ep.attributes.get("routing_stats", {}).get("num_running_reqs", 0)),
            lower_is_better=True
        )


@register_scorer("kv_cache")
class KVCacheScorer(ScorerPlugin):
    """Scores candidate endpoints based on KV cache utilization."""
    def score(self, cycle_state: Any, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        return score_by_metric(
            endpoints,
            metric_extractor=lambda ep: float(ep.attributes.get("routing_stats", {}).get("kv", 0.0)),
            lower_is_better=True
        )
