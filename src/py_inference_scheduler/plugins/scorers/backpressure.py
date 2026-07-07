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

from typing import Mapping

from py_inference_scheduler.framework import (
    CycleState,
    Endpoint,
    LLMRequest,
    ScorerPlugin,
    register_scorer,
    score_by_metric,
)


@register_scorer("queue_length")
class QueueLengthScorer(ScorerPlugin):
    """Scores endpoints based on a 'waiting_queue_size' attribute.

    Lower queue size is better.
    """

    def __init__(self, attribute_key: str = "waiting_queue_size") -> None:
        self.attribute_key = attribute_key

    def score(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, ep in pods.items():
            raw = ep.attributes.get(self.attribute_key, 0)
            try:
                size = int(raw)  # type: ignore[call-overload]
            except Exception:  # noqa: BLE001
                size = 0
            result[name] = float(-size)
        return result


@register_scorer("least_queue")
class LeastQueueScorer(ScorerPlugin):
    """Scores endpoints based on their real-time Ray Serve actor queue length."""

    def score(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> dict[str, float]:
        return score_by_metric(
            pods,
            metric_extractor=lambda ep: float(ep.attributes.get("queue_len", 0)),  # type: ignore[arg-type]
            lower_is_better=True,
        )


@register_scorer("waiting_queue")
class WaitingQueueScorer(ScorerPlugin):
    """Scores candidate endpoints based on the number of waiting requests inside the vLLM engine."""

    def score(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> dict[str, float]:
        return score_by_metric(
            pods,
            metric_extractor=lambda ep: float(
                ep.attributes.get("routing_stats", {}).get("num_waiting_reqs", 0)  # type: ignore[attr-defined]
            ),
            lower_is_better=True,
        )


@register_scorer("running_queue")
class RunningQueueScorer(ScorerPlugin):
    """Scores candidate endpoints based on the number of running requests inside the vLLM engine."""

    def score(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> dict[str, float]:
        return score_by_metric(
            pods,
            metric_extractor=lambda ep: float(
                ep.attributes.get("routing_stats", {}).get("num_running_reqs", 0)  # type: ignore[attr-defined]
            ),
            lower_is_better=True,
        )


@register_scorer("kv_cache")
class KVCacheScorer(ScorerPlugin):
    """Scores candidate endpoints based on KV cache utilization."""

    def score(
        self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]
    ) -> dict[str, float]:
        return score_by_metric(
            pods,
            metric_extractor=lambda ep: float(
                ep.attributes.get("routing_stats", {}).get("kv", 0.0)  # type: ignore[attr-defined]
            ),
            lower_is_better=True,
        )
