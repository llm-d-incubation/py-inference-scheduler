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
import threading
from ...framework import (
    ScorerPlugin, 
    Endpoint, 
    LLMRequest, 
    register_scorer
)


@register_scorer("constant")
class ConstantScorer(ScorerPlugin):
    """Scorer that returns a constant score."""

    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, cycle_state: Any, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Dict[str, float]:
        if isinstance(endpoints, Mapping):
            return {name: float(self.value) for name in endpoints.keys()}
        return {p.name: float(self.value) for p in endpoints}


@register_scorer("round_robin")
class RoundRobinScorer(ScorerPlugin):
    """A scorer that cycles through endpoints in a round-robin fashion."""

    def __init__(self) -> None:
        self._counter = 0
        self._lock = threading.Lock()

    def score(self, cycle_state: Any, request: LLMRequest, pods: Mapping[str, Endpoint]) -> Dict[str, float]:
        if not pods:
            return {}

        names = sorted(pods.keys(), key=str)
        with self._lock:
            idx = self._counter % len(names)
            self._counter += 1
        selected_name = names[idx]

        return {selected_name: 1.0}
