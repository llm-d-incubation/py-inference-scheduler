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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class LLMRequest:
    request_id: str
    target_model: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None


@dataclass
class Endpoint:
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredEndpoint:
    endpoint: Endpoint
    score: float


@dataclass
class ProfileRunResult:
    # list of chosen pods (may be empty)
    endpoint_list: List[ScoredEndpoint] = field(default_factory=list)
    # arbitrary result metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulingResult:
    profile_results: Dict[str, ProfileRunResult] = field(default_factory=dict)
    primary_profile_name: Optional[str] = None


class CycleState:
    """Per-request ephemeral state that plugins may use."""

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value
