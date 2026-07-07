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


@dataclass
class LLMRequest:
    request_id: str
    target_model: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    body: object | None = None


@dataclass
class Endpoint:
    name: str
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass
class ScoredEndpoint:
    endpoint: Endpoint
    score: float


@dataclass
class ProfileRunResult:
    # list of chosen pods (may be empty)
    endpoint_list: list[ScoredEndpoint] = field(default_factory=list)
    # arbitrary result metadata
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class SchedulingResult:
    profile_results: dict[str, ProfileRunResult] = field(default_factory=dict)
    primary_profile_name: str | None = None


class CycleState:
    """Per-request ephemeral state that plugins may use."""

    def __init__(self) -> None:
        self._state: dict[str, object] = {}

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._state.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._state[key] = value
