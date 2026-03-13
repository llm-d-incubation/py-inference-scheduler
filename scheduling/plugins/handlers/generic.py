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
from typing import Any, Dict, Optional, Sequence, Mapping
from ...framework import (
    FilterPlugin, 
    ProfileHandler, 
    SchedulerProfile,
    Endpoint, 
    CycleState, 
    LLMRequest, 
    ProfileRunResult,
    register_filter,
    register_profile_handler
)


@register_filter("simple")
class SimpleFilter(FilterPlugin):
    """A filter that keeps endpoints whose attribute `key` equals `value` (if provided)."""

    def __init__(self, key: str, value: Optional[Any] = None) -> None:
        self.key = key
        self.value = value

    def filter(self, cycle_state: CycleState, request: LLMRequest, endpoints: Sequence[Endpoint]) -> Mapping[str, Endpoint]:
        if isinstance(endpoints, Mapping):
            result: Dict[str, Endpoint] = {}
            for name, p in endpoints.items():
                v = p.attributes.get(self.key)
                if self.value is None or v == self.value:
                    result[name] = p
            return result

        out: List[Endpoint] = []
        for p in endpoints:
            v = p.attributes.get(self.key)
            if self.value is None or v == self.value:
                out.append(p)
        return {p.name: p for p in out}


@register_profile_handler("single_profile")
class SingleProfileHandler(ProfileHandler):
    """Simple profile handler that runs all profiles and returns the first as primary."""

    def pick(self, cycle_state: CycleState, request: LLMRequest, profiles: Dict[str, SchedulerProfile], profile_results: Dict[str, Optional[ProfileRunResult]]) -> Dict[str, SchedulerProfile]:
        return profiles.copy()

    def process_results(self, cycle_state: CycleState, request: LLMRequest, profile_results: Dict[str, Optional[ProfileRunResult]]) -> Optional[str]:
        for name, res in profile_results.items():
            if res is not None and res.endpoint_list:
                return name
        if profile_results:
            return next(iter(profile_results))
        return None
