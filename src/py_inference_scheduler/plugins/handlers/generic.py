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
    FilterPlugin,
    LLMRequest,
    ProfileHandler,
    ProfileRunResult,
    SchedulerProfile,
    register_filter,
    register_profile_handler,
)


@register_filter("simple")
class SimpleFilter(FilterPlugin):
    """A filter that keeps endpoints whose attribute `key` equals `value` (if provided)."""

    def __init__(self, key: str, value: object | None = None) -> None:
        self.key = key
        self.value = value

    def filter(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        pods: Mapping[str, Endpoint],
    ) -> Mapping[str, Endpoint]:
        result: dict[str, Endpoint] = {}
        for name, p in pods.items():
            v = p.attributes.get(self.key)
            if self.value is None or v == self.value:
                result[name] = p
        return result


@register_profile_handler("single_profile")
class SingleProfileHandler(ProfileHandler):
    """Simple profile handler that runs all profiles and returns the first as primary."""

    def pick(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        profiles: dict[str, SchedulerProfile],
        profile_results: dict[str, ProfileRunResult | None],
    ) -> dict[str, SchedulerProfile]:
        return profiles.copy()

    def process_results(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        profile_results: dict[str, ProfileRunResult | None],
    ) -> str | None:
        for name, res in profile_results.items():
            if res is not None and res.endpoint_list:
                return name
        if profile_results:
            return next(iter(profile_results))
        return None
