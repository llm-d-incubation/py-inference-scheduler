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

import os
import pathlib
from typing import Sequence

import yaml

from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.framework import (
    CycleState,
    Endpoint,
    FlowControlPlugin,
    LLMRequest,
    ProfileHandler,
    ProfileRunResult,
    SchedulerProfile,
    SchedulingResult,
    ScoredEndpoint,
)


class Scheduler:
    config_path: str | None
    last_mtime: float
    profile_handler: ProfileHandler
    profiles: dict[str, SchedulerProfile]

    def __init__(self, config_path: str | None = None) -> None:
        if config_path:
            self.config_path = config_path
        else:
            self.config_path = os.environ.get("ROUTER_CONFIG_PATH")
            if not self.config_path:
                raise ValueError(
                    "ROUTER_CONFIG_PATH environment variable is missing and no "
                    "config_path provided. Ensure the ConfigMap is mounted or "
                    "path is passed."
                )

        self.last_mtime = 0.0

    @classmethod
    def new_with_config(cls, config: SchedulerConfig) -> Scheduler:
        instance = object.__new__(cls)
        instance.config_path = None
        instance.last_mtime = 0.0
        instance.profile_handler = config.profile_handler
        instance.profiles = config.profiles
        return instance

    def has_flow_control(self) -> bool:
        """Returns True if any profile has any flow control plugins."""
        return bool(self.get_flow_control_plugins())

    def get_flow_control_plugins(self) -> list[FlowControlPlugin]:
        """Returns all flow control plugins from all profiles."""
        all_plugins = []
        if hasattr(self, "profiles") and self.profiles:
            for profile in self.profiles.values():
                if profile.flow_controls:
                    all_plugins.extend(profile.flow_controls)
        return all_plugins

    def _maybe_reload_config(self) -> None:
        if self.config_path is None:
            return
        mtime = pathlib.Path(self.config_path).stat().st_mtime
        if mtime > self.last_mtime:
            print(f"Reloading scheduler config from {self.config_path}")
            with pathlib.Path(self.config_path).open(encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                raise ValueError("Parsed configuration is not a valid dictionary.")
            config = SchedulerConfig.from_dict(config_dict)
            self.profile_handler = config.profile_handler
            self.profiles = config.profiles
            self.last_mtime = mtime

    def schedule(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> SchedulingResult:
        self._maybe_reload_config()
        if not candidates:
            raise ValueError("no scheduling candidates provided")

        cycle_state = CycleState()
        profile_results: dict[str, ProfileRunResult | None] = {}

        # ask profile handler which profiles to run
        selected = self.profile_handler.pick(
            cycle_state, request, self.profiles, profile_results
        )
        assert selected is not None  # noqa: S101

        def run_profile(
            profile_name: str, profile: SchedulerProfile
        ) -> ProfileRunResult | None:
            try:
                return profile.run(request, cycle_state, candidates)
            except Exception as e:  # noqa: BLE001
                print(f"Error running profile {profile_name}: ")
                print(repr(e))
                return None

        for name, profile in selected.items():
            profile_results[name] = run_profile(name, profile)

        primary = self.profile_handler.process_results(
            cycle_state, request, profile_results
        )

        # Build SchedulingResult
        result = SchedulingResult(
            profile_results={k: v or ProfileRunResult() for k, v in profile_results.items()},
            primary_profile_name=primary,
        )

        selected_eps = (
            result.profile_results[primary].endpoint_list[:1]
            if primary is not None and primary in result.profile_results
            else []
        )
        print(f"Selected endpoint {selected_eps}")
        if selected_eps and primary is not None:
            for w in self.profiles[primary].scorers:
                if hasattr(w.scorer, "pre_request"):
                    w.scorer.pre_request(cycle_state, request, selected_eps[0].endpoint)  # type: ignore[attr-defined]
        return result

    def run(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[ScoredEndpoint]:
        scheduler_output = self.schedule(request, candidates)
        profile_name = scheduler_output.primary_profile_name
        profile_results = (
            scheduler_output.profile_results.get(profile_name)
            if profile_name is not None
            else None
        )

        print(f"Profile {profile_name} results: {profile_results}")
        selected_endpoint = profile_results.endpoint_list[:1] if profile_results else []

        if len(selected_endpoint) > 0:
            return selected_endpoint  # pick top 1
        print("No endpoint selected, defaulting to framework routing logic")
        return []
