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

import os
import yaml
from typing import Dict, Optional, Sequence
from ..framework import (
    LLMRequest, 
    Endpoint, 
    SchedulingResult, 
    CycleState, 
    ProfileRunResult,
    ProfileHandler,
    ScoredEndpoint
)
from .config import SchedulerConfig


class Scheduler:
    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path:
            self.config_path = config_path
        else:
            self.config_path = os.environ.get("ROUTER_CONFIG_PATH")
            if not self.config_path:
                raise ValueError("ROUTER_CONFIG_PATH environment variable is missing and no config_path provided. Ensure the ConfigMap is mounted or path is passed.")

        self.last_mtime = 0
        self._maybe_reload_config()

    @classmethod
    def new_with_config(cls, config: SchedulerConfig) -> "Scheduler":
        instance = object.__new__(cls)
        instance.config_path = None
        instance.last_mtime = 0
        instance.profile_handler = config.profile_handler
        instance.profiles = config.profiles
        return instance

    def _maybe_reload_config(self):
        if self.config_path is None:
            return
        mtime = os.path.getmtime(self.config_path)
        if mtime > self.last_mtime:
            print(f"Reloading scheduler config from {self.config_path}")
            with open(self.config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            if not isinstance(config_dict, dict):
                raise ValueError("Parsed configuration is not a valid dictionary.")
            config = SchedulerConfig.from_dict(config_dict)
            self.profile_handler = config.profile_handler
            self.profiles = config.profiles
            self.last_mtime = mtime

    def schedule(self, request: LLMRequest, candidates: Sequence[Endpoint]) -> SchedulingResult:
        if not candidates:
            raise ValueError("no scheduling candidates provided")

        cycle_state = CycleState()
        profile_results: Dict[str, Optional[ProfileRunResult]] = {}

        # ask profile handler which profiles to run
        selected = self.profile_handler.pick(cycle_state, request, self.profiles, profile_results)
        assert selected is not None

        for name, profile in selected.items():
            try:
                res = profile.run(request, cycle_state, candidates)
                profile_results[name] = res
            except Exception as e:
                print(f"Error running profile {name}: ")
                print(repr(e))
                profile_results[name] = None

        primary = self.profile_handler.process_results(cycle_state, request, profile_results)

        # Build SchedulingResult
        result = SchedulingResult(profile_results={k: v or ProfileRunResult() for k, v in profile_results.items()}, primary_profile_name=primary)
        selected_eps = result.profile_results[primary].endpoint_list[:1] if primary in result.profile_results else []
        print(f"Selected endpoint {selected_eps}")
        if selected_eps:
            for w in self.profiles[primary].scorers:
                if hasattr(w.scorer, "pre_request"):
                    w.scorer.pre_request(cycle_state, request, selected_eps[0].endpoint)
        return result

    def run(self, request: LLMRequest, candidates: Sequence[Endpoint]) -> Sequence[ScoredEndpoint]:
        self._maybe_reload_config()
        scheduler_output = self.schedule(request, candidates)
        profile_name = scheduler_output.primary_profile_name
        profile_results = scheduler_output.profile_results.get(profile_name)

        print(f"Profile {profile_name} results: {profile_results}")
        selected_endpoint = profile_results.endpoint_list[:1]

        if len(selected_endpoint) > 0:
            return selected_endpoint  # pick top 1
        print("No endpoint selected, defaulting to framework routing logic")
        return []
