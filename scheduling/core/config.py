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

from dataclasses import dataclass
from typing import Dict, Any
from ..framework import (
    ProfileHandler, 
    SchedulerProfile, 
    WeightedScorer,
    build_profile_handler, 
    build_scorer, 
    build_picker, 
    build_filter
)


@dataclass
class SchedulerConfig:
    profile_handler: ProfileHandler
    profiles: Dict[str, SchedulerProfile]

    def __str__(self) -> str:
        return f"{{ProfileHandler: {type(self.profile_handler).__name__}, Profiles: {list(self.profiles.keys())}}}"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SchedulerConfig":
        """
        Parses a nested dictionary (e.g. from YAML or JSON - sticking to YAML for now but the code works for both) into a full 
        SchedulerConfig object containing instantiated Scorer, Picker, and Filter plugins.
        """
        if not config_dict:
            raise ValueError("Provided configuration dictionary is empty.")

        ph_config = config_dict.get("profile_handler")
        if not ph_config or "type" not in ph_config:
            raise ValueError("Scheduler configuration must include a 'profile_handler' dictionary with a 'type' key.")
        
        ph_type = ph_config.pop("type")
        profile_handler = build_profile_handler(ph_type, **ph_config)

        profiles_dict = config_dict.get("profiles")
        if not profiles_dict:
            raise ValueError("Scheduler configuration must include a 'profiles' dictionary.")

        parsed_profiles: Dict[str, SchedulerProfile] = {}
        for profile_name, prof_data in profiles_dict.items():
            profile = SchedulerProfile(name=profile_name)

            for filter_config in prof_data.get("filters", []):
                cfg = dict(filter_config)
                f_type = cfg.pop("type")
                f_instance = build_filter(f_type, **cfg)
                profile.with_filters(f_instance)

            for scorer_config in prof_data.get("scorers", []):
                cfg = dict(scorer_config)
                s_type = cfg.pop("type")
                weight = cfg.pop("weight", 1.0)
                s_instance = build_scorer(s_type, **cfg)
                profile.with_scorers(WeightedScorer(scorer=s_instance, weight=float(weight)))

            picker_config = prof_data.get("picker")
            if picker_config:
                cfg = dict(picker_config)
                p_type = cfg.pop("type")
                p_instance = build_picker(p_type, **cfg)
                profile.with_picker(p_instance)

            parsed_profiles[profile_name] = profile

        return cls(profile_handler=profile_handler, profiles=parsed_profiles)
