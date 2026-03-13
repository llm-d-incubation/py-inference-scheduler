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

import pytest
from scheduling.core.config import SchedulerConfig
from scheduling.framework import _SCORERS, _PICKERS, _PROFILE_HANDLERS
from scheduling.plugins import WaitingQueueScorer, MaxScorePicker, SingleProfileHandler
import scheduling.plugins

def test_registry_populated():
    assert "waiting_queue" in _SCORERS
    assert _SCORERS["waiting_queue"] == WaitingQueueScorer
    
    assert "max_score" in _PICKERS
    assert _PICKERS["max_score"] == MaxScorePicker

    assert "single_profile" in _PROFILE_HANDLERS
    assert _PROFILE_HANDLERS["single_profile"] == SingleProfileHandler

def test_scheduler_config_from_dict():
    config_dict = {
        "profile_handler": {
            "type": "single_profile"
        },
        "profiles": {
            "test_profile": {
                "scorers": [
                    {"type": "waiting_queue", "weight": 2.5},
                    {"type": "constant", "value": 5.0}
                ],
                "picker": {
                    "type": "max_score",
                    "max_num": 3
                }
            }
        }
    }

    config = SchedulerConfig.from_dict(config_dict)

    assert isinstance(config.profile_handler, SingleProfileHandler)

    assert "test_profile" in config.profiles
    profile = config.profiles["test_profile"]

    assert len(profile.scorers) == 2
    
    assert isinstance(profile.scorers[0].scorer, WaitingQueueScorer)
    assert profile.scorers[0].weight == 2.5
    
    assert profile.scorers[1].scorer.value == 5.0

    assert isinstance(profile.picker, MaxScorePicker)
    assert profile.picker.max_num == 3

def test_scheduler_config_invalid_type():
    config_dict = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "test_profile": {
                "scorers": [{"type": "does_not_exist"}]
            }
        }
    }

    with pytest.raises(ValueError, match="Unknown plugin type 'does_not_exist'"):
        SchedulerConfig.from_dict(config_dict)
