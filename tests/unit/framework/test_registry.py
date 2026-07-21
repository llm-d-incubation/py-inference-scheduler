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

import pathlib
import textwrap

import pytest

from py_inference_scheduler.core.config import SchedulerConfig
from py_inference_scheduler.core.scheduler import Scheduler
from py_inference_scheduler.framework import _PICKERS, _PROFILE_HANDLERS, _SCORERS
from py_inference_scheduler.plugins import (
    MaxScorePicker,
    PrefixCacheScorer,
    SingleProfileHandler,
    WaitingQueueScorer,
)


def test_registry_populated():
    assert "waiting_queue" in _SCORERS
    assert _SCORERS["waiting_queue"] == WaitingQueueScorer

    assert "max_score" in _PICKERS
    assert _PICKERS["max_score"] == MaxScorePicker

    assert "single_profile" in _PROFILE_HANDLERS
    assert _PROFILE_HANDLERS["single_profile"] == SingleProfileHandler


def test_scheduler_config_from_dict():
    config_dict = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {
            "test_profile": {
                "scorers": [
                    {"type": "waiting_queue", "weight": 2.5},
                    {"type": "constant", "value": 5.0},
                ],
                "picker": {"type": "max_score", "max_num": 3},
            }
        },
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


def test_prefix_cache_config_propagates_from_yaml_file(tmp_path: pathlib.Path):
    """End-to-end config propagation for the prefix_cache scorer.

    Mirrors the production path: the Scheduler reads the mounted scheduler.yaml
    (the inner document of the ConfigMap, not the manifest wrapper) with
    yaml.safe_load, and SchedulerConfig.from_dict pops type/weight and passes
    every remaining key straight into the registered scorer's constructor.
    """
    scheduler_yaml = textwrap.dedent(
        """
        profile_handler:
          type: single_profile
        profiles:
          default:
            scorers:
              - type: prefix_cache
                weight: 10.0
                block_size: 8
                max_prefix_blocks: 32
                lru_capacity_per_server: 123
                min_match_ratio: 0.25
            picker:
              type: max_score
        """
    )
    config_file = tmp_path / "scheduler.yaml"
    config_file.write_text(scheduler_yaml, encoding="utf-8")

    scheduler = Scheduler(config_path=str(config_file))
    scheduler._maybe_reload_config()

    weighted = scheduler.profiles["default"].scorers[0]
    assert weighted.weight == 10.0

    scorer = weighted.scorer
    assert isinstance(scorer, PrefixCacheScorer)
    assert scorer.block_size == 8
    assert scorer.max_prefix_blocks == 32
    assert scorer.min_match_ratio == 0.25
    assert scorer.indexer._lru_capacity_per_server == 123


def test_scheduler_config_invalid_type():
    config_dict = {
        "profile_handler": {"type": "single_profile"},
        "profiles": {"test_profile": {"scorers": [{"type": "does_not_exist"}]}},
    }

    with pytest.raises(ValueError, match="Unknown plugin type 'does_not_exist'"):
        SchedulerConfig.from_dict(config_dict)
