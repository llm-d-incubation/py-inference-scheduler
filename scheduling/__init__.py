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

"""Simple Python port of the scheduling package from the Go implementation.

This package provides a lightweight Scheduler, SchedulerConfig, basic plugin
interfaces (Scorer/Filter/Picker/ProfileHandler) and core types used by the
scheduler. It is intentionally small and focuses on the same high-level
concepts so it can be used for experimentation and unit testing in Python.
"""

from .core.scheduler import Scheduler
from .core.config import SchedulerConfig
from .framework import (
    LLMRequest,
    Endpoint,
    ScoredEndpoint,
    SchedulingResult,
    ProfileRunResult,
    CycleState,
    SchedulerProfile,
    ProfileHandler,
)
from .plugins.scorers.prefix_plugin import PrefixCacheScorer
from .plugins.scorers.generic import RoundRobinScorer
from .plugins import LeastQueueScorer, WaitingQueueScorer, RunningQueueScorer, KVCacheScorer

__all__ = [
    "Scheduler",
    "SchedulerConfig",
    "LLMRequest",
    "Endpoint",
    "ScoredEndpoint",
    "SchedulingResult",
    "ProfileRunResult",
    "CycleState",
    "SchedulerProfile",
    "ProfileHandler",
    "PrefixCacheScorer",
    "RoundRobinScorer",
    "LeastQueueScorer",
    "WaitingQueueScorer",
    "RunningQueueScorer",
    "KVCacheScorer",
]
