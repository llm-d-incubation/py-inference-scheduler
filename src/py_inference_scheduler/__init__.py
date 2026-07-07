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

from .core.config import SchedulerConfig
from .core.scheduler import Scheduler
from .framework import (
    CycleState,
    Endpoint,
    LLMRequest,
    ProfileHandler,
    ProfileRunResult,
    SchedulerProfile,
    SchedulingResult,
    ScoredEndpoint,
)
from .plugins import (
    KVCacheScorer,
    LeastQueueScorer,
    RunningQueueScorer,
    WaitingQueueScorer,
)
from .plugins.scorers.generic import RoundRobinScorer
from .plugins.scorers.prefix_plugin import PrefixCacheScorer
from .plugins.scorers.sticky_session import StickySessionScorer

__all__ = [
    "CycleState",
    "Endpoint",
    "KVCacheScorer",
    "LLMRequest",
    "LeastQueueScorer",
    "PrefixCacheScorer",
    "ProfileHandler",
    "ProfileRunResult",
    "RoundRobinScorer",
    "RunningQueueScorer",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerProfile",
    "SchedulingResult",
    "ScoredEndpoint",
    "StickySessionScorer",
    "WaitingQueueScorer",
]
