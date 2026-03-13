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

# Import sub-modules to trigger registration side-effects
from .scorers import prefix_plugin, backpressure, generic as generic_scorers
from .pickers import generic as generic_pickers
from .handlers import generic as generic_handlers

# Re-export key plugins if needed, but registry handles dynamic lookup
from .scorers.prefix_plugin import PrefixCacheScorer
from .scorers.backpressure import WaitingQueueScorer, RunningQueueScorer, KVCacheScorer, LeastQueueScorer, QueueLengthScorer
from .scorers.generic import RoundRobinScorer, ConstantScorer
from .pickers.generic import RandomPicker, MaxScorePicker
from .handlers.generic import SingleProfileHandler, SimpleFilter
