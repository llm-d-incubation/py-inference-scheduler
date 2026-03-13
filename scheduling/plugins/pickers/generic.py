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
import random
from typing import Optional, Sequence
from ...framework import (
    PickerPlugin, 
    Endpoint, 
    ScoredEndpoint, 
    CycleState, 
    LLMRequest, 
    register_picker
)


@register_picker("random")
class RandomPicker(PickerPlugin):
    def __init__(self, max_num: int = 1) -> None:
        self.max_num = max_num

    def pick(self, cycle_state: CycleState, request: LLMRequest, scored_endpoints: Sequence[ScoredEndpoint]) -> Optional[ScoredEndpoint]:
        if not scored_endpoints:
            return None
        top = scored_endpoints[: self.max_num]
        return random.choice(top)


@register_picker("max_score")
class MaxScorePicker(PickerPlugin):
    def __init__(self, max_num: int = 1) -> None:
        self.max_num = max_num

    def pick(self, cycle_state: CycleState, request: LLMRequest, scored_endpoints: Sequence[ScoredEndpoint]) -> Optional[ScoredEndpoint]:
        if not scored_endpoints:
            return None
        return scored_endpoints[0]
