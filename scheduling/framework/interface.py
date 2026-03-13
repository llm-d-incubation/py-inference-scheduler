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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Mapping
from .types import Endpoint, ScoredEndpoint, CycleState, LLMRequest, ProfileRunResult


class FilterPlugin(Protocol):
    def filter(self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]) -> Mapping[str, Endpoint]:
        ...


class ScorerPlugin(Protocol):
    def score(self, cycle_state: CycleState, request: LLMRequest, pods: Mapping[str, Endpoint]) -> Dict[str, float]:
        ...


class PickerPlugin(Protocol):
    def pick(self, cycle_state: CycleState, request: LLMRequest, scored_pods: Sequence[ScoredEndpoint]) -> Optional[ScoredEndpoint]:
        ...


class ProfileHandler(Protocol):
    def pick(self, cycle_state: CycleState, request: LLMRequest, profiles: Dict[str, "SchedulerProfile"], profile_results: Dict[str, Optional[ProfileRunResult]]) -> Dict[str, "SchedulerProfile"]:
        ...

    def process_results(self, cycle_state: CycleState, request: LLMRequest, profile_results: Dict[str, Optional[ProfileRunResult]]) -> Optional[str]:
        ...


@dataclass
class WeightedScorer:
    scorer: ScorerPlugin
    weight: float = 1.0


@dataclass
class SchedulerProfile:
    name: str
    filters: List[FilterPlugin] = field(default_factory=list)
    scorers: List[WeightedScorer] = field(default_factory=list)
    picker: Optional[PickerPlugin] = None

    def with_filters(self, *fs: FilterPlugin) -> "SchedulerProfile":
        self.filters.extend(fs)
        return self

    def with_scorers(self, *ss: WeightedScorer) -> "SchedulerProfile":
        self.scorers.extend(ss)
        return self

    def with_picker(self, p: PickerPlugin) -> "SchedulerProfile":
        self.picker = p
        return self

    def run(self, request: LLMRequest, cycle_state: CycleState, candidates: Sequence[Endpoint]) -> ProfileRunResult:
        # normalize candidates into a mapping name->Endpoint
        endpoints_map: Dict[str, Endpoint] = {e.name: e for e in candidates}

        # run filters: each filter returns a mapping of name->Endpoint
        for f in self.filters:
            endpoints_map = dict(f.filter(cycle_state, request, endpoints_map))

        # if no endpoints left, return empty result
        if not endpoints_map:
            return ProfileRunResult()

        # combine scores (score plugins return name->float)
        total_scores: Dict[str, float] = {name: 0.0 for name in endpoints_map.keys()}
        for w in self.scorers:
            raw_sc = w.scorer.score(cycle_state, request, endpoints_map)
            print(f"Scorer {w.scorer} raw scores: {raw_sc}")
            # normalize raw_sc values to [0,1] across endpoints present in endpoints_map
            vals = [float(raw_sc.get(name, 0.0)) for name in endpoints_map.keys()]
            if vals:
                min_v = min(vals)
                max_v = max(vals)
                if max_v > min_v:
                    for name in endpoints_map.keys():
                        v = float(raw_sc.get(name, 0.0))
                        norm = (v - min_v) / (max_v - min_v)
                        total_scores[name] += norm * w.weight
                else:
                    # all values equal -> give full score to all
                    for name in endpoints_map.keys():
                        total_scores[name] += 1.0 * w.weight

        # create ScoredEndpoint list preserving endpoint info
        scored = [ScoredEndpoint(endpoint=endpoints_map[name], score=total_scores.get(name, 0.0)) for name in endpoints_map.keys()]
        # sort descending
        scored.sort(key=lambda sp: sp.score, reverse=True)

        chosen = []
        if self.picker is not None:
            picked = self.picker.pick(cycle_state, request, scored)
            if picked is not None:
                chosen = [picked]
        else:
            # default: take highest-scoring endpoint
            if scored:
                chosen = [scored[0]]

        return ProfileRunResult(endpoint_list=chosen)
