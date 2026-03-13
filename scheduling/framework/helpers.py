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

from typing import Dict, Sequence, Callable
from .types import Endpoint

def score_by_metric(
    endpoints: Sequence[Endpoint],
    metric_extractor: Callable[[Endpoint], float],
    lower_is_better: bool = True
) -> Dict[str, float]:
    """
    Helper function to score endpoints relative to each other based on a numeric metric.
    
    Args:
        endpoints: A sequence or dictionary values of Endpoint objects.
        metric_extractor: A callable that takes an Endpoint and returns a float metric.
        lower_is_better: If True, lower metric values receive higher scores (close to 1.0).
                         If False, higher metric values receive higher scores.
                         
    Returns:
        A dictionary mapping endpoint names to a normalized score between 0.0 and 1.0.
    """
    eps = endpoints.values() if isinstance(endpoints, dict) else endpoints
    
    min_val = float('inf')
    max_val = float('-inf')

    for ep in eps:
        val = metric_extractor(ep)
        if val < min_val: min_val = val
        if val > max_val: max_val = val

    scores: Dict[str, float] = {}
    for ep in eps:
        val = metric_extractor(ep)
        if max_val == min_val:
            scores[ep.name] = 1.0
        else:
            if lower_is_better:
                scores[ep.name] = (max_val - val) / (max_val - min_val)
            else:
                scores[ep.name] = (val - min_val) / (max_val - min_val)
                
    return scores
