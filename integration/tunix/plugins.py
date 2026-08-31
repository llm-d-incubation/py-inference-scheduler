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

import logging
from typing import Mapping

from py_inference_scheduler.framework import (
    CycleState,
    Endpoint,
    FilterPlugin,
    LLMRequest,
    register_filter,
)

logger = logging.getLogger(__name__)


@register_filter("worker_state")
class WorkerStateFilter(FilterPlugin):
    """Drops endpoints whose tunix worker state makes them unroutable.

    tunix rollout workers report a lifecycle state through heartbeats (READY,
    SYNCING while a weight-sync round quiesces them, ERROR, ...). The
    orchestrator ships that state in each candidate's attributes; this filter
    keeps only the states allowed to serve.
    """

    def __init__(self, allowed_states: list[str] | None = None) -> None:
        self.allowed = set(allowed_states or ["READY"])

    def filter(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        pods: Mapping[str, Endpoint],
    ) -> Mapping[str, Endpoint]:
        eligible: dict[str, Endpoint] = {}
        dropped: list[str] = []
        for name, ep in pods.items():
            state = str(ep.attributes.get("state", "UNKNOWN"))
            if state in self.allowed:
                eligible[name] = ep
            else:
                dropped.append(f"{name}({state})")

        # A filter must never leave the ballot empty: routing somewhere beats
        # deadlock, since the worker just queues the request.
        if not eligible:
            logger.warning(
                "all %d endpoints unroutable by state: filter disabled for this decision",
                len(pods),
            )
            return pods
        if dropped:
            logger.info("worker_state filter dropped %s", dropped)
        return eligible
