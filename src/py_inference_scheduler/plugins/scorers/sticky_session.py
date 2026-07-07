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

import hashlib
from typing import Mapping

from scheduling.framework import (
    CycleState,
    Endpoint,
    LLMRequest,
    ScorerPlugin,
    register_scorer,
)


@register_scorer("sticky_session")
class StickySessionScorer(ScorerPlugin):
    def __init__(self, header_name: str) -> None:
        if not header_name.strip():
            raise ValueError("Sticky session header name must be non-empty.")
        self.header_name = header_name.strip().lower()

    def score(
        self,
        cycle_state: CycleState,
        request: LLMRequest,
        pods: Mapping[str, Endpoint],
    ) -> dict[str, float]:
        if not pods:
            return {}

        session_id = _session_id_from_headers(request.headers, self.header_name)
        if session_id is None:
            return dict.fromkeys(pods.keys(), 0.0)

        selected_name = max(
            pods,
            key=lambda name: _rendezvous_score(
                session_id=session_id,
                endpoint_name=name,
            ),
        )
        scores = dict.fromkeys(pods.keys(), 0.0)
        scores[selected_name] = 1.0
        return scores


def _session_id_from_headers(
    headers: dict[str, str],
    header_name: str,
) -> str | None:
    raw = next(
        (value for key, value in headers.items() if key.lower() == header_name),
        None,
    )
    if raw is None:
        return None
    session_id = raw.strip()
    return session_id or None


def _rendezvous_score(
    *,
    session_id: str,
    endpoint_name: str,
) -> int:
    key = f"{session_id}\0{endpoint_name}".encode()
    return int.from_bytes(hashlib.sha256(key).digest(), "big")
