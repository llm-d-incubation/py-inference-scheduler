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

import math
import sys
from collections import OrderedDict
from typing import NamedTuple, Sequence

from py_inference_scheduler.framework import (
    Endpoint,
    FlowControlPlugin,
    LLMRequest,
    register_flow_control,
)
from py_inference_scheduler.framework.helpers import prefill_tokens

# Far more trajectories than one rollout holds.
_MAX_REMEMBERED_TRAJECTORIES = 65536


class _Reservation(NamedTuple):
    """Tokens an admitted, unfinished request has booked on one engine."""

    engine: str
    tokens: int


class _LastTurn(NamedTuple):
    """Where a trajectory's last turn ran, and its finished turns' total output and count."""

    engine: str
    total_output: int
    turns: int


@register_flow_control("kv_saturation")
class KVSaturationPlugin(FlowControlPlugin):
    """Admits a request only where its tokens fit in an engine's KV budget.

    - A request needs its prompt plus its trajectory's average output, or default_osl on turn one.
    - An engine's budget is budget_fraction of its KV capacity, minus what admitted, unfinished
      requests reserved.
    - Offers the engine that served the trajectory's last turn if it fits, otherwise every engine
      that fits.
    - Offers nothing when no engine fits, so the caller queues the request.
    """

    def __init__(self, default_osl: int = 1024, budget_fraction: float = 1.0) -> None:
        if default_osl < 0:
            raise ValueError("default_osl must be >= 0.")
        if not 0 < budget_fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1].")
        self.default_osl = default_osl
        self.budget_fraction = budget_fraction
        self._reserved_by_engine: dict[str, int] = {}
        self._reservations: dict[str, _Reservation] = {}
        self._last_turns: OrderedDict[str, _LastTurn] = OrderedDict()

    def get_allowed_candidates(
        self, request: LLMRequest, candidates: Sequence[Endpoint]
    ) -> Sequence[Endpoint]:
        if candidates and all(_reports_no_capacity(ep) for ep in candidates):
            raise RuntimeError("No engine reports its KV capacity, which kv_saturation needs.")
        need = self.kv_tokens_needed(request)
        fitting = [ep for ep in candidates if self._fits(ep, need, request.request_id)]
        last = self._last_turns.get(request.request_id)
        home = [ep for ep in fitting if last and ep.name == last.engine]
        return home or fitting

    def reserve(self, request: LLMRequest, selected: Endpoint) -> None:
        # Ray Serve routes a request again when a replica turns it away. Drop the old reservation
        # first so the request is never booked on two engines.
        self._drop_reservation(request.request_id)
        engine = selected.name
        tokens = min(self.kv_tokens_needed(request), self._kv_token_budget(selected))
        self._reserved_by_engine[engine] = self._reserved_by_engine.get(engine, 0) + tokens
        self._reservations[request.request_id] = _Reservation(engine, tokens)

    def release(
        self, request: LLMRequest, endpoint_name: str, output_tokens: int | None = None
    ) -> None:
        engine = self._drop_reservation(request.request_id)
        if engine is None or output_tokens is None:
            return
        previous = self._last_turns.get(request.request_id)
        total, turns = (previous.total_output, previous.turns) if previous else (0, 0)
        self._last_turns[request.request_id] = _LastTurn(engine, total + output_tokens, turns + 1)
        self._last_turns.move_to_end(request.request_id)
        if len(self._last_turns) > _MAX_REMEMBERED_TRAJECTORIES:
            self._last_turns.popitem(last=False)

    def kv_tokens_needed(self, request: LLMRequest) -> int:
        """The request's prompt plus its trajectory's average output, or default_osl on turn one."""
        last = self._last_turns.get(request.request_id)
        output = math.ceil(last.total_output / last.turns) if last else self.default_osl
        return prefill_tokens(request.body) + output

    def max_admissible_kv_tokens(self, ep: Endpoint) -> int:
        """The largest request the engine can take now.

        An idle engine takes any request, as an oversize one reserves a whole engine.
        """
        budget = self._kv_token_budget(ep)
        if budget <= 0:
            return 0
        reserved = self._reserved_by_engine.get(ep.name, 0)
        return budget - reserved if reserved else sys.maxsize

    def _drop_reservation(self, request_id: str) -> str | None:
        reservation = self._reservations.pop(request_id, None)
        if reservation is None:
            return None
        self._reserved_by_engine[reservation.engine] -= reservation.tokens
        return reservation.engine

    def _fits(self, ep: Endpoint, need: int, request_id: str) -> bool:
        budget = self._kv_token_budget(ep)
        # A request routed again must not compete with its own old reservation.
        reservation = self._reservations.get(request_id)
        own = reservation.tokens if reservation and reservation.engine == ep.name else 0
        reserved = self._reserved_by_engine.get(ep.name, 0) - own
        # An oversize request reserves the whole budget, so it waits for an idle engine.
        return budget > 0 and reserved + min(need, budget) <= budget

    def _kv_token_budget(self, ep: Endpoint) -> int:
        return int(_capacity(ep) * self.budget_fraction)


def _capacity(ep: Endpoint) -> int:
    value = ep.attributes.get("kv_cache_size")
    return int(value) if isinstance(value, (int, float)) else 0


def _reports_no_capacity(ep: Endpoint) -> bool:
    # Engines not yet polled have empty stats and are simply waited for.
    stats = ep.attributes.get("routing_stats")
    return isinstance(stats, dict) and bool(stats) and not stats.get("error") and _capacity(ep) <= 0
