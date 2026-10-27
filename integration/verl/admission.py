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

import bisect
from dataclasses import dataclass
from typing import Callable, Sequence

from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.framework import Endpoint, FlowControlPlugin, LLMRequest
from py_inference_scheduler.framework.helpers import prefill_tokens

# Receives a queued request's admission, or the error that stopped it.
Deliver = Callable[["Endpoint | Exception"], None]


@dataclass(eq=False)
class _Waiting:
    """A queued request, with the sizes its retries compare."""

    ticket: str
    request: LLMRequest
    ranking: Sequence[str]
    deliver: Deliver
    prompt: int
    # Fixed while queued: the trajectory's previous turn, which set it, has already finished.
    need: int

    def __lt__(self, other: _Waiting) -> bool:
        # Longest prompt first, and in arrival order among equals.
        return self.prompt > other.prompt


class Admission:
    """Admits requests with a flow-control plugin, in the order each worker ranked the engines.

    - Reserves and counts each admission in the same step, so later decisions see it.
    - Keeps requests that fit nowhere in one queue, longest prompt first, and hands each admission
      to the callback its request queued with.
    """

    def __init__(
        self,
        plugin: FlowControlPlugin,
        counts: InflightStore,
        endpoints: Callable[[], Sequence[Endpoint]],
    ) -> None:
        self._plugin = plugin
        self._counts = counts
        self._endpoints = endpoints
        # Longest prompt first: late turns are the step's critical path and the costliest to stall.
        self._waiting: list[_Waiting] = []

    def admit(
        self, request: LLMRequest, ranking: Sequence[str], ticket: str, deliver: Deliver
    ) -> Endpoint | None:
        """Admit the request to its highest-ranked engine with room, or queue it until one has."""
        winner = self._try_admit(request, ranking, self._endpoints())
        if winner is None:
            prompt, need = prefill_tokens(request.body), self._plugin.need(request)
            waiting = _Waiting(ticket, request, ranking, deliver, prompt, need)
            bisect.insort(self._waiting, waiting)
        return winner

    def cancel(self, ticket: str) -> None:
        """Drop a queued request whose caller gave up."""
        self._waiting = [w for w in self._waiting if w.ticket != ticket]

    def has_waiting(self) -> bool:
        return bool(self._waiting)

    def finish(self, request_id: str, endpoint_name: str, output_tokens: int | None) -> None:
        """Release the request's reservation and retry the requests waiting for one."""
        self._counts.decrement(endpoint_name)
        self._plugin.release(LLMRequest(request_id=request_id), endpoint_name, output_tokens)
        # Only this engine gained room, so only it can take a queued request now.
        freed = next((ep for ep in self._endpoints() if ep.name == endpoint_name), None)
        if freed is not None:
            self.retry(freed)

    def _try_admit(
        self, request: LLMRequest, ranking: Sequence[str], candidates: Sequence[Endpoint]
    ) -> Endpoint | None:
        allowed = {ep.name: ep for ep in self._plugin.get_allowed_candidates(request, candidates)}
        if not allowed:
            return None
        # An engine the worker has not discovered yet is not ranked, so it comes last.
        winner = allowed[next((name for name in ranking if name in allowed), next(iter(allowed)))]
        self._plugin.reserve(request, winner)
        self._counts.increment(winner.name)
        return winner

    def retry(self, engine: Endpoint | None = None) -> None:
        """Admit the queued requests that now fit on this engine, or on any engine by default."""
        engines = [engine] if engine is not None else self._endpoints()
        largest = max((self._plugin.room(ep) for ep in engines), default=0)
        for waiting in list(self._waiting):
            # A size check first, so a long queue that still fits nowhere costs little. With no
            # room anywhere, the plugin still sees each request, to report engines it cannot use.
            if 0 < largest < waiting.need:
                continue
            try:
                winner = self._try_admit(waiting.request, waiting.ranking, engines)
            except Exception as e:  # noqa: BLE001 - the waiting caller must see the failure
                self._waiting.remove(waiting)
                waiting.deliver(e)
                continue
            if winner is not None:
                self._waiting.remove(waiting)
                waiting.deliver(winner)
                largest = max((self._plugin.room(ep) for ep in engines), default=0)
