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
"""verl integration hook: delegate rollout routing to py-inference-scheduler.

Supports two verl layouts, auto-detected at import time:

- **legacy** (v0.7.1): ``AsyncLLMServerManager`` lives in
  ``verl.experimental.agent_loop.agent_loop`` and owns the server list.
- **modern** (v0.9.x): ``LLMServerClient`` lives in
  ``verl.workers.rollout.llm_server``; a ``GlobalRequestLoadBalancer`` Ray
  actor owns the server registry and does atomic acquire. The fleet actor
  (fleet.py) recovers the engine handles by draining the balancer once for
  every worker.

Both layouts expose the same entrypoint for the trainer flag:
``+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager``
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import ray
from omegaconf import DictConfig  # type: ignore[import-not-found]

try:  # legacy layout (verl v0.7.x)
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
        AgentLoopManager,
        AgentLoopWorker,
    )
    from verl.experimental.agent_loop.agent_loop import (
        AsyncLLMServerManager as _LegacyServerManager,
    )

    _VERL_LAYOUT = "legacy"
except ImportError:  # modern layout (verl v0.9.x)
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
        AgentLoopManager,
        AgentLoopWorker,
    )
    from verl.workers.rollout.llm_server import (  # type: ignore[import-not-found]
        LLMServerClient as _ModernServerClient,
    )

    _VERL_LAYOUT = "modern"

from backends.verl.sglang import SglangEnginePatch
from backends.verl.vllm import VllmEnginePatch
from integration.verl.fleet import FleetSnapshot, fleet_actor
from py_inference_scheduler import Scheduler
from py_inference_scheduler.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)
logger.info("py-inference-scheduler verl hook: %s layout detected", _VERL_LAYOUT)

# Must apply at module level to patch classes before use across distributed
# Ray workers without modifying verl.
VllmEnginePatch.apply()
SglangEnginePatch.apply()


def _rollout_config(config: DictConfig):
    if config.get("actor_rollout_ref"):
        return config.actor_rollout_ref.rollout
    return config.rollout


class _SchedulerCore:
    """Per-worker routing on state shared by the whole fleet.

    - Reads engine metrics and fleet-wide in-flight counts from the fleet actor once per decision.
    - Holds a lock so a decision and the dispatch it counts never interleave with another.
    """

    def __init__(self) -> None:
        self.scheduler = Scheduler()
        self.fleet = fleet_actor()
        self.endpoints: list[Endpoint] = []
        self.lb_acquired_requests: set[str] = set()
        self.lock = asyncio.Lock()

    def set_endpoints(self, handles: dict[str, ray.actor.ActorHandle]) -> None:
        if set(handles) != {ep.name for ep in self.endpoints}:
            self.endpoints = [
                Endpoint(name=name, attributes={"replica_obj": handle, "routing_stats": {}})
                for name, handle in handles.items()
            ]

    async def schedule(self, request_id: str, prompt_ids: list[int] | None) -> Endpoint | None:
        """Pick an endpoint on the latest fleet snapshot; None means fall back to verl's LB."""
        async with self.lock:
            # Ray runs one caller's actor calls in order, so this worker's last
            # dispatch is already counted in the snapshot.
            snapshot: FleetSnapshot = await self.fleet.snapshot.remote()
            endpoints = self.endpoints
            for ep in endpoints:
                ep.attributes["routing_stats"] = snapshot.stats.get(ep.name, {})
                ep.attributes["queue_len"] = snapshot.inflight.get(ep.name, 0)
            request = LLMRequest(request_id=request_id, body=prompt_ids)
            selected = self.scheduler.run(request, candidates=endpoints)
            if not selected:
                return None
            winner: Endpoint = selected[0].endpoint
            self.note_dispatch(winner.name)
            return winner

    def note_dispatch(self, endpoint_name: str) -> None:
        self.fleet.increment.remote(endpoint_name)

    def release(self, endpoint_name: str) -> None:
        self.fleet.decrement.remote(endpoint_name)


if _VERL_LAYOUT == "legacy":

    class InferenceSchedulerServerManager(_LegacyServerManager):  # type: ignore[misc]
        """Delegate routing to py-inference-scheduler. Compatible with verl v0.7.1."""

        def __init__(
            self,
            config: DictConfig,
            servers: list[tuple[str, ray.actor.ActorHandle]],
            load_balancer_handle: ray.actor.ActorHandle,
            *args: object,
            **kwargs: object,
        ) -> None:
            super().__init__(config, servers, load_balancer_handle, *args, **kwargs)
            self.rollout_config = _rollout_config(config)
            self.core = _SchedulerCore()
            self.core.set_endpoints(dict(servers))
            self.core.fleet.watch.remote(dict(servers))

        async def _acquire_server(
            self,
            request_id: str,
            prompt_ids: list[int] | None = None,
        ) -> tuple[str, ray.actor.ActorHandle]:
            winner = await self.core.schedule(request_id, prompt_ids)
            if winner is None:
                logger.warning(
                    "py-inference-scheduler returned no endpoints, falling back to verl global LB."
                )
                self.core.lb_acquired_requests.add(request_id)
                server_id, handle = await super()._acquire_server(request_id)  # type: ignore[no-any-return]
                self.core.note_dispatch(server_id)
                return server_id, handle
            return winner.name, winner.attributes["replica_obj"]

        def _release_server(self, server_id: str, request_id: str | None = None) -> None:
            self.core.release(server_id)
            if request_id and request_id in self.core.lb_acquired_requests:
                super()._release_server(server_id)
                self.core.lb_acquired_requests.remove(request_id)

        async def generate(
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, object],
            image_data: list[object] | None = None,
            video_data: list[object] | None = None,
        ) -> object:
            # Yield CPU so queued metric/scheduling tasks can interleave.
            await asyncio.sleep(0)
            server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)

            # vLLMAsyncServer ignores ignore_eos from config, so pass it explicitly.
            # A fresh request_id per generation avoids vLLM KV-cache collisions
            # with verl's sticky multi-turn request ids.
            ignore_eos = self.rollout_config.get("ignore_eos", False)
            if isinstance(sampling_params, dict):
                sampling_params["ignore_eos"] = ignore_eos
            elif hasattr(sampling_params, "ignore_eos"):
                sampling_params.ignore_eos = ignore_eos

            try:
                return await server.generate.remote(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                )
            finally:
                self._release_server(server_id, request_id)

    class PyInferenceAgentLoopWorker(AgentLoopWorker):  # type: ignore[misc]
        """Inject the custom ServerManager before calling super().__init__."""

        def __init__(
            self,
            config: DictConfig,
            servers: list[tuple[str, ray.actor.ActorHandle]],
            load_balancer_handle: ray.actor.ActorHandle,
            reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
        ) -> None:
            self.server_manager = InferenceSchedulerServerManager(
                config, servers, load_balancer_handle
            )
            super().__init__(config, servers, load_balancer_handle, reward_loop_worker_handles)

else:  # modern layout

    class InferenceSchedulerServerClient(_ModernServerClient):  # type: ignore[misc]
        """Delegate routing to py-inference-scheduler. Compatible with verl v0.9.x.

        verl's balancer owns the engine registry but enumerates only ids; the
        fleet actor recovers the handles once for every worker.
        """

        def __init__(
            self,
            config: DictConfig,
            load_balancer_handle: ray.actor.ActorHandle = None,
            **kwargs: object,
        ) -> None:
            super().__init__(config, load_balancer_handle, **kwargs)
            self.rollout_config = _rollout_config(config)
            self.core = _SchedulerCore()
            self._view_complete = False

        async def _ensure_endpoints(self) -> None:
            if self._view_complete:
                return
            server_ids = await self._load_balancer.get_all_servers.remote()
            handles = await self.core.fleet.discover.remote(self._load_balancer, len(server_ids))
            self.core.set_endpoints(handles)
            self._view_complete = len(handles) >= len(server_ids)

        async def _acquire_server(
            self,
            request_id: str,
            prompt_ids: list[int] | None = None,
        ) -> tuple[str, ray.actor.ActorHandle]:
            await self._ensure_endpoints()
            winner = await self.core.schedule(request_id, prompt_ids)
            if winner is None:
                logger.warning(
                    "py-inference-scheduler returned no endpoints, falling back to verl global LB."
                )
                self.core.lb_acquired_requests.add(request_id)
                server_id, handle = await super()._acquire_server(request_id)
                self.core.note_dispatch(server_id)
                return server_id, handle
            return winner.name, winner.attributes["replica_obj"]

        def _release_server(self, server_id: str, request_id: str | None = None) -> None:
            self.core.release(server_id)
            if request_id and request_id in self.core.lb_acquired_requests:
                super()._release_server(server_id)
                self.core.lb_acquired_requests.remove(request_id)

        async def generate(  # noqa: PLR0913
            self,
            request_id: str,
            *,
            prompt_ids: list[int],
            sampling_params: dict[str, object],
            image_data: list[object] | None = None,
            video_data: list[object] | None = None,
            audio_data: list[object] | None = None,
            mm_processor_kwargs: dict[str, object] | None = None,
            **kwargs: object,
        ) -> object:
            await asyncio.sleep(0)
            server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)

            ignore_eos = self.rollout_config.get("ignore_eos", False)
            if isinstance(sampling_params, dict):
                sampling_params["ignore_eos"] = ignore_eos

            multimodal_kwargs: dict[str, object] = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            try:
                return await server.generate.remote(
                    request_id=uuid.uuid4().hex,  # fresh id per turn, mirrors upstream
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                    **multimodal_kwargs,
                    **kwargs,
                )
            finally:
                self._release_server(server_id, request_id)

    class PyInferenceAgentLoopWorker(AgentLoopWorker):  # type: ignore[misc,no-redef]
        """Swap the incoming LLMServerClient for the scheduler-backed client."""

        def __init__(
            self,
            config: DictConfig,
            llm_client: Any,  # noqa: ANN401 - verl LLMServerClient; type unavailable off-cluster
            teacher_client: dict | None = None,
            reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
        ) -> None:
            scheduler_client = InferenceSchedulerServerClient(
                config, load_balancer_handle=llm_client._load_balancer
            )
            super().__init__(config, scheduler_client, teacher_client, reward_loop_worker_handles)


class PyInferenceAgentLoopManager(AgentLoopManager):
    """Main hook entrypoint loaded by ray_trainer.py.

    Overrides the worker actor class that verl spawns across the cluster.
    Works on both supported verl layouts (the worker class above is selected
    at import time).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        # Ray ties an actor's life to its creator: created here, the fleet outlives any one worker.
        self._rls_fleet = fleet_actor()
        self.agent_loop_workers_class = ray.remote(PyInferenceAgentLoopWorker)
        super().__init__(*args, **kwargs)
