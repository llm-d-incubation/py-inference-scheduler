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
  actor owns the server registry and does atomic acquire. The scheduler client
  bootstraps its endpoint set by draining the balancer once at first use
  (acquire every server with unique request ids, record the handles, release).

Both layouts expose the same entrypoint for the trainer flag:
``+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager``
"""

from __future__ import annotations

import asyncio
import logging
import uuid

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
from py_inference_scheduler import Scheduler
from py_inference_scheduler.datalayer.metrics.datastore import InflightStore
from py_inference_scheduler.datalayer.metrics.verl.fetch_metrics import fetch_worker_metrics
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
    """Layout-independent scheduling state: engine, inflight tracking, metrics."""

    def __init__(self) -> None:
        self.scheduler = Scheduler()
        self.inflight_store = InflightStore()
        self.endpoints: list[Endpoint] = []
        self.lb_acquired_requests: set[str] = set()
        self.lock = asyncio.Lock()

    async def schedule(self, request_id: str, prompt_ids: list[int] | None) -> Endpoint | None:
        """Refresh metrics and pick an endpoint; None means fall back to verl's LB.

        The lock makes metric refresh part of the scheduling task itself:
        verl composes the whole batch before any task runs, so an independent
        poller task would never be interleaved by the FIFO event loop.
        """
        async with self.lock:
            await asyncio.gather(
                *(fetch_worker_metrics(ep, self.inflight_store) for ep in self.endpoints)
            )
            for ep in self.endpoints:
                ep.attributes["queue_len"] = self.inflight_store.get(ep.name)

            request = LLMRequest(request_id=request_id, body=prompt_ids)
            selected = self.scheduler.run(request, candidates=self.endpoints)
            if not selected:
                return None
            winner: Endpoint = selected[0].endpoint
            self.inflight_store.increment(winner.name)
            return winner


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
            self.core.endpoints = [
                Endpoint(name=server_id, attributes={"replica_obj": handle, "routing_stats": {}})
                for server_id, handle in servers
            ]

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
                self.core.inflight_store.increment(server_id)
                return server_id, handle
            return winner.name, winner.attributes["replica_obj"]

        def _release_server(self, server_id: str, request_id: str | None = None) -> None:
            self.core.inflight_store.decrement(server_id)
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

        The GlobalRequestLoadBalancer actor owns the (server_id -> handle)
        registry but exposes no enumeration API, so the endpoint set is
        bootstrapped once by draining it: with all inflight counters equal,
        consecutive acquires with unique request ids visit every server.
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

        async def _ensure_endpoints(self) -> None:
            if self.core.endpoints:
                return
            server_ids = await self._load_balancer.get_all_servers.remote()
            handles: dict[str, ray.actor.ActorHandle] = {}
            acquired: list[str] = []
            for _ in range(max(1, len(server_ids)) * 3):
                server_id, handle = await self._load_balancer.acquire_server.remote(
                    request_id=f"pyis-bootstrap-{uuid.uuid4().hex}"
                )
                acquired.append(server_id)
                handles[server_id] = handle
                if len(handles) >= len(server_ids):
                    break
            for server_id in acquired:
                self._load_balancer.release_server.remote(server_id=server_id)
            self.core.endpoints = [
                Endpoint(name=server_id, attributes={"replica_obj": handle, "routing_stats": {}})
                for server_id, handle in handles.items()
            ]
            logger.info(
                "py-inference-scheduler bootstrapped %d endpoints from global LB", len(handles)
            )

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
                self.core.inflight_store.increment(server_id)
                return server_id, handle
            return winner.name, winner.attributes["replica_obj"]

        def _release_server(self, server_id: str, request_id: str | None = None) -> None:
            self.core.inflight_store.decrement(server_id)
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
            llm_client: object,
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
        self.agent_loop_workers_class = ray.remote(PyInferenceAgentLoopWorker)
        super().__init__(*args, **kwargs)
