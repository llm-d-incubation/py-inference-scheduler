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

import asyncio
import logging
import uuid

import ray
from omegaconf import DictConfig  # type: ignore[import-not-found]
from verl.experimental.agent_loop.agent_loop import (  # type: ignore[import-not-found]
    AgentLoopManager,
    AgentLoopWorker,
    AsyncLLMServerManager,
)

from backends.verl.sglang import SglangEnginePatch
from backends.verl.vllm import VllmEnginePatch
from datalayer.metrics.verl.datastore import InflightStore
from datalayer.metrics.verl.fetch_metrics import fetch_worker_metrics
from scheduling import Scheduler
from scheduling.framework import Endpoint, LLMRequest

logger = logging.getLogger(__name__)

# Must apply at module level to patch classes before use across distributed
# Ray workers without modifying verl.
VllmEnginePatch.apply()
SglangEnginePatch.apply()


class InferenceSchedulerServerManager(AsyncLLMServerManager):
    """
    Delegate routing to the native py-inference-scheduler engine.

    Compatible with verl v0.7.1.
    """
    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(config, servers, load_balancer_handle, *args, **kwargs)
        # Extract rollout config to support configurable ignore_eos
        if config.get("actor_rollout_ref"):
            self.rollout_config = config.actor_rollout_ref.rollout
        else:
            self.rollout_config = config.rollout

        self.ray_request_scheduler = Scheduler()
        self.inflight_store = InflightStore()
        self.endpoints = []
        self._lb_acquired_requests = set()  # type: ignore[var-annotated]
        self._scheduling_lock = asyncio.Lock()

        # Reconstruct endpoints from the new (id, handle) tuple structure
        for server_id, handle in servers:
            ep = Endpoint(
                name=server_id,
                attributes={
                    "replica_obj": handle,
                    "routing_stats": {},
                },
            )
            self.endpoints.append(ep)

        self._metrics_task = None

    async def _acquire_server(
        self,
        request_id: str,
        prompt_ids: list[int] | None = None,
    ) -> tuple[str, ray.actor.ActorHandle]:
        """Overrides Verl's Native Global Load Balancer with py-inference-scheduler logic."""
        # We use the lock because of python:
        # -- vERL composes the batch of requests before we get to schedule any tasks.
        # -- we cannot interleave metric tasks in between of scheduled tasks
        # -- due to python being FIFO.
        # -- so we just make it part of the scheduling task instead of
        # -- having an independent metric poller task.
        async with self._scheduling_lock:
            tasks = [fetch_worker_metrics(ep, self.inflight_store) for ep in self.endpoints]
            await asyncio.gather(*tasks)

            for ep in self.endpoints:
                ep.attributes["queue_len"] = self.inflight_store.get(ep.name)

            req = LLMRequest(request_id=request_id, body=prompt_ids)
            selected_endpoints = self.ray_request_scheduler.run(
                req, candidates=self.endpoints
            )

            if selected_endpoints:
                winning_endpoint: Endpoint = selected_endpoints[0].endpoint
                self.inflight_store.increment(winning_endpoint.name)

        if not selected_endpoints:
            logger.warning(
                "py-inference-scheduler returned no endpoints, falling back to verl global LB."
            )
            self._lb_acquired_requests.add(request_id)
            server_id, handle = await super()._acquire_server(request_id)  # type: ignore[no-any-return]
            self.inflight_store.increment(server_id)
            return server_id, handle

        return winning_endpoint.name, winning_endpoint.attributes["replica_obj"]

    def _release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Decrements local inflight tracking and notifies global LB if it originated there."""
        self.inflight_store.decrement(server_id)
        if request_id and request_id in self._lb_acquired_requests:
            super()._release_server(server_id)
            self._lb_acquired_requests.remove(request_id)

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, object],
        image_data: list[object] | None = None,
        video_data: list[object] | None = None,
    ) -> object:
        """Overrides Verl's generate to manage lifecycle with scheduler."""
        # Yield CPU to check for metrics poller
        await asyncio.sleep(0)

        server_id, server = await self._acquire_server(request_id, prompt_ids=prompt_ids)

        # vLLM needs a fresh request_id per generation to avoid KV cache collisions.
        # verl has sticky request_ids for multi-turn rollouts.
        vllm_request_id = uuid.uuid4().hex

        # vLLMAsyncServer ignores ignore_eos from config, so we must pass it explicitly.
        ignore_eos = self.rollout_config.get("ignore_eos", False)
        if isinstance(sampling_params, dict):
            sampling_params["ignore_eos"] = ignore_eos
        elif hasattr(sampling_params, "ignore_eos"):
            sampling_params.ignore_eos = ignore_eos

        try:
            return await server.generate.remote(
                request_id=vllm_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
        finally:
            self._release_server(server_id, request_id)


class PyInferenceAgentLoopWorker(AgentLoopWorker):
    """
    Inject the custom ServerManager before calling super().__init__.

    Compatible with verl v0.7.1.
    """
    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
    ) -> None:
        # Inject our manager
        self.server_manager = InferenceSchedulerServerManager(config, servers, load_balancer_handle)
        super().__init__(config, servers, load_balancer_handle, reward_loop_worker_handles)


class PyInferenceAgentLoopManager(AgentLoopManager):
    """
    Main hook entrypoint loaded by ray_trainer.py.

    Overrides the worker actor class that verl spawns across the cluster.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.agent_loop_workers_class = ray.remote(PyInferenceAgentLoopWorker)
        super().__init__(*args, **kwargs)
