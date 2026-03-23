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


import asyncio
import logging
import ray
import os
import uuid
import yaml
from typing import Dict, List, Any
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AsyncLLMServerManager,
    AgentLoopWorker,
    AgentLoopManager
)

from scheduling.core.scheduler import Scheduler
from scheduling.core.config import SchedulerConfig
from scheduling.framework import LLMRequest, Endpoint, SchedulerProfile
from datalayer.verl.metrics import verl_metrics_polling_loop
from datalayer.verl.datastore import InflightStore
from scheduling.plugins import (
    SimpleFilter,
    WaitingQueueScorer,
    MaxScorePicker,
    SingleProfileHandler
)
from scheduling import Scheduler

logger = logging.getLogger(__name__)

class InferenceSchedulerServerManager(AsyncLLMServerManager):
    """
    Subclass of verl's AsyncLLMServerManager that delegates routing
    to the native py-inference-scheduler engine.
    """
    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], *args, **kwargs):
        super().__init__(config, server_handles, *args, **kwargs)
        self.ray_request_scheduler = Scheduler()
        self.inflight_store = InflightStore()
        self.endpoints = []

        for i, rep_handle in enumerate(self.server_handles):
            ep = Endpoint(
                name=f"verl-worker-{i}",
                attributes={
                    "replica_obj": rep_handle,
                    "routing_stats": {}
                }
            )
            self.endpoints.append(ep)

        self._metrics_task = None


    async def generate(self, request_id: str, **kwargs):
        """Overrides Verl's native generate to forcefully yield to the metrics poller"""

        # Yield CPU to check for metrics poller
        await asyncio.sleep(0)

        prompt_ids = kwargs.get("prompt_ids", None)
        winning_endpoint = self._choose_server(request_id, prompt_ids=prompt_ids)

        if isinstance(winning_endpoint, Endpoint):
            server = winning_endpoint.attributes["replica_obj"]
            endpoint_name = winning_endpoint.name
            self.inflight_store.increment(endpoint_name)
            stats = winning_endpoint.attributes.setdefault("routing_stats", {})
            stats["num_running_reqs"] = max(stats.get("num_running_reqs", 0), self.inflight_store.get(endpoint_name))
            stats["queue_len"] = stats.get("num_waiting_reqs", 0) + stats["num_running_reqs"]
        else:
            server = winning_endpoint
            endpoint_name = None

        # vLLM requires a fresh request_id per generation to prevent KV cache collisions.
        kwargs["request_id"] = uuid.uuid4().hex

        # vLLMAsyncServer ignores ignore_eos from config, so we must pass it explicitly.
        sampling_params = kwargs.get("sampling_params", {})
        if isinstance(sampling_params, dict):
            sampling_params["ignore_eos"] = True
            kwargs["sampling_params"] = sampling_params
        elif hasattr(sampling_params, "ignore_eos"):
             setattr(sampling_params, "ignore_eos", True)

        try:
            return await server.generate.remote(**kwargs)
        finally:
            if endpoint_name:
                self.inflight_store.decrement(endpoint_name)

    def _choose_server(self, request_id: str, prompt_ids: List[int] = None) -> ray.actor.ActorHandle:
        """Overrides Verl's Native LRU Router"""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(verl_metrics_polling_loop(self.endpoints, self.inflight_store))

        req = LLMRequest(request_id=request_id, body=prompt_ids)
        selected_endpoints = self.ray_request_scheduler.run(req, candidates=self.endpoints)

        if not selected_endpoints:
            logger.warning("py-inference-scheduler returned no endpoints, falling back to basic LRU.")
            return super()._choose_server(request_id)

        winning_endpoint: Endpoint = selected_endpoints[0].endpoint
        stats = winning_endpoint.attributes.get('routing_stats', {})
        logger.info(f"[{request_id[:6]}] Routed to {winning_endpoint.name} (stats: {stats})")
        return winning_endpoint


class PyInferenceAgentLoopWorker(AgentLoopWorker):
    """
    Overrides the Ray worker actor to inject the custom ServerManager
    before calling super().__init__
    """
    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], *args, **kwargs):
        self.server_manager = InferenceSchedulerServerManager(config, server_handles)
        super().__init__(config, server_handles, *args, **kwargs)


class PyInferenceAgentLoopManager(AgentLoopManager):
    """
    The main hook entrypoint loaded by ray_trainer.py
    Overrides the worker actor class that verl spawns across the cluster.
    """
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(PyInferenceAgentLoopWorker)
        super().__init__(*args, **kwargs)
