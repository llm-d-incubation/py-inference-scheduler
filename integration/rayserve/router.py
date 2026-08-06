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
import os
import random
import uuid
from collections import OrderedDict
from typing import Callable, Sequence

from ray import serve
from ray.actor import ActorHandle
from ray.llm._internal.serve.core.configs.openai_api_models import (  # noqa: PLC2701
    to_model_metadata,
)
from ray.llm._internal.serve.core.ingress.builder import (  # noqa: PLC2701
    LLMServingArgs,
    make_fastapi_ingress,
)
from ray.llm._internal.serve.core.server.builder import build_llm_deployment  # noqa: PLC2701
from ray.serve._private.common import (
    DeploymentHandleSource,
    DeploymentID,
    RunningReplicaInfo,
)
from ray.serve.llm import LLMConfig
from ray.serve.request_router import (
    PendingRequest,
    ReplicaID,
    ReplicaResult,
    RequestRouter,
    RunningReplica,
)

from py_inference_scheduler.core.scheduler import Scheduler
from py_inference_scheduler.datalayer.connectors.mooncake.kv import (
    mooncake_enabled,
    mooncake_engine_kwargs,
    mooncake_env_vars,
)
from py_inference_scheduler.datalayer.rayserve.engine import MetricsAwareLLMServer
from py_inference_scheduler.datalayer.rayserve.relocation import (
    ENABLE_ENV as ENABLE_RELOCATION_ENV,
)
from py_inference_scheduler.datalayer.rayserve.relocation import (
    PULL_OF_HEADER,
    TARGET_HEADER,
    RelocatingIngress,
    relocation_enabled,
)
from py_inference_scheduler.framework import (
    Endpoint,
    FlowControlPlugin,
    LLMRequest,
)


class FlowControlManager:
    def __init__(self, router: IGWRouter) -> None:
        self.router = router
        self.scheduler = router.scheduler
        self.admission_queue: list[asyncio.Future] = []
        self.loop: asyncio.AbstractEventLoop | None = None

    def _get_plugins(self) -> list[FlowControlPlugin]:
        return self.scheduler.get_flow_control_plugins()

    async def admit(
        self,
        request: LLMRequest,
        endpoints: Sequence[Endpoint],
        candidate_replicas: list[RunningReplica],
        pending_request: PendingRequest,
    ) -> tuple[list[RunningReplica], Sequence[Endpoint]]:
        if self.loop is None:
            self.loop = asyncio.get_event_loop()

        plugins = self._get_plugins()
        if not plugins:
            return candidate_replicas, endpoints

        while True:
            allowed_eps: Sequence[Endpoint] = endpoints
            for plugin in plugins:
                allowed_eps = plugin.get_allowed_candidates(request, allowed_eps)

            if allowed_eps:
                allowed_names = {e.name for e in allowed_eps}
                allowed_replicas = [
                    r for r in candidate_replicas if str(r.replica_id) in allowed_names
                ]
                return allowed_replicas, allowed_eps

            fut = self.loop.create_future()
            self.admission_queue.append(fut)
            await fut

            # Refetch stats because we blocked and conditions changed
            endpoints = await self.router.build_endpoints(candidate_replicas, pending_request)

    def commit(self, request: LLMRequest, selected: Endpoint) -> None:
        for plugin in self._get_plugins():
            plugin.reserve(request, selected)

    def release(self, request: LLMRequest, replica_id: str) -> None:
        plugins = self._get_plugins()
        for plugin in plugins:
            plugin.release(request, replica_id)

        if plugins and self.admission_queue:
            waiter = self.admission_queue.pop(0)
            if not waiter.done():
                waiter.set_result(True)

    def attach_learning_callback(
        self,
        request: LLMRequest,
        result: ReplicaResult,
        is_streaming: bool,  # noqa: FBT001
    ) -> None:
        plugins = self._get_plugins()
        if not plugins:
            return

        # we only have one flow control plugin right now
        plugin = plugins[0]
        rollout_request_id, _ = plugin._get_rollout_request_id(request.body)  # type: ignore[attr-defined]

        if not is_streaming:
            original_get_async = result.get_async

            async def patched_get_async():
                response = await original_get_async()
                if response and hasattr(response, "usage") and response.usage:
                    for plugin in plugins:
                        plugin.update_learned_stats(  # type: ignore[attr-defined]
                            rollout_request_id,
                            response.usage.prompt_tokens,
                            response.usage.completion_tokens,
                        )
                return response

            result.get_async = patched_get_async
        else:
            original_anext = result.__anext__

            async def patched_anext():
                chunk = await original_anext()
                if chunk and hasattr(chunk, "usage") and chunk.usage:
                    for plugin in plugins:
                        plugin.update_learned_stats(  # type: ignore[attr-defined]
                            rollout_request_id,
                            chunk.usage.prompt_tokens,
                            chunk.usage.completion_tokens,
                        )
                return chunk
            result.__anext__ = patched_anext


class IGWRouter(RequestRouter):
    def __init__(  # noqa: PLR0913
        self,
        deployment_id: DeploymentID,
        handle_source: DeploymentHandleSource,
        *,
        self_actor_id: str | None = None,
        self_actor_handle: ActorHandle | None = None,
        use_replica_queue_len_cache: bool = False,
        get_curr_time_s: Callable[[], float] | None = None,
        create_replica_wrapper_func: Callable[[RunningReplicaInfo], RunningReplica] | None = None,
        **kwargs: object,
    ) -> None:
        RequestRouter.__init__(
            self,
            deployment_id=deployment_id,
            handle_source=handle_source,
            self_actor_id=self_actor_id,
            self_actor_handle=self_actor_handle,
            use_replica_queue_len_cache=True,
            get_curr_time_s=get_curr_time_s,
            create_replica_wrapper_func=create_replica_wrapper_func,
            **kwargs,
        )
        self.scheduler = Scheduler()
        self.deployment_name = deployment_id.name
        self.fc_manager = FlowControlManager(self)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._relocation_on = relocation_enabled()
        self._request_replica: OrderedDict[str, str] = OrderedDict()
        self._request_replica_cap = int(
            os.environ.get("MOONCAKE_RELOCATION_REGISTRY_CAP", "4096")
        )

    async def _get_routing_stats(
        self, replicas: list[RunningReplica], pending_request: PendingRequest
    ) -> list[dict]:
        """Fetch routing stats (KV usage, etc.) from a list of replicas."""
        futures = [
            r._get_replica_wrapper(pending_request)._actor_handle.record_routing_stats.remote()
            for r in replicas
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)
        return [res if not isinstance(res, BaseException) else {} for res in results]

    async def build_endpoints(
        self, replicas: list[RunningReplica], pending_request: PendingRequest
    ) -> list[Endpoint]:
        metrics_results = await self._get_routing_stats(replicas, pending_request)

        endpoints = []
        for replica, routing_stats in zip(replicas, metrics_results):
            queue_len = 0
            if self.replica_queue_len_cache:
                cached_val = self.replica_queue_len_cache.get(replica.replica_id)
                if cached_val is not None:
                    queue_len = cached_val

            kv_cache_size = routing_stats.get("kv_cache_size", -1)

            endpoints.append(
                Endpoint(
                    name=str(replica.replica_id),
                    attributes={
                        "queue_len": queue_len,
                        "routing_stats": routing_stats,
                        "kv_cache_size": kv_cache_size,
                    },
                )
            )
        return endpoints

    def _fallback_random_choice(
        self, candidate_replicas: list[RunningReplica]
    ) -> list[list[RunningReplica]]:
        """Helper to pick a random replica as fallback."""
        index = random.randint(0, len(candidate_replicas) - 1)  # noqa: S311
        return [[candidate_replicas[index]]]

    def _parse_to_llm_request(self, pending_request: PendingRequest) -> LLMRequest:
        """Converts Ray request to LLMRequest."""
        req_id = pending_request.metadata.request_id if pending_request else str(uuid.uuid4())

        if not pending_request or not pending_request.args:
            print("No pending request or args, defaulting to random choice")
            return LLMRequest(request_id=req_id, body="", target_model=self.deployment_name)
        request_args = pending_request.args[0]

        if hasattr(request_args, "messages"):
            body = request_args.messages
        elif hasattr(request_args, "prompt"):
            body = request_args.prompt
        else:
            body = request_args

        target_model = getattr(request_args, "model", self.deployment_name)

        headers: dict[str, str] = {}
        if len(pending_request.args) > 1 and pending_request.args[1] is not None:
            headers = dict(getattr(pending_request.args[1], "headers", {}) or {})

        return LLMRequest(
            request_id=req_id, body=body, target_model=target_model, headers=headers
        )

    def _apply_relocation_headers(
        self, headers: dict[str, str], candidate_replicas: list[RunningReplica]
    ) -> tuple[RunningReplica | None, list[RunningReplica]]:
        """Returns (target replica or None, remaining candidates)."""
        if not self._relocation_on:
            return None, candidate_replicas
        # A pull must never land back on the replica it was pushed out of
        pushed_from = self._request_replica.get(headers.get(PULL_OF_HEADER, ""))
        if pushed_from and len(candidate_replicas) > 1:
            candidate_replicas = [
                r for r in candidate_replicas if str(r.replica_id) != pushed_from
            ]
        target_name = headers.get(TARGET_HEADER)
        if target_name:
            for replica in candidate_replicas:
                if str(replica.replica_id) == target_name:
                    return replica, candidate_replicas
        return None, candidate_replicas

    async def choose_replicas(  # noqa: PLR0911, PLR0912
        self,
        candidate_replicas: list[RunningReplica],
        pending_request: PendingRequest | None = None,
    ) -> list[list[RunningReplica]]:
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        if not candidate_replicas:
            return []

        llm_req = self._parse_to_llm_request(pending_request)

        # for health check empty requests
        if not pending_request or not pending_request.args or not llm_req.body:
            return self._fallback_random_choice(candidate_replicas)

        target_replica, candidate_replicas = self._apply_relocation_headers(
            llm_req.headers or {}, candidate_replicas
        )
        if target_replica:
            return [[target_replica]]

        try:
            endpoints: Sequence[Endpoint] = await self.build_endpoints(
                candidate_replicas, pending_request
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[ROUTER ERROR] Failed to build endpoints: {e!r}. Falling back to random choice."
            )
            return self._fallback_random_choice(candidate_replicas)

        try:
            if self.scheduler.has_flow_control():
                candidate_replicas, endpoints = await self.fc_manager.admit(
                    llm_req, endpoints, candidate_replicas, pending_request
                )
        except Exception as e:  # noqa: BLE001
            print(
                f"[ROUTER ERROR] Flow control admission failed: {e!r}. "
                f"Falling back to random choice."
            )
            return self._fallback_random_choice(candidate_replicas)

        try:
            selected_endpoints = self.scheduler.run(llm_req, endpoints)
            index = -1
            for i, replica in enumerate(candidate_replicas):
                if (
                    len(selected_endpoints) > 0
                    and str(replica.replica_id)
                    == selected_endpoints[0].endpoint.name
                ):
                    index = i
                    break
            if index == -1:
                index = random.randint(0, len(candidate_replicas) - 1)  # noqa: S311
            target_replica = candidate_replicas[index]
        except Exception as e:  # noqa: BLE001
            print(
                f"[ROUTER ERROR] Scheduling or mapping failed: {e!r}. "
                f"Falling back to random choice."
            )
            return self._fallback_random_choice(candidate_replicas)

        try:
            if self.scheduler.has_flow_control() and selected_endpoints:
                self.fc_manager.commit(llm_req, selected_endpoints[0].endpoint)
        except Exception as e:  # noqa: BLE001
            print(f"[ROUTER WARNING] Failed to commit reservation: {e!r}")

        return [[target_replica]]

    def on_request_routed(
        self,
        pending_request: PendingRequest,
        replica_id: ReplicaID,
        result: ReplicaResult,
    ) -> None:
        llm_req = self._parse_to_llm_request(pending_request)
        is_streaming = pending_request.metadata.is_streaming

        # Key=x-request-id since push and pull share the same Serve request_id
        if self._relocation_on:
            ingress_request_id = (llm_req.headers or {}).get("x-request-id")
            if ingress_request_id:
                self._request_replica[ingress_request_id] = str(replica_id)
                if len(self._request_replica) > self._request_replica_cap:
                    self._request_replica.popitem(last=False)

        if self.scheduler.has_flow_control():
            result.add_done_callback(lambda _: self.fc_manager.release(llm_req, str(replica_id)))
            self.fc_manager.attach_learning_callback(llm_req, result, is_streaming)


# Hooking into Ray Serve's Request Router

engine_kwargs: dict[str, object] = {
    "enable_prefix_caching": True,
    "tensor_parallel_size": 2,
}
env_vars: dict[str, str] = {
    "NCCL_NET_PLUGIN": "/usr/local/gib/lib64/libnccl-net_internal.so",
    "NCCL_CROSS_NIC": "0",
    "NCCL_NET_GDR_LEVEL": "PIX",
    "NCCL_P2P_NET_CHUNKSIZE": "131072",
    "NCCL_NVLS_CHUNKSIZE": "524288",
    "NCCL_IB_ADAPTIVE_ROUTING": "1",
    "NCCL_IB_QPS_PER_CONNECTION": "4",
    "NCCL_IB_TC": "52",
    "NCCL_IB_FIFO_TC": "84",
    "NCCL_TUNER_CONFIG_PATH": "/usr/local/gib/configs/tuner_config_a3u.txtpb",
}

# Opt-in Mooncake shared KV tier (set ENABLE_MOONCAKE_KV=1). Off by default so
# the normal deployment is unchanged; when on, every replica pushes/pulls KV to a
# cluster-wide Mooncake Store and saves decode KV via DecodeKVSavingConnector.
if mooncake_enabled():
    engine_kwargs.update(mooncake_engine_kwargs())
    env_vars.update(mooncake_env_vars())

# Pass through to engine side.
if ENABLE_RELOCATION_ENV in os.environ:
    env_vars[ENABLE_RELOCATION_ENV] = os.environ[ENABLE_RELOCATION_ENV]

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "qwen-32b",
        "model_source": "Qwen/Qwen2.5-32B-Instruct",
    },
    engine_kwargs=engine_kwargs,
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 1,
        },
        "request_router_config": {
            # Note our custom IGWRouter here
            "request_router_class": IGWRouter,
        },
        "ray_actor_options": {"num_cpus": 1},
    },
    runtime_env={"env_vars": env_vars},
)


def build_custom_openai_app(builder_config: dict[str, object]) -> object:
    # Same internal logic as build_openai_app, but we map our deployment_cls
    builder_args = LLMServingArgs.model_validate(builder_config)
    llm_configs = builder_args.llm_configs

    llm_deployments = {
        c.model_id: build_llm_deployment(c, deployment_cls=MetricsAwareLLMServer)
        for c in llm_configs
    }
    model_cards = {c.model_id: to_model_metadata(c.model_id, c) for c in llm_configs}
    lora_paths = {
        c.model_id: c.lora_config.dynamic_lora_loading_path
        for c in llm_configs
        if c.lora_config is not None
    }

    ingress_cls_config = builder_args.ingress_cls_config
    base_ingress_cls = (
        RelocatingIngress if relocation_enabled() else ingress_cls_config.ingress_cls
    )
    ingress_options = base_ingress_cls.get_deployment_options(llm_configs)
    ingress_cls = make_fastapi_ingress(base_ingress_cls)

    return serve.deployment(
        ingress_cls,
        **ingress_options,
    ).bind(
        llm_deployments=llm_deployments,
        model_cards=model_cards,
        lora_paths=lora_paths,
        **ingress_cls_config.ingress_extra_kwargs,
    )


app = build_custom_openai_app({
    "llm_configs": [llm_config],
})

if __name__ == "__main__":
    serve.run(app, blocking=True)
