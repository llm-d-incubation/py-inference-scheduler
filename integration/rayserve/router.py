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


from scheduling.framework import (
    ScorerPlugin, 
    PickerPlugin, 
    SchedulerProfile, 
    WeightedScorer,
    Endpoint, 
    ScoredEndpoint, 
    CycleState, 
    LLMRequest
)
from scheduling.plugins import (
    SingleProfileHandler, 
    LeastQueueScorer, 
    WaitingQueueScorer, 
    RunningQueueScorer, 
    KVCacheScorer
)
from scheduling.core.config import SchedulerConfig
from scheduling import PrefixCacheScorer, RoundRobinScorer
from scheduling.plugins.scorers.prefix_plugin import _hash_prompt_bytes, _get_user_input_bytes
from scheduling.core.scheduler import Scheduler
from typing import Dict, Optional, Sequence
import asyncio
import random
import time
import os
import yaml

from ray.serve.request_router import (
    PendingRequest,
    RequestRouter,
    ReplicaID,
    ReplicaResult,
    RunningReplica,
)
from ray.serve.exceptions import BackPressureError
from ray.actor import ActorHandle
from ray.serve._private.common import (
    DeploymentHandleSource,
    DeploymentID,
    ReplicaID,
    ReplicaQueueLengthInfo,
    RequestMetadata,
    RunningReplicaInfo,
)
from typing import (
    List,
    Optional,
    Callable
)

class IGWRouter(RequestRouter):
    def __init__(
        self,
        deployment_id: DeploymentID,
        handle_source: DeploymentHandleSource,
        self_actor_id: Optional[str] = None,
        self_actor_handle: Optional[ActorHandle] = None,
        use_replica_queue_len_cache: bool = False,
        get_curr_time_s: Optional[Callable[[], float]] = None,
        create_replica_wrapper_func: Optional[
            Callable[[RunningReplicaInfo], RunningReplica]
        ] = None,
        *args,
        **kwargs,
    ):
        RequestRouter.__init__(
            self,
            deployment_id=deployment_id,
            handle_source=handle_source,
            self_actor_id=self_actor_id,
            self_actor_handle=self_actor_handle,
            use_replica_queue_len_cache=True,
            get_curr_time_s=get_curr_time_s,
            create_replica_wrapper_func=create_replica_wrapper_func,
            *args,
            **kwargs,
        )
        # Initialize the RayRequestScheduler
        self.scheduler = Scheduler()

    async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:    
        self.scheduler._maybe_reload_config()

        if not pending_request or not pending_request.args:
            print("No pending request or args, defaulting to random choice")
            index = random.randint(0, len(candidate_replicas) - 1)
            final = time.time()
            return [[candidate_replicas[index]]]

        try:
            print("Using scheduling library to route request")
            print("Pending request:", pending_request)
            initial = time.time()
            
            request_args = pending_request.args[0]
            if hasattr(request_args, "messages"):
                body = request_args.messages
            elif hasattr(request_args, "prompt"):
                body = request_args.prompt
            else:
                body = request_args

            igw_req = LLMRequest(request_id="1", body=body, target_model="qwen-32b")
            print("IGW Request body:", igw_req.body)
            
            metrics_futures = [
                replica._get_replica_wrapper(pending_request)._actor_handle.record_routing_stats.remote() 
                for replica in candidate_replicas
            ]

            metrics_results = await asyncio.gather(*metrics_futures, return_exceptions=True)

            candidates = []
            for replica, routing_stats in zip(candidate_replicas, metrics_results):
                queue_len = 0
                rid = replica.replica_id
                if self.replica_queue_len_cache:
                    cached_val = self.replica_queue_len_cache.get(rid)
                    if cached_val is not None:
                        queue_len = cached_val
                
                if isinstance(routing_stats, Exception):
                    print(f"Failed to fetch metrics via RPC for {rid}: {routing_stats}")
                    routing_stats = {}

                candidates.append(
                    Endpoint(
                        name=str(replica.replica_id),
                        attributes={
                            "queue_len": queue_len,
                            "routing_stats": routing_stats,
                        },
                    )
                )

            selected_endpoints = self.scheduler.run(igw_req, candidates)
            
            if len(selected_endpoints) > 0:
                print(f"Routed to endpoint: {selected_endpoints[0].endpoint.name}\n")
            else:
                print("No endpoint selected by scheduler\n")
            index = -1
            for i, replica in enumerate(candidate_replicas):
                if len(selected_endpoints) > 0 and str(replica.replica_id) == selected_endpoints[0].endpoint.name:
                    index = i
                    break
            if index == -1:
                index = random.randint(0, len(candidate_replicas) - 1)
            final = time.time()
            print(f"Scheduling took {final - initial} seconds")
            return [[candidate_replicas[index]]]
        except Exception as e:
            print(f"Error of: {repr(e)} during scheduling: {e}, defaulting to random choice")
            index = random.randint(0, len(candidate_replicas) - 1)
            return [[candidate_replicas[index]]]

    def on_request_routed(
        self,
        pending_request: PendingRequest,
        replica_id: ReplicaID,
        result: ReplicaResult,
    ):
        # Not currently used, but could be hooked into for the PreRequest hook. 
        # But intentionally keeping the py-scheduler framework isolated from Ray Serve
        print("on_request_routed callback is called")

#defining an IGW-like scheduler
from scheduling import Scheduler

from datalayer.rayserve.engine import MetricsAwareLLMServer
from ray.llm._internal.serve.core.ingress.builder import LLMServingArgs, make_fastapi_ingress
from ray.llm._internal.serve.core.server.builder import build_llm_deployment

def build_custom_openai_app(builder_config: dict):
    # Same internal logic as build_openai_app, but we map our deployment_cls
    builder_config = LLMServingArgs.model_validate(builder_config)
    llm_configs = builder_config.llm_configs
    
    llm_deployments = [
        build_llm_deployment(c, deployment_cls=MetricsAwareLLMServer) 
        for c in llm_configs
    ]
    
    ingress_cls_config = builder_config.ingress_cls_config
    ingress_options = ingress_cls_config.ingress_cls.get_deployment_options(llm_configs)
    ingress_cls = make_fastapi_ingress(ingress_cls_config.ingress_cls)
    return serve.deployment(ingress_cls, **ingress_options).bind(
        llm_deployments=llm_deployments, **ingress_cls_config.ingress_extra_kwargs
    )

# Hooking into Ray Serve's Request Router

from ray import serve
from ray.serve.llm.request_router import PrefixCacheAffinityRouter
from ray.serve.llm import LLMConfig

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-32b",
        model_source="Qwen/Qwen2.5-32B-Instruct",
    ),
    engine_kwargs=dict(
        enable_prefix_caching=True,
    ),
    deployment_config=dict(
        # max_ongoing_requests=10,
        autoscaling_config=dict(
            min_replicas=6, max_replicas=6,
        ),
        request_router_config=dict(
            # Note our custom IGWRouter here
            request_router_class=IGWRouter,
        ),
        ray_actor_options=dict(num_cpus=1),
    ),
)

app = build_custom_openai_app({
    "llm_configs": [llm_config],
})

if __name__ == "__main__":
    serve.run(app, blocking=True)