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

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector as UpstreamMooncakeStoreConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig


class MooncakeStoreConnector(UpstreamMooncakeStoreConnector):
    """vLLM's Mooncake store connector with two fixes it needs on vLLM 0.29.

    - Engines on the same host each get their own lookup socket.
    - Store jobs from a step with no forward pass still run, so their KV blocks are freed.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        # Upstream names the lookup socket after the host and DP rank, so engines on one host
        # collide on it. The engine's instance id is unique.
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        extra.setdefault("lookup_rpc_port", f"i{vllm_config.instance_id}")
        super().__init__(vllm_config, role, kv_cache_config)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        # The worker side reads this in start_load_kv.
        meta.rls_no_forward_step = scheduler_output.total_num_scheduled_tokens == 0  # type: ignore[attr-defined]
        return meta

    def start_load_kv(self, forward_context: object, **kwargs: object) -> None:  # type: ignore[override]
        super().start_load_kv(forward_context, **kwargs)
        # vLLM skips wait_for_save on a step with no forward pass, but upstream queues store jobs
        # only there, so their blocks would stay pinned. Queue them on exactly those steps.
        if getattr(self._get_connector_metadata(), "rls_no_forward_step", False):
            super().wait_for_save()
