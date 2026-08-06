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
    MooncakeStoreConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig


def should_save_decode_request(
    *,
    is_resumed: bool,
    num_computed_tokens: int,
    prefill_end_tokens: int,
) -> bool:
    """True only for decode since vllm already saves prefill."""
    # is_resumed means vllm's own preemption-resume step (bulk block realloc,
    # skipped for one step), not a relocated request - those save normally.
    if is_resumed:
        return False
    return num_computed_tokens >= prefill_end_tokens


class DecodeKVSavingConnector(MooncakeStoreConnector):
    """
    vllm only saves prompt KV to the store; this also saves decode KV.

    So other replicas can reuse generated tokens. Enabled by save_decode_kv.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        # extra_config values arrive as JSON bools or strings.
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = (
            kv_transfer_config.kv_connector_extra_config if kv_transfer_config else {}
        )
        value = extra_config.get("save_decode_kv", False)
        self.save_decode_kv = value is True or str(value).lower() == "true"

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        if not isinstance(meta, MooncakeStoreConnectorMetadata):
            return meta  # upstream returned a different metadata type
        if not self.save_decode_kv or self.kv_role == "kv_consumer":
            return meta

        sched = self.connector_scheduler
        if sched is None:  # set for the scheduler role only
            return meta
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            new_block_ids = cached.new_block_ids[i]
            if not new_block_ids:
                continue
            tracker = sched._request_trackers.get(req_id)
            if tracker is None:
                continue
            if not should_save_decode_request(
                is_resumed=req_id in cached.resumed_req_ids,
                num_computed_tokens=cached.num_computed_tokens[i],
                prefill_end_tokens=tracker.prefill_end_tokens,
            ):
                continue
            req_tuple = sched._unfinished_requests.get(req_id)
            if not req_tuple:
                continue
            unfinished_req = req_tuple[0]

            # The stock scheduler only advances tracker.token_len on
            # block-allocating steps, so in decode it lags the real token count;
            true_token_len = (
                cached.num_computed_tokens[i]
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            tracker.token_len = max(tracker.token_len, true_token_len)

            tracker.update(new_block_ids)

            # returns None until a new full block has completed.
            req_meta = ReqMeta.from_request_tracker(
                tracker,
                sched._block_size,
                load_spec=None,
                skip_save=False,
                block_hashes=unfinished_req.block_hashes,
                is_last_chunk=False,
                original_block_size=sched.original_block_size,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
        return meta
