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

import logging
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import ReqMeta
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
        KVCacheStoreSendingThread,
    )

logger = logging.getLogger(__name__)

# Times a store job whose put failed is tried again before the scheduler gets its blocks back.
SAVE_RETRIES = 2


class MooncakeStoreConnector(UpstreamMooncakeStoreConnector):
    """vLLM's Mooncake store connector with three fixes it needs on vLLM 0.29.

    - Engines on the same host each get their own lookup socket.
    - Store jobs from a step with no forward pass still run, so their KV blocks are freed.
    - A store job whose put fails is retried before its KV blocks are released.
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

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        super().register_kv_caches(kv_caches)
        # Only ranks that save KV have a send thread.
        send_thread = self.connector_worker.kv_send_thread if self.connector_worker else None
        if send_thread is not None:
            retry = _FailedSaveRetry(send_thread)
            send_thread._record_operation_cb = retry.record_operation
            send_thread.finish_store_job = retry.finish_store_job  # type: ignore[method-assign]

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


class _FailedSaveRetry:
    """Holds back the done report of a store job whose put failed, and queues the job again.

    Upstream reports every store job done, failed or not, and the scheduler then drops the
    job's GPU block references. A rank saves a failed range again only on the request's next
    store job, so when the failed job is the request's last one (the end of prefill with
    save_decode_cache off), the finished turn's only KV copy becomes evictable. While the
    report is held back, the scheduler keeps the blocks referenced. After SAVE_RETRIES the
    job is reported anyway: held blocks can't be evicted, and they keep the engine stepping.
    """

    def __init__(self, send_thread: KVCacheStoreSendingThread) -> None:
        self._send_thread = send_thread
        self._upstream_record_operation = send_thread._record_operation_cb
        self._upstream_finish_store_job = send_thread.finish_store_job
        # Only the send thread itself uses these.
        self._job_failed = False
        self._retries: dict[int | None, int] = {}  # store_job_id -> retries queued

    def record_operation(self, *, operation: str, status: str, **stats: object) -> None:
        # The thread runs one job at a time, so a failed store call belongs to the job in hand.
        if operation in {"save_exists", "save_put"} and status != "ok":
            self._job_failed = True
        if self._upstream_record_operation is not None:
            self._upstream_record_operation(operation=operation, status=status, **stats)

    def finish_store_job(self, req_meta: ReqMeta) -> None:
        failed, self._job_failed = self._job_failed, False
        retries = self._retries.pop(req_meta.store_job_id, 0)
        if failed and self._is_newest_live_job(req_meta):
            if retries < SAVE_RETRIES:
                self._retries[req_meta.store_job_id] = retries + 1
                logger.info(
                    "Retrying Mooncake store job %s for request %s (%d of %d)",
                    req_meta.store_job_id,
                    req_meta.req_id,
                    retries + 1,
                    SAVE_RETRIES,
                )
                self._send_thread.request_queue.put(req_meta)
                return
            logger.warning(
                "Mooncake store job %s for request %s failed %d times; releasing its KV blocks",
                req_meta.store_job_id,
                req_meta.req_id,
                retries + 1,
            )
        self._upstream_finish_store_job(req_meta)

    def _is_newest_live_job(self, req_meta: ReqMeta) -> bool:
        # A newer job of the request resumes from the last saved offset, so it saves this
        # range too, and it references the same blocks. A preempted request's jobs are no
        # longer live, and the request saves again from the start when it resumes.
        job_id: int | None = req_meta.store_job_id
        with self._send_thread.done_task_lock:
            live_jobs: set[int] = self._send_thread.stored_requests.get(req_meta.req_id, set())
            return max(live_jobs, default=None) == job_id
