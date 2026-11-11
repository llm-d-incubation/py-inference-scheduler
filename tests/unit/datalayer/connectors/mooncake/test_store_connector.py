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

import threading
from types import SimpleNamespace

import pytest

# vLLM is an engine-runtime dependency required by these tests.
pytest.importorskip("vllm")

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreSendingThread,
    MooncakeStoreCoordinator,
    MooncakeStoreWorker,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import KVConnectorBlockState
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput

from py_inference_scheduler.datalayer.connectors.mooncake.store_connector import (
    MooncakeStoreConnector,
)

BLOCK_SIZE = 16
BLOCK_BYTES = 256
REQ_ID = "turn-1"
PROMPT_LEN = 2 * BLOCK_SIZE

# Per-key statuses batch_put_from_multi_buffers returns: bytes written, or a
# negative Mooncake ErrorCode. -800 is TRANSFER_FAIL (mooncake-store/include/types.h).
PUT_OK = BLOCK_BYTES
TRANSFER_FAIL = -800


class FakeMooncakeStore:
    """Stands in for MooncakeDistributedStore: no key exists yet, every put gets `put_status`."""

    def __init__(self, put_status: int) -> None:
        self.put_status = put_status
        self.put_keys: list[str] = []

    def batch_is_exist(self, keys):
        return [0] * len(keys)

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, replicate_config):
        self.put_keys.extend(keys)
        return [self.put_status] * len(keys)


def _scheduler_connector(pool: BlockPool) -> MooncakeStoreConnector:
    # Scheduler half with the attributes build_connector_meta and
    # update_connector_output read on vLLM 0.29; the lookup client is never used.
    sched = object.__new__(MooncakeStoreScheduler)
    sched.kv_role = "kv_both"
    sched.save_decode_cache = False  # what the overscheduling guide sets
    sched.enable_kv_events = False
    sched._block_size = BLOCK_SIZE
    sched._hash_block_size = BLOCK_SIZE
    sched.kv_cache_config = SimpleNamespace(select_transfer_block_ids=tuple)
    sched.load_specs = {}
    sched._request_trackers = {}
    sched._unfinished_requests = {}
    sched._unfinished_request_ids = set()
    sched._gpu_block_pool = pool
    sched._num_workers = 1
    sched._next_store_job_id = 0
    sched._pinned_saves = {}
    sched._boundary_state_group_ids = frozenset()

    conn = object.__new__(MooncakeStoreConnector)
    conn.connector_scheduler = sched
    conn._kv_cache_events = None
    return conn


def _worker_connector(store: FakeMooncakeStore) -> MooncakeStoreConnector:
    # Worker half: the real send thread over one full-attention group, run inline.
    spec = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=8, head_size=64, dtype=None)
    coord = MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
    )
    db = ChunkedTokenDatabase(KeyMetadata("test-model", 0, 0, 0, 0), block_size=BLOCK_SIZE)
    db.set_kv_caches_base_addr([0x1000])
    db.set_block_len([BLOCK_BYTES])
    send_thread = KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=[db],
        block_size=BLOCK_SIZE,
        tp_rank=0,
        group_put_steps=[1],
        kv_role="kv_both",
        ready_event=threading.Event(),
    )
    worker = object.__new__(MooncakeStoreWorker)
    worker.kv_send_thread = send_thread

    conn = object.__new__(MooncakeStoreConnector)
    conn.connector_worker = worker
    return conn


def _prefill_step(block_ids: list[int]) -> SimpleNamespace:
    """Scheduler output for the step that prefills the turn's whole prompt."""
    new_req = SimpleNamespace(
        req_id=REQ_ID,
        num_computed_tokens=0,
        prompt_token_ids=list(range(PROMPT_LEN)),
        prefill_token_ids=None,
        block_ids=(block_ids,),
    )
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={REQ_ID: PROMPT_LEN},
        total_num_scheduled_tokens=PROMPT_LEN,
        kv_connector_block_state=KVConnectorBlockState(
            block_ids={REQ_ID: (block_ids,)}, boundary_state_offloads={}
        ),
    )


def _run_turn(put_status: int) -> tuple[list[int], FakeMooncakeStore]:
    """Save one finished turn's prompt KV and return its blocks' ref counts.

    Order of events: the prefill step emits the store job, the turn finishes and
    vLLM frees the request's own block references, then the worker runs the put
    and reports the job done on a later step.
    """
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=BLOCK_SIZE)
    blocks = pool.get_new_blocks(2)  # the request's own reference
    block_ids = [b.block_id for b in blocks]
    sched_conn = _scheduler_connector(pool)
    store = FakeMooncakeStore(put_status)
    worker_conn = _worker_connector(store)

    request = SimpleNamespace(
        request_id=REQ_ID,
        all_token_ids=list(range(PROMPT_LEN)),
        block_hashes=[b"h0", b"h1"],
    )
    sched_conn.connector_scheduler.update_state_after_alloc(request, None, num_external_tokens=0)
    meta = sched_conn.build_connector_meta(_prefill_step(block_ids))
    (save,) = [m for m in meta.requests if m.can_save]

    # The turn ends: on vLLM 0.29 request_finished never delays the free, so the
    # store job's reference is all that keeps these blocks out of the free queue.
    pool.free_blocks(blocks)

    send_thread = worker_conn.connector_worker.kv_send_thread
    send_thread.add_request(save)
    send_thread._handle_request(send_thread.request_queue.get())
    sched_conn.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=worker_conn.build_connector_worker_meta())
    )
    return [pool.blocks[i].ref_cnt for i in block_ids], store


def test_saved_turn_releases_its_blocks():
    ref_cnts, store = _run_turn(PUT_OK)

    assert len(store.put_keys) == 2
    # The store holds the KV, so HBM may evict these blocks.
    assert ref_cnts == [0, 0]


def test_failed_put_keeps_turn_blocks_referenced():
    """#58: a put that fails must not hand the turn's only KV copy to the allocator."""
    ref_cnts, store = _run_turn(TRANSFER_FAIL)

    assert len(store.put_keys) == 2
    # Both puts failed, so the store has nothing for this turn. If HBM evicts
    # these blocks, the next turn recomputes the whole prompt.
    assert ref_cnts == [1, 1]
