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
    SAVE_RETRIES,
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
    """Stands in for MooncakeDistributedStore.

    Put batch n gives every key status n of `batch_results` (the last one repeats);
    an exception there is raised instead.
    """

    def __init__(self, *batch_results: int | Exception) -> None:
        self.batch_results = batch_results
        self.put_batches: list[list[str]] = []
        self.stored: set[str] = set()

    def batch_is_exist(self, keys):
        return [int(key in self.stored) for key in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, replicate_config):
        result = self.batch_results[min(len(self.put_batches), len(self.batch_results) - 1)]
        self.put_batches.append(list(keys))
        if isinstance(result, Exception):
            raise result
        if result >= 0:
            self.stored.update(keys)
        return [result] * len(keys)


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
    worker.kv_send_thread = None
    # The real register_kv_caches needs a live store; of what it does, only creating the
    # send thread matters here.
    worker.register_kv_caches = lambda kv_caches: setattr(worker, "kv_send_thread", send_thread)

    conn = object.__new__(MooncakeStoreConnector)
    conn.connector_worker = worker
    conn.register_kv_caches({})
    return conn


def _prefill_step(block_ids: list[int], num_computed: int, num_new: int) -> SimpleNamespace:
    """Scheduler output for a step that prefills `num_new` more prompt tokens."""
    cached_reqs = SimpleNamespace(
        req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()
    )
    new_reqs = []
    if num_computed == 0:
        new_reqs.append(
            SimpleNamespace(
                req_id=REQ_ID,
                num_computed_tokens=0,
                prompt_token_ids=list(range(PROMPT_LEN)),
                prefill_token_ids=None,
                block_ids=(block_ids,),
            )
        )
    else:
        cached_reqs.req_ids = [REQ_ID]
        cached_reqs.new_block_ids = [None]
        cached_reqs.num_computed_tokens = [num_computed]
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens={REQ_ID: num_new},
        total_num_scheduled_tokens=num_new,
        kv_connector_block_state=KVConnectorBlockState(
            block_ids={REQ_ID: (block_ids,)}, boundary_state_offloads={}
        ),
    )


def _start_turn(store: FakeMooncakeStore, chunks: tuple[int, ...] = (PROMPT_LEN,)):
    """Prefill one turn's prompt in `chunks`, queue each step's store job, then end the turn.

    On vLLM 0.29 request_finished never delays the free, so once the turn ends the
    store jobs' references are all that keep its blocks out of the free queue.
    """
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=BLOCK_SIZE)
    blocks = pool.get_new_blocks(2)  # the request's own reference
    turn = SimpleNamespace(
        pool=pool,
        block_ids=[b.block_id for b in blocks],
        sched_conn=_scheduler_connector(pool),
        worker_conn=_worker_connector(store),
    )
    turn.send_thread = turn.worker_conn.connector_worker.kv_send_thread

    request = SimpleNamespace(
        request_id=REQ_ID,
        all_token_ids=list(range(PROMPT_LEN)),
        block_hashes=[b"h0", b"h1"],
    )
    turn.sched_conn.connector_scheduler.update_state_after_alloc(
        request, None, num_external_tokens=0
    )
    num_computed = 0
    for num_new in chunks:
        meta = turn.sched_conn.build_connector_meta(
            _prefill_step(turn.block_ids, num_computed, num_new)
        )
        for req_meta in meta.requests:
            if req_meta.can_save:
                turn.send_thread.add_request(req_meta)  # as wait_for_save does
        num_computed += num_new

    pool.free_blocks(blocks)  # the turn ends
    return turn


def _run_store_jobs(turn, max_jobs: int = 10) -> list[int]:
    """Run up to `max_jobs` queued store jobs, then report to the scheduler as a step would.

    Returns the ref counts of the turn's blocks.
    """
    queue = turn.send_thread.request_queue
    for _ in range(max_jobs):
        if queue.empty():
            break
        turn.send_thread._handle_request(queue.get())
    turn.sched_conn.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=turn.worker_conn.build_connector_worker_meta())
    )
    return [turn.pool.blocks[i].ref_cnt for i in turn.block_ids]


def test_saved_turn_releases_its_blocks():
    store = FakeMooncakeStore(PUT_OK)
    turn = _start_turn(store)

    # The store holds the KV, so HBM may evict these blocks.
    assert _run_store_jobs(turn) == [0, 0]
    assert len(store.put_batches) == 1
    assert len(store.stored) == 2


def test_failed_put_keeps_turn_blocks_referenced():
    """#58: a put that fails must not hand the turn's only KV copy to the allocator."""
    store = FakeMooncakeStore(TRANSFER_FAIL)
    turn = _start_turn(store)

    # Both puts failed, so the store has nothing for this turn. If HBM evicts
    # these blocks, the next turn recomputes the whole prompt.
    assert _run_store_jobs(turn, max_jobs=1) == [1, 1]
    assert not store.stored


def test_failed_put_is_retried_until_it_lands():
    store = FakeMooncakeStore(TRANSFER_FAIL, PUT_OK)
    turn = _start_turn(store)

    assert _run_store_jobs(turn) == [0, 0]
    assert len(store.put_batches) == 2
    assert len(store.stored) == 2


def test_put_that_raises_is_retried():
    store = FakeMooncakeStore(RuntimeError("transfer engine went away"), PUT_OK)
    turn = _start_turn(store)

    assert _run_store_jobs(turn) == [0, 0]
    assert len(store.put_batches) == 2
    assert len(store.stored) == 2


def test_failed_job_is_left_to_a_newer_job_of_the_request():
    # Under chunked prefill the request's next job is already queued. It resumes from
    # the last saved offset, so it saves the failed chunk too, and it references the
    # same blocks.
    store = FakeMooncakeStore(TRANSFER_FAIL, PUT_OK)
    turn = _start_turn(store, chunks=(BLOCK_SIZE, BLOCK_SIZE))

    # The first chunk's put fails; from here only the newer job holds the blocks.
    assert _run_store_jobs(turn, max_jobs=1) == [1, 1]
    assert _run_store_jobs(turn) == [0, 0]
    assert len(store.put_batches) == 2
    assert len(store.stored) == 2


def test_retries_stop_and_release_the_blocks():
    # A pin that never drops starves HBM, and has_pending_push_work keeps the engine
    # stepping while any pin is held.
    store = FakeMooncakeStore(TRANSFER_FAIL)
    turn = _start_turn(store)

    assert _run_store_jobs(turn) == [0, 0]
    assert turn.send_thread.request_queue.empty()
    assert len(store.put_batches) == 1 + SAVE_RETRIES
    assert not store.stored
