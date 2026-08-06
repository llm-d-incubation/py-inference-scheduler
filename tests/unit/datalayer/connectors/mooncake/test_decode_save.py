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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# vLLM is an engine-runtime dependency required by these tests.
pytest.importorskip("vllm")

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)

from py_inference_scheduler.datalayer.connectors.mooncake.decode_save import (
    DecodeKVSavingConnector,
    should_save_decode_request,
)

BLOCK_SIZE = 16

# vLLM 0.21.x trackers hold a flat list[int] of block ids; 0.23+ holds a
# tuple-of-lists (one per KV cache group). Shape the fixture to the installed pin.
_TUPLE_GROUPS = "tuple" in str(
    RequestTracker.__dataclass_fields__["allocated_block_ids"].type
)


def _block_ids(ids):
    return (list(ids),) if _TUPLE_GROUPS else list(ids)


def test_resumed_request_left_to_base():
    assert not should_save_decode_request(
        is_resumed=True, num_computed_tokens=100, prefill_end_tokens=16,
    )


def test_chunked_prefill_left_to_base():
    # num_computed < prefill_end means still prefill; the base saves it, not us.
    assert not should_save_decode_request(
        is_resumed=False, num_computed_tokens=8, prefill_end_tokens=16,
    )


def test_decode_step_is_handled():
    # num_computed >= prefill_end with new blocks means pure decode (base clamps),
    # so we save.
    assert should_save_decode_request(
        is_resumed=False, num_computed_tokens=16, prefill_end_tokens=16,
    )
    assert should_save_decode_request(
        is_resumed=False, num_computed_tokens=64, prefill_end_tokens=16,
    )


def _make_connector(*, save_decode_kv: bool) -> DecodeKVSavingConnector:
    # A stock scheduler half with the minimal attrs build_connector_meta
    sched = object.__new__(MooncakeStoreScheduler)
    sched.client = Mock()
    sched.load_specs = {}
    sched._request_trackers = {}
    sched._preempted_req_ids = set()
    sched._unfinished_requests = {}
    sched._unfinished_request_ids = set()
    sched.kv_role = "kv_both"
    sched._block_size = BLOCK_SIZE
    sched.original_block_size = BLOCK_SIZE  # set by the real __init__ (0.22.0)

    conn = object.__new__(DecodeKVSavingConnector)
    conn.connector_scheduler = sched
    conn.kv_role = "kv_both"
    conn.save_decode_kv = save_decode_kv
    return conn


def _register(conn, req_id, *, token_len, num_saved, prefill_end):
    sched = conn.connector_scheduler
    tracker = RequestTracker(
        req_id=req_id,
        token_len=token_len,
        allocated_block_ids=_block_ids([0, 1]),
        num_saved_tokens=num_saved,
        token_ids=list(range(token_len)),
        prefill_end_tokens=prefill_end,
    )
    sched._request_trackers[req_id] = tracker
    fake_req = SimpleNamespace(
        all_token_ids=list(range(token_len + 64)),
        block_hashes=[b"h0", b"h1", b"h2", b"h3", b"h4"],
    )
    sched._unfinished_requests[req_id] = (fake_req, _block_ids([0, 1]))
    sched._unfinished_request_ids.add(req_id)
    return tracker


def _cached_step(req_id, *, new_block_ids, num_computed_tokens, num_new=1, resumed=False):
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[req_id],
            new_block_ids=[new_block_ids],
            num_computed_tokens=[num_computed_tokens],
            resumed_req_ids={req_id} if resumed else set(),
        ),
        num_scheduled_tokens={req_id: num_new},
    )


def test_decode_step_emits_block_save():
    """A decode step that completes a full block emits exactly one save."""
    conn = _make_connector(save_decode_kv=True)
    # token_len=17 is stale: the stock scheduler advances it only on block alloc
    _register(conn, "r1", token_len=17, num_saved=16, prefill_end=16)
    # 31 computed + 1 scheduled = 32 tokens = two full 16-token blocks
    so = _cached_step("r1", new_block_ids=([2],), num_computed_tokens=31, num_new=1)

    meta = conn.build_connector_meta(so)

    saves = [r for r in meta.requests if r.req_id == "r1" and r.can_save]
    assert len(saves) == 1
    assert saves[0].token_len_chunk == 32  # rounded to the block boundary
    trackers = conn.connector_scheduler._request_trackers
    assert trackers["r1"].num_saved_tokens == 32
    assert trackers["r1"].token_len == 32  # resynced


def test_disabled_emits_no_decode_save():
    # Same stale-decode step but save_decode_kv off: base clamps, override no-op.
    conn = _make_connector(save_decode_kv=False)
    _register(conn, "r1", token_len=17, num_saved=16, prefill_end=16)
    so = _cached_step("r1", new_block_ids=([2],), num_computed_tokens=31, num_new=1)

    meta = conn.build_connector_meta(so)

    assert [r for r in meta.requests if r.req_id == "r1"] == []
    assert conn.connector_scheduler._request_trackers["r1"].num_saved_tokens == 16


def test_chunked_prefill_not_double_processed():
    # num_computed < prefill_end is a chunked-prefill step the base saves. The
    # override must not add a duplicate (exactly one save total).
    conn = _make_connector(save_decode_kv=True)
    _register(conn, "r1", token_len=16, num_saved=0, prefill_end=64)
    so = _cached_step("r1", new_block_ids=([2],), num_computed_tokens=0, num_new=16)

    meta = conn.build_connector_meta(so)

    saves = [r for r in meta.requests if r.req_id == "r1" and r.can_save]
    assert len(saves) == 1  # the base's save, not duplicated by the override


def test_connector_config_resolves_to_this_class():
    """Resolve the connector exactly as vLLM's factory does from the config
    strings, so a module rename fails here instead of at engine boot"""
    import importlib

    from py_inference_scheduler.datalayer.connectors.mooncake.kv import (
        mooncake_engine_kwargs,
    )

    cfg = mooncake_engine_kwargs()["kv_transfer_config"]
    module = importlib.import_module(cfg["kv_connector_module_path"])
    assert getattr(module, cfg["kv_connector"]) is DecodeKVSavingConnector


def test_no_new_blocks_emits_no_decode_save():
    # A decode step that allocated no new block has nothing new to save.
    conn = _make_connector(save_decode_kv=True)
    _register(conn, "r1", token_len=17, num_saved=16, prefill_end=16)
    so = _cached_step("r1", new_block_ids=None, num_computed_tokens=31, num_new=1)

    meta = conn.build_connector_meta(so)

    assert [r for r in meta.requests if r.req_id == "r1"] == []
    assert conn.connector_scheduler._request_trackers["r1"].num_saved_tokens == 16
